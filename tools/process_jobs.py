"""
Daily pipeline orchestrator.

Flow:
  1. Run all scrapers IN PARALLEL:
     LinkedIn, Glassdoor, Indeed, Wellfound, RemoteBoards, ATS (13 platforms)
  2. Phase 1 (fast, sequential): dedup by ID + title+company, excluded-title filter
  3. Phase 2 (PARALLEL): score_job + enrich + send Telegram — all jobs at once
  4. Send daily summary to Telegram

Score threshold configurable via /threshold bot command.
Custom dealbreakers via /redflags bot command.

Usage:
    python tools/process_jobs.py
"""

import os
import sys
import json
import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

# Import sibling tools
sys.path.insert(0, os.path.dirname(__file__))
import traceback

from db import get_supabase
from log_setup import get_logger
import apify_client
from apify_client import SourceResult
import run_apify_search
import run_glassdoor_search
import run_indeed_search
import run_wellfound_search
import run_remoteboards_search
import run_ats_search
from score_job import score_job, is_excluded_by_title, cost_from_usage, DEFAULT_SCORE_THRESHOLD
from notify_telegram import send_job_card, send_daily_summary, send_message

# Wellfound runs only in this UTC slot (low yield; see plan). None/other slots skip it.
WELLFOUND_SLOT = "09:00"
# How far back the cross-run fuzzy dedup looks for an already-seen posting.
CROSSRUN_DEDUP_DAYS = 4

load_dotenv()

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Smart deduplication helpers
# ---------------------------------------------------------------------------

# Title abbreviation map for normalisation
_TITLE_SYNONYMS = {
    "sr": "senior", "sr.": "senior",
    "jr": "junior", "jr.": "junior",
    "eng": "engineer", "eng.": "engineer",
    "dev": "developer", "dev.": "developer",
    "mgr": "manager", "mgr.": "manager",
    "dir": "director", "dir.": "director",
    "vp": "vice president",
    "assoc": "associate", "assoc.": "associate",
    "asst": "assistant", "asst.": "assistant",
    "admin": "administrator",
    "ops": "operations",
    "swe": "software engineer",
    "sde": "software development engineer",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "qa": "quality assurance",
    "ui": "user interface",
    "ux": "user experience",
    "fe": "frontend",
    "be": "backend",
}

# Company suffixes to strip
_COMPANY_SUFFIXES = [
    ", inc.", ", inc", " inc.", " inc",
    ", llc", " llc",
    ", ltd.", ", ltd", " ltd.", " ltd",
    ", corp.", ", corp", " corp.", " corp",
    ", co.", ", co", " co.",
    ", gmbh", " gmbh",
    ", s.a.", " s.a.",
    ", plc", " plc",
    ", ag", " ag",
    ", pty", " pty",
]

# Trade descriptors stripped from the company tail so that
# "NineTwoThree AI Studio" (LinkedIn) and "ninetwothree, Inc." (Glassdoor)
# collapse to the same dedup key. Only applied when the remaining stem
# stays ≥ 3 chars and ≥ 1 word — protects single-word names like "AI".
_COMPANY_TRADE_WORDS = {
    "ai", "studio", "studios", "labs", "lab",
    "software", "technologies", "technology", "tech",
    "solutions", "systems", "group", "holdings",
    "digital", "media",
}


def _normalise_title(title: str) -> str:
    """Normalise a job title for dedup comparison."""
    if not title:
        return ""
    t = title.lower().strip()
    # Remove common punctuation noise. Commas and pipes are also stripped so the
    # fingerprint (norm_title|norm_company) is safe inside a PostgREST in.() list.
    t = t.replace("-", " ").replace("–", " ").replace("/", " ").replace(",", " ").replace("|", " ")
    # Replace abbreviations with full forms
    words = t.split()
    words = [_TITLE_SYNONYMS.get(w, w) for w in words]
    # Remove filler words
    fillers = {"a", "an", "the", "and", "&", "of", "for", "with", "in", "at", "to", "-", "–", "—"}
    words = [w for w in words if w not in fillers]
    return " ".join(words)


def _normalise_company(company: str) -> str:
    """Normalise a company name for dedup comparison."""
    if not company:
        return ""
    c = company.lower().strip()
    for suffix in _COMPANY_SUFFIXES:
        if c.endswith(suffix):
            c = c[: -len(suffix)].strip()
            break
    # Remove trailing punctuation; strip commas/pipes so the fingerprint is safe
    # inside a PostgREST in.() list.
    c = c.rstrip(".,").replace(",", " ").replace("|", " ")
    # Strip trade descriptors from the tail ("ninetwothree ai studio" → "ninetwothree").
    # Guard: never collapse to a stem shorter than 3 chars — preserves short real names.
    words = c.split()
    while len(words) > 1 and words[-1] in _COMPANY_TRADE_WORDS:
        stem = " ".join(words[:-1])
        if len(stem) < 3:
            break
        words = words[:-1]
    return " ".join(words)


def _description_similarity(desc_a: str, desc_b: str) -> float:
    """Compute word-level Jaccard similarity between two descriptions.
    Returns float 0.0-1.0.
    """
    if not desc_a or not desc_b:
        return 0.0
    # Use a set of meaningful words (>= 4 chars to skip noise)
    words_a = {w for w in desc_a.lower().split() if len(w) >= 4}
    words_b = {w for w in desc_b.lower().split() if len(w) >= 4}
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


DEDUP_SIMILARITY_THRESHOLD = 0.45
# When two jobs share the same company but have different titles, deduplicate if
# their descriptions are nearly identical — catches template variants like
# "Python AI Trainer" vs "JavaScript AI Trainer" from the same employer.
COMPANY_VARIANT_SIM_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def get_profile() -> dict | None:
    result = get_supabase().table("profile").select("parsed").order("updated_at", desc=True).limit(1).execute()
    if result.data:
        return result.data[0]["parsed"]
    return None


def fetch_existing_ids(job_ids: list[str]) -> set[str]:
    """Batch-check which job IDs already exist in Supabase.
    Queries in chunks to avoid URL length limits.
    """
    existing: set[str] = set()
    chunk_size = 100
    for i in range(0, len(job_ids), chunk_size):
        chunk = job_ids[i : i + chunk_size]
        result = get_supabase().table("jobs").select("id").in_("id", chunk).execute()
        existing.update(row["id"] for row in result.data)
    return existing


def _base_source(source: str) -> str:
    """Collapse 'ats-greenhouse' / 'remoteboards-remoteok' to the base bucket."""
    s = (source or "").lower()
    for b in ("linkedin", "glassdoor", "indeed", "wellfound", "ats", "remoteboards"):
        if s.startswith(b):
            return b
    return s.split("-")[0] or "unknown"


def _fingerprint(job: dict) -> str:
    """Cross-run dedup key: normalised title|company (same as in-run dedup_key)."""
    return f"{_normalise_title(job.get('title', ''))}|{_normalise_company(job.get('company', ''))}"


def _job_row(job: dict, status: str) -> dict:
    """Build a Supabase `jobs` row from a job dict."""
    return {
        "id": job.get("id") or job.get("jobId") or job.get("url", "")[:200],
        "source": job.get("source", "linkedin"),
        "title": job.get("title", ""),
        "company": job.get("company", ""),
        "url": job.get("url") or job.get("jobUrl", ""),
        "salary_text": job.get("salary") or job.get("salaryText", ""),
        "is_remote": job.get("is_remote", True),
        "description": job.get("description", "")[:10000],
        "posted_at": job.get("postedAt") or job.get("publishedAt"),
        "score": job.get("score"),
        "match_summary": job.get("match_summary", ""),
        "red_flags": json.dumps(job.get("red_flags", [])),
        "score_breakdown": json.dumps(job.get("score_breakdown", {})),
        # Reuse the fingerprint computed in Phase 1 if present (avoids recomputing
        # the title/company normalisation for every saved row).
        "fingerprint": job.get("_fingerprint") or _fingerprint(job),
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _upsert_rows(rows: list[dict]) -> None:
    """Upsert a batch of job rows. Schema (incl. fingerprint, score_breakdown) is
    defined in migrations/ and applied once; see migrations/002_*.sql."""
    if not rows:
        return
    get_supabase().table("jobs").upsert(rows).execute()


def save_jobs_batch(rows: list[dict], chunk_size: int = 25) -> None:
    """Batch-upsert job rows in chunks (replaces hundreds of per-job round-trips).

    Smaller chunks + per-chunk retry keep large writes under the client timeout
    on a flaky/remote DB connection. Raises only if a chunk still fails after
    retries (caller treats persistence failure as non-fatal)."""
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                _upsert_rows(chunk)
                last_err = None
                break
            except Exception as e:
                last_err = e
                logger.warning(f"Batch upsert chunk {i // chunk_size} attempt {attempt + 1} failed: {e}")
                time.sleep(2 * (attempt + 1))
        if last_err is not None:
            raise last_err


def save_job(job: dict, status: str) -> None:
    """Single-row convenience wrapper around the batch saver."""
    save_jobs_batch([_job_row(job, status)])


def fetch_recent_fingerprints(fingerprints: list[str], days: int = CROSSRUN_DEDUP_DAYS) -> dict[str, list[str]]:
    """For cross-run fuzzy dedup: map each given fingerprint to descriptions of
    recently-saved jobs that share it. Empty if the `fingerprint` column does
    not exist yet (migration not run) — degrades to exact-id dedup only.
    """
    out: dict[str, list[str]] = {}
    fps = [f for f in set(fingerprints) if f and f != "|"]
    if not fps:
        return out
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    chunk_size = 100
    try:
        for i in range(0, len(fps), chunk_size):
            chunk = fps[i : i + chunk_size]
            res = (
                get_supabase().table("jobs")
                .select("fingerprint,description")
                .in_("fingerprint", chunk)
                .gte("created_at", cutoff)
                .execute()
            )
            for row in res.data:
                fp = row.get("fingerprint")
                if fp:
                    out.setdefault(fp, []).append(row.get("description") or "")
    except Exception as e:
        logger.warning(f"Cross-run fingerprint lookup skipped ({e}) — exact-id dedup only")
        return {}
    return out


def save_pipeline_run(row: dict) -> None:
    """Persist a pipeline_runs audit row. Never raises (observability must not
    break the pipeline); logs if the table is missing."""
    try:
        get_supabase().table("pipeline_runs").insert(row).execute()
    except Exception as e:
        logger.warning(f"Could not write pipeline_runs ({e}) — run migration 001 to enable audit log")


# ---------------------------------------------------------------------------
# Normalise Apify job fields
# ---------------------------------------------------------------------------

def normalise_job(raw: dict) -> dict:
    """Pass through already-normalised fields from source scrapers.
    All source scrapers normalise their output,
    so this just ensures required keys exist with sensible defaults.

    Uses `... or ""` instead of `dict.get(k, "")` because scrapers occasionally
    put explicit None values in their output (e.g. a missing employer name from
    an upstream payload). dict.get returns None in that case, which would then
    blow up downstream on `.lower()` etc.
    """
    return {
        "id": raw.get("id") or "",
        "title": raw.get("title") or "",
        "company": raw.get("company") or "",
        "url": raw.get("url") or "",
        "salary": raw.get("salary") or "",
        "description": raw.get("description") or "",
        "location": raw.get("location") or "",
        "postedAt": raw.get("postedAt"),
        "is_remote": raw.get("is_remote", True),
        "source": raw.get("source") or "linkedin",
    }


# ---------------------------------------------------------------------------
# Parallel scraper fetch
# ---------------------------------------------------------------------------

def _fetch_all_sources(
    keywords: list[str], profile: dict, lookback: int, run_wellfound: bool,
) -> tuple[list[dict], dict[str, SourceResult]]:
    """Run all scrapers and return (merged job list, per-source SourceResults).

    The lightweight scrapers run in parallel first. ATS runs after them so it
    doesn't compete for the 8192 MB per-account memory cap (ATS alone needs
    4096 MB; the others together ~5760 MB → all simultaneous would exceed it).
    Each scraper returns a SourceResult (never raises); the runner handles
    retries/cost/cap/errors uniformly.
    """
    results: dict[str, SourceResult] = {}

    def _safe(name, fn):
        try:
            return fn()
        except Exception as e:  # defensive — runner shouldn't raise
            logger.error(f"{name} unexpected error: {e}")
            return SourceResult(source=name, error=f"unexpected: {e}")

    # Phase 1: lightweight scrapers in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            "linkedin": executor.submit(_safe, "linkedin", lambda: run_apify_search.fetch(keywords, lookback=lookback)),
            "glassdoor": executor.submit(_safe, "glassdoor", lambda: run_glassdoor_search.fetch(keywords, lookback=lookback)),
            "indeed": executor.submit(_safe, "indeed", lambda: run_indeed_search.fetch(keywords, lookback=lookback)),
            "remoteboards": executor.submit(_safe, "remoteboards", lambda: run_remoteboards_search.fetch(keywords, lookback=lookback, profile=profile)),
        }
        if run_wellfound:
            futures["wellfound"] = executor.submit(_safe, "wellfound", lambda: run_wellfound_search.fetch(keywords, lookback=lookback, profile=profile))
        for name, fut in futures.items():
            results[name] = fut.result()

    if not run_wellfound:
        logger.info(f"Wellfound skipped this slot (runs only in {WELLFOUND_SLOT} UTC)")

    # Phase 2: ATS after the others (Apify memory freed)
    results["ats"] = _safe("ats", lambda: run_ats_search.fetch(keywords, lookback=lookback))

    raw_jobs: list[dict] = []
    for name, r in results.items():
        raw_jobs.extend(r.items)
        logger.info(f"{name}: {len(r.items)} jobs, ${r.cost_usd:.4f}, capped={r.capped}, error={r.error}")

    return raw_jobs, results


# ---------------------------------------------------------------------------
# Single-job processor (runs inside thread pool)
# ---------------------------------------------------------------------------

def _process_single_job(job: dict, profile: dict, threshold: int) -> tuple[dict, str]:
    """Score + analyse + send one job. Returns (job, status_label).

    Single LLM call: model writes match_summary + red_flags first (chain-of-thought),
    then assigns sub-scores consistent with its analysis.
    Runs in a worker thread — no shared mutable state.
    """
    try:
        score_job(job, profile)  # mutates job in place: score, match_summary, red_flags, _usage
    except Exception as e:
        logger.error(f"Scoring failed for {job['title']}: {e}")
        return job, "score_error"

    score = job.get("score", 0)
    logger.info(f"[{score}/100] {job['title']} @ {job['company']} ({job['source']})")

    if score < threshold:
        return job, "low_score"

    # Send to Telegram
    if send_job_card(job):
        return job, "sent"
    else:
        return job, "notify_failed"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(lookback: int | None = None, slot: str | None = None) -> None:
    """Run the full pipeline for one slot.

    lookback: search window in seconds (default from LOOKBACK_SECONDS env, ~7h).
    slot: '03:00'/'09:00'/... — controls low-yield Wellfound (runs only in
          WELLFOUND_SLOT). None = manual run (Wellfound included).

    Every outcome — including a fatal crash — is persisted to `pipeline_runs`
    in Supabase via the `finally` block, so errors are always queryable.
    """
    started_at = datetime.now(timezone.utc)
    if lookback is None:
        try:
            lookback = int(os.environ.get("LOOKBACK_SECONDS", "25200"))  # 7h default
        except ValueError:
            lookback = 25200
    run_wellfound = slot in (None, WELLFOUND_SLOT)
    logger.info(f"=== Pipeline started (slot={slot}, lookback={lookback}s, wellfound={run_wellfound}) ===")

    per_source: dict[str, dict] = {}
    errors_log: list[dict] = []
    totals: dict = {}
    keywords_used: list | None = None
    ok = True
    # OpenAI scoring spend — aggregated across all scored candidates this run.
    openai_usage = {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0,
                    "completion_tokens": 0, "cost_usd": 0.0}

    try:
        # Load profile
        try:
            profile = get_profile()
        except Exception as e:
            errors_log.append({"source": "supabase", "stage": "load_profile", "message": str(e)[:500]})
            ok = False
            msg = f"⚠️ Pipeline failed — cannot reach Supabase: {e}\n\nResume the DB if paused."
            logger.error(msg)
            try:
                send_message(msg, parse_mode=None)
            except Exception:
                pass
            raise
        if not profile:
            logger.error("No resume profile in Supabase. Send PDF to Telegram bot first.")
            errors_log.append({"source": "pipeline", "stage": "profile", "message": "no profile"})
            ok = False
            return

        threshold = profile.get("score_threshold") or DEFAULT_SCORE_THRESHOLD
        logger.info(f"Score threshold: {threshold}%")

        keywords = profile.get("search_keywords")
        if not keywords:
            logger.error("No search keywords set. Use /keywords in the bot first.")
            errors_log.append({"source": "pipeline", "stage": "keywords", "message": "no keywords"})
            ok = False
            send_message("⚠️ Pipeline skipped — no search keywords set.\nUse /keywords to add them.")
            return
        keywords_used = keywords
        logger.info(f"Search keywords ({len(keywords)}): {keywords}")

        excluded_title_kw = profile.get("excluded_title_keywords") or []
        logger.info(
            f"Excluded title keywords ({len(excluded_title_kw)}): {excluded_title_kw}"
            if excluded_title_kw else "No excluded title keywords — title filter disabled"
        )

        # ── Phase 0: fetch from all sources ──────────────────────────
        raw_jobs, results = _fetch_all_sources(keywords, profile, lookback, run_wellfound)

        for name, r in results.items():
            per_source[name] = {
                "fetched": r.fetched, "new": 0, "sent": 0,
                "cost_usd": round(r.cost_usd, 4), "capped": r.capped,
                "error": r.error, "ms": r.ms, "attempts": r.attempts,
            }
            if r.error:
                errors_log.append({"source": name, "stage": "fetch", "message": r.error[:500]})
        total_cost = round(sum(p["cost_usd"] for p in per_source.values()), 4)

        if not raw_jobs:
            logger.error("No jobs fetched from any source.")
            ok = len(errors_log) == 0
            totals = {"fetched": 0, "new": 0, "sent": 0, "cost_usd": total_cost}
            msg = "⚠️ Pipeline finished — 0 jobs fetched."
            if errors_log:
                msg += "\n\nErrors:\n" + "\n".join(f"• {e['source']}: {e['message'][:200]}" for e in errors_log)
            send_message(msg, parse_mode=None)
            return

        logger.info(f"Total raw jobs: {len(raw_jobs)}")

        # ── Phase 1: dedup + title filter (no API calls) ─────────────
        candidates: list[dict] = []
        rows_to_save: list[dict] = []
        skipped_excluded = 0
        dupes_crossrun = 0        # exact id already in Supabase from previous runs
        dupes_crossrun_fuzzy = 0  # same fingerprint seen in DB in last N days
        dupes_local = 0           # same id twice this run
        dupes_fuzzy = 0           # same title+company, similar description (this run)

        # Normalise every raw job once, and derive the dedup fingerprint
        # (norm_title|norm_company) once too — both were previously recomputed
        # 2–4× per job across the fingerprint pass and the loop below.
        jobs = [normalise_job(r) for r in raw_jobs]
        norm_keys = [
            (_normalise_title(j["title"]), _normalise_company(j["company"]))
            for j in jobs
        ]
        all_fps = [f"{t}|{c}" for t, c in norm_keys]

        all_raw_ids = [j["id"] for j in jobs if j["id"]]
        already_seen_ids = fetch_existing_ids(all_raw_ids) if all_raw_ids else set()
        # Cross-run fuzzy: pull recent fingerprints once (degrades gracefully if column missing)
        recent_fp_map = fetch_recent_fingerprints(all_fps)
        logger.info(
            f"Batch dedup: {len(already_seen_ids)} exact-id seen / {len(all_raw_ids)} ids; "
            f"{len(recent_fp_map)} fingerprints seen in last {CROSSRUN_DEDUP_DAYS}d"
        )

        processed_ids: set[str] = set()
        fuzzy_groups: dict[str, list[dict]] = {}
        company_jobs: dict[str, list[dict]] = {}

        for job, (norm_title, norm_company), dedup_key in zip(jobs, norm_keys, all_fps):
            if not job["id"] or not job["title"]:
                continue

            if job["id"] in already_seen_ids:
                dupes_crossrun += 1
                continue
            if job["id"] in processed_ids:
                dupes_local += 1
                continue
            processed_ids.add(job["id"])

            is_duplicate = False

            # Cross-run fuzzy: same fingerprint seen recently in the DB
            if dedup_key in recent_fp_map:
                for desc in recent_fp_map[dedup_key]:
                    if not desc or not job.get("description"):
                        is_duplicate = True
                        break
                    if _description_similarity(job.get("description", ""), desc) >= DEDUP_SIMILARITY_THRESHOLD:
                        is_duplicate = True
                        break
                if is_duplicate:
                    dupes_crossrun_fuzzy += 1

            # Same-run: same title+company with similar description
            if not is_duplicate and dedup_key in fuzzy_groups:
                for existing_job in fuzzy_groups[dedup_key]:
                    if _description_similarity(job.get("description", ""), existing_job.get("description", "")) >= DEDUP_SIMILARITY_THRESHOLD:
                        logger.info(f"[DEDUP] '{job['title']}' @ {job['company']} ≈ '{existing_job['title']}'")
                        is_duplicate = True
                        dupes_fuzzy += 1
                        break

            # Same-run: same company, near-identical description (template variants)
            if not is_duplicate and norm_company in company_jobs:
                for existing_job in company_jobs[norm_company]:
                    if _description_similarity(job.get("description", ""), existing_job.get("description", "")) >= COMPANY_VARIANT_SIM_THRESHOLD:
                        logger.info(f"[DEDUP-VARIANT] '{job['title']}' @ {job['company']} ≈ '{existing_job['title']}'")
                        is_duplicate = True
                        dupes_fuzzy += 1
                        break

            if is_duplicate:
                continue

            fuzzy_groups.setdefault(dedup_key, []).append(job)
            company_jobs.setdefault(norm_company, []).append(job)
            job["_fingerprint"] = dedup_key  # cache for _job_row (see I2)

            # genuinely new posting
            base = _base_source(job["source"])
            if base in per_source:
                per_source[base]["new"] += 1

            if is_excluded_by_title(job, excluded_title_kw):
                logger.info(f"[EXCLUDED] {job['title']} @ {job['company']}")
                rows_to_save.append(_job_row(job, "filtered_excluded"))
                skipped_excluded += 1
                continue

            candidates.append(job)

        skipped_dupe = dupes_crossrun + dupes_crossrun_fuzzy + dupes_local + dupes_fuzzy
        logger.info(
            f"Phase 1 done — {len(candidates)} candidates, {skipped_dupe} dupes "
            f"(id-prev: {dupes_crossrun}, fp-prev: {dupes_crossrun_fuzzy}, same-run id: {dupes_local}, "
            f"same-run fuzzy: {dupes_fuzzy}), {skipped_excluded} excluded"
        )

        # ── Phase 2: score + enrich + send (parallel) ────────────────
        sent = 0
        skipped_score = 0
        errors = 0

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(_process_single_job, job, profile, threshold): job for job in candidates}
            for future in as_completed(futures):
                job, status = future.result()
                rows_to_save.append(_job_row(job, status))
                u = job.get("_usage")
                if u:
                    openai_usage["calls"] += 1
                    openai_usage["prompt_tokens"] += u.get("prompt_tokens", 0) or 0
                    openai_usage["cached_tokens"] += u.get("cached_tokens", 0) or 0
                    openai_usage["completion_tokens"] += u.get("completion_tokens", 0) or 0
                    openai_usage["cost_usd"] += cost_from_usage(u)
                base = _base_source(job["source"])
                if status == "sent":
                    sent += 1
                    if base in per_source:
                        per_source[base]["sent"] += 1
                elif status == "low_score":
                    skipped_score += 1
                elif status in ("notify_failed", "score_error"):
                    errors += 1
                    errors_log.append({"source": base, "stage": status, "message": f"{job.get('title','')[:80]}"})

        # batch-save everything at once (was hundreds of per-job upserts).
        # Persistence failure is non-fatal: cards were already sent, and the run
        # outcome is still recorded in pipeline_runs below.
        try:
            save_jobs_batch(rows_to_save)
        except Exception as e:
            logger.error(f"Batch save failed (continuing): {e}")
            errors_log.append({"source": "supabase", "stage": "save_jobs", "message": str(e)[:300]})
            ok = False

        openai_cost = round(openai_usage["cost_usd"], 4)
        totals = {
            "fetched": sum(p["fetched"] for p in per_source.values()),
            "new": sum(p["new"] for p in per_source.values()),
            "sent": sent,
            "cost_usd": total_cost,                       # Apify only (kept for back-compat)
            "apify_cost_usd": total_cost,
            "openai_cost_usd": openai_cost,
            "openai_calls": openai_usage["calls"],
            "openai_tokens": {
                "prompt": openai_usage["prompt_tokens"],
                "cached": openai_usage["cached_tokens"],
                "completion": openai_usage["completion_tokens"],
            },
            "total_cost_usd": round(total_cost + openai_cost, 4),
        }
        logger.info(
            f"Pipeline done — sent: {sent}, low score: {skipped_score}, "
            f"excluded: {skipped_excluded}, dupes: {skipped_dupe}, "
            f"Apify ${total_cost:.2f} + OpenAI ${openai_cost:.2f} "
            f"({openai_usage['calls']} calls) = ${total_cost + openai_cost:.2f}"
        )

        source_errors = {name: p["error"] for name, p in per_source.items() if p["error"]}
        send_daily_summary(
            sent=sent,
            skipped_score=skipped_score,
            skipped_excluded=skipped_excluded,
            skipped_dupe=skipped_dupe,
            dupes_crossrun=dupes_crossrun + dupes_crossrun_fuzzy,
            dupes_local=dupes_local,
            dupes_fuzzy=dupes_fuzzy,
            threshold=threshold,
            source_errors=source_errors,
            per_source=per_source,
            total_cost=total_cost,
            openai_cost=openai_cost,
            openai_calls=openai_usage["calls"],
        )

    except Exception:
        tb = traceback.format_exc()
        errors_log.append({"source": "pipeline", "stage": "fatal", "message": tb[:2000]})
        ok = False
        logger.error(f"Pipeline fatal:\n{tb}")
        raise
    finally:
        # Always persist the run — this is the always-queryable audit log.
        save_pipeline_run({
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "slot_utc": slot,
            "keywords": keywords_used,
            "per_source": per_source,
            "totals": totals,
            "errors": errors_log,
            "ok": ok,
        })


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=None, help="search window seconds")
    ap.add_argument("--slot", default=None, help="UTC slot label, e.g. 09:00")
    args = ap.parse_args()
    run_pipeline(lookback=args.lookback, slot=args.slot)
