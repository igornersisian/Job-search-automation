"""
Daily pipeline orchestrator.

Flow:
  1. Run all scrapers IN PARALLEL:
     LinkedIn, Glassdoor, Indeed, Wellfound, RemoteBoards, ATS (13 platforms)
  2. Phase 1 (fast, sequential): dedup by ID + title+company, junior filter
  3. Phase 2 (PARALLEL): quick_score + enrich + send Telegram — all jobs at once
  4. Send daily summary to Telegram

Score threshold configurable via /threshold bot command.
Custom dealbreakers via /redflags bot command.

Usage:
    python tools/process_jobs.py
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from supabase import create_client, Client
from dotenv import load_dotenv

# Import sibling tools
sys.path.insert(0, os.path.dirname(__file__))
from run_apify_search import run_search as run_linkedin_search
from run_glassdoor_search import run_search as run_glassdoor_search
from run_indeed_search import run_search as run_indeed_search
from run_wellfound_search import run_search as run_wellfound_search
from run_remoteboards_search import run_search as run_remoteboards_search
from run_ats_search import run_search as run_ats_search
from score_job import quick_score, is_excluded_by_title
from notify_telegram import send_job_card, send_daily_summary, send_message

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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


def _normalise_title(title: str) -> str:
    """Normalise a job title for dedup comparison."""
    t = title.lower().strip()
    # Remove common punctuation noise
    t = t.replace("-", " ").replace("–", " ").replace("/", " ")
    # Replace abbreviations with full forms
    words = t.split()
    words = [_TITLE_SYNONYMS.get(w, w) for w in words]
    # Remove filler words
    fillers = {"a", "an", "the", "and", "&", "of", "for", "with", "in", "at", "to", "-", "–", "—"}
    words = [w for w in words if w not in fillers]
    return " ".join(words)


def _normalise_company(company: str) -> str:
    """Normalise a company name for dedup comparison."""
    c = company.lower().strip()
    for suffix in _COMPANY_SUFFIXES:
        if c.endswith(suffix):
            c = c[: -len(suffix)].strip()
            break
    # Remove trailing punctuation
    c = c.rstrip(".,")
    return c


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


DEFAULT_SCORE_THRESHOLD = 70

_supabase: Client | None = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        )
    return _supabase


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


def save_job(job: dict, status: str) -> None:
    """Insert job record into Supabase."""
    row = {
        "id": job.get("id") or job.get("jobId") or job.get("url", "")[:200],
        "source": job.get("source", "linkedin"),
        "title": job.get("title", ""),
        "company": job.get("company", ""),
        "url": job.get("url") or job.get("jobUrl", ""),
        "salary_text": job.get("salary") or job.get("salaryText", ""),
        "is_remote": True,
        "description": job.get("description", "")[:10000],
        "posted_at": job.get("postedAt") or job.get("publishedAt"),
        "score": job.get("score"),
        "match_summary": job.get("match_summary", ""),
        "red_flags": json.dumps(job.get("red_flags", [])),
        "score_breakdown": json.dumps(job.get("score_breakdown", {})),
        "typical_qa": "[]",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        get_supabase().table("jobs").upsert(row).execute()
    except Exception as e:
        if "score_breakdown" in str(e):
            logger.warning("score_breakdown column missing — saving without it")
            row.pop("score_breakdown", None)
            get_supabase().table("jobs").upsert(row).execute()
        else:
            raise


# ---------------------------------------------------------------------------
# Normalise Apify job fields
# ---------------------------------------------------------------------------

def normalise_job(raw: dict) -> dict:
    """Pass through already-normalised fields from source scrapers.
    All source scrapers normalise their output,
    so this just ensures required keys exist with sensible defaults.
    """
    return {
        "id": raw.get("id", ""),
        "title": raw.get("title", ""),
        "company": raw.get("company", ""),
        "url": raw.get("url", ""),
        "salary": raw.get("salary", ""),
        "description": raw.get("description", ""),
        "location": raw.get("location", ""),
        "postedAt": raw.get("postedAt"),
        "source": raw.get("source", "linkedin"),
    }


# ---------------------------------------------------------------------------
# Parallel scraper fetch
# ---------------------------------------------------------------------------

def _fetch_all_sources(keywords: list[str], profile: dict) -> list[dict]:
    """Run all scrapers in parallel, return merged job list."""
    raw_jobs: list[dict] = []

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_linkedin = executor.submit(run_linkedin_search, keywords)
        future_glassdoor = executor.submit(run_glassdoor_search, keywords)
        future_indeed = executor.submit(run_indeed_search, keywords)
        future_wellfound = executor.submit(run_wellfound_search, keywords, profile)
        future_remoteboards = executor.submit(run_remoteboards_search, keywords)
        future_ats = executor.submit(run_ats_search, keywords)

        for name, future in [
            ("LinkedIn", future_linkedin),
            ("Glassdoor", future_glassdoor),
            ("Indeed", future_indeed),
            ("Wellfound", future_wellfound),
            ("RemoteBoards", future_remoteboards),
            ("ATS", future_ats),
        ]:
            try:
                jobs = future.result()
                raw_jobs.extend(jobs)
                logger.info(f"{name}: {len(jobs)} jobs")
            except Exception as e:
                logger.error(f"{name} search failed: {e}")

    return raw_jobs


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
        score, breakdown = quick_score(job, profile)
    except Exception as e:
        logger.error(f"Scoring failed for {job['title']}: {e}")
        return job, "score_error"

    # quick_score now also sets match_summary + red_flags on the job dict
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

def run_pipeline() -> None:
    logger.info("=== Daily job search pipeline started ===")

    # Load profile
    profile = get_profile()
    if not profile:
        logger.error("No resume profile in Supabase. Send PDF to Telegram bot first.")
        return

    # Configurable threshold from profile (set via /threshold bot command)
    threshold = profile.get("score_threshold") or DEFAULT_SCORE_THRESHOLD
    logger.info(f"Score threshold: {threshold}%")

    # Keywords must be set by user via /keywords bot command
    keywords = profile.get("search_keywords")
    if not keywords:
        logger.error("No search keywords set. Use /keywords command in Telegram bot first.")
        send_message("⚠️ Pipeline skipped — no search keywords set.\nUse /keywords to add them.")
        return
    logger.info(f"Search keywords ({len(keywords)}): {keywords}")

    excluded_title_kw = profile.get("excluded_title_keywords") or []
    if excluded_title_kw:
        logger.info(f"Excluded title keywords ({len(excluded_title_kw)}): {excluded_title_kw}")
    else:
        logger.info("No excluded title keywords set — title filter disabled")

    # ── Phase 0: Fetch jobs from all sources in parallel ─────────────
    raw_jobs = _fetch_all_sources(keywords, profile)

    if not raw_jobs:
        logger.error("No jobs fetched from any source.")
        return

    logger.info(f"Total raw jobs: {len(raw_jobs)}")

    # ── Phase 1: Dedup + title exclusion filter (fast, sequential, no API calls) ──
    candidates: list[dict] = []
    skipped_excluded = 0
    dupes_crossrun = 0   # already in Supabase from previous runs
    dupes_local = 0      # same ID appeared multiple times this run
    dupes_fuzzy = 0      # same title+company with similar description

    # Batch-fetch existing IDs from Supabase (1-2 queries instead of N)
    all_raw_ids = [r.get("id", "") for r in raw_jobs if r.get("id")]
    already_seen_ids = fetch_existing_ids(all_raw_ids) if all_raw_ids else set()
    logger.info(f"Batch dedup: {len(already_seen_ids)} already seen out of {len(all_raw_ids)} raw IDs")

    processed_ids: set[str] = set()
    # Smart dedup: group jobs by normalised title+company, then check description similarity
    # Key: "normalised_title|normalised_company" → list of jobs with that key
    fuzzy_groups: dict[str, list[dict]] = {}

    for raw in raw_jobs:
        job = normalise_job(raw)

        if not job["id"] or not job["title"]:
            continue

        # Dedup: Supabase (cross-run, catches jobs from previous days)
        if job["id"] in already_seen_ids:
            dupes_crossrun += 1
            continue

        # Dedup: local by ID (within this run — same job from same source)
        if job["id"] in processed_ids:
            dupes_local += 1
            continue
        processed_ids.add(job["id"])

        # Smart fuzzy dedup: normalise title + company, then compare descriptions
        norm_title = _normalise_title(job["title"])
        norm_company = _normalise_company(job["company"])
        dedup_key = f"{norm_title}|{norm_company}"

        is_duplicate = False
        if dedup_key in fuzzy_groups:
            # Same normalised title+company exists — check if descriptions are similar
            for existing_job in fuzzy_groups[dedup_key]:
                sim = _description_similarity(
                    job.get("description", ""),
                    existing_job.get("description", ""),
                )
                if sim >= DEDUP_SIMILARITY_THRESHOLD:
                    logger.info(
                        f"[DEDUP] '{job['title']}' @ {job['company']} ({job.get('source')}) "
                        f"≈ '{existing_job['title']}' @ {existing_job['company']} ({existing_job.get('source')}) "
                        f"[similarity={sim:.0%}]"
                    )
                    is_duplicate = True
                    break

        if is_duplicate:
            dupes_fuzzy += 1
            continue

        # Not a duplicate — register in fuzzy group
        fuzzy_groups.setdefault(dedup_key, []).append(job)

        # Title exclusion filter
        if is_excluded_by_title(job, excluded_title_kw):
            logger.info(f"[EXCLUDED] {job['title']} @ {job['company']}")
            save_job(job, "filtered_excluded")
            skipped_excluded += 1
            continue

        candidates.append(job)

    skipped_dupe = dupes_crossrun + dupes_local + dupes_fuzzy
    logger.info(
        f"Phase 1 done — {len(candidates)} candidates, "
        f"{skipped_dupe} dupes (prev runs: {dupes_crossrun}, same run: {dupes_local}, fuzzy: {dupes_fuzzy}), "
        f"{skipped_excluded} excluded by title"
    )

    # ── Phase 2: Score + enrich + send IN PARALLEL ───────────────────
    sent = 0
    skipped_score = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_process_single_job, job, profile, threshold): job
            for job in candidates
        }

        for future in as_completed(futures):
            job, status = future.result()

            if status == "sent":
                save_job(job, "sent")
                sent += 1
            elif status == "low_score":
                save_job(job, "low_score")
                skipped_score += 1
            elif status == "notify_failed":
                save_job(job, "notify_failed")
                errors += 1
            elif status == "score_error":
                save_job(job, "score_error")
                errors += 1

    logger.info(
        f"Pipeline done — sent: {sent}, "
        f"low score: {skipped_score}, "
        f"excluded: {skipped_excluded}, "
        f"dupes: {skipped_dupe}"
    )

    send_daily_summary(
        sent=sent,
        skipped_score=skipped_score,
        skipped_excluded=skipped_excluded,
        skipped_dupe=skipped_dupe,
        dupes_crossrun=dupes_crossrun,
        dupes_local=dupes_local,
        dupes_fuzzy=dupes_fuzzy,
        threshold=threshold,
    )


if __name__ == "__main__":
    run_pipeline()
