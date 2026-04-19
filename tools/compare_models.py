"""
Compare gpt-4.1-mini (historical scores from Supabase) against gpt-5-mini
run twice (reasoning_effort = minimal and medium) on the SAME prompt.

Pulls N random already-scored jobs from Supabase, re-scores them with both
gpt-5-mini variants, saves a side-by-side comparison to .tmp/, and prints
a short analysis (quality divergences + estimated token costs).

The 4.1-mini result is NOT recomputed — we use what is stored in the DB and
estimate tokens via tiktoken on the exact same prompt (build_scoring_prompt
is shared, so the prompt is byte-identical to what was sent historically).

Usage:
    python tools/compare_models.py            # 20 jobs
    python tools/compare_models.py --n 10     # custom sample size
"""

import os
import sys
import json
import random
import argparse
import logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import tiktoken
from supabase import create_client
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from score_job import score_job, build_scoring_prompt

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# OpenAI list prices (USD per token, 2026-Q1).
# https://openai.com/api/pricing — update if prices change.
PRICING = {
    "gpt-4.1-mini": {"in": 0.40 / 1_000_000, "out": 1.60 / 1_000_000},
    "gpt-5-mini":   {"in": 0.25 / 1_000_000, "out": 2.00 / 1_000_000},
}

SAMPLE_POOL_SIZE = 500  # fetch this many recent jobs, then random.sample from them
DEFAULT_SAMPLE = 20

# tiktoken: gpt-5-mini uses o200k_base; gpt-4.1-mini also uses o200k_base.
_ENC = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def estimate_cost(model: str, prompt_tokens: int, output_tokens: int) -> float:
    p = PRICING[model]
    return prompt_tokens * p["in"] + output_tokens * p["out"]


def get_supabase():
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )


def fetch_profile(sb) -> dict:
    r = sb.table("profile").select("parsed").order("updated_at", desc=True).limit(1).execute()
    if not r.data:
        raise RuntimeError("No profile in Supabase — send resume PDF to the Telegram bot first.")
    return r.data[0]["parsed"]


def fetch_random_scored_jobs(
    sb,
    n: int,
    exclude_ids: set[str] | None = None,
    status: str | None = None,
) -> list[dict]:
    """Pull a pool of recent analysed jobs, random.sample(n) from it.

    Analysed = has a non-null score AND a non-empty match_summary (i.e. the
    LLM actually ran — not an auto-0 for empty description, not a filter-out).

    If `status` is given (e.g. 'sent'), filter by jobs.status.
    """
    q = (
        sb.table("jobs")
        .select("id,source,title,company,url,description,posted_at,"
                "score,match_summary,red_flags,score_breakdown,status,created_at")
        .not_.is_("score", "null")
        .not_.is_("match_summary", "null")
        .neq("match_summary", "")
        .neq("match_summary", "No job description available.")
    )
    if status:
        q = q.eq("status", status)
    res = q.order("created_at", desc=True).limit(SAMPLE_POOL_SIZE).execute()
    excluded = exclude_ids or set()
    pool = [
        j for j in (res.data or [])
        if (j.get("description") or "").strip() and j.get("id") not in excluded
    ]
    if len(pool) < n:
        logger.warning(f"Only {len(pool)} analysed jobs available — using all of them")
        return pool
    return random.sample(pool, n)


def _parse_stored_breakdown(row: dict) -> dict:
    raw = row.get("score_breakdown")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _parse_stored_red_flags(row: dict) -> list:
    raw = row.get("red_flags")
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def estimate_historical_41_cost(row: dict, profile: dict) -> tuple[int, int, float]:
    """Estimate (prompt_tokens, output_tokens, cost_usd) for the historical
    4.1-mini run on this job, using the exact same prompt build function.
    """
    system, user = build_scoring_prompt(row, profile)
    # Chat format adds ~4 tokens of framing per message — close enough for comparison.
    prompt_tokens = count_tokens(system) + count_tokens(user) + 8

    # Reconstruct what the model output looked like (we stored everything):
    reconstructed = {
        "match_summary": row.get("match_summary", ""),
        "red_flags": _parse_stored_red_flags(row),
    }
    bd = _parse_stored_breakdown(row)
    if "block1" in bd:
        reconstructed["block1"] = bd["block1"]
    if "block2" in bd:
        reconstructed["block2"] = bd["block2"]
    output_tokens = count_tokens(json.dumps(reconstructed, ensure_ascii=False))

    return prompt_tokens, output_tokens, estimate_cost(
        "gpt-4.1-mini", prompt_tokens, output_tokens
    )


def run_one(row: dict, profile: dict, effort: str) -> dict:
    """Score one job with gpt-5-mini at the given reasoning_effort level."""
    # score_job mutates the dict — use a shallow copy with only the fields it needs.
    job_input = {
        "title": row.get("title", ""),
        "company": row.get("company", ""),
        "description": row.get("description", "") or "",
    }
    enriched = score_job(
        job_input,
        profile,
        model="gpt-5-mini",
        reasoning_effort=effort,
    )
    usage = enriched.get("_usage", {})
    cost = estimate_cost(
        "gpt-5-mini",
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),  # completion already includes reasoning
    )
    return {
        "score": enriched.get("score"),
        "match_summary": enriched.get("match_summary", ""),
        "red_flags": enriched.get("red_flags", []),
        "score_breakdown": enriched.get("score_breakdown", {}),
        "usage": usage,
        "latency_ms": enriched.get("_latency_ms"),
        "cost_usd": cost,
    }


def score_row(row: dict, profile: dict, variants: list[str]) -> dict:
    """Score one row with the requested 5-mini variants; add 4.1 cost estimate."""
    p41, o41, c41 = estimate_historical_41_cost(row, profile)

    v5_min = None
    v5_med = None
    if "minimal" in variants:
        try:
            v5_min = run_one(row, profile, "minimal")
        except Exception as e:
            logger.error(f"5-mini minimal failed for {row.get('title')!r}: {e}")
            v5_min = {"error": str(e)}
    if "medium" in variants:
        try:
            v5_med = run_one(row, profile, "medium")
        except Exception as e:
            logger.error(f"5-mini medium failed for {row.get('title')!r}: {e}")
            v5_med = {"error": str(e)}

    logger.info(
        f"Done: {row.get('title')[:60]!r:<62} "
        f"| 4.1={row.get('score')} "
        f"| 5min={v5_min.get('score') if v5_min else '-'} "
        f"| 5med={v5_med.get('score') if v5_med else '-'}"
    )

    out = {
        "id": row.get("id"),
        "source": row.get("source"),
        "title": row.get("title"),
        "company": row.get("company"),
        "url": row.get("url"),
        "description_len": len(row.get("description") or ""),
        "v41_mini_historical": {
            "score": row.get("score"),
            "match_summary": row.get("match_summary"),
            "red_flags": _parse_stored_red_flags(row),
            "score_breakdown": _parse_stored_breakdown(row),
            "usage_estimated": {
                "prompt_tokens": p41,
                "completion_tokens": o41,
                "reasoning_tokens": 0,
            },
            "cost_usd_estimated": c41,
        },
    }
    if v5_min is not None:
        out["v5_mini_minimal"] = v5_min
    if v5_med is not None:
        out["v5_mini_medium"] = v5_med
    return out


def _avg(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def summarise(results: list[dict]) -> dict:
    def _valid(r, key):
        v = r.get(key)
        return v is not None and "error" not in v

    ok = [
        r for r in results
        if (("v5_mini_minimal" not in r) or _valid(r, "v5_mini_minimal"))
        and (("v5_mini_medium" not in r) or _valid(r, "v5_mini_medium"))
    ]

    has_min = any("v5_mini_minimal" in r for r in ok)
    has_med = any("v5_mini_medium" in r for r in ok)

    scores_41 = [r["v41_mini_historical"]["score"] for r in ok]
    rf_41 = [len(r["v41_mini_historical"]["red_flags"]) for r in ok]
    cost_41 = sum(r["v41_mini_historical"]["cost_usd_estimated"] for r in ok)

    out: dict = {
        "n": len(ok),
        "errors": len(results) - len(ok),
        "avg_score": {"gpt-4.1-mini": _avg(scores_41)},
        "avg_red_flag_count": {"gpt-4.1-mini": _avg(rf_41)},
        "avg_reasoning_tokens": {},
        "avg_latency_ms": {},
        "total_cost_usd": {"gpt-4.1-mini (estimated)": cost_41},
        "cost_per_job_usd": {
            "gpt-4.1-mini (estimated)": cost_41 / len(ok) if ok else 0.0,
        },
        "score_disagreement_count_vs_41": {},
    }

    def disagree_count(scores_a, scores_b, delta=15):
        return sum(1 for a, b in zip(scores_a, scores_b) if abs(a - b) >= delta)

    if has_min:
        scores = [r["v5_mini_minimal"]["score"] for r in ok]
        rf = [len(r["v5_mini_minimal"]["red_flags"]) for r in ok]
        cost = sum(r["v5_mini_minimal"]["cost_usd"] for r in ok)
        lat = [r["v5_mini_minimal"]["latency_ms"] for r in ok]
        reas = [r["v5_mini_minimal"]["usage"]["reasoning_tokens"] for r in ok]
        out["avg_score"]["gpt-5-mini/minimal"] = _avg(scores)
        out["avg_red_flag_count"]["gpt-5-mini/minimal"] = _avg(rf)
        out["avg_latency_ms"]["gpt-5-mini/minimal"] = _avg(lat)
        out["avg_reasoning_tokens"]["gpt-5-mini/minimal"] = _avg(reas)
        out["total_cost_usd"]["gpt-5-mini/minimal"] = cost
        out["cost_per_job_usd"]["gpt-5-mini/minimal"] = cost / len(ok) if ok else 0.0
        out["score_disagreement_count_vs_41"]["gpt-5-mini/minimal >=15pt delta"] = disagree_count(scores_41, scores)

    if has_med:
        scores = [r["v5_mini_medium"]["score"] for r in ok]
        rf = [len(r["v5_mini_medium"]["red_flags"]) for r in ok]
        cost = sum(r["v5_mini_medium"]["cost_usd"] for r in ok)
        lat = [r["v5_mini_medium"]["latency_ms"] for r in ok]
        reas = [r["v5_mini_medium"]["usage"]["reasoning_tokens"] for r in ok]
        out["avg_score"]["gpt-5-mini/medium"] = _avg(scores)
        out["avg_red_flag_count"]["gpt-5-mini/medium"] = _avg(rf)
        out["avg_latency_ms"]["gpt-5-mini/medium"] = _avg(lat)
        out["avg_reasoning_tokens"]["gpt-5-mini/medium"] = _avg(reas)
        out["total_cost_usd"]["gpt-5-mini/medium"] = cost
        out["cost_per_job_usd"]["gpt-5-mini/medium"] = cost / len(ok) if ok else 0.0
        out["score_disagreement_count_vs_41"]["gpt-5-mini/medium >=15pt delta"] = disagree_count(scores_41, scores)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_SAMPLE, help="sample size")
    ap.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    ap.add_argument("--append", type=str, default=None,
                    help="path to existing comparison JSON; new rows are appended to it (excluding IDs already present)")
    ap.add_argument("--status", type=str, default=None,
                    help="filter sample to jobs with this status (e.g. 'sent')")
    ap.add_argument("--variants", type=str, default="minimal,medium",
                    help="comma-separated gpt-5-mini reasoning levels to run (default: minimal,medium)")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if args.seed is not None:
        random.seed(args.seed)

    sb = get_supabase()
    profile = fetch_profile(sb)
    logger.info("Profile loaded")

    existing_results: list[dict] = []
    exclude_ids: set[str] = set()
    if args.append:
        with open(args.append, encoding="utf-8") as f:
            existing_data = json.load(f)
        existing_results = existing_data.get("results", [])
        exclude_ids = {r.get("id") for r in existing_results if r.get("id")}
        logger.info(f"Loaded {len(existing_results)} existing results from {args.append}")

    rows = fetch_random_scored_jobs(sb, args.n, exclude_ids=exclude_ids, status=args.status)
    logger.info(f"Sampled {len(rows)} new jobs (status filter={args.status!r}, variants={variants})")

    new_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(score_row, row, profile, variants): row for row in rows}
        for fut in futures:
            try:
                new_results.append(fut.result())
            except Exception as e:
                logger.error(f"Row failed completely: {e}")

    results = existing_results + new_results
    summary = summarise(results)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(".tmp", f"model_comparison_{ts}.json")
    os.makedirs(".tmp", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"Saved → {out_path}")

    # Pretty-print summary to console.
    print("\n" + "=" * 72)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nFull side-by-side data:", out_path)


if __name__ == "__main__":
    main()
