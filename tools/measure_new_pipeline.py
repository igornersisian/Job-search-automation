"""
Measure real cost + latency of the new scoring pipeline on 20 random jobs.

Runs score_job() with production defaults (gpt-5-mini + minimal + flex +
instructions-based profile + prompt caching). Captures usage data, computes
actual cost using public OpenAI pricing (standard and flex), and compares
against the 4.1-mini baseline from the 50-job historical run.

Usage:
    python tools/measure_new_pipeline.py --n 20 --seed 42
"""

import argparse
import io
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.score_job import score_job  # noqa: E402

load_dotenv()

# OpenAI public pricing (per 1M tokens), as of 2026-04
# Cached input: 90% discount on input; flex tier: 50% off everything.
PRICING = {
    "gpt-5-mini": {
        "standard": {"in": 0.25 / 1_000_000, "cached_in": 0.025 / 1_000_000, "out": 2.00 / 1_000_000},
        "flex":     {"in": 0.125 / 1_000_000, "cached_in": 0.0125 / 1_000_000, "out": 1.00 / 1_000_000},
    },
}

# 4.1-mini baseline from the 50-job historical run (see .tmp/model_comparison_*.json)
BASELINE_41_MINI_PER_JOB = 0.00199


def compute_cost(usage: dict, model: str, tier: str) -> float:
    p = PRICING[model][tier]
    non_cached = max(usage["prompt_tokens"] - usage["cached_tokens"], 0)
    cached = usage["cached_tokens"]
    out = usage["completion_tokens"]
    return non_cached * p["in"] + cached * p["cached_in"] + out * p["out"]


def fetch_random_jobs(sb, n: int, seed: int) -> list[dict]:
    """Pull random jobs that have real descriptions (so scoring actually runs)."""
    data = (
        sb.table("jobs")
          .select("id,title,company,description,status")
          .not_.is_("description", "null")
          .execute()
          .data
    )
    # filter out too-short descriptions (would trigger the short-circuit path)
    data = [d for d in data if d.get("description") and len(d["description"]) >= 200]
    random.seed(seed)
    return random.sample(data, min(n, len(data)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sb = create_client(
        os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    )
    profile = (
        sb.table("profile").select("parsed")
          .order("updated_at", desc=True).limit(1).execute()
          .data[0]["parsed"]
    )

    rows = fetch_random_jobs(sb, args.n, args.seed)
    print(f"Scoring {len(rows)} jobs with gpt-5-mini/minimal/flex + prompt cache...")

    results = []
    t0_total = time.perf_counter()
    for i, row in enumerate(rows, 1):
        job = {
            "title": row["title"],
            "company": row["company"],
            "description": row["description"],
        }
        try:
            scored = score_job(job, profile)
            usage = scored["_usage"]
            cost_flex = compute_cost(usage, "gpt-5-mini", "flex")
            cost_std = compute_cost(usage, "gpt-5-mini", "standard")
            results.append({
                "id": row["id"],
                "title": row["title"],
                "company": row["company"],
                "score": scored["score"],
                "red_flags": scored["red_flags"],
                "usage": usage,
                "latency_ms": scored["_latency_ms"],
                "cost_flex": cost_flex,
                "cost_std": cost_std,
            })
            print(
                f"  [{i:>2}/{len(rows)}] score={scored['score']:>3}  "
                f"in={usage['prompt_tokens']:>5}  cached={usage['cached_tokens']:>5}  "
                f"out={usage['completion_tokens']:>4}  "
                f"${cost_flex*1000:.4f}/k (flex)  {scored['_latency_ms']:>5}ms  "
                f"— {row['title'][:45]}"
            )
        except Exception as e:
            print(f"  [{i:>2}/{len(rows)}] ERROR: {e}  — {row['title'][:45]}")
            results.append({
                "id": row["id"],
                "title": row["title"],
                "error": str(e),
            })

    wall_s = time.perf_counter() - t0_total

    ok = [r for r in results if "error" not in r]
    total_flex = sum(r["cost_flex"] for r in ok)
    total_std = sum(r["cost_std"] for r in ok)
    total_in = sum(r["usage"]["prompt_tokens"] for r in ok)
    total_cached = sum(r["usage"]["cached_tokens"] for r in ok)
    total_out = sum(r["usage"]["completion_tokens"] for r in ok)
    avg_latency = sum(r["latency_ms"] for r in ok) / len(ok) if ok else 0
    cache_ratio = total_cached / total_in if total_in else 0

    per_job_flex = total_flex / len(ok) if ok else 0
    per_job_std = total_std / len(ok) if ok else 0
    baseline = BASELINE_41_MINI_PER_JOB
    savings_flex_pct = (1 - per_job_flex / baseline) * 100 if baseline else 0
    ratio_flex = baseline / per_job_flex if per_job_flex else 0

    print("\n" + "=" * 70)
    print(f"Summary (n={len(ok)}, wall={wall_s:.1f}s)")
    print("=" * 70)
    print(f"Tokens total:   in={total_in:,}   cached={total_cached:,} ({cache_ratio*100:.1f}%)   out={total_out:,}")
    print(f"Avg latency:    {avg_latency:.0f}ms per job")
    print()
    print(f"Cost per job (gpt-5-mini + cache):")
    print(f"  standard tier: ${per_job_std:.6f}")
    print(f"  flex tier:     ${per_job_flex:.6f}")
    print()
    print(f"Baseline (gpt-4.1-mini from 50-job historical run): ${baseline:.6f}/job")
    print(f"Savings:  flex = {savings_flex_pct:.1f}%   ({ratio_flex:.2f}x cheaper)")
    print()
    print(f"Total for this run: ${total_flex:.4f} (flex) vs ${total_std:.4f} (std)")
    print(f"Implied cost per 1k jobs: ${per_job_flex*1000:.2f} (flex) vs ${baseline*1000:.2f} (baseline)")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / ".tmp" / f"new_pipeline_measure_{ts}.json"
    out_path.write_text(
        json.dumps(
            {
                "n": len(ok),
                "seed": args.seed,
                "wall_seconds": wall_s,
                "totals": {
                    "prompt_tokens": total_in,
                    "cached_tokens": total_cached,
                    "completion_tokens": total_out,
                    "cost_flex_usd": total_flex,
                    "cost_std_usd": total_std,
                },
                "per_job": {
                    "cost_flex_usd": per_job_flex,
                    "cost_std_usd": per_job_std,
                    "baseline_41_mini_usd": baseline,
                    "savings_vs_baseline_pct_flex": savings_flex_pct,
                    "ratio_flex": ratio_flex,
                },
                "cache_hit_ratio": cache_ratio,
                "avg_latency_ms": avg_latency,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
