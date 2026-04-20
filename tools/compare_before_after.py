"""
Before/after scoring comparison on the same 50 jobs.

Runs each job through TWO pipelines on the same model (gpt-5-mini):

  BEFORE (semi-naive, mimics pre-optimization code):
    Chat Completions, system=rules, user=PROFILE+JOB,
    no service_tier, no prompt_cache_key.

  AFTER  (production, post-optimization):
    Responses API, instructions=rules+PROFILE, input=JOB,
    service_tier=flex, prompt_cache_key=score_job_v1.

Both use reasoning_effort="minimal" so only the wrapper differs, not the
model's inner effort.

Logs per-job: tokens (in/cached/out/reasoning), latency, score, red_flags,
service_tier. Computes cost from published pricing. Writes full JSON to
.tmp/ and prints a terminal summary.

Dashboard cross-check: BEFORE cost shows up in the `gpt-5-mini` rows,
AFTER cost shows up in the `flex | gpt-5-mini` rows — different buckets,
so both runs can happen back-to-back and the dashboard separates them.

Usage:
    python tools/compare_before_after.py --n 50 --seed 42
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
from tools.score_job_semi_naive import score_job_semi_naive  # noqa: E402

load_dotenv()

# Use the dedicated test key so dashboard spend from this run is isolated.
if os.environ.get("OPENAI_API_KEY_TEST"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_TEST"]

# OpenAI public pricing (per token), as of 2026-04
PRICING = {
    "gpt-5-mini": {
        "standard": {"in": 0.25 / 1_000_000, "cached_in": 0.025 / 1_000_000, "out": 2.00 / 1_000_000},
        "flex":     {"in": 0.125 / 1_000_000, "cached_in": 0.0125 / 1_000_000, "out": 1.00 / 1_000_000},
    },
}


def compute_cost(usage: dict, model: str, tier: str) -> float:
    p = PRICING[model][tier]
    non_cached = max(usage["prompt_tokens"] - usage["cached_tokens"], 0)
    cached = usage["cached_tokens"]
    out = usage["completion_tokens"]
    return non_cached * p["in"] + cached * p["cached_in"] + out * p["out"]


def fetch_jobs(sb, n: int, seed: int) -> list[dict]:
    data = (
        sb.table("jobs")
          .select("id,title,company,description")
          .not_.is_("description", "null")
          .execute()
          .data
    )
    data = [d for d in data if d.get("description") and len(d["description"]) >= 200]
    random.seed(seed)
    return random.sample(data, min(n, len(data)))


def run_pipeline(name: str, jobs: list[dict], profile: dict, scorer, tier_for_cost: str):
    """Run `scorer(job, profile)` over all jobs, collecting usage + cost."""
    print(f"\n{'=' * 70}")
    print(f"Running {name} pipeline ({len(jobs)} jobs) ...")
    print("=" * 70)
    out = []
    t0 = time.perf_counter()
    for i, row in enumerate(jobs, 1):
        job = {
            "title": row["title"],
            "company": row["company"],
            "description": row["description"],
        }
        try:
            scored = scorer(job, profile)
            usage = scored["_usage"]
            cost = compute_cost(usage, "gpt-5-mini", tier_for_cost)
            out.append({
                "id": row["id"],
                "title": row["title"],
                "company": row["company"],
                "score": scored["score"],
                "red_flags": scored["red_flags"],
                "usage": usage,
                "latency_ms": scored["_latency_ms"],
                "cost_usd": cost,
            })
            print(
                f"  [{i:>2}/{len(jobs)}] score={scored['score']:>3}  "
                f"in={usage['prompt_tokens']:>5}  cached={usage['cached_tokens']:>5}  "
                f"out={usage['completion_tokens']:>4}  "
                f"tier={str(usage['service_tier']):>8}  "
                f"${cost*1000:.4f}/k  {scored['_latency_ms']:>5}ms  "
                f"— {row['title'][:40]}"
            )
        except Exception as e:
            print(f"  [{i:>2}/{len(jobs)}] ERROR: {e}  — {row['title'][:40]}")
            out.append({"id": row["id"], "title": row["title"], "error": str(e)})
    wall_s = time.perf_counter() - t0
    return out, wall_s


def summarize(name: str, results: list[dict], wall_s: float):
    ok = [r for r in results if "error" not in r]
    if not ok:
        print(f"\n[{name}] no successful runs")
        return {}
    total_in = sum(r["usage"]["prompt_tokens"] for r in ok)
    total_cached = sum(r["usage"]["cached_tokens"] for r in ok)
    total_out = sum(r["usage"]["completion_tokens"] for r in ok)
    total_reasoning = sum(r["usage"]["reasoning_tokens"] for r in ok)
    total_cost = sum(r["cost_usd"] for r in ok)
    avg_latency = sum(r["latency_ms"] for r in ok) / len(ok)
    cache_hit = total_cached / total_in if total_in else 0
    per_job = total_cost / len(ok)
    tiers = {r["usage"]["service_tier"] for r in ok}
    return {
        "n": len(ok),
        "wall_s": wall_s,
        "tokens_in": total_in,
        "tokens_cached": total_cached,
        "tokens_out": total_out,
        "tokens_reasoning": total_reasoning,
        "cache_hit_ratio": cache_hit,
        "avg_latency_ms": avg_latency,
        "cost_total_usd": total_cost,
        "cost_per_job_usd": per_job,
        "service_tiers_seen": sorted(t or "none" for t in tiers),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
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
    jobs = fetch_jobs(sb, args.n, args.seed)
    print(f"Fetched {len(jobs)} jobs (seed={args.seed}). Same set is used in both runs.")

    # BEFORE: semi-naive (Chat Completions, standard tier bucket on dashboard)
    before_results, before_wall = run_pipeline(
        "BEFORE (semi-naive)", jobs, profile, score_job_semi_naive, tier_for_cost="standard",
    )

    # AFTER: production (Responses + flex + cache_key, flex bucket on dashboard)
    after_results, after_wall = run_pipeline(
        "AFTER (optimized)", jobs, profile, score_job, tier_for_cost="flex",
    )

    before_sum = summarize("BEFORE", before_results, before_wall)
    after_sum = summarize("AFTER", after_results, after_wall)

    ratio = before_sum.get("cost_per_job_usd", 0) / after_sum.get("cost_per_job_usd", 1) if after_sum.get("cost_per_job_usd") else 0
    savings_pct = (1 - after_sum.get("cost_per_job_usd", 0) / before_sum.get("cost_per_job_usd", 1)) * 100 if before_sum.get("cost_per_job_usd") else 0

    # Score agreement: how often do the two pipelines give the same score?
    by_id_after = {r["id"]: r for r in after_results if "error" not in r}
    matched = [(b, by_id_after[b["id"]]) for b in before_results
               if "error" not in b and b["id"] in by_id_after]
    exact = sum(1 for b, a in matched if b["score"] == a["score"])
    within_5 = sum(1 for b, a in matched if abs(b["score"] - a["score"]) <= 5)
    within_10 = sum(1 for b, a in matched if abs(b["score"] - a["score"]) <= 10)
    score_diffs = [a["score"] - b["score"] for b, a in matched]
    avg_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0
    mad = sum(abs(d) for d in score_diffs) / len(score_diffs) if score_diffs else 0

    print("\n" + "=" * 70)
    print("BEFORE vs AFTER (gpt-5-mini, reasoning_effort=minimal)")
    print("=" * 70)
    print(f"{'metric':<32}{'BEFORE':>18}{'AFTER':>18}")
    for k in ("n", "wall_s", "tokens_in", "tokens_cached", "tokens_out",
              "tokens_reasoning", "cache_hit_ratio", "avg_latency_ms",
              "cost_total_usd", "cost_per_job_usd"):
        bv = before_sum.get(k)
        av = after_sum.get(k)
        if isinstance(bv, float) and "ratio" in k:
            bv_s, av_s = f"{bv*100:.1f}%", f"{av*100:.1f}%"
        elif isinstance(bv, float):
            bv_s, av_s = f"{bv:.6f}" if "cost" in k else f"{bv:.1f}", f"{av:.6f}" if "cost" in k else f"{av:.1f}"
        else:
            bv_s, av_s = f"{bv:,}" if isinstance(bv, int) else str(bv), f"{av:,}" if isinstance(av, int) else str(av)
        print(f"{k:<32}{bv_s:>18}{av_s:>18}")

    print(f"\nservice_tiers_seen  BEFORE={before_sum.get('service_tiers_seen')}   AFTER={after_sum.get('service_tiers_seen')}")
    print(f"\nRatio BEFORE/AFTER cost per job: {ratio:.2f}x cheaper")
    print(f"Savings: {savings_pct:.1f}%")
    print(f"\nImplied cost per 1k jobs:  BEFORE=${before_sum.get('cost_per_job_usd', 0)*1000:.2f}  AFTER=${after_sum.get('cost_per_job_usd', 0)*1000:.2f}")

    print(f"\nScore agreement over {len(matched)} matched jobs:")
    print(f"  exact match:    {exact}/{len(matched)} ({exact/max(len(matched),1)*100:.0f}%)")
    print(f"  within ±5:      {within_5}/{len(matched)} ({within_5/max(len(matched),1)*100:.0f}%)")
    print(f"  within ±10:     {within_10}/{len(matched)} ({within_10/max(len(matched),1)*100:.0f}%)")
    print(f"  avg(after-before) = {avg_diff:+.2f}")
    print(f"  mean abs diff     = {mad:.2f}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / ".tmp" / f"before_after_{ts}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "n": len(jobs),
                "seed": args.seed,
                "before_summary": before_sum,
                "after_summary": after_sum,
                "cost_ratio_before_over_after": ratio,
                "savings_pct": savings_pct,
                "score_agreement": {
                    "matched": len(matched),
                    "exact": exact,
                    "within_5": within_5,
                    "within_10": within_10,
                    "avg_diff_after_minus_before": avg_diff,
                    "mean_abs_diff": mad,
                },
                "before_results": before_results,
                "after_results": after_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
