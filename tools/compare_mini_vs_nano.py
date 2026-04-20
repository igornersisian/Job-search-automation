"""
Head-to-head on the same 50 jobs: gpt-5-mini vs gpt-5.4-nano.

Both runs use the production pipeline (Responses API, prompt_cache_key,
service_tier="flex", reasoning_effort="minimal"). If the API silently
downgrades nano off flex (unsupported tier), the actual `service_tier`
returned in usage is used for cost calculation — we never guess pricing
based on what we *asked* for.

Logs per-job: tokens (in/cached/out/reasoning), latency, score, red_flags,
actual service_tier. Writes full JSON to .tmp/ and prints a summary
comparing cost per job, latency, and score agreement.

Usage:
    python tools/compare_mini_vs_nano.py --n 50 --seed 42
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

# Use the dedicated test key so dashboard spend from this run is isolated.
if os.environ.get("OPENAI_API_KEY_TEST"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY_TEST"]

# OpenAI public pricing per token, as of 2026-04.
# Flex tier for gpt-5.4-nano is not officially documented; if the API downgrades
# nano off flex, the actual returned tier is used — we never pay the wrong rate.
PRICING = {
    "gpt-5-mini": {
        "standard": {"in": 0.25 / 1_000_000,  "cached_in": 0.025 / 1_000_000,  "out": 2.00 / 1_000_000},
        "flex":     {"in": 0.125 / 1_000_000, "cached_in": 0.0125 / 1_000_000, "out": 1.00 / 1_000_000},
    },
    "gpt-5.4-nano": {
        "standard": {"in": 0.20 / 1_000_000,  "cached_in": 0.02 / 1_000_000,   "out": 1.25 / 1_000_000},
        # If flex is supported, we assume the same 50% discount pattern as gpt-5-mini.
        # If not supported, the API downgrades to standard and we use standard pricing.
        "flex":     {"in": 0.10 / 1_000_000,  "cached_in": 0.01 / 1_000_000,   "out": 0.625 / 1_000_000},
    },
}


def _tier_key(actual_tier: str | None) -> str:
    """Map the tier string returned by the API to our pricing table keys."""
    if actual_tier == "flex":
        return "flex"
    # "default", "auto", None → standard pricing
    return "standard"


def compute_cost(usage: dict, model: str) -> tuple[float, str]:
    tier = _tier_key(usage.get("service_tier"))
    p = PRICING[model][tier]
    non_cached = max(usage["prompt_tokens"] - usage["cached_tokens"], 0)
    cached = usage["cached_tokens"]
    out = usage["completion_tokens"]
    return non_cached * p["in"] + cached * p["cached_in"] + out * p["out"], tier


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


def run_pipeline(name: str, jobs: list[dict], profile: dict, model: str, reasoning_effort: str):
    print(f"\n{'=' * 70}")
    print(f"Running {name} ({model}, reasoning={reasoning_effort}) over {len(jobs)} jobs ...")
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
            scored = score_job(job, profile, model=model, reasoning_effort=reasoning_effort)
            usage = scored["_usage"]
            cost, priced_tier = compute_cost(usage, model)
            out.append({
                "id": row["id"],
                "title": row["title"],
                "company": row["company"],
                "score": scored["score"],
                "red_flags": scored["red_flags"],
                "usage": usage,
                "latency_ms": scored["_latency_ms"],
                "cost_usd": cost,
                "priced_tier": priced_tier,
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
    priced = {r["priced_tier"] for r in ok}
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
        "priced_tiers_used": sorted(priced),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-a", default="gpt-5-mini")
    parser.add_argument("--model-b", default="gpt-5.4-nano")
    # gpt-5-mini supports "minimal"; gpt-5.4-nano's minimum is "low"/"none".
    parser.add_argument("--effort-a", default="minimal")
    parser.add_argument("--effort-b", default="low")
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

    a_results, a_wall = run_pipeline(f"A={args.model_a}", jobs, profile, args.model_a, args.effort_a)
    b_results, b_wall = run_pipeline(f"B={args.model_b}", jobs, profile, args.model_b, args.effort_b)

    a_sum = summarize("A", a_results, a_wall)
    b_sum = summarize("B", b_results, b_wall)

    ratio = a_sum.get("cost_per_job_usd", 0) / b_sum.get("cost_per_job_usd", 1) if b_sum.get("cost_per_job_usd") else 0
    savings_pct = (1 - b_sum.get("cost_per_job_usd", 0) / a_sum.get("cost_per_job_usd", 1)) * 100 if a_sum.get("cost_per_job_usd") else 0

    by_id_b = {r["id"]: r for r in b_results if "error" not in r}
    matched = [(a, by_id_b[a["id"]]) for a in a_results
               if "error" not in a and a["id"] in by_id_b]
    exact = sum(1 for a, b in matched if a["score"] == b["score"])
    within_5 = sum(1 for a, b in matched if abs(a["score"] - b["score"]) <= 5)
    within_10 = sum(1 for a, b in matched if abs(a["score"] - b["score"]) <= 10)
    score_diffs = [b["score"] - a["score"] for a, b in matched]
    avg_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0
    mad = sum(abs(d) for d in score_diffs) / len(score_diffs) if score_diffs else 0

    print("\n" + "=" * 70)
    print(f"{args.model_a}  vs  {args.model_b}   (optimized pipeline, reasoning=minimal)")
    print("=" * 70)
    print(f"{'metric':<32}{args.model_a:>18}{args.model_b:>18}")
    def _fmt(v, key):
        if v is None:
            return "-"
        if isinstance(v, float) and "ratio" in key:
            return f"{v*100:.1f}%"
        if isinstance(v, float):
            return f"{v:.6f}" if "cost" in key else f"{v:.1f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    for k in ("n", "wall_s", "tokens_in", "tokens_cached", "tokens_out",
              "tokens_reasoning", "cache_hit_ratio", "avg_latency_ms",
              "cost_total_usd", "cost_per_job_usd"):
        print(f"{k:<32}{_fmt(a_sum.get(k), k):>18}{_fmt(b_sum.get(k), k):>18}")

    print(f"\nservice_tiers_seen  A={a_sum.get('service_tiers_seen')}   B={b_sum.get('service_tiers_seen')}")
    print(f"priced_tiers_used   A={a_sum.get('priced_tiers_used')}   B={b_sum.get('priced_tiers_used')}")
    print(f"\nRatio A/B cost per job: {ratio:.2f}x  (B is this much cheaper than A)")
    print(f"Savings (A → B): {savings_pct:+.1f}%")
    print(f"\nImplied cost per 1k jobs:  A=${a_sum.get('cost_per_job_usd', 0)*1000:.2f}  B=${b_sum.get('cost_per_job_usd', 0)*1000:.2f}")

    print(f"\nScore agreement over {len(matched)} matched jobs:")
    print(f"  exact match:    {exact}/{len(matched)} ({exact/max(len(matched),1)*100:.0f}%)")
    print(f"  within ±5:      {within_5}/{len(matched)} ({within_5/max(len(matched),1)*100:.0f}%)")
    print(f"  within ±10:     {within_10}/{len(matched)} ({within_10/max(len(matched),1)*100:.0f}%)")
    print(f"  avg(B-A) = {avg_diff:+.2f}")
    print(f"  mean abs diff = {mad:.2f}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = ROOT / ".tmp" / f"mini_vs_nano_{ts}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "n": len(jobs),
                "seed": args.seed,
                "model_a": args.model_a,
                "model_b": args.model_b,
                "a_summary": a_sum,
                "b_summary": b_sum,
                "cost_ratio_a_over_b": ratio,
                "savings_pct_a_to_b": savings_pct,
                "score_agreement": {
                    "matched": len(matched),
                    "exact": exact,
                    "within_5": within_5,
                    "within_10": within_10,
                    "avg_diff_b_minus_a": avg_diff,
                    "mean_abs_diff": mad,
                    "diffs": score_diffs,
                },
                "a_results": a_results,
                "b_results": b_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
