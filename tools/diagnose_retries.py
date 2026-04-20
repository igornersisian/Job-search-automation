"""
Diagnose whether silent SDK retries inflate OpenAI spend.

Hooks httpx to log every outgoing HTTP request (including SDK-level retries),
disables the openai-python internal retries (max_retries=0), and runs 10 jobs
through each pipeline. Compare the HTTP-level request count to the "logical"
call count; compare computed cost against dashboard delta.

Usage:
    python tools/diagnose_retries.py --n 10 --seed 42
"""

import argparse
import io
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.score_job import build_scoring_prompt, _build_rules_text  # noqa: E402

load_dotenv()

PRICING = {
    "gpt-5-mini": {
        "standard": {"in": 0.25 / 1_000_000, "cached_in": 0.025 / 1_000_000, "out": 2.00 / 1_000_000},
        "flex":     {"in": 0.125 / 1_000_000, "cached_in": 0.0125 / 1_000_000, "out": 1.00 / 1_000_000},
    },
}


# ─── Instrumentation: log every HTTP request to OpenAI ────────────────────
HTTP_EVENTS: list[dict] = []


def log_request(request: httpx.Request):
    HTTP_EVENTS.append({
        "ts": time.time(),
        "type": "request",
        "method": request.method,
        "url": str(request.url),
    })


def log_response(response: httpx.Response):
    # Ensure body is available (streaming responses need read())
    try:
        response.read()
    except Exception:
        pass
    HTTP_EVENTS.append({
        "ts": time.time(),
        "type": "response",
        "status": response.status_code,
        "url": str(response.request.url),
    })


def make_client() -> OpenAI:
    http_client = httpx.Client(
        timeout=httpx.Timeout(60.0),
        event_hooks={"request": [log_request], "response": [log_response]},
    )
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        http_client=http_client,
        max_retries=0,  # ← disable silent SDK retries
    )


# ─── Pipelines ────────────────────────────────────────────────────────────
def call_before(client: OpenAI, job: dict, profile: dict) -> dict:
    profile_text = json.dumps(profile, indent=2)
    job_text = f"Title: {job['title']}\nCompany: {job['company']}\nDescription:\n{job['description']}"
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        reasoning_effort="minimal",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _build_rules_text(profile)},
            {"role": "user", "content": f"PROFILE:\n{profile_text}\n\nJOB:\n{job_text}"},
        ],
    )
    u = resp.usage
    return {
        "prompt_tokens": u.prompt_tokens,
        "cached_tokens": (u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else 0),
        "completion_tokens": u.completion_tokens,
        "reasoning_tokens": (u.completion_tokens_details.reasoning_tokens if u.completion_tokens_details else 0),
        "service_tier": getattr(resp, "service_tier", None),
    }


def call_after(client: OpenAI, job: dict, profile: dict) -> dict:
    instructions, user_input = build_scoring_prompt(job, profile)
    resp = client.responses.create(
        model="gpt-5-mini",
        instructions=instructions,
        input=user_input,
        text={"format": {"type": "json_object"}},
        reasoning={"effort": "minimal"},
        service_tier="flex",
        prompt_cache_key="score_job_v1",
        store=False,
    )
    u = resp.usage
    return {
        "prompt_tokens": u.input_tokens,
        "cached_tokens": (u.input_tokens_details.cached_tokens if u.input_tokens_details else 0),
        "completion_tokens": u.output_tokens,
        "reasoning_tokens": (u.output_tokens_details.reasoning_tokens if u.output_tokens_details else 0),
        "service_tier": getattr(resp, "service_tier", None),
    }


def cost(usage: dict, tier: str) -> float:
    p = PRICING["gpt-5-mini"][tier]
    non_cached = max(usage["prompt_tokens"] - usage["cached_tokens"], 0)
    return (
        non_cached * p["in"]
        + usage["cached_tokens"] * p["cached_in"]
        + usage["completion_tokens"] * p["out"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    profile = (
        sb.table("profile").select("parsed").order("updated_at", desc=True)
        .limit(1).execute().data[0]["parsed"]
    )
    rows = (
        sb.table("jobs").select("id,title,company,description")
        .not_.is_("description", "null").execute().data
    )
    rows = [r for r in rows if r.get("description") and len(r["description"]) >= 200]
    random.seed(args.seed)
    jobs = random.sample(rows, min(args.n, len(rows)))

    client = make_client()

    print(f"\n=== BEFORE (chat.completions, standard tier, max_retries=0) — {len(jobs)} jobs ===")
    before = []
    for i, j in enumerate(jobs, 1):
        try:
            u = call_before(client, j, profile)
            before.append(u)
            print(f"  [{i:>2}] in={u['prompt_tokens']} cached={u['cached_tokens']} "
                  f"out={u['completion_tokens']} reasoning={u['reasoning_tokens']} "
                  f"tier={u['service_tier']}")
        except Exception as e:
            print(f"  [{i:>2}] ERROR: {e}")

    print(f"\n=== AFTER (responses, flex tier, max_retries=0) — {len(jobs)} jobs ===")
    after = []
    for i, j in enumerate(jobs, 1):
        try:
            u = call_after(client, j, profile)
            after.append(u)
            print(f"  [{i:>2}] in={u['prompt_tokens']} cached={u['cached_tokens']} "
                  f"out={u['completion_tokens']} reasoning={u['reasoning_tokens']} "
                  f"tier={u['service_tier']}")
        except Exception as e:
            print(f"  [{i:>2}] ERROR: {e}")

    # ─── HTTP-level analysis ──────────────────────────────────────────────
    requests = [e for e in HTTP_EVENTS if e["type"] == "request"]
    responses = [e for e in HTTP_EVENTS if e["type"] == "response"]
    status_counts = Counter(e["status"] for e in responses)
    url_counts = Counter(e["url"].split("?")[0] for e in requests)

    print("\n" + "=" * 70)
    print("HTTP-LEVEL SUMMARY")
    print("=" * 70)
    print(f"Total HTTP requests sent   : {len(requests)}")
    print(f"Total HTTP responses       : {len(responses)}")
    print(f"Status codes               : {dict(status_counts)}")
    print(f"Per-endpoint request count : {dict(url_counts)}")
    print(f"Logical calls attempted    : BEFORE={len(jobs)}  AFTER={len(jobs)}  total={2*len(jobs)}")
    print(f"Retries implied            : {len(requests) - 2*len(jobs)}")

    # ─── Cost ─────────────────────────────────────────────────────────────
    def sum_usage(rows):
        if not rows:
            return dict(prompt_tokens=0, cached_tokens=0, completion_tokens=0, reasoning_tokens=0)
        return {
            k: sum(r[k] for r in rows) for k in ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens")
        }

    before_tot = sum_usage(before)
    after_tot = sum_usage(after)
    c_before = cost(before_tot, "standard")
    c_after = cost(after_tot, "flex")

    print("\n" + "=" * 70)
    print("COST (computed from usage × published pricing)")
    print("=" * 70)
    print(f"BEFORE  tokens: in={before_tot['prompt_tokens']:,}  cached={before_tot['cached_tokens']:,}  "
          f"out={before_tot['completion_tokens']:,}  reasoning={before_tot['reasoning_tokens']:,}")
    print(f"BEFORE  cost  : ${c_before:.6f}  per job: ${c_before/max(len(before),1):.6f}")
    print(f"AFTER   tokens: in={after_tot['prompt_tokens']:,}  cached={after_tot['cached_tokens']:,}  "
          f"out={after_tot['completion_tokens']:,}  reasoning={after_tot['reasoning_tokens']:,}")
    print(f"AFTER   cost  : ${c_after:.6f}  per job: ${c_after/max(len(after),1):.6f}")

    # ─── Save ─────────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / ".tmp" / f"diagnose_retries_{ts}.json"
    out.write_text(json.dumps({
        "n": len(jobs),
        "http": {
            "requests": len(requests),
            "responses": len(responses),
            "status_counts": dict(status_counts),
            "per_endpoint": dict(url_counts),
            "events": HTTP_EVENTS,
        },
        "before_usage_total": before_tot,
        "after_usage_total": after_tot,
        "cost_before": c_before,
        "cost_after": c_after,
        "before_calls": before,
        "after_calls": after,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
