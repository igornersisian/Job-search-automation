"""
Dump the FULL raw `usage` object for one call through each pipeline.

Goal: verify whether reasoning_tokens / any other billed-but-undercounted
fields are hiding in the response. If reasoning_tokens truly = 0, then the
dashboard discrepancy comes from elsewhere (silent retries, other activity).
If non-zero, our cost calc is under-reporting.
"""

import io
import json
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.score_job import build_scoring_prompt, get_openai, _build_rules_text  # noqa: E402

load_dotenv()


def dump_object(obj, label):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    # Pydantic v2: .model_dump() — full serialisable view of all fields,
    # including ones we may have ignored.
    try:
        print(json.dumps(obj.model_dump(), indent=2, default=str))
    except Exception:
        print(repr(obj))


def main():
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    profile = (
        sb.table("profile").select("parsed")
          .order("updated_at", desc=True).limit(1).execute()
          .data[0]["parsed"]
    )
    rows = (
        sb.table("jobs").select("id,title,company,description")
          .not_.is_("description", "null").execute().data
    )
    row = next(r for r in rows if r.get("description") and len(r["description"]) >= 200)
    job = {"title": row["title"], "company": row["company"], "description": row["description"]}

    client = get_openai()

    # --- BEFORE-style call (Chat Completions, standard tier) ---
    profile_text = json.dumps(profile, indent=2)
    job_text = (
        f"Title: {job['title']}\nCompany: {job['company']}\n"
        f"Description:\n{job['description']}"
    )
    before_resp = client.chat.completions.create(
        model="gpt-5-mini",
        reasoning_effort="minimal",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _build_rules_text(profile)},
            {"role": "user", "content": f"PROFILE:\n{profile_text}\n\nJOB:\n{job_text}"},
        ],
    )
    dump_object(before_resp.usage, "BEFORE (chat.completions) — raw usage")
    print("\nBEFORE response.service_tier =", getattr(before_resp, "service_tier", "<not set>"))

    # --- AFTER-style call (Responses, flex) ---
    instructions, user_input = build_scoring_prompt(job, profile)
    after_resp = client.responses.create(
        model="gpt-5-mini",
        instructions=instructions,
        input=user_input,
        text={"format": {"type": "json_object"}},
        reasoning={"effort": "minimal"},
        service_tier="flex",
        prompt_cache_key="score_job_v1",
        store=False,
    )
    dump_object(after_resp.usage, "AFTER (responses) — raw usage")
    print("\nAFTER response.service_tier =", getattr(after_resp, "service_tier", "<not set>"))

    # Top-level response fields, for billing/routing info
    print("\n" + "=" * 70)
    print("AFTER top-level attrs of interest:")
    for attr in ("id", "model", "service_tier", "status", "billing"):
        print(f"  {attr}: {getattr(after_resp, attr, '<missing>')}")


if __name__ == "__main__":
    main()
