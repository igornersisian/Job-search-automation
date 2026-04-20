"""
Semi-naive scoring baseline for the LinkedIn before/after experiment.

Reconstructs the PRE-optimization pipeline (commit d8a3fae and earlier):
  - Chat Completions API
  - system message = scoring rubric only (no profile body)
  - user message   = "PROFILE:\\n{profile}\\n\\nJOB:\\n{job}"
  - No service_tier (standard), no prompt_cache_key
  - response_format = json_object

But with the MODEL swapped to gpt-5-mini + reasoning_effort="minimal",
so the comparison against the optimized version isolates the effect of:
  1. prompt structure (profile-in-user vs profile-in-instructions),
  2. API choice (Chat Completions vs Responses),
  3. service_tier (standard vs flex),
  4. prompt_cache_key (absent vs set).

Do NOT use this in production — it exists purely as the baseline half of
`tools/compare_before_after.py`.
"""

import json
import time

from tools.score_job import (
    RED_FLAG_PENALTY,
    _CAPS,
    _EMPTY_BREAKDOWN,
    _build_rules_text,
    get_openai,
    logger,
)


def _chat_usage(response) -> dict:
    u = getattr(response, "usage", None)
    if u is None:
        return {"prompt_tokens": 0, "completion_tokens": 0,
                "reasoning_tokens": 0, "cached_tokens": 0,
                "service_tier": None}
    in_details = getattr(u, "prompt_tokens_details", None)
    out_details = getattr(u, "completion_tokens_details", None)
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "reasoning_tokens": getattr(out_details, "reasoning_tokens", 0) or 0,
        "cached_tokens": getattr(in_details, "cached_tokens", 0) or 0,
        "service_tier": getattr(response, "service_tier", None),
    }


def score_job_semi_naive(
    job: dict,
    profile: dict,
    model: str = "gpt-5-mini",
    reasoning_effort: str | None = "minimal",
) -> dict:
    """Score via the pre-optimization path: Chat Completions, profile-in-user."""
    description = (job.get("description") or "").strip()
    if len(description) < 50:
        logger.info(f"Empty/short description for '{job.get('title')}' — auto-score 0")
        job["score"] = 0
        job["score_breakdown"] = _EMPTY_BREAKDOWN
        job["match_summary"] = "No job description available."
        job["red_flags"] = ["No description"]
        return job

    job_text = (
        f"Title: {job.get('title', 'N/A')}\n"
        f"Company: {job.get('company', 'N/A')}\n"
        f"Description:\n{description}"
    )
    profile_text = json.dumps(profile, indent=2)
    system_content = _build_rules_text(profile)
    user_content = f"PROFILE:\n{profile_text}\n\nJOB:\n{job_text}"

    kwargs: dict = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
    }
    if reasoning_effort and (model.startswith("gpt-5") or model.startswith("o")):
        kwargs["reasoning_effort"] = reasoning_effort

    t0 = time.perf_counter()
    response = get_openai().chat.completions.create(**kwargs)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    usage = _chat_usage(response)

    result = json.loads(response.choices[0].message.content)

    job["_usage"] = usage
    job["_latency_ms"] = latency_ms
    job["match_summary"] = result.get("match_summary", "")
    job["red_flags"] = result.get("red_flags", [])

    b1 = result.get("block1", {})
    b2 = result.get("block2", {})
    b1 = {k: min(max(b1.get(k, 0), 0), _CAPS[k]) for k in ("domain", "patterns", "role")}
    b2 = {k: min(max(b2.get(k, 0), 0), _CAPS[k]) for k in ("tools", "experience")}

    base = sum(b1.values()) + sum(b2.values())
    penalty = len(job["red_flags"]) * RED_FLAG_PENALTY
    computed = max(base - penalty, 0)

    job["score"] = computed
    job["score_breakdown"] = {
        "block1": dict(b1),
        "block2": dict(b2),
        "penalty": penalty,
        "red_flag_count": len(job["red_flags"]),
    }
    return job
