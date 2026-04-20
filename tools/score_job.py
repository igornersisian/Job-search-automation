"""
Score a single job against the user's resume profile using OpenAI.

Input:  job dict (from scraper), profile dict (from Supabase)
Output: enriched job dict with score, match_summary, red_flags, score_breakdown

Single-pass scoring with chain-of-thought:
  The model first writes match_summary + red_flags (forcing it to reason),
  then assigns sub-scores that must be consistent with its own analysis.

Production path: gpt-5-mini + reasoning.effort=minimal + service_tier=flex
via the Responses API. The full profile and all instructions live in the
`instructions` field (stable prefix -> OpenAI prompt caching kicks in when
scoring many jobs in a run).

Usage (standalone):
    python tools/score_job.py '{"title": "...", "description": "..."}'
"""

import os
import sys
import json
import time
import logging

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_openai_client: OpenAI | None = None
_openrouter_client: OpenAI | None = None

# Stable key so OpenAI routes repeated scoring calls to the same cache shard.
# Bump this string whenever the instructions template changes — the cache
# becomes useless the moment the prefix shifts by a single token.
PROMPT_CACHE_KEY = "score_job_v1"


def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def _get_openrouter() -> OpenAI | None:
    """Return OpenRouter client if API key is configured, else None."""
    global _openrouter_client
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    if _openrouter_client is None:
        _openrouter_client = OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _openrouter_client


def _is_reasoning_model(model: str) -> bool:
    return model.startswith("gpt-5") or model.startswith("o")


def _responses_call(
    instructions: str,
    user_input: str,
    model: str,
    reasoning_effort: str | None,
    service_tier: str | None,
):
    kwargs: dict = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "text": {"format": {"type": "json_object"}},
        "prompt_cache_key": PROMPT_CACHE_KEY,
        "store": False,
    }
    # gpt-5 / o-series reject `temperature` — they sample internally.
    # Only pass it for non-reasoning models (e.g. gpt-4.1-mini).
    if not _is_reasoning_model(model):
        kwargs["temperature"] = 0
    if reasoning_effort and _is_reasoning_model(model):
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if service_tier:
        kwargs["service_tier"] = service_tier
    return get_openai().responses.create(**kwargs)


def _extract_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0,
                "reasoning_tokens": 0, "cached_tokens": 0,
                "service_tier": None}
    in_details = getattr(usage, "input_tokens_details", None)
    out_details = getattr(usage, "output_tokens_details", None)
    return {
        "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
        "reasoning_tokens": getattr(out_details, "reasoning_tokens", 0) or 0,
        "cached_tokens": getattr(in_details, "cached_tokens", 0) or 0,
        "service_tier": getattr(response, "service_tier", None),
    }


def _openrouter_fallback(
    instructions: str,
    user_input: str,
    model: str,
    reasoning_effort: str | None,
) -> tuple[str, dict]:
    """Last-resort fallback through OpenRouter's chat.completions endpoint."""
    client = _get_openrouter()
    if client is None:
        return None  # type: ignore[return-value]
    or_kwargs: dict = {
        "model": f"openai/{model}",
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_input},
        ],
        "response_format": {"type": "json_object"},
    }
    if not _is_reasoning_model(model):
        or_kwargs["temperature"] = 0
    if reasoning_effort and _is_reasoning_model(model):
        or_kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
    logger.info(f"Falling back to OpenRouter ({or_kwargs['model']})")
    resp = client.chat.completions.create(**or_kwargs)
    text = resp.choices[0].message.content
    u = getattr(resp, "usage", None)
    usage = {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0 if u else 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0 if u else 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
    }
    return text, usage


def _call_llm(
    instructions: str,
    user_input: str,
    model: str,
    reasoning_effort: str | None,
    service_tier: str | None,
    max_retries: int = 3,
) -> tuple[str, dict]:
    """Execute the scoring call. Returns (json_text, usage_dict).

    Retry strategy:
      1. Up to `max_retries` attempts with the requested service_tier
      2. If still failing on 429/capacity and we were on flex, try standard tier once
      3. Final fallback: OpenRouter via chat.completions
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = _responses_call(instructions, user_input, model, reasoning_effort, service_tier)
            actual_tier = getattr(resp, "service_tier", None)
            if service_tier and actual_tier and actual_tier != service_tier:
                logger.warning(
                    f"Requested service_tier={service_tier} but got {actual_tier} (silent downgrade)"
                )
            return resp.output_text, _extract_usage(resp)
        except Exception as e:
            msg = str(e).lower()
            transient = "429" in msg or "capacity" in msg or "resource_unavailable" in msg
            if not transient:
                raise
            last_error = e
            wait = min(2 ** attempt, 8)
            logger.warning(
                f"Transient API error ({service_tier or 'default'}), retry in {wait}s "
                f"({attempt+1}/{max_retries}): {e}"
            )
            time.sleep(wait)

    if service_tier == "flex":
        try:
            logger.info("Flex exhausted — falling back to standard tier")
            resp = _responses_call(instructions, user_input, model, reasoning_effort, None)
            return resp.output_text, _extract_usage(resp)
        except Exception as e:
            if "429" not in str(e):
                raise
            last_error = e

    fb = _openrouter_fallback(instructions, user_input, model, reasoning_effort)
    if fb is not None:
        return fb

    assert last_error is not None
    raise last_error


def _build_rules_text(profile: dict) -> str:
    """Scoring rubric + dealbreakers only (no profile body).

    Shared helper so experiment scripts (e.g. semi-naive baseline) can reuse
    the exact same rubric, isolating prompt-structure differences from any
    drift in the rules themselves.
    """
    dealbreakers = profile.get("custom_red_flags") or []

    text = (
        "You are a strict job-candidate fit evaluator. "
        "Score ONLY using facts explicitly stated in the candidate profile. "
        "Do NOT infer, assume, or guess any skills, experience, or qualifications.\n\n"
    )

    if dealbreakers:
        items = "\n".join(f"- {d}" for d in dealbreakers)
        text += (
            "DEALBREAKERS (HARD BLOCKERS) — CHECK BEFORE ANYTHING ELSE:\n"
            f"{items}\n\n"
            "How to apply them:\n"
            "- Go through EACH dealbreaker one by one. For each, ask: does the job "
            "description, ANYWHERE in its full text, match this dealbreaker in meaning?\n"
            "- Match by MEANING, not by exact wording. The job will rarely use the "
            "same phrasing as the dealbreaker. Translate terms from other languages, "
            "convert units when needed, and resolve implicit signals against the "
            "dealbreaker's intent.\n"
            "- Read the FULL description. Do NOT rely on the 'Remote' tag, job "
            "title, or any header metadata — the blocker is often only stated "
            "deep in the description.\n"
            "- When in doubt, flag it. Missing a real dealbreaker is worse than "
            "over-flagging.\n\n"
            "IF ANY DEALBREAKER MATCHES:\n"
            "  1. Add it to red_flags, quoting the dealbreaker and the job phrase "
            "that triggered it.\n"
            "  2. Set ALL FIVE sub-scores to 0 (domain, patterns, role, tools, experience).\n"
            "  3. Tech stack / tool match does NOT override a dealbreaker. No exceptions.\n\n"
        )

    text += (
        "YOUR TASK — analyse the job against the candidate in THREE steps:\n\n"

        "STEP 1: Write match_summary (1-2 sentences).\n"
        "Address the candidate directly (use 'you/your'). "
        "State clearly whether this job fits or not and WHY. "
        "Mention the most important match or mismatch.\n\n"

        "STEP 2: List red_flags.\n"
        "CRITICAL: Red flags are the most important part of your evaluation. "
        "Each red flag will deduct 15 points from the final score. "
        "Be thorough — missing a real red flag means the candidate wastes time "
        "on a job they can't get.\n\n"

        "You MUST flag ALL of the following if they apply:\n"
        "- Location mismatch: if the job description mentions hybrid, in-office, "
        "on-site, specific office days, or requires physical presence — flag it "
        "REGARDLESS of what the job title or metadata says. Read the FULL description.\n"
        "- Unpaid/volunteer/bootcamp: if the job is unpaid, this violates any "
        "minimum salary dealbreaker. '$0/hr < $25/hr' — flag it.\n"
        "- Seniority mismatch: if the job title includes Senior/Staff/Principal/"
        "Director/VP/Head/Lead/Founding and the candidate lacks the years of "
        "experience for that level — flag it.\n"
        "- Domain mismatch: if the job requires traditional software engineering "
        "(writing production code in React/Node/Python/Go/Java etc.) but the "
        "candidate uses AI-assisted development and no-code/low-code tools — "
        "flag 'requires traditional software engineering'.\n"
        "- Primary function mismatch: if the PRIMARY function of the role is "
        "sales/BDR, customer support, customer success, marketing, recruiting, "
        "operations, product management, or any non-engineering function — "
        "flag it as 'primary function is <X>, not automation engineering'. "
        "Tool mentions (n8n, automation, AI) in the job ad do NOT change the "
        "primary function. Ask: what is this person hired to DO most of the time?\n"
        "- Tool mismatch: if the job lists specific languages/frameworks the "
        "candidate does not know and cannot substitute — flag it.\n"
        "- Any dealbreaker from the list above that applies.\n\n"

        "If zero real concerns exist, return an empty list. "
        "Do NOT duplicate the same concern in different words.\n\n"

        "STEP 3: Assign sub-scores.\n"
        "Your scores MUST be consistent with your match_summary and red_flags above. "
        "If you identified a problem in Steps 1-2, the corresponding score MUST be LOW.\n\n"

        "SUB-SCORES (5 dimensions, max 100 base points):\n\n"

        "BLOCK 1 — Domain & Capability Fit (max 70 points):\n"
        "  domain (0-30): How well does the candidate's actual domain match the "
        "PRIMARY FUNCTION of this role?\n"
        "    FIRST identify the primary function — what will the hired person "
        "spend 70%+ of their time doing? Read the role's top responsibilities and "
        "the job title, NOT the 'nice-to-have' or 'tools you'll use' sections. "
        "A Customer Support role that uses n8n internally is still a support role. "
        "A Sales/BDR role that sends automated cold emails is still a sales role. "
        "A Marketing role that runs AI workflows is still a marketing role.\n"
        "    Tool mentions (n8n, automation, AI, Python) in a job ad do NOT make "
        "the domain match if the primary function is something else (sales, "
        "support, marketing, PM, ops, recruiting, etc.).\n"
        "    Be precise — adjacent-sounding fields can be very different. "
        "Using AI APIs/integrations ≠ building/training ML models. "
        "AI-assisted development (vibe coding) ≠ traditional software engineering. "
        "Product management ≠ engineering. Sales/BDR ≠ automation engineering. "
        "Customer support/success ≠ automation engineering.\n"
        "    Scoring:\n"
        "      0-5: primary function is a fundamentally different domain "
        "(sales, support, marketing, PM, ops) — even if automation/AI tools "
        "are mentioned as secondary requirements.\n"
        "      8-15: related but not the same primary function (e.g. traditional "
        "SWE role vs candidate's no-code/AI-assisted approach).\n"
        "      20-30: primary function IS building automation / AI agents / "
        "LLM integrations / workflow systems — matches candidate's core work.\n"
        "    If you flag 'primary function is X, not automation engineering' in "
        "red_flags, domain MUST be 0-5.\n"
        "  patterns (0-25): Does the candidate have experience with the architectural "
        "patterns the job needs? Use semantic matching — similar patterns in different "
        "tools count partially. If the job requires writing production code from scratch "
        "but the candidate builds with no-code/low-code + AI assistance, score 0-5. "
        "Completely unrelated patterns score 0-3.\n"
        "  role (0-15): Does the candidate's seniority match? "
        "Compare actual years of experience against the job's level. "
        "If the job is 3+ levels above (Junior → Director/VP/Head/Principal), cap at 2. "
        "If 2 levels above (Junior → Senior/Staff), cap at 5. "
        "If close match, score 10-15.\n\n"

        "BLOCK 2 — Formal Requirements (max 30 points):\n"
        "  tools (0-20): Does the candidate know the specific tools/languages/frameworks? "
        "Compare ACTUAL toolset. Similar tools = partial credit. "
        "If the candidate's approach is fundamentally different (no-code vs traditional "
        "engineering), score 0-3 unless the job explicitly values their approach.\n"
        "  experience (0-10): Years of experience match. If the job states N+ years "
        "and the candidate has fewer, apply -5 per missing year (starting from 10, "
        "floor 0). If no YOE mentioned, give 8.\n\n"

        "CONSISTENCY RULE:\n"
        "Before returning, re-read your red_flags.\n"
        "- If ANY red flag is a dealbreaker match, ALL FIVE sub-scores MUST be 0 "
        "(domain, patterns, role, tools, experience). No partial credit.\n"
        "- For non-dealbreaker red flags: the corresponding sub-score must be LOW "
        "(0-5). If you flagged domain mismatch but domain > 5, FIX IT. If you "
        "flagged seniority mismatch but role > 5, FIX IT.\n"
        "Your text analysis is the truth.\n\n"

        "SCORING: The base score = sum of all sub-scores (max 100). "
        "Then each red flag you listed deducts 15 points. "
        "So if you list 3 red flags, the candidate loses 45 points. "
        "This means red flags have MASSIVE impact — only flag genuine concerns, "
        "but flag ALL genuine concerns.\n\n"

        "Return ONLY this JSON (match_summary and red_flags FIRST, then scores):\n"
        "{\n"
        '  "match_summary": "<1-2 sentences>",\n'
        '  "red_flags": ["<flag1>", ...],\n'
        '  "block1": {"domain": <int>, "patterns": <int>, "role": <int>},\n'
        '  "block2": {"tools": <int>, "experience": <int>}\n'
        "}"
    )

    return text


def build_scoring_prompt(job: dict, profile: dict) -> tuple[str, str]:
    """Build (instructions, user_input) for the Responses API.

    `instructions` is stable across jobs in a single run — it contains the
    full scoring rubric PLUS the user's profile (dealbreakers + resume). This
    lets OpenAI's automatic prefix cache kick in after the first call.
    `user_input` carries ONLY the job being scored — the variable part.

    Exposed publicly so other tools (model comparison, batch jobs, etc.) can
    reuse the exact same prompt.
    """
    description = (job.get("description") or "").strip()
    job_text = (
        f"Title: {job.get('title', 'N/A')}\n"
        f"Company: {job.get('company', 'N/A')}\n"
        f"Description:\n{description}"
    )
    profile_text = json.dumps(profile, indent=2)

    instructions = _build_rules_text(profile) + f"\n\nCANDIDATE PROFILE:\n{profile_text}"
    user_input = f"Score this job and return JSON per the instructions.\n\nJOB:\n{job_text}"
    return instructions, user_input


def is_excluded_by_title(job: dict, excluded_keywords: list[str]) -> bool:
    """Return True if job title contains any of the user-defined excluded keywords."""
    if not excluded_keywords:
        return False
    title = job.get("title", "").lower()
    return any(kw in title for kw in excluded_keywords)


_EMPTY_BREAKDOWN = {
    "block1": {"domain": 0, "patterns": 0, "role": 0},
    "block2": {"tools": 0, "experience": 0},
    "penalty": 0,
    "red_flag_count": 0,
}

# Sub-score caps for server-side clamping (total base max = 100)
_CAPS = {
    "domain": 30, "patterns": 25, "role": 15,
    "tools": 20, "experience": 10,
}

# Each red flag deducts this many points from the base score
RED_FLAG_PENALTY = 15


def score_job(
    job: dict,
    profile: dict,
    model: str = "gpt-5-mini",
    reasoning_effort: str | None = "minimal",
    service_tier: str | None = "flex",
) -> dict:
    """Single-pass scoring with chain-of-thought via the Responses API.

    Args:
        model: OpenAI model id. Default gpt-5-mini.
        reasoning_effort: for gpt-5*/o-series — "minimal"|"low"|"medium"|"high".
            Ignored for models that don't support reasoning.
        service_tier: "flex" (default; ~50% cheaper, slower) or None for default tier.

    Returns the job dict enriched with:
      score, score_breakdown, match_summary, red_flags, and for diagnostics
      _usage {prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens},
      _latency_ms.
    """
    description = (job.get("description") or "").strip()
    if len(description) < 50:
        logger.info(f"Empty/short description for '{job.get('title')}' — auto-score 0")
        job["score"] = 0
        job["score_breakdown"] = _EMPTY_BREAKDOWN
        job["match_summary"] = "No job description available."
        job["red_flags"] = ["No description"]
        return job

    instructions, user_input = build_scoring_prompt(job, profile)

    t0 = time.perf_counter()
    json_text, usage = _call_llm(
        instructions=instructions,
        user_input=user_input,
        model=model,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    result = json.loads(json_text)

    job["_usage"] = usage
    job["_latency_ms"] = latency_ms

    job["match_summary"] = result.get("match_summary", "")
    job["red_flags"] = result.get("red_flags", [])

    # Server-side recomputation (don't trust LLM arithmetic)
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


def quick_score(job: dict, profile: dict) -> tuple[int, dict]:
    """Score a job and return (score, breakdown). Also sets match_summary/red_flags on the job dict."""
    enriched = score_job(job, profile)
    return enriched["score"], enriched["score_breakdown"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python score_job.py '<job_json>'")
        sys.exit(1)

    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    profile_result = sb.table("profile").select("parsed").order("updated_at", desc=True).limit(1).execute()
    if not profile_result.data:
        print("No profile found in Supabase. Send your PDF to the Telegram bot first.")
        sys.exit(1)

    profile = profile_result.data[0]["parsed"]
    job = json.loads(sys.argv[1])
    enriched = score_job(job, profile)
    print(json.dumps(enriched, ensure_ascii=False, indent=2))
