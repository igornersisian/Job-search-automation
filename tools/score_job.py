"""
Score a single job against the user's resume profile using OpenAI.

Input:  job dict (from scraper), profile dict (from Supabase)
Output: enriched job dict with score, match_summary, red_flags, score_breakdown

Single-pass scoring with chain-of-thought:
  The model first writes match_summary + red_flags (forcing it to reason),
  then assigns sub-scores that must be consistent with its own analysis.

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


def _chat_completion(messages: list, max_retries: int = 3, **kwargs):
    """OpenAI call with retry on 429 + OpenRouter fallback."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return get_openai().chat.completions.create(messages=messages, **kwargs)
        except Exception as e:
            if "429" in str(e):
                last_error = e
                wait = min(2 ** attempt, 8)
                logger.warning(f"Rate limited, retry in {wait}s ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise

    # Retries exhausted — try OpenRouter
    fallback = _get_openrouter()
    if fallback:
        model = kwargs.pop("model", "gpt-4.1-mini")
        kwargs["model"] = f"openai/{model}"
        logger.info(f"Falling back to OpenRouter ({kwargs['model']})")
        return fallback.chat.completions.create(messages=messages, **kwargs)

    raise last_error


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


def score_job(job: dict, profile: dict) -> dict:
    """Single-pass scoring with chain-of-thought.

    The model first writes match_summary + red_flags (its reasoning),
    then assigns sub-scores consistent with that analysis.

    Returns the job dict enriched with:
      score (int 0-100), score_breakdown (dict), match_summary (str), red_flags (list)
    """
    # ── Deterministic guard: empty / too-short description ──
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

    # Custom dealbreakers from profile
    dealbreakers = profile.get("custom_red_flags") or []

    # ── Build system prompt ──
    system_content = (
        "You are a strict job-candidate fit evaluator. "
        "Score ONLY using facts explicitly stated in the candidate profile. "
        "Do NOT infer, assume, or guess any skills, experience, or qualifications.\n\n"
    )

    if dealbreakers:
        items = "\n".join(f"- {d}" for d in dealbreakers)
        system_content += (
            "DEALBREAKERS — CHECK FIRST:\n"
            "If ANY of these apply to this job, set ALL sub-scores to 0. "
            "No exceptions:\n"
            f"{items}\n\n"
        )

    system_content += (
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
        "Before returning, re-read your red_flags. For EACH red flag, verify that "
        "the corresponding sub-score is LOW (0-5). If you flagged domain mismatch "
        "but domain > 5, FIX IT. If you flagged seniority mismatch but role > 5, "
        "FIX IT. Your text analysis is the truth.\n\n"

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

    response = _chat_completion(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"PROFILE:\n{profile_text}\n\nJOB:\n{job_text}"},
        ],
    )
    result = json.loads(response.choices[0].message.content)

    # ── Extract text fields ──
    job["match_summary"] = result.get("match_summary", "")
    job["red_flags"] = result.get("red_flags", [])

    # ── Server-side recomputation (don't trust LLM arithmetic) ──
    b1 = result.get("block1", {})
    b2 = result.get("block2", {})

    # Clamp sub-scores to valid ranges
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


# Keep quick_score as a thin wrapper for backward compatibility with process_jobs.py
def quick_score(job: dict, profile: dict) -> tuple[int, dict]:
    """Score a job and return (score, breakdown). Also sets match_summary/red_flags on the job dict."""
    enriched = score_job(job, profile)
    return enriched["score"], enriched["score_breakdown"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python score_job.py '<job_json>'")
        sys.exit(1)

    # For testing: load profile from Supabase
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
