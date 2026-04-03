"""
Score a single job against the user's resume profile using OpenAI.

Input:  job dict (from Apify), profile dict (from Supabase)
Output: enriched job dict with score, match_summary, red_flags

Two-step scoring:
  1. quick_score() — returns just an int 0-100 (cheap, fast)
  2. score_job()   — returns score + match_summary + red_flags (only for jobs above threshold)

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

# Only filter actual internships/trainee positions — not entry-level or junior
INTERN_KEYWORDS = [
    "intern",
    "internship",
    "trainee",
    "graduate program",
    "apprentice",
    "apprenticeship",
]


def is_junior_or_intern(job: dict) -> bool:
    """Return True if job title indicates an internship/trainee position."""
    title = job.get("title", "").lower()
    return any(kw in title for kw in INTERN_KEYWORDS)


def quick_score(job: dict, profile: dict) -> int:
    """Fast score-only evaluation. Returns int 0-100.

    Uses a compact prompt to minimise tokens — called for every job.
    Only jobs that pass the threshold get a full score_job() call.
    """
    job_text = (
        f"Title: {job.get('title', 'N/A')}\n"
        f"Company: {job.get('company', 'N/A')}\n"
        f"Description:\n{job.get('description', '')[:4000]}"
    )
    profile_text = json.dumps(profile, indent=2)

    # Custom dealbreakers from profile
    dealbreakers = profile.get("custom_red_flags") or []

    # Build system prompt — dealbreakers go here for maximum weight
    system_content = (
        "Rate how well this candidate REALISTICALLY fits the job. "
        "Use ONLY facts explicitly stated in the profile — do not infer, assume, or guess.\n\n"
    )

    if dealbreakers:
        items = "\n".join(f"- {d}" for d in dealbreakers)
        system_content += (
            "DEALBREAKERS — HIGHEST PRIORITY:\n"
            "If ANY of the following apply to this job, the score MUST be 25 or below. "
            "No exceptions. Check these FIRST before evaluating anything else:\n"
            f"{items}\n\n"
        )

    system_content += (
        "SCORING RULES:\n"
        "- If the job requires N+ years of experience and the candidate has significantly less "
        "(e.g. job asks 5+ years, candidate has <2), cap score at 40\n"
        "- If the job requires specific programming languages or frameworks the candidate "
        "doesn't demonstrate, deduct 15-20 points per critical missing skill\n"
        "- If the job is senior/lead/staff level and candidate profile shows junior-level "
        "experience, cap score at 45\n"
        "- Score 70+ ONLY if the candidate could realistically compete for this role\n"
        "- Score 90+ ONLY if the candidate meets virtually ALL stated requirements\n"
        "- No-code/low-code experience does NOT count as software engineering experience "
        "unless the job specifically asks for no-code skills\n"
        "- When requirements are ambiguous, lean slightly lower rather than higher\n\n"
        'Return JSON: {"score": <int 0-100>}'
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
    return result.get("score", 0)


def score_job(job: dict, profile: dict) -> dict:
    """
    Enrichment-only analysis for jobs that already passed quick_score threshold.
    Adds match_summary and red_flags. Does NOT override score — quick_score is authoritative.
    """
    job_text = (
        f"Title: {job.get('title', 'N/A')}\n"
        f"Company: {job.get('company', 'N/A')}\n"
        f"Location/Remote: {job.get('location', 'N/A')}\n"
        f"Salary: {job.get('salary', 'Not listed')}\n"
        f"Description:\n{job.get('description', '')[:4000]}"
    )

    profile_text = json.dumps(profile, indent=2)

    # Custom dealbreakers from profile
    dealbreakers = profile.get("custom_red_flags") or []
    dealbreakers_section = ""
    if dealbreakers:
        items = "\n".join(f"- {d}" for d in dealbreakers)
        dealbreakers_section = (
            f"\n\nYOUR DEALBREAKERS (flag in red_flags if any apply):\n{items}"
        )

    response = _chat_completion(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are writing a short job evaluation directly TO the candidate (use 'you/your', "
                    "never third person like 'the candidate' or 'Igor').\n\n"
                    "CRITICAL RULES:\n"
                    "- Use ONLY facts explicitly stated in the candidate profile. "
                    "Do NOT infer, assume, or invent any details.\n"
                    "- If something is not mentioned in the profile, treat it as absent.\n"
                    "- Address the candidate directly: 'This role fits you because...' or "
                    "'You lack the required...'\n\n"
                    "Return a JSON object with:\n"
                    "- match_summary: string (1-2 sentences explaining why this job does or doesn't "
                    "fit YOU, focused on the job's requirements vs your profile)\n"
                    "- red_flags: list of strings — ONLY genuine concerns, mismatches, or dealbreakers. "
                    "Include as many or as few as actually exist. Do NOT pad to a fixed number. "
                    "If there are zero real concerns, return an empty list. "
                    "Do NOT duplicate the same concern in different words.\n\n"
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CANDIDATE PROFILE:\n{profile_text}\n\n"
                    f"JOB POSTING:\n{job_text}{dealbreakers_section}"
                ),
            },
        ],
    )

    result = json.loads(response.choices[0].message.content)
    # Do NOT override job["score"] — quick_score is the authoritative score
    job["match_summary"] = result.get("match_summary", "")
    job["red_flags"] = result.get("red_flags", [])
    return job


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
