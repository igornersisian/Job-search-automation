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


def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client

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
    dealbreakers_text = ""
    if dealbreakers:
        items = "\n".join(f"- {d}" for d in dealbreakers)
        dealbreakers_text = (
            f"\n\nCANDIDATE DEALBREAKERS (if ANY of these apply, score below 30):\n{items}"
        )

    response = get_openai().chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Rate how well this candidate fits the job. "
                    "Use ONLY facts from the profile, do not infer or assume.\n"
                    'Return JSON: {"score": <int 0-100>}\n'
                    "90-100: excellent match, 70-89: good, 50-69: partial, 0-49: poor."
                ),
            },
            {
                "role": "user",
                "content": f"PROFILE:\n{profile_text}\n\nJOB:\n{job_text}{dealbreakers_text}",
            },
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
    dealbreakers_text = ""
    if dealbreakers:
        items = "\n".join(f"- {d}" for d in dealbreakers)
        dealbreakers_text = (
            f"\n\nCANDIDATE DEALBREAKERS (flag these in red_flags if they apply):\n{items}"
        )

    response = get_openai().chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a career advisor evaluating job fit for a candidate.\n\n"
                    "CRITICAL RULES:\n"
                    "- Use ONLY facts explicitly stated in the candidate profile. "
                    "Do NOT infer, assume, or invent any details about the candidate's experience, "
                    "years of work, skills, or projects that are not written in the profile.\n"
                    "- If something is not mentioned in the profile, treat it as absent — "
                    "do not guess or extrapolate.\n"
                    "- When citing candidate experience, quote the actual profile data.\n\n"
                    "Return a JSON object with:\n"
                    "- match_summary: string (1-2 sentences on why this is or isn't a good fit)\n"
                    "- red_flags: list of strings (concerns, missing requirements, or mismatches)\n\n"
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CANDIDATE PROFILE:\n{profile_text}\n\n"
                    f"JOB POSTING:\n{job_text}{dealbreakers_text}"
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
