"""
Score a single job against the user's resume profile using OpenAI.

Input:  job dict (from Apify), profile dict (from Supabase)
Output: enriched job dict with score, match_summary, red_flags, typical_qa

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

# Keywords that indicate internship or clearly junior roles — filter these out
JUNIOR_KEYWORDS = [
    "intern", "internship", "entry level", "entry-level",
    "no experience required", "0-1 year", "0 to 1 year",
    "fresh graduate", "recent graduate", "graduate program",
    "junior developer", "junior engineer",
]


def is_junior_or_intern(job: dict) -> bool:
    """Return True if job title indicates internship/junior role."""
    title = job.get("title", "").lower()
    return any(kw in title for kw in JUNIOR_KEYWORDS)


def score_job(job: dict, profile: dict) -> dict:
    """
    Analyze job against profile. Returns job dict enriched with:
      score, match_summary, red_flags, typical_qa
    """
    job_text = (
        f"Title: {job.get('title', 'N/A')}\n"
        f"Company: {job.get('company', 'N/A')}\n"
        f"Location/Remote: {job.get('location', 'N/A')}\n"
        f"Salary: {job.get('salary', 'Not listed')}\n"
        f"Description:\n{job.get('description', '')[:4000]}"
    )

    profile_text = json.dumps(profile, indent=2)

    response = get_openai().chat.completions.create(
        model="gpt-4o-mini",
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
                    "- When citing candidate experience in red_flags or match_summary, "
                    "quote the actual profile data.\n\n"
                    "Return a JSON object with:\n"
                    "- score: int 0-100 (how well the candidate fits this role)\n"
                    "- match_summary: string (1-2 sentences on why this is or isn't a good fit)\n"
                    "- red_flags: list of strings (concerns, missing requirements, or mismatches)\n"
                    "- typical_qa: list of 3-5 objects [{question: string, answer: string}] "
                    "  (likely application screening questions with suggested answers written "
                    "  in first person as the candidate, based on their actual experience)\n\n"
                    "Scoring guide:\n"
                    "90-100: Excellent match, candidate clearly qualifies\n"
                    "70-89: Good match, meets most requirements\n"
                    "50-69: Partial match, some gaps\n"
                    "0-49: Poor match\n\n"
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CANDIDATE PROFILE:\n{profile_text}\n\n"
                    f"JOB POSTING:\n{job_text}"
                ),
            },
        ],
    )

    result = json.loads(response.choices[0].message.content)
    job["score"] = result.get("score", 0)
    job["match_summary"] = result.get("match_summary", "")
    job["red_flags"] = result.get("red_flags", [])
    job["typical_qa"] = result.get("typical_qa", [])
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
