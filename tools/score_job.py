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
    "block2": {"tools": 0, "experience": 0, "location": 0, "red_flags": 0},
}

# Sub-score caps for server-side clamping
_CAPS = {
    "domain": 25, "patterns": 20, "role": 15,
    "tools": 15, "experience": 10, "location": 10, "red_flags": 5,
}


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
        f"Description:\n{description[:4000]}"
    )
    profile_text = json.dumps(profile, indent=2)

    # Custom dealbreakers from profile
    dealbreakers = profile.get("custom_red_flags") or []

    # ── Build system prompt ──
    system_content = (
        "You are a job-candidate fit evaluator. "
        "Score ONLY using facts explicitly stated in the candidate profile. "
        "Do NOT infer, assume, or guess any skills, experience, or qualifications.\n\n"
    )

    if dealbreakers:
        items = "\n".join(f"- {d}" for d in dealbreakers)
        system_content += (
            "DEALBREAKERS — CHECK FIRST:\n"
            "If ANY of these apply to this job, set ALL sub-scores to 0 "
            "and total score to 0. No exceptions:\n"
            f"{items}\n\n"
        )

    system_content += (
        "YOUR TASK — analyse the job against the candidate in THREE steps:\n\n"

        "STEP 1: Write match_summary (1-2 sentences).\n"
        "Address the candidate directly (use 'you/your'). "
        "State clearly whether this job fits or not and WHY. "
        "Mention the most important match or mismatch.\n\n"

        "STEP 2: List red_flags.\n"
        "Every genuine concern, mismatch, or dealbreaker. "
        "Include as many or as few as actually exist. "
        "If zero real concerns, return an empty list. "
        "Do NOT duplicate the same concern in different words.\n\n"

        "STEP 3: Assign sub-scores.\n"
        "Your scores MUST be consistent with your match_summary and red_flags above. "
        "If you wrote that something is a problem, the corresponding score must be LOW. "
        "If you flagged 'hybrid' as a red flag, location must be 0-2. "
        "If you flagged seniority mismatch, role must reflect that.\n\n"

        "SUB-SCORES:\n\n"

        "BLOCK 1 — Domain & Capability Fit (max 60 points):\n"
        "  domain (0-25): How well does the candidate's actual domain match the job's "
        "domain? Be precise about domain boundaries — adjacent-sounding fields can be "
        "very different in practice. For example: using AI APIs/integrations is different "
        "from building/training ML models; front-end web dev is different from embedded "
        "systems; DevOps is different from software architecture; "
        "product management is different from engineering. "
        "If the candidate works in a fundamentally different domain, score 0-5. "
        "If related but not the same, score 8-15. If strong match, score 18-25.\n"
        "  patterns (0-20): Does the candidate have experience with the architectural "
        "patterns and categories of work the job needs? Use semantic matching — "
        "similar patterns in different tools count partially. "
        "Completely unrelated patterns score 0-3.\n"
        "  role (0-15): Does the candidate's seniority match the job's level? "
        "Compare the candidate's actual years of experience and role history against "
        "the job's seniority expectations. "
        "If the job is 3+ levels above the candidate (e.g. candidate is junior, "
        "job is Director/VP/Head/Principal), cap at 2. "
        "If the job is 2 levels above (e.g. candidate is junior, job is Senior/Staff), "
        "cap at 5. If close match, score 10-15. "
        "This also works in reverse: if the candidate is very senior and the job is "
        "clearly junior, cap at 5 (overqualified).\n\n"

        "BLOCK 2 — Formal Requirements Fit (max 40 points):\n"
        "  tools (0-15): Does the candidate know the specific tools, languages, and "
        "frameworks the job requires? Compare the candidate's ACTUAL toolset against "
        "what's listed in the job. Similar/adjacent tools give partial credit (e.g. "
        "React vs Vue = partial, Python vs Java = low). "
        "If the candidate's entire approach to work is fundamentally different from "
        "what the job requires (e.g. no-code vs traditional engineering, or vice versa), "
        "score 0-3 unless the job explicitly values their approach.\n"
        "  experience (0-10): Years of experience match. If the job states N+ years "
        "required and the candidate has fewer, apply -5 per each missing year "
        "(starting from 10, floor at 0). Example: job asks 5+, candidate has 1 → "
        "4 years short → penalty -20 → capped at 0. "
        "If the job does not mention years of experience, give 8.\n"
        "  location (0-10): Does the candidate's location/remote preference match? "
        "If the job is hybrid, on-site, office-based, or requires specific country "
        "residence/clearance/visa that the candidate cannot meet, give 0. "
        "If job is fully remote with no geographic restrictions, give 10.\n"
        "  red_flags (0-5): 5 = no concerns. "
        "Deduct 1 for each red flag you listed above. 0 = multiple serious concerns.\n\n"

        "CONSISTENCY RULE:\n"
        "Before returning, re-read your match_summary and red_flags. "
        "If you described a serious mismatch but gave a high score for that dimension, "
        "FIX THE SCORE to match your analysis. Your text analysis is the truth — "
        "the numbers must follow it, not the other way around.\n\n"

        "RULES:\n"
        "- Be conservative: when uncertain, score lower rather than higher\n"
        "- Score 70+ total ONLY if the candidate could realistically compete\n\n"

        "Return ONLY this JSON (match_summary and red_flags FIRST, then scores):\n"
        "{\n"
        '  "match_summary": "<1-2 sentences>",\n'
        '  "red_flags": ["<flag1>", ...],\n'
        '  "block1": {"domain": <int>, "patterns": <int>, "role": <int>},\n'
        '  "block2": {"tools": <int>, "experience": <int>, "location": <int>, "red_flags": <int>}\n'
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
    b2 = {k: min(max(b2.get(k, 0), 0), _CAPS[k]) for k in ("tools", "experience", "location", "red_flags")}

    computed = sum(b1.values()) + sum(b2.values())

    job["score"] = computed
    job["score_breakdown"] = {"block1": dict(b1), "block2": dict(b2)}
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
