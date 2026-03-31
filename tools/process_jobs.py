"""
Daily pipeline orchestrator.

Flow:
  1. Run Apify LinkedIn + Glassdoor searches → merge job lists
  2. For each job:
     a. Deduplicate via Supabase
     b. Filter internship/junior
     c. Score vs resume profile
     d. If score >= 70 → send Telegram card + save as "sent"
     e. Else → save as "low_score"
  3. Send daily summary to Telegram

Usage:
    python tools/process_jobs.py
"""

import os
import sys
import json
import logging
from datetime import datetime

from supabase import create_client, Client
from dotenv import load_dotenv

# Import sibling tools
sys.path.insert(0, os.path.dirname(__file__))
from run_apify_search import run_search as run_linkedin_search
from run_glassdoor_search import run_search as run_glassdoor_search
from score_job import score_job, is_junior_or_intern
from notify_telegram import send_job_card, send_daily_summary

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SCORE_THRESHOLD = 70

_supabase: Client | None = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        )
    return _supabase


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def get_profile() -> dict | None:
    result = get_supabase().table("profile").select("parsed").order("updated_at", desc=True).limit(1).execute()
    if result.data:
        return result.data[0]["parsed"]
    return None


def is_already_seen(job_id: str) -> bool:
    result = get_supabase().table("jobs").select("id").eq("id", job_id).execute()
    return len(result.data) > 0


def save_job(job: dict, status: str) -> None:
    """Insert job record into Supabase."""
    get_supabase().table("jobs").upsert({
        "id": job.get("id") or job.get("jobId") or job.get("url", "")[:200],
        "source": "linkedin",
        "title": job.get("title", ""),
        "company": job.get("company", ""),
        "url": job.get("url") or job.get("jobUrl", ""),
        "salary_text": job.get("salary") or job.get("salaryText", ""),
        "is_remote": True,
        "description": job.get("description", "")[:5000],
        "posted_at": job.get("postedAt") or job.get("publishedAt"),
        "score": job.get("score"),
        "match_summary": job.get("match_summary", ""),
        "red_flags": json.dumps(job.get("red_flags", [])),
        "typical_qa": json.dumps(job.get("typical_qa", [])),
        "status": status,
        "created_at": datetime.utcnow().isoformat(),
    }).execute()


# ---------------------------------------------------------------------------
# Normalise Apify job fields
# ---------------------------------------------------------------------------

def normalise_job(raw: dict) -> dict:
    """Map Apify actor output fields to our standard schema.
    Glassdoor jobs are already normalised by run_glassdoor_search.py,
    so we only remap LinkedIn-style fields when the canonical ones are missing.
    """
    return {
        "id": raw.get("id") or raw.get("jobId") or raw.get("trackingUrn", ""),
        "title": raw.get("title") or raw.get("jobTitle", ""),
        "company": raw.get("company") or raw.get("companyName", ""),
        "url": raw.get("url") or raw.get("jobUrl", ""),
        "salary": raw.get("salary") or raw.get("salaryText", ""),
        "description": raw.get("description") or raw.get("jobDescription", ""),
        "location": raw.get("location") or raw.get("locationText", ""),
        "postedAt": raw.get("postedAt") or raw.get("publishedAt"),
        "source": raw.get("source", "linkedin"),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    logger.info("=== Daily job search pipeline started ===")

    # Load profile
    profile = get_profile()
    if not profile:
        logger.error("No resume profile in Supabase. Send PDF to Telegram bot first.")
        return

    # Fetch jobs from LinkedIn and Glassdoor in sequence
    raw_jobs: list[dict] = []

    logger.info("Fetching jobs from LinkedIn (Apify)...")
    try:
        linkedin_jobs = run_linkedin_search()
        for job in linkedin_jobs:
            job.setdefault("source", "linkedin")
        raw_jobs.extend(linkedin_jobs)
        logger.info(f"LinkedIn: {len(linkedin_jobs)} jobs")
    except Exception as e:
        logger.error(f"LinkedIn search failed: {e}")

    logger.info("Fetching jobs from Glassdoor (Apify)...")
    try:
        glassdoor_jobs = run_glassdoor_search()
        raw_jobs.extend(glassdoor_jobs)
        logger.info(f"Glassdoor: {len(glassdoor_jobs)} jobs")
    except Exception as e:
        logger.error(f"Glassdoor search failed: {e}")

    if not raw_jobs:
        logger.error("No jobs fetched from any source.")
        return

    logger.info(f"Total raw jobs: {len(raw_jobs)}")

    sent = 0
    skipped_dupe = 0
    skipped_junior = 0
    skipped_score = 0

    for raw in raw_jobs:
        job = normalise_job(raw)

        if not job["id"] or not job["title"]:
            continue

        # Deduplication
        if is_already_seen(job["id"]):
            skipped_dupe += 1
            continue

        # Junior/intern filter
        if is_junior_or_intern(job):
            logger.info(f"[JUNIOR] {job['title']} @ {job['company']}")
            save_job(job, "filtered_junior")
            skipped_junior += 1
            continue

        # Score vs profile
        try:
            job = score_job(job, profile)
        except Exception as e:
            logger.error(f"Scoring failed for {job['title']}: {e}")
            save_job(job, "score_error")
            continue

        score = job.get("score", 0)
        logger.info(f"[{score}/100] {job['title']} @ {job['company']}")

        if score < SCORE_THRESHOLD:
            save_job(job, "low_score")
            skipped_score += 1
            continue

        # Send to Telegram
        if send_job_card(job):
            save_job(job, "sent")
            sent += 1
        else:
            save_job(job, "notify_failed")

    logger.info(
        f"Pipeline done — sent: {sent}, "
        f"low score: {skipped_score}, "
        f"junior: {skipped_junior}, "
        f"dupes: {skipped_dupe}"
    )

    send_daily_summary(
        sent=sent,
        skipped_score=skipped_score,
        skipped_junior=skipped_junior,
        skipped_dupe=skipped_dupe,
    )


if __name__ == "__main__":
    run_pipeline()
