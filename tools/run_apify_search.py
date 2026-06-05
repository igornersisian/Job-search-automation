"""
LinkedIn jobs via Apify, on the shared runner.

Actor: cheap_scraper/linkedin-job-scraper (id 2rJKkhh7vjpX7pvjg)
Pricing: pay-per-result (~$0.70/1000 free tier) — narrows with the lookback window.

Exposes `fetch(keywords, *, lookback)` -> apify_client.SourceResult.
The start/poll/fetch/retry/cap/cost lifecycle lives in apify_client.run_actor_job.
"""

import json
import logging

from dotenv import load_dotenv

import apify_client

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ACTOR_ID = "2rJKkhh7vjpX7pvjg"  # cheap_scraper/linkedin-job-scraper
MAX_ITEMS = 150                 # our cap → also the truncation threshold


def normalise_linkedin(raw: dict) -> dict:
    """Map cheap_scraper/linkedin-job-scraper output to the shared job schema."""
    salary_info = raw.get("salaryInfo") or []
    salary_text = " - ".join(salary_info) if salary_info else ""

    return {
        "id": raw.get("jobId", ""),
        "title": raw.get("jobTitle", ""),
        "company": raw.get("companyName", ""),
        "url": raw.get("jobUrl") or raw.get("applyUrl", ""),
        "salary": salary_text,
        "description": raw.get("jobDescription", ""),
        "location": raw.get("location", ""),
        "postedAt": raw.get("publishedAt") or raw.get("postedTime", ""),
        "is_remote": "remote" in (raw.get("workType") or "").lower(),
        "source": "linkedin",
    }


def fetch(keywords: list[str], *, lookback: int = 86400) -> apify_client.SourceResult:
    """Run the LinkedIn actor for the given keywords over the lookback window."""
    actor_input = {
        "keyword": keywords,
        "publishedAt": f"r{lookback}",   # relative seconds, e.g. r25200 = last 7h
        "workType": ["remote"],
        "maxItems": MAX_ITEMS,
        "saveOnlyUniqueItems": True,
    }
    return apify_client.run_actor_job(
        ACTOR_ID, actor_input,
        source="linkedin", normalise=normalise_linkedin, cap=MAX_ITEMS,
    )


if __name__ == "__main__":
    r = fetch(["automation engineer", "AI engineer"], lookback=86400)
    print(json.dumps(r.items, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(r.items)} LinkedIn jobs, ${r.cost_usd:.4f}, capped={r.capped}, error={r.error}")
