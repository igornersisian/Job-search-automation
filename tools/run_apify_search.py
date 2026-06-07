"""
LinkedIn jobs via Apify, on the shared runner.

Actor: cheap_scraper/linkedin-job-scraper (id 2rJKkhh7vjpX7pvjg)
Pricing: pay-per-result (~$0.70/1000 free tier) — narrows with the lookback window.

Exposes `fetch(keywords, *, lookback)` -> apify_client.SourceResult.
The start/poll/fetch/retry/cap/cost lifecycle lives in apify_client.run_actor_job.
"""

import json

from dotenv import load_dotenv

import apify_client
from log_setup import get_logger

load_dotenv()

logger = get_logger(__name__)

ACTOR_ID = "2rJKkhh7vjpX7pvjg"  # cheap_scraper/linkedin-job-scraper
MAX_ITEMS = 400                 # our cap → also the truncation threshold
                                # (raised 150→400 to stop truncating the 24h window)


def _published_at(lookback: int) -> str:
    """LinkedIn's publishedAt is an ENUM ('', r86400, r604800, r2592000) — it
    rejects arbitrary seconds (HTTP 400). Snap lookback up to the nearest
    allowed bucket. So LinkedIn effectively can't go below 24h."""
    if lookback <= 86400:
        return "r86400"      # last 24 hours (minimum granularity)
    if lookback <= 604800:
        return "r604800"     # last 7 days
    return "r2592000"        # last 30 days


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
        "publishedAt": _published_at(lookback),   # enum-only; 24h minimum
        "workType": ["remote"],
        # Source-side seniority filter: keep everything up to mid-senior, drop only
        # the clearly-too-senior tier (director). Recall-safe — mid-senior still
        # includes the 2-4yr roles that actually match; scoring handles the rest.
        "experienceLevel": ["internship", "entry-level", "associate", "mid-senior"],
        "jobType": ["full-time", "contract"],
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
