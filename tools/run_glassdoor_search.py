"""
Glassdoor jobs via Apify, on the shared runner.

Actor: valig/glassdoor-jobs-scraper (id 5OaooRg0FxlRF0L1B)
One actor run per keyword, fanned out in PARALLEL and merged. daysOld is
whole-days only, so the window effectively stays 24h; cross-run dedup removes
slot overlap. High match volume at low cost (~$0.08/run).

Exposes `fetch(keywords, *, lookback)` -> apify_client.SourceResult.
"""

import json
import math
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

import apify_client
from normalise_utils import format_salary
from log_setup import get_logger

load_dotenv()

logger = get_logger(__name__)

ACTOR_ID = "5OaooRg0FxlRF0L1B"  # valig/glassdoor-jobs-scraper
LIMIT = 100  # per keyword → cap / truncation threshold (raised 25→100; cheap source)


def normalise_glassdoor(raw: dict) -> dict:
    """Map Glassdoor output fields to the shared job schema."""
    pay = raw.get("pay") or {}
    period = (pay.get("period") or "").capitalize()
    salary_text = format_salary(
        pay.get("min"), pay.get("max"),
        currency=pay.get("currency") or "USD",
        sep="–",  # en-dash, as before
        suffix=f" {period}",
    )

    location = raw.get("location") or {}
    location_str = location.get("name", "")
    employer = raw.get("employer") or {}
    job_url = raw.get("seoUrl") or raw.get("url", "")

    posted_at = None
    age = raw.get("ageInDays")
    if age is not None:
        posted_at = (datetime.now(timezone.utc) - timedelta(days=age)).isoformat()

    return {
        "id": str(raw.get("id", "")) or job_url[:200],
        "title": raw.get("title", ""),
        "company": employer.get("name", ""),
        "url": job_url,
        "salary": salary_text,
        "description": raw.get("description") or "",
        "location": location_str,
        "postedAt": posted_at,
        "is_remote": False,
        "source": "glassdoor",
    }


def fetch(keywords: list[str], *, lookback: int = 86400) -> apify_client.SourceResult:
    """Run one Glassdoor search per keyword in parallel and merge."""
    days_old = max(1, math.ceil(lookback / 86400))  # whole days; min 1
    return apify_client.fan_out_keywords(
        ACTOR_ID, keywords,
        lambda kw: {"keywords": kw, "location": "United States", "daysOld": days_old, "limit": LIMIT},
        source="glassdoor", normalise=normalise_glassdoor, cap=LIMIT,
    )


if __name__ == "__main__":
    r = fetch(["automation engineer", "AI engineer"], lookback=86400)
    print(json.dumps(r.items, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(r.items)} Glassdoor jobs, ${r.cost_usd:.4f}, capped={r.capped}, error={r.error}")
