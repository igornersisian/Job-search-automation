"""
Indeed jobs via Apify, on the shared runner.

Actor: valig/indeed-jobs-scraper (id valig~indeed-jobs-scraper)
One actor run per keyword (the actor takes a single title), fanned out in
PARALLEL and merged. datePosted granularity is whole days only, so the lookback
window effectively stays 24h; cross-run dedup removes the slot overlap.

Exposes `fetch(keywords, *, lookback)` -> apify_client.SourceResult.
"""

import json
import math

from dotenv import load_dotenv

import apify_client
from normalise_utils import format_salary
from log_setup import get_logger

load_dotenv()

logger = get_logger(__name__)

ACTOR_ID = "valig~indeed-jobs-scraper"
LIMIT = 100  # per keyword → cap / truncation threshold (raised 25→100; free source)


def normalise_indeed(raw: dict) -> dict:
    """Map Indeed output fields to the shared job schema."""
    base = raw.get("baseSalary") or {}
    unit = (base.get("unitOfWork") or "").capitalize()
    salary_text = format_salary(
        base.get("min"), base.get("max"),
        currency=base.get("currencyCode") or "USD",
        suffix=f"/{unit}",
    )

    loc = raw.get("location") or {}
    parts = [loc.get("city"), loc.get("admin1Code"), loc.get("countryCode")]
    location_str = ", ".join(p for p in parts if p)

    desc = raw.get("description") or {}
    description = desc.get("text", "") if isinstance(desc, dict) else str(desc)
    job_url = raw.get("jobUrl") or raw.get("url", "")

    return {
        "id": raw.get("key") or job_url[:200],
        "title": raw.get("title", ""),
        "company": (raw.get("employer") or {}).get("name", ""),
        "url": job_url,
        "salary": salary_text,
        "description": description,
        "location": location_str,
        "postedAt": raw.get("datePublished"),
        "is_remote": True,   # we query location=remote
        "source": "indeed",
    }


def fetch(keywords: list[str], *, lookback: int = 86400) -> apify_client.SourceResult:
    """Run one Indeed search per keyword in parallel and merge."""
    date_posted = str(max(1, math.ceil(lookback / 86400)))  # whole days; min 1
    return apify_client.fan_out_keywords(
        ACTOR_ID, keywords,
        lambda kw: {"title": kw, "country": "us", "location": "remote", "datePosted": date_posted, "limit": LIMIT},
        source="indeed", normalise=normalise_indeed, cap=LIMIT,
    )


if __name__ == "__main__":
    r = fetch(["automation engineer", "AI workflow"], lookback=86400)
    print(json.dumps(r.items, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(r.items)} Indeed jobs, ${r.cost_usd:.4f}, capped={r.capped}, error={r.error}")
