"""
Indeed jobs via Apify, on the shared runner.

Actor: valig/indeed-jobs-scraper (id valig~indeed-jobs-scraper)
One actor run per keyword (the actor takes a single title), fanned out in
PARALLEL and merged. datePosted granularity is whole days only, so the lookback
window effectively stays 24h; cross-run dedup removes the slot overlap.

Exposes `fetch(keywords, *, lookback)` -> apify_client.SourceResult.
"""

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

import apify_client

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ACTOR_ID = "valig~indeed-jobs-scraper"
LIMIT = 25  # per keyword → cap / truncation threshold


def normalise_indeed(raw: dict) -> dict:
    """Map Indeed output fields to the shared job schema."""
    salary_text = ""
    base = raw.get("baseSalary") or {}
    if base:
        lo = base.get("min")
        hi = base.get("max")
        currency = base.get("currencyCode") or "USD"
        unit = (base.get("unitOfWork") or "").capitalize()
        symbol = "$" if currency == "USD" else currency + " "
        if lo and hi:
            salary_text = f"{symbol}{lo:,}-{symbol}{hi:,}/{unit}".strip()
        elif lo:
            salary_text = f"{symbol}{lo:,}+/{unit}".strip()

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


def _fetch_one(keyword: str, date_posted: str) -> apify_client.SourceResult:
    actor_input = {
        "title": keyword,
        "country": "us",
        "location": "remote",
        "datePosted": date_posted,
        "limit": LIMIT,
    }
    return apify_client.run_actor_job(
        ACTOR_ID, actor_input,
        source="indeed", normalise=normalise_indeed, cap=LIMIT,
    )


def fetch(keywords: list[str], *, lookback: int = 86400) -> apify_client.SourceResult:
    """Run one Indeed search per keyword in parallel and merge."""
    if not keywords:
        return apify_client.SourceResult(source="indeed")
    date_posted = str(max(1, math.ceil(lookback / 86400)))  # whole days; min 1
    with ThreadPoolExecutor(max_workers=min(5, len(keywords))) as ex:
        results = list(ex.map(lambda kw: _fetch_one(kw, date_posted), keywords))
    return apify_client.merge_results("indeed", results)


if __name__ == "__main__":
    r = fetch(["automation engineer", "AI workflow"], lookback=86400)
    print(json.dumps(r.items, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(r.items)} Indeed jobs, ${r.cost_usd:.4f}, capped={r.capped}, error={r.error}")
