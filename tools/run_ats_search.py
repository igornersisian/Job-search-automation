"""
ATS jobs via Apify, on the shared runner.

Actor: jobo.world/ats-jobs-search (id NDli5o5pYKW1atJAY)
Searches 13 ATS platforms (Greenhouse, Lever, Workday, Ashby, Workable,
SmartRecruiters, BambooHR, Rippling, Personio, JazzHR, Breezy HR, Recruitee,
Polymer). Direct company sources, zero ghost jobs. Highest avg match quality.

Pricing: pay-per-result (~$4/1000 effective on free tier). page_size=100 is OUR
cap → `capped` flags possible truncation. The actor caps `queries` at 5 items.

Exposes `fetch(keywords, *, lookback)` -> apify_client.SourceResult.
"""

import json
import logging
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

import apify_client
from normalise_utils import format_salary

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ACTOR_ID = "NDli5o5pYKW1atJAY"  # jobo.world/ats-jobs-search
PAGE_SIZE = 100                 # results per page (actor max)
MAX_PAGES = 3                   # ceiling → up to 300 results/run before we stop
MAX_QUERIES = 5                 # actor rejects >5 (HTTP 400)


def normalise_ats(raw: dict) -> dict:
    """Map ATS Jobs Search output to the shared job schema."""
    comp = raw.get("compensation") or {}
    period = (comp.get("period") or "").capitalize()
    salary_text = format_salary(
        comp.get("min"), comp.get("max"),
        currency=comp.get("currency") or "USD",
        suffix=f" {period}",
    )

    locations = raw.get("locations") or []
    location_parts = []
    for loc in locations[:3]:
        loc_str = loc.get("location") or ""
        if not loc_str:
            parts = [loc.get("city"), loc.get("state"), loc.get("country")]
            loc_str = ", ".join(p for p in parts if p)
        if loc_str:
            location_parts.append(loc_str)
    location_str = "; ".join(location_parts)

    company = raw.get("company") or {}
    company_name = company.get("name", "") if isinstance(company, dict) else str(company)
    ats_source = raw.get("source", "unknown")

    return {
        "id": raw.get("id") or raw.get("source_id", ""),
        "title": raw.get("title", ""),
        "company": company_name,
        "url": raw.get("listing_url") or raw.get("apply_url", ""),
        "salary": salary_text,
        "description": raw.get("description") or "",
        "location": location_str,
        "postedAt": raw.get("date_posted"),
        "is_remote": raw.get("is_remote", False),
        "source": f"ats-{ats_source}",
    }


def fetch(keywords: list[str], *, lookback: int = 86400) -> apify_client.SourceResult:
    """Run the ATS actor for the first 5 keywords over the lookback window,
    paginating until a page comes back not-full (window exhausted) or we hit
    MAX_PAGES. `capped` on the merged result means we stopped at the ceiling
    with a still-full last page — i.e. there were more we didn't fetch."""
    since = (datetime.now(timezone.utc) - timedelta(seconds=lookback)).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = {
        "queries": keywords[:MAX_QUERIES],
        "is_remote": True,
        "posted_after": since,
        "page_size": PAGE_SIZE,
    }
    pages: list[apify_client.SourceResult] = []
    truncated = False
    for page in range(1, MAX_PAGES + 1):
        # cap=None per page: a full page is normal here, not a truncation signal —
        # we decide truncation ourselves from whether the LAST page was full.
        r = apify_client.run_actor_job(
            ACTOR_ID, {**base, "page": page},
            source="ats", normalise=normalise_ats, cap=None,
        )
        pages.append(r)
        if r.error or r.fetched < PAGE_SIZE:
            break  # actor error, or window exhausted (partial page) → stop
        if page == MAX_PAGES:
            truncated = True  # last allowed page was full → more remain beyond ceiling

    merged = apify_client.merge_results("ats", pages)
    merged.capped = truncated
    logger.info(f"ATS paginated {len(pages)} page(s), {merged.fetched} fetched, truncated={truncated}")
    return merged


if __name__ == "__main__":
    r = fetch(["automation engineer", "AI engineer"], lookback=86400)
    print(json.dumps(r.items, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(r.items)} ATS jobs, ${r.cost_usd:.4f}, capped={r.capped}, error={r.error}")
