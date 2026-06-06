"""
Wellfound (AngelList) jobs via Apify, on the shared runner.

Actor: clearpath/wellfound-api-ppe (id clearpath~wellfound-api-ppe)
Wellfound has no free-text search — only fixed role-category URLs
(wellfound.com/role/r/{slug}). We build URLs from `wellfound_roles` in the
profile (set via /wellfound). NOTE: generic roles like "Software Engineer" pull
traditional senior-SWE jobs that score ~0 for an AI-automation profile — set
AI/ML-relevant roles for this source to be worth anything.

Exposes `fetch(keywords, *, lookback, profile)` -> apify_client.SourceResult.
"""

import json
import re
from datetime import datetime, timezone

from dotenv import load_dotenv

import apify_client
from normalise_utils import format_salary
from log_setup import get_logger

load_dotenv()

logger = get_logger(__name__)

ACTOR_ID = "clearpath~wellfound-api-ppe"


def _role_to_slug(role: str) -> str:
    """'Software Engineer' -> 'software-engineer'."""
    slug = role.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _build_urls(roles: list[str]) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for role in roles:
        slug = _role_to_slug(role)
        if slug and slug not in seen:
            seen.add(slug)
            urls.append(f"https://wellfound.com/role/r/{slug}")
    return urls


def normalise_wellfound(raw: dict) -> dict:
    """Map Wellfound output fields to the shared job schema."""
    base = raw.get("base_salary") or {}
    lo, hi = base.get("min_value"), base.get("max_value")
    salary_text = format_salary(
        int(lo) if lo else None,
        int(hi) if hi else None,
        currency=base.get("currency") or "USD",
        suffix="/Year",
    )
    if not salary_text:
        salary_text = raw.get("compensation", "")

    equity = raw.get("equity_parsed") or {}
    if equity.get("has_equity"):
        eq_min = equity.get("min_percentage", 0)
        eq_max = equity.get("max_percentage", 0)
        if eq_min and eq_max:
            salary_text = f"{salary_text} + {eq_min}%-{eq_max}% equity".strip()

    locations = raw.get("location_names") or []
    location_str = ", ".join(locations[:3])

    posted_at = raw.get("posted_at") or None
    if not posted_at:
        ts = raw.get("live_start_at")
        if ts:
            posted_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    job_id = raw.get("id", "")
    slug = raw.get("slug", "")
    if job_id and slug:
        job_url = f"https://wellfound.com/jobs/{job_id}-{slug}"
    else:
        job_url = raw.get("url", "")

    return {
        "id": str(job_id or job_url[:200]),
        "title": raw.get("title", ""),
        "company": raw.get("company_name", ""),
        "url": job_url,
        "salary": salary_text,
        "description": raw.get("description") or "",
        "location": location_str,
        "postedAt": posted_at,
        "is_remote": raw.get("remote", False),
        "source": "wellfound",
    }


def fetch(keywords: list[str], *, lookback: int = 86400, profile: dict | None = None) -> apify_client.SourceResult:
    """Run Wellfound for the configured roles. `keywords`/`lookback` unused
    (Wellfound has no free-text search and no time filter — it sorts by
    LAST_POSTED and we take one page)."""
    roles = (profile or {}).get("wellfound_roles") or []
    if not roles:
        logger.info("No Wellfound roles configured — skipping. Use /wellfound to set roles.")
        return apify_client.SourceResult(source="wellfound")

    search_urls = _build_urls(roles)
    actor_input = {
        "urls": search_urls,
        "pageLimit": 1,
        "onlyRemoteJobs": True,
        "sortBy": "LAST_POSTED",
    }
    logger.info(f"Wellfound roles ({len(roles)}): {roles}")
    return apify_client.run_actor_job(
        ACTOR_ID, actor_input,
        source="wellfound", normalise=normalise_wellfound, cap=None,
    )


if __name__ == "__main__":
    r = fetch([], profile={"wellfound_roles": ["Machine Learning Engineer", "AI Engineer"]})
    print(json.dumps(r.items, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(r.items)} Wellfound jobs, ${r.cost_usd:.4f}, error={r.error}")
