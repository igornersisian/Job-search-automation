"""
Trigger Apify Wellfound (AngelList) Scraper and return job list.

Actor: clearpath/wellfound-api-ppe
Docs:  https://apify.com/clearpath/wellfound-api-ppe

This actor takes Wellfound URLs (not keywords).  We convert search_keywords
into role-slug URLs like https://wellfound.com/role/r/automation-engineer.

Usage:
    python tools/run_wellfound_search.py
"""

import os
import json
import time
import logging
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ACTOR_ID = "clearpath~wellfound-api-ppe"


def _get_token() -> str:
    return os.environ["APIFY_API_TOKEN"]


# Valid Wellfound role slugs that match our search profile.
# These are actual predefined roles on wellfound.com — arbitrary slugs return 404.
# Update this list if search_keywords change significantly.
DEFAULT_WELLFOUND_ROLES = [
    "automation-engineer",
    "artificial-intelligence-engineer",
    "software-engineer",
    "devops-engineer",
    "full-stack-engineer",
    "backend-engineer",
    "data-engineer",
    "machine-learning-engineer",
]


def _build_urls(profile: dict | None = None) -> list[str]:
    """Build Wellfound search URLs from profile config or defaults.

    If profile contains 'wellfound_urls' (list of full URLs), use those directly.
    Otherwise fall back to DEFAULT_WELLFOUND_ROLES.
    """
    if profile and profile.get("wellfound_urls"):
        return profile["wellfound_urls"]

    return [
        f"https://wellfound.com/role/r/{slug}"
        for slug in DEFAULT_WELLFOUND_ROLES
    ]


def run_actor(profile: dict | None = None) -> str:
    url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/runs"
    params = {"token": _get_token()}
    search_urls = _build_urls(profile)

    actor_input = {
        "urls": search_urls,
        "pageLimit": 3,
        "onlyRemoteJobs": True,
        "sortBy": "LAST_POSTED",
    }
    logger.info(f"Starting Wellfound actor run with {len(search_urls)} URLs...")
    resp = httpx.post(url, json=actor_input, params=params, timeout=30)
    if resp.status_code >= 400:
        logger.error(f"Wellfound actor start failed ({resp.status_code}): {resp.text[:500]}")
    resp.raise_for_status()
    run_id = resp.json()["data"]["id"]
    logger.info(f"Wellfound actor run started: {run_id}")
    return run_id


def wait_for_run(run_id: str, timeout_seconds: int = 600) -> str:
    url = f"https://api.apify.com/v2/actor-runs/{run_id}"
    params = {"token": _get_token()}
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        resp = httpx.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()["data"]
        status = data["status"]

        if status == "SUCCEEDED":
            dataset_id = data["defaultDatasetId"]
            logger.info(f"Wellfound run succeeded. Dataset: {dataset_id}")
            return dataset_id
        elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
            raise RuntimeError(f"Wellfound actor run ended with status: {status}")

        logger.info(f"Wellfound run status: {status}. Waiting...")
        time.sleep(10)

    raise TimeoutError(f"Wellfound actor run {run_id} did not finish within {timeout_seconds}s")


def fetch_dataset(dataset_id: str) -> list[dict]:
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    params = {"token": _get_token(), "format": "json", "clean": "true"}
    logger.info(f"Downloading Wellfound dataset {dataset_id}...")
    resp = httpx.get(url, params=params, timeout=60)
    resp.raise_for_status()
    items = resp.json()
    logger.info(f"Downloaded {len(items)} Wellfound jobs")
    return items


def normalise_wellfound(raw: dict) -> dict:
    """Map Wellfound output fields to the shared job schema."""
    # Salary — use parsed base_salary if available, else raw compensation string
    salary_text = ""
    base = raw.get("base_salary") or {}
    if base and base.get("min_value"):
        lo = base.get("min_value")
        hi = base.get("max_value")
        currency = base.get("currency", "USD")
        symbol = "$" if currency == "USD" else currency + " "
        if lo and hi:
            salary_text = f"{symbol}{int(lo):,}-{symbol}{int(hi):,}/Year"
        elif lo:
            salary_text = f"{symbol}{int(lo):,}+/Year"
    if not salary_text:
        salary_text = raw.get("compensation", "")

    # Equity info — append if present
    equity = raw.get("equity_parsed") or {}
    if equity.get("has_equity"):
        eq_min = equity.get("min_percentage", 0)
        eq_max = equity.get("max_percentage", 0)
        if eq_min and eq_max:
            salary_text = f"{salary_text} + {eq_min}%-{eq_max}% equity".strip()

    # Location
    locations = raw.get("location_names") or []
    location_str = ", ".join(locations[:3])

    # Posted date
    posted_at = raw.get("posted_at") or None
    if not posted_at:
        ts = raw.get("live_start_at")
        if ts:
            posted_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    # URL — construct from id+slug for working links
    # Wellfound format: https://wellfound.com/jobs/{id}-{slug}
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
        "description": (raw.get("description") or "")[:5000],
        "location": location_str,
        "postedAt": posted_at,
        "is_remote": raw.get("remote", False),
        "source": "wellfound",
    }


def run_search(keywords: list[str], profile: dict | None = None) -> list[dict]:
    """Full flow: trigger → wait → fetch → normalise.

    keywords arg is accepted for interface consistency with other scrapers
    but ignored — Wellfound uses predefined role URLs, not free-text search.
    Pass profile to allow custom wellfound_urls from profile config.
    """
    run_id = run_actor(profile)
    dataset_id = wait_for_run(run_id)
    raw_items = fetch_dataset(dataset_id)
    return [normalise_wellfound(item) for item in raw_items]


if __name__ == "__main__":
    jobs = run_search([])
    print(json.dumps(jobs, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(jobs)} Wellfound jobs fetched.")
