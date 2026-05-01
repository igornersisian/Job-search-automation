"""
Trigger Apify Wellfound (AngelList) Scraper and return job list.

Actor: clearpath/wellfound-api-ppe
Docs:  https://apify.com/clearpath/wellfound-api-ppe

This actor takes Wellfound URLs (not keywords).  We convert wellfound_roles
(set via /wellfound bot command) into role-slug URLs like
https://wellfound.com/role/r/software-engineer.

Usage:
    python tools/run_wellfound_search.py
"""

import json
import re
import time
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv

import apify_client

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ACTOR_ID = "clearpath~wellfound-api-ppe"


def _role_to_slug(role: str) -> str:
    """Convert a Wellfound role name to a URL slug.

    'Software Engineer' → 'software-engineer'
    'H.R.' → 'hr'
    'Finance/Accounting' → 'finance-accounting'
    """
    slug = role.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _build_urls(roles: list[str]) -> list[str]:
    """Build Wellfound search URLs from role names.

    Uses validated role names from the /wellfound bot command.
    Deduplicates slugs.
    """
    seen: set[str] = set()
    urls: list[str] = []
    for role in roles:
        slug = _role_to_slug(role)
        if slug and slug not in seen:
            seen.add(slug)
            urls.append(f"https://wellfound.com/role/r/{slug}")
    return urls


def run_actor(roles: list[str]) -> tuple[str, str]:
    url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/runs"
    search_urls = _build_urls(roles)

    actor_input = {
        "urls": search_urls,
        "pageLimit": 1,
        "onlyRemoteJobs": True,
        "sortBy": "LAST_POSTED",
    }
    logger.info(f"Starting Wellfound actor run with {len(search_urls)} keyword URLs (pageLimit=1, {apify_client.active_slot_summary()})...")
    resp, token = apify_client.post(url, json_body=actor_input, timeout=30)
    if resp.status_code >= 400:
        logger.error(f"Wellfound actor start failed ({resp.status_code}): {resp.text[:500]}")
    resp.raise_for_status()
    run_id = resp.json()["data"]["id"]
    logger.info(f"Wellfound actor run started: {run_id}")
    return run_id, token


def wait_for_run(run_id: str, token: str, timeout_seconds: int = 600) -> str:
    url = f"https://api.apify.com/v2/actor-runs/{run_id}"
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        resp = apify_client.get(url, token, timeout=15)
        resp.raise_for_status()
        data = resp.json()["data"]
        status = data["status"]

        if status == "SUCCEEDED":
            dataset_id = data["defaultDatasetId"]
            logger.info(f"Wellfound run succeeded. Dataset: {dataset_id}")
            return dataset_id
        elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
            status_msg = data.get("statusMessage") or ""
            apify_client.report_run_failure(token, status, status_msg)
            raise RuntimeError(f"Wellfound actor run ended with status: {status} ({status_msg[:200]})")

        logger.info(f"Wellfound run status: {status}. Waiting...")
        time.sleep(10)

    raise TimeoutError(f"Wellfound actor run {run_id} did not finish within {timeout_seconds}s")


def fetch_dataset(dataset_id: str, token: str) -> list[dict]:
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    logger.info(f"Downloading Wellfound dataset {dataset_id}...")
    resp = apify_client.get(url, token, params={"format": "json", "clean": "true"}, timeout=60)
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
        "description": raw.get("description") or "",
        "location": location_str,
        "postedAt": posted_at,
        "is_remote": raw.get("remote", False),
        "source": "wellfound",
    }


def run_search(keywords: list[str], profile: dict | None = None) -> list[dict]:
    """Full flow: trigger → wait → fetch → normalise.

    Uses wellfound_roles from profile (set via /wellfound bot command).
    Falls back to empty list if no roles configured — skips Wellfound entirely.
    keywords arg is ignored (kept for interface consistency with other scrapers).
    """
    roles = (profile or {}).get("wellfound_roles") or []
    if not roles:
        logger.info("No Wellfound roles configured — skipping Wellfound search. Use /wellfound in bot to set roles.")
        return []

    logger.info(f"Wellfound roles ({len(roles)}): {roles}")
    run_id, token = run_actor(roles)
    dataset_id = wait_for_run(run_id, token)
    raw_items = fetch_dataset(dataset_id, token)
    return [normalise_wellfound(item) for item in raw_items]


if __name__ == "__main__":
    test_roles = ["Software Engineer", "Data Scientist"]
    # Simulate profile with wellfound_roles for standalone testing
    jobs = run_search([], {"wellfound_roles": test_roles})
    print(json.dumps(jobs, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(jobs)} Wellfound jobs fetched.")
