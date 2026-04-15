"""
Trigger Apify ATS Jobs Search actor and return normalised job list.

Actor: jobo.world/ats-jobs-search
ID:    NDli5o5pYKW1atJAY
Docs:  https://console.apify.com/actors/NDli5o5pYKW1atJAY

Searches 13 ATS platforms (Greenhouse, Lever, Workday, Ashby, Workable,
SmartRecruiters, BambooHR, Rippling, Personio, JazzHR, Breezy HR,
Recruitee, Polymer) for remote jobs. Direct company sources, zero ghost jobs.

Usage:
    python tools/run_ats_search.py
"""

import os
import json
import time
import logging
from datetime import datetime, timezone, timedelta

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ACTOR_ID = "NDli5o5pYKW1atJAY"  # jobo.world/ats-jobs-search


def _get_token() -> str:
    return os.environ["APIFY_API_TOKEN"]


def run_actor(keywords: list[str]) -> str:
    """Start the ATS Jobs Search actor run and return run_id."""
    url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/runs"
    params = {"token": _get_token()}

    # posted_after = 24 hours ago
    since = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    actor_input = {
        "queries": keywords[:10],
        "is_remote": True,
        "posted_after": since,
        "page_size": 100,
        "page": 1,
    }

    logger.info(f"Starting ATS actor run with {len(actor_input['queries'])} queries...")
    resp = httpx.post(url, json=actor_input, params=params, timeout=30)
    if resp.status_code >= 400:
        logger.error(f"ATS actor start failed ({resp.status_code}): {resp.text[:500]}")
    resp.raise_for_status()

    run_id = resp.json()["data"]["id"]
    logger.info(f"ATS actor run started: {run_id}")
    return run_id


def wait_for_run(run_id: str, timeout_seconds: int = 600) -> str:
    """Poll until the run is SUCCEEDED, return dataset_id."""
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
            logger.info(f"ATS run succeeded. Dataset: {dataset_id}")
            return dataset_id
        elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
            raise RuntimeError(f"ATS actor run {run_id} ended with status: {status}")

        logger.info(f"ATS run status: {status}. Waiting...")
        time.sleep(10)

    raise TimeoutError(f"ATS actor run {run_id} did not finish within {timeout_seconds}s")


def fetch_dataset(dataset_id: str) -> list[dict]:
    """Download all items from an Apify dataset."""
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    params = {"token": _get_token(), "format": "json", "clean": "true"}

    logger.info(f"Downloading ATS dataset {dataset_id}...")
    resp = httpx.get(url, params=params, timeout=60)
    resp.raise_for_status()

    items = resp.json()
    logger.info(f"Downloaded {len(items)} ATS jobs")
    return items


def normalise_ats(raw: dict) -> dict:
    """Map ATS Jobs Search output to the shared job schema."""
    # Salary from compensation object
    salary_text = ""
    comp = raw.get("compensation") or {}
    if comp:
        lo = comp.get("min")
        hi = comp.get("max")
        currency = comp.get("currency") or "USD"
        period = (comp.get("period") or "").capitalize()
        symbol = "$" if currency == "USD" else currency + " "
        if lo and hi:
            salary_text = f"{symbol}{lo:,}-{symbol}{hi:,} {period}".strip()
        elif lo:
            salary_text = f"{symbol}{lo:,}+ {period}".strip()

    # Location from locations array
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

    # Company name
    company = raw.get("company") or {}
    company_name = company.get("name", "") if isinstance(company, dict) else str(company)

    # ATS source (greenhouse, lever_co, workable, etc.)
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


def run_search(keywords: list[str]) -> list[dict]:
    """Full flow: trigger -> wait -> fetch -> normalise. Returns list of job dicts."""
    run_id = run_actor(keywords)
    dataset_id = wait_for_run(run_id)
    raw_items = fetch_dataset(dataset_id)
    return [normalise_ats(item) for item in raw_items]


if __name__ == "__main__":
    jobs = run_search(["automation engineer", "AI engineer"])
    print(json.dumps(jobs, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(jobs)} ATS jobs fetched.")
