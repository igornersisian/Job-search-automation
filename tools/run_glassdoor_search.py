"""
Trigger Apify Glassdoor Jobs Scraper and return job list.

Actor: silentflow/glassdoor-jobs-scraper-ppr
Docs:  https://console.apify.com/actors/QWGmrJFfdRhAzjZVu

Usage:
    python tools/run_glassdoor_search.py
"""

import os
import json
import time
import logging

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ACTOR_ID = "QWGmrJFfdRhAzjZVu"  # silentflow/glassdoor-jobs-scraper-ppr


def _get_token() -> str:
    return os.environ["APIFY_API_TOKEN"]

def run_actor(keywords: list[str]) -> str:
    url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/runs"
    params = {"token": _get_token()}
    actor_input = {
        "keywords": keywords,
        "country": "US",
        "location": "United States",
        "remoteWorkType": "true",
        "jobType": "fulltime",
        "fromAge": "3",       # last 3 days
        "maxItems": 100,
    }
    logger.info(f"Starting Glassdoor actor run with {len(keywords)} keywords...")
    resp = httpx.post(url, json=actor_input, params=params, timeout=30)
    if resp.status_code >= 400:
        logger.error(f"Glassdoor actor start failed ({resp.status_code}): {resp.text[:500]}")
    resp.raise_for_status()
    run_id = resp.json()["data"]["id"]
    logger.info(f"Glassdoor actor run started: {run_id}")
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
            logger.info(f"Glassdoor run succeeded. Dataset: {dataset_id}")
            return dataset_id
        elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
            raise RuntimeError(f"Glassdoor actor run ended with status: {status}")

        logger.info(f"Glassdoor run status: {status}. Waiting...")
        time.sleep(10)

    raise TimeoutError(f"Glassdoor actor run {run_id} did not finish within {timeout_seconds}s")


def fetch_dataset(dataset_id: str) -> list[dict]:
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    params = {"token": _get_token(), "format": "json", "clean": "true"}
    logger.info(f"Downloading Glassdoor dataset {dataset_id}...")
    resp = httpx.get(url, params=params, timeout=60)
    resp.raise_for_status()
    items = resp.json()
    logger.info(f"Downloaded {len(items)} Glassdoor jobs")
    return items


def normalise_glassdoor(raw: dict) -> dict:
    """Map Glassdoor output fields to the shared job schema."""
    salary = raw.get("job_salary") or {}
    salary_text = ""
    if salary:
        lo = salary.get("min")
        hi = salary.get("max")
        currency = salary.get("currency_symbol", "$")
        period = salary.get("pay_period", "")
        if lo and hi:
            salary_text = f"{currency}{lo:,}–{currency}{hi:,} {period}".strip()
        elif lo:
            salary_text = f"{currency}{lo:,}+ {period}".strip()

    location = raw.get("job_location") or {}
    location_str = location.get("unknown") or location.get("city") or ""

    return {
        "id": raw.get("job_url", "")[:200],
        "title": raw.get("job_title", ""),
        "company": raw.get("company_name", ""),
        "url": raw.get("job_url") or raw.get("job_apply_url", ""),
        "salary": salary_text,
        "description": raw.get("job_description", ""),
        "location": location_str,
        "postedAt": raw.get("job_posted_date"),
        "is_remote": raw.get("job_is_remote", False),
        "source": "glassdoor",
    }


def run_search(keywords: list[str]) -> list[dict]:
    """Full flow: trigger → wait → fetch → normalise. Returns list of job dicts."""
    run_id = run_actor(keywords)
    dataset_id = wait_for_run(run_id)
    raw_items = fetch_dataset(dataset_id)
    return [normalise_glassdoor(item) for item in raw_items]


if __name__ == "__main__":
    jobs = run_search()
    print(json.dumps(jobs, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(jobs)} Glassdoor jobs fetched.")
