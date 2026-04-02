"""
Trigger Apify Glassdoor Jobs Scraper and return job list.

Actor: valig/glassdoor-jobs-scraper
ID:    5OaooRg0FxlRF0L1B
Docs:  https://apify.com/valig/glassdoor-jobs-scraper

Usage:
    python tools/run_glassdoor_search.py
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

ACTOR_ID = "5OaooRg0FxlRF0L1B"


def _get_token() -> str:
    return os.environ["APIFY_API_TOKEN"]


def run_actor(keyword: str) -> str:
    """Start one Glassdoor actor run for a single keyword."""
    url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/runs"
    params = {"token": _get_token()}
    actor_input = {
        "keywords": keyword,
        "location": "United States",
        "daysOld": 1,
        "limit": 25,
    }
    logger.info(f"Starting Glassdoor actor run: '{keyword}'")
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
    # Salary from pay object
    pay = raw.get("pay") or {}
    salary_text = ""
    if pay:
        lo = pay.get("min")
        hi = pay.get("max")
        currency = pay.get("currency") or "USD"
        period = (pay.get("period") or "").capitalize()
        symbol = "$" if currency == "USD" else currency + " "
        if lo and hi:
            salary_text = f"{symbol}{lo:,}–{symbol}{hi:,} {period}".strip()
        elif lo:
            salary_text = f"{symbol}{lo:,}+ {period}".strip()

    # Location
    location = raw.get("location") or {}
    location_str = location.get("name", "")

    # Company from employer object
    employer = raw.get("employer") or {}

    # URL — prefer seoUrl for human-readable links
    job_url = raw.get("seoUrl") or raw.get("url", "")

    # Posted date — convert ageInDays to approximate ISO date
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
        "description": (raw.get("description") or "")[:5000],
        "location": location_str,
        "postedAt": posted_at,
        "is_remote": False,
        "source": "glassdoor",
    }


def _search_one(keyword: str) -> list[dict]:
    """Run a single keyword search end-to-end."""
    run_id = run_actor(keyword)
    dataset_id = wait_for_run(run_id)
    return fetch_dataset(dataset_id)


def run_search(keywords: list[str]) -> list[dict]:
    """Run one Glassdoor search per keyword (sequentially), deduplicate results."""
    all_jobs: list[dict] = []
    seen_ids: set[str] = set()

    for kw in keywords:
        try:
            raw_items = _search_one(kw)
            added = 0
            for item in raw_items:
                job = normalise_glassdoor(item)
                if job["id"] and job["id"] not in seen_ids:
                    seen_ids.add(job["id"])
                    all_jobs.append(job)
                    added += 1
            logger.info(f"Glassdoor '{kw}': {len(raw_items)} raw, {added} new")
        except Exception as e:
            logger.error(f"Glassdoor search failed for '{kw}': {e}")

    return all_jobs


if __name__ == "__main__":
    jobs = run_search(["automation engineer", "AI workflow"])
    print(json.dumps(jobs, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(jobs)} Glassdoor jobs fetched.")
