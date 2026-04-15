"""
Trigger Apify Indeed Jobs Scraper and return job list.

Actor: valig/indeed-jobs-scraper
Docs:  https://apify.com/valig/indeed-jobs-scraper

Usage:
    python tools/run_indeed_search.py
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

ACTOR_ID = "valig~indeed-jobs-scraper"


def _get_token() -> str:
    return os.environ["APIFY_API_TOKEN"]


def run_actor(keyword: str) -> str:
    """Start one Indeed actor run for a single keyword."""
    url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/runs"
    params = {"token": _get_token()}
    actor_input = {
        "title": keyword,
        "country": "us",
        "location": "remote",
        "datePosted": "1",   # last 24 hours
        "limit": 25,
    }
    logger.info(f"Starting Indeed actor run: '{keyword}'")
    resp = httpx.post(url, json=actor_input, params=params, timeout=30)
    if resp.status_code >= 400:
        logger.error(f"Indeed actor start failed ({resp.status_code}): {resp.text[:500]}")
    resp.raise_for_status()
    run_id = resp.json()["data"]["id"]
    logger.info(f"Indeed actor run started: {run_id}")
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
            logger.info(f"Indeed run succeeded. Dataset: {dataset_id}")
            return dataset_id
        elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
            raise RuntimeError(f"Indeed actor run ended with status: {status}")

        logger.info(f"Indeed run status: {status}. Waiting...")
        time.sleep(10)

    raise TimeoutError(f"Indeed actor run {run_id} did not finish within {timeout_seconds}s")


def fetch_dataset(dataset_id: str) -> list[dict]:
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    params = {"token": _get_token(), "format": "json", "clean": "true"}
    logger.info(f"Downloading Indeed dataset {dataset_id}...")
    resp = httpx.get(url, params=params, timeout=60)
    resp.raise_for_status()
    items = resp.json()
    logger.info(f"Downloaded {len(items)} Indeed jobs")
    return items


def normalise_indeed(raw: dict) -> dict:
    """Map Indeed output fields to the shared job schema."""
    # Salary
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

    # Location
    loc = raw.get("location") or {}
    parts = [loc.get("city"), loc.get("admin1Code"), loc.get("countryCode")]
    location_str = ", ".join(p for p in parts if p)

    # Description - prefer plain text over HTML
    desc = raw.get("description") or {}
    description = desc.get("text", "") if isinstance(desc, dict) else str(desc)

    # Job URL - prefer direct employer URL, fall back to Indeed URL
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
        "is_remote": False,  # Indeed doesn't have a dedicated remote flag
        "source": "indeed",
    }


def _search_one(keyword: str) -> list[dict]:
    """Run a single keyword search end-to-end."""
    run_id = run_actor(keyword)
    dataset_id = wait_for_run(run_id)
    return fetch_dataset(dataset_id)


def run_search(keywords: list[str]) -> list[dict]:
    """Run one Indeed search per keyword (sequentially), deduplicate results."""
    all_jobs: list[dict] = []
    seen_ids: set[str] = set()

    for kw in keywords:
        try:
            raw_items = _search_one(kw)
            added = 0
            for item in raw_items:
                job = normalise_indeed(item)
                if job["id"] and job["id"] not in seen_ids:
                    seen_ids.add(job["id"])
                    all_jobs.append(job)
                    added += 1
            logger.info(f"Indeed '{kw}': {len(raw_items)} raw, {added} new")
        except Exception as e:
            logger.error(f"Indeed search failed for '{kw}': {e}")

    return all_jobs


if __name__ == "__main__":
    jobs = run_search(["automation engineer", "AI workflow"])
    print(json.dumps(jobs, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(jobs)} Indeed jobs fetched.")
