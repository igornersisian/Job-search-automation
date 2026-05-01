"""
Trigger Apify LinkedIn Jobs Scraper and return normalised job list.

Actor: cheap_scraper/linkedin-job-scraper
Docs:  https://console.apify.com/actors/2rJKkhh7vjpX7pvjg

Usage:
    python tools/run_apify_search.py
"""

import json
import time
import logging

from dotenv import load_dotenv

import apify_client

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ACTOR_ID = "2rJKkhh7vjpX7pvjg"  # cheap_scraper/linkedin-job-scraper


def run_actor(keywords: list[str]) -> tuple[str, str]:
    """Start the Apify actor run. Returns (run_id, token_used)."""
    url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/runs"
    actor_input = {
        "keyword": keywords,
        "publishedAt": "r86400",          # last 24 hours
        "workType": ["remote"],
        "maxItems": 150,
        "saveOnlyUniqueItems": True,
    }

    logger.info(f"Starting LinkedIn actor run with {len(keywords)} keywords ({apify_client.active_slot_summary()})...")
    resp, token = apify_client.post(url, json_body=actor_input, timeout=30)
    if resp.status_code >= 400:
        logger.error(f"LinkedIn actor start failed ({resp.status_code}): {resp.text[:500]}")
    resp.raise_for_status()

    run_id = resp.json()["data"]["id"]
    logger.info(f"LinkedIn actor run started: {run_id}")
    return run_id, token


def wait_for_run(run_id: str, token: str, timeout_seconds: int = 600) -> str:
    """Poll until the run is SUCCEEDED, return dataset_id."""
    url = f"https://api.apify.com/v2/actor-runs/{run_id}"
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        resp = apify_client.get(url, token, timeout=15)
        resp.raise_for_status()
        data = resp.json()["data"]
        status = data["status"]

        if status == "SUCCEEDED":
            dataset_id = data["defaultDatasetId"]
            logger.info(f"LinkedIn run succeeded. Dataset: {dataset_id}")
            return dataset_id
        elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
            status_msg = data.get("statusMessage") or ""
            apify_client.report_run_failure(token, status, status_msg)
            raise RuntimeError(f"LinkedIn actor run {run_id} ended with status: {status} ({status_msg[:200]})")

        logger.info(f"LinkedIn run status: {status}. Waiting...")
        time.sleep(10)

    raise TimeoutError(f"LinkedIn actor run {run_id} did not finish within {timeout_seconds}s")


def fetch_dataset(dataset_id: str, token: str) -> list[dict]:
    """Download all items from an Apify dataset."""
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    logger.info(f"Downloading LinkedIn dataset {dataset_id}...")
    resp = apify_client.get(url, token, params={"format": "json", "clean": "true"}, timeout=60)
    resp.raise_for_status()

    items = resp.json()
    logger.info(f"Downloaded {len(items)} LinkedIn jobs")
    return items


def normalise_linkedin(raw: dict) -> dict:
    """Map cheap_scraper/linkedin-job-scraper output to the shared job schema."""
    salary_info = raw.get("salaryInfo") or []
    salary_text = " - ".join(salary_info) if salary_info else ""

    return {
        "id": raw.get("jobId", ""),
        "title": raw.get("jobTitle", ""),
        "company": raw.get("companyName", ""),
        "url": raw.get("jobUrl") or raw.get("applyUrl", ""),
        "salary": salary_text,
        "description": raw.get("jobDescription", ""),
        "location": raw.get("location", ""),
        "postedAt": raw.get("publishedAt") or raw.get("postedTime", ""),
        "is_remote": "remote" in (raw.get("workType") or "").lower(),
        "source": "linkedin",
    }


def run_search(keywords: list[str]) -> list[dict]:
    """Full flow: trigger -> wait -> fetch -> normalise. Returns list of job dicts."""
    run_id, token = run_actor(keywords)
    dataset_id = wait_for_run(run_id, token)
    raw_items = fetch_dataset(dataset_id, token)
    return [normalise_linkedin(item) for item in raw_items]


if __name__ == "__main__":
    jobs = run_search()
    print(json.dumps(jobs, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(jobs)} LinkedIn jobs fetched.")
