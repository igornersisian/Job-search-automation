"""
Trigger Apify LinkedIn Jobs Scraper and return the dataset ID.

Actor: curious_coder/linkedin-jobs-scraper
Docs:  https://console.apify.com/actors/hKByXkMQaC5Qt9UMN

Usage:
    python tools/run_apify_search.py
    → prints dataset_id to stdout

Returns dataset_id so process_jobs.py can download results.
"""

import os
import sys
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

APIFY_TOKEN = os.environ["APIFY_API_TOKEN"]
ACTOR_ID = "hKByXkMQaC5Qt9UMN"  # curious_coder/linkedin-jobs-scraper

# Search queries — multiple runs, results merged
SEARCH_QUERIES = [
    "AI workflow",
    "n8n",
    "automation engineer",
    "no-code automation engineer",
    "AI automation engineer",
    "workflow automation",
    "AI agent developer",
    "vibe coding developer",
]

ACTOR_INPUT = {
    "queries": SEARCH_QUERIES,
    "location": "Worldwide",
    "remote": True,
    "datePosted": "past24Hours",
    "limit": 20,  # per query, total ~120 results
    "proxy": {"useApifyProxy": True},
}


def run_actor() -> str:
    """Start the Apify actor run and return run_id."""
    url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/runs"
    headers = {"Content-Type": "application/json"}
    params = {"token": APIFY_TOKEN}

    logger.info("Starting Apify actor run...")
    resp = httpx.post(url, json={"input": ACTOR_INPUT}, params=params, headers=headers, timeout=30)
    resp.raise_for_status()

    run_id = resp.json()["data"]["id"]
    logger.info(f"Actor run started: {run_id}")
    return run_id


def wait_for_run(run_id: str, timeout_seconds: int = 300) -> str:
    """Poll until the run is SUCCEEDED, return dataset_id."""
    url = f"https://api.apify.com/v2/actor-runs/{run_id}"
    params = {"token": APIFY_TOKEN}
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        resp = httpx.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()["data"]
        status = data["status"]

        if status == "SUCCEEDED":
            dataset_id = data["defaultDatasetId"]
            logger.info(f"Run succeeded. Dataset ID: {dataset_id}")
            return dataset_id
        elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
            raise RuntimeError(f"Actor run {run_id} ended with status: {status}")

        logger.info(f"Run status: {status}. Waiting...")
        time.sleep(10)

    raise TimeoutError(f"Actor run {run_id} did not finish within {timeout_seconds}s")


def fetch_dataset(dataset_id: str) -> list[dict]:
    """Download all items from an Apify dataset."""
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    params = {"token": APIFY_TOKEN, "format": "json", "clean": "true"}

    logger.info(f"Downloading dataset {dataset_id}...")
    resp = httpx.get(url, params=params, timeout=60)
    resp.raise_for_status()

    items = resp.json()
    logger.info(f"Downloaded {len(items)} items from dataset")
    return items


def run_search() -> list[dict]:
    """Full flow: trigger → wait → fetch. Returns list of job dicts."""
    run_id = run_actor()
    dataset_id = wait_for_run(run_id)
    return fetch_dataset(dataset_id)


if __name__ == "__main__":
    jobs = run_search()
    # Print dataset contents as JSON for piping to process_jobs.py
    print(json.dumps(jobs, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(jobs)} jobs fetched.")
