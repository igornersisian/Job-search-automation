"""
Trigger Apify JobsFlow actor and return normalised job list.

Actor: zyncodltd/JobsFlow
ID:    l09aaDKNa1G3SVFzr
Docs:  https://console.apify.com/actors/l09aaDKNa1G3SVFzr

Aggregates remote jobs from RemoteOK, Remotive, and WeWorkRemotely.

Usage:
    python tools/run_remoteboards_search.py
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

ACTOR_ID = "l09aaDKNa1G3SVFzr"  # zyncodltd/JobsFlow


def run_actor(keywords: list[str]) -> tuple[str, str]:
    """Start the JobsFlow actor run. Returns (run_id, token_used)."""
    url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/runs"

    # JobsFlow accepts a single searchKeywords string
    # Join multiple keywords with comma for broader results
    search_str = ", ".join(keywords[:5]) if keywords else ""

    actor_input = {
        "maxJobs": 150,
    }
    if search_str:
        actor_input["searchKeywords"] = search_str

    logger.info(f"Starting JobsFlow actor run (keywords: '{search_str}', {apify_client.active_slot_summary()})...")
    resp, token = apify_client.post(url, json_body=actor_input, timeout=30)
    if resp.status_code >= 400:
        logger.error(f"JobsFlow actor start failed ({resp.status_code}): {resp.text[:500]}")
    resp.raise_for_status()

    run_id = resp.json()["data"]["id"]
    logger.info(f"JobsFlow actor run started: {run_id}")
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
            logger.info(f"JobsFlow run succeeded. Dataset: {dataset_id}")
            return dataset_id
        elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
            status_msg = data.get("statusMessage") or ""
            apify_client.report_run_failure(token, status, status_msg)
            raise RuntimeError(f"JobsFlow actor run {run_id} ended with status: {status} ({status_msg[:200]})")

        logger.info(f"JobsFlow run status: {status}. Waiting...")
        time.sleep(10)

    raise TimeoutError(f"JobsFlow actor run {run_id} did not finish within {timeout_seconds}s")


def fetch_dataset(dataset_id: str, token: str) -> list[dict]:
    """Download all items from an Apify dataset."""
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
    logger.info(f"Downloading JobsFlow dataset {dataset_id}...")
    resp = apify_client.get(url, token, params={"format": "json", "clean": "true"}, timeout=60)
    resp.raise_for_status()

    items = resp.json()
    logger.info(f"Downloaded {len(items)} JobsFlow jobs")
    return items


def normalise_remoteboards(raw: dict) -> dict:
    """Map JobsFlow output to the shared job schema.

    JobsFlow fields: id, title, company, location, salary, tags, url, source, posted_at, description
    """
    return {
        "id": str(raw.get("id", "")),
        "title": raw.get("title", ""),
        "company": raw.get("company", ""),
        "url": raw.get("url", ""),
        "salary": raw.get("salary", ""),
        "description": raw.get("description") or "",
        "location": raw.get("location", ""),
        "postedAt": raw.get("posted_at"),
        "is_remote": True,
        "source": f"remoteboards-{raw.get('source', 'unknown')}",
    }


def run_search(keywords: list[str]) -> list[dict]:
    """Full flow: trigger -> wait -> fetch -> normalise. Returns list of job dicts."""
    run_id, token = run_actor(keywords)
    dataset_id = wait_for_run(run_id, token)
    raw_items = fetch_dataset(dataset_id, token)
    return [normalise_remoteboards(item) for item in raw_items]


if __name__ == "__main__":
    jobs = run_search(["automation engineer", "AI workflow"])
    print(json.dumps(jobs, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(jobs)} remote board jobs fetched.")
