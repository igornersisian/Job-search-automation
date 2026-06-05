"""
Remote job boards (RemoteOK, Remotive, WeWorkRemotely) via Apify JobsFlow,
on the shared runner.

Actor: silicatelabs/JobsFlow (id l09aaDKNa1G3SVFzr). NOTE: this actor changed
owner from zyncodltd and its input schema is now
{searchKeywords: string, techStack: string, maxJobs: int}. These boards tag
jobs by tech stack (python/n8n/react), not by role phrases — sending only
phrase-style searchKeywords returned 0 results. We now also send `techStack`.
Free actor ($0). Validate with a test run; tune `techStack` via the profile
override `remoteboards_tech_stack` if results are empty.

Exposes `fetch(keywords, *, lookback, profile)` -> apify_client.SourceResult.
"""

import json
import logging

from dotenv import load_dotenv

import apify_client

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

ACTOR_ID = "l09aaDKNa1G3SVFzr"  # silicatelabs/JobsFlow
MAX_JOBS = 150


def _tech_stack(keywords: list[str], profile: dict | None) -> str:
    """techStack string for the tag-based boards. Profile override wins;
    otherwise derive from keywords (the actor matches against job tech tags)."""
    override = (profile or {}).get("remoteboards_tech_stack")
    if override:
        return override if isinstance(override, str) else ", ".join(override)
    return ", ".join(keywords[:8])


def normalise_remoteboards(raw: dict) -> dict:
    """Map JobsFlow output to the shared job schema."""
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


def fetch(keywords: list[str], *, lookback: int = 86400, profile: dict | None = None) -> apify_client.SourceResult:
    """Run JobsFlow over the remote boards. lookback unused (actor has no time
    filter)."""
    actor_input = {"maxJobs": MAX_JOBS}
    search_str = ", ".join(keywords[:5]) if keywords else ""
    if search_str:
        actor_input["searchKeywords"] = search_str
    tech = _tech_stack(keywords, profile)
    if tech:
        actor_input["techStack"] = tech

    logger.info(f"JobsFlow input: searchKeywords='{search_str}', techStack='{tech}'")
    return apify_client.run_actor_job(
        ACTOR_ID, actor_input,
        source="remoteboards", normalise=normalise_remoteboards, cap=MAX_JOBS,
    )


if __name__ == "__main__":
    r = fetch(["automation engineer", "n8n", "python"], profile=None)
    print(json.dumps(r.items, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(r.items)} remoteboards jobs, ${r.cost_usd:.4f}, error={r.error}")
