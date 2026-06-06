"""
Remote job boards (RemoteOK, Remotive, WeWorkRemotely) via Apify JobsFlow,
on the shared runner.

Actor: silicatelabs/JobsFlow (id l09aaDKNa1G3SVFzr). The actor changed owner
from zyncodltd; its input is now {searchKeywords: string, techStack: string,
maxJobs: int}. Probed behaviour (2026-06-05):
  - `searchKeywords` is a loose per-term title/description match. Sending ONE
    comma-joined string of all keywords matched nothing → 0 results (the old bug).
  - `techStack` matches job tags, but the niche tags (n8n/nocode/llm) don't exist
    on these boards, so it's not useful here.
So we fan out ONE run per keyword on `searchKeywords` (free actor, $0) and merge,
mirroring Indeed/Glassdoor. The scoring threshold filters the loose matches.

Exposes `fetch(keywords, *, lookback, profile)` -> apify_client.SourceResult.
"""

import json

from dotenv import load_dotenv

import apify_client
from log_setup import get_logger

load_dotenv()

logger = get_logger(__name__)

ACTOR_ID = "l09aaDKNa1G3SVFzr"  # silicatelabs/JobsFlow
MAX_JOBS = 50           # per keyword
MAX_KEYWORDS = 8        # cap fan-out breadth (free, but keep run count sane)


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
    """One JobsFlow run per keyword (parallel) over the remote boards, merged.
    lookback unused (actor has no time filter)."""
    kws = [k for k in (keywords or []) if k][:MAX_KEYWORDS]
    return apify_client.fan_out_keywords(
        ACTOR_ID, kws,
        lambda kw: {"maxJobs": MAX_JOBS, "searchKeywords": kw},
        source="remoteboards", normalise=normalise_remoteboards, cap=MAX_JOBS,
    )


if __name__ == "__main__":
    r = fetch(["n8n", "AI engineer", "workflow automation", "full stack"])
    print(json.dumps(r.items, ensure_ascii=False, indent=2))
    logger.info(f"Done. {len(r.items)} remoteboards jobs, ${r.cost_usd:.4f}, error={r.error}")
