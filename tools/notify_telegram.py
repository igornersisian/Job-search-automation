"""
Send a job card to Telegram.

Input: enriched job dict (with score, match_summary, red_flags)
Usage:
    python tools/notify_telegram.py '<job_json>'
    or imported as a module: send_job_card(job)
"""

import os
import sys
import json
import logging

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def _telegram_api() -> str:
    return f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}"


def send_message(text: str, parse_mode: str | None = "Markdown") -> dict:
    """Send a text message to the configured chat."""
    payload: dict = {
        "chat_id": os.environ["TELEGRAM_CHAT_ID"],
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    resp = httpx.post(
        f"{_telegram_api()}/sendMessage",
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _esc_md(text: str) -> str:
    """Escape Telegram Markdown v1 special characters."""
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text


def format_job_card(job: dict) -> str:
    """Format a job dict into a Telegram message."""
    score = job.get("score", 0)
    score_emoji = "✅" if score >= 80 else "🟡" if score >= 70 else "🔴"

    title = _esc_md(job.get("title", "Unknown Position"))
    company = _esc_md(job.get("company", "Unknown Company"))
    url = job.get("url", job.get("jobUrl", ""))
    salary = _esc_md(job.get("salary", job.get("salaryText", "Not listed")) or "Not listed")
    source = _esc_md(job.get("source", ""))
    match_summary = _esc_md(job.get("match_summary", ""))
    red_flags = job.get("red_flags", [])

    lines = [
        f"*{title}*",
        f"🏢 {company}",
        f"💰 {salary}",
    ]
    if source:
        lines.append(f"🔗 {source}")
    lines.append(f"Score: {score}/100 {score_emoji}")

    breakdown = job.get("score_breakdown", {})
    if breakdown:
        b1 = breakdown.get("block1", {})
        b2 = breakdown.get("block2", {})
        if b1 or b2:
            lines.append("")
            lines.append("*Breakdown:*")
            if b1:
                lines.append(
                    f"  Domain: {b1.get('domain', 0)}/30 | "
                    f"Patterns: {b1.get('patterns', 0)}/25 | "
                    f"Role: {b1.get('role', 0)}/15"
                )
            if b2:
                lines.append(
                    f"  Tools: {b2.get('tools', 0)}/20 | "
                    f"YoE: {b2.get('experience', 0)}/10"
                )
            penalty = breakdown.get("penalty", 0)
            if penalty:
                count = breakdown.get("red_flag_count", 0)
                lines.append(f"  Red flags: {count} x (-15) = -{penalty}")

    lines += [
        "",
        f"_{match_summary}_",
    ]

    if red_flags:
        lines.append("")
        lines.append("*Red flags:*")
        for flag in red_flags:
            lines.append(f"⚠️ {_esc_md(flag)}")

    if url:
        lines.append(f"[Open job]({url})")

    return "\n".join(lines)


def send_job_card(job: dict) -> bool:
    """
    Format and send a job card to Telegram.
    Returns True on success, False on failure.
    """
    try:
        text = format_job_card(job)
        send_message(text)
        logger.info(f"Sent job card: {job.get('title')} @ {job.get('company')}")
        return True
    except Exception as e:
        logger.error(f"Failed to send job card: {e}")
        return False


_TG_LIMIT = 4000  # Telegram caps messages at 4096 chars; keep a small buffer


def _send_chunked(text: str, parse_mode: str | None = None) -> None:
    """Send a (possibly very long) plain-text payload, splitting on line/char
    boundaries so Telegram never rejects it for length."""
    while text:
        chunk = text[:_TG_LIMIT]
        if len(text) > _TG_LIMIT:
            # try to break on a newline so error blocks stay readable
            cut = chunk.rfind("\n")
            if cut > _TG_LIMIT // 2:
                chunk = chunk[:cut]
        send_message(chunk, parse_mode=parse_mode)
        text = text[len(chunk):].lstrip("\n")


def send_daily_summary(
    sent: int,
    skipped_score: int,
    skipped_excluded: int,
    skipped_dupe: int,
    threshold: int = 70,
    dupes_crossrun: int = 0,
    dupes_local: int = 0,
    dupes_fuzzy: int = 0,
    source_errors: dict | None = None,
    per_source: dict | None = None,
    total_cost: float | None = None,
) -> None:
    """Send a brief pipeline summary, including a per-source breakdown.

    The summary itself uses Markdown. Source errors are sent as separate
    plain-text follow-up messages so response bodies (which often contain
    `_`, `*`, `[` that break Markdown) survive intact and the full error
    is preserved instead of being truncated.
    """
    dupe_detail = ""
    if skipped_dupe:
        parts = []
        if dupes_crossrun:
            parts.append(f"prev runs: {dupes_crossrun}")
        if dupes_local:
            parts.append(f"same run: {dupes_local}")
        if dupes_fuzzy:
            parts.append(f"fuzzy: {dupes_fuzzy}")
        if parts:
            dupe_detail = f" ({', '.join(parts)})"

    cost_line = f" — Apify ${total_cost:.2f}" if total_cost is not None else ""
    text = (
        f"*Job search run complete*{cost_line}\n"
        f"✅ Sent to you: {sent}\n"
        f"🔕 Low score (<{threshold}%): {skipped_score}\n"
        f"🚫 Excluded by title: {skipped_excluded}\n"
        f"♻️ Duplicates skipped: {skipped_dupe}{dupe_detail}"
    )

    # Per-source breakdown: fetched / new / sent (+ cost, cap flag)
    if per_source:
        lines = []
        order = ["linkedin", "ats", "glassdoor", "indeed", "wellfound", "remoteboards"]
        for name in order:
            p = per_source.get(name)
            if not p:
                continue
            # "capped" = the source returned as many as our limit allowed, so
            # there were probably MORE we didn't fetch (truncation). It is NOT a
            # cap on what gets sent — sending is score-gated only.
            flag = " ⚠️ hit our limit — more exist" if p.get("capped") else ""
            err = " ❌" if p.get("error") else ""
            lines.append(
                f"• {name}: {p.get('fetched', 0)} fetched, {p.get('new', 0)} new, "
                f"{p.get('sent', 0)} sent (${p.get('cost_usd', 0):.2f}){flag}{err}"
            )
        if lines:
            text += "\n\n*By source* (fetched / new / sent):\n" + "\n".join(lines)

    if source_errors:
        text += f"\n\n⚠️ Errors: {len(source_errors)} — details below"
    try:
        send_message(text)
    except Exception as e:
        logger.error(f"Failed to send summary: {e}")

    # Send each error as its own plain-text message so we never truncate and
    # never lose response bodies to Markdown parse errors.
    if source_errors:
        for name, err in source_errors.items():
            try:
                _send_chunked(f"⚠️ {name} failed:\n\n{err}", parse_mode=None)
            except Exception as e:
                logger.error(f"Failed to send error detail for {name}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python notify_telegram.py '<job_json>'")
        sys.exit(1)
    job = json.loads(sys.argv[1])
    success = send_job_card(job)
    sys.exit(0 if success else 1)
