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


def send_message(text: str, parse_mode: str = "Markdown") -> dict:
    """Send a text message to the configured chat."""
    resp = httpx.post(
        f"{_telegram_api()}/sendMessage",
        json={
            "chat_id": os.environ["TELEGRAM_CHAT_ID"],
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        },
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
        f"🌍 Remote" + (f" • {source}" if source else ""),
        f"Score: {score}/100 {score_emoji}",
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


def send_daily_summary(
    sent: int,
    skipped_score: int,
    skipped_junior: int,
    skipped_dupe: int,
    threshold: int = 70,
    dupes_crossrun: int = 0,
    dupes_local: int = 0,
    dupes_fuzzy: int = 0,
) -> None:
    """Send a brief daily pipeline summary."""
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
    text = (
        f"*Daily job search complete*\n"
        f"✅ Sent to you: {sent}\n"
        f"🔕 Low score (<{threshold}%): {skipped_score}\n"
        f"🚫 Junior/intern filtered: {skipped_junior}\n"
        f"♻️ Duplicates skipped: {skipped_dupe}{dupe_detail}"
    )
    try:
        send_message(text)
    except Exception as e:
        logger.error(f"Failed to send summary: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python notify_telegram.py '<job_json>'")
        sys.exit(1)
    job = json.loads(sys.argv[1])
    success = send_job_card(job)
    sys.exit(0 if success else 1)
