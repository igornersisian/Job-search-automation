"""
Send a job card to Telegram.

Input: enriched job dict (with score, match_summary, red_flags, typical_qa)
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

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def send_message(text: str, parse_mode: str = "Markdown") -> dict:
    """Send a text message to the configured chat."""
    resp = httpx.post(
        f"{TELEGRAM_API}/sendMessage",
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def format_job_card(job: dict) -> str:
    """Format a job dict into a Telegram message."""
    score = job.get("score", 0)
    score_emoji = "✅" if score >= 80 else "🟡" if score >= 70 else "🔴"

    title = job.get("title", "Unknown Position")
    company = job.get("company", "Unknown Company")
    url = job.get("url", job.get("jobUrl", ""))
    salary = job.get("salary", job.get("salaryText", "Not listed"))
    match_summary = job.get("match_summary", "")
    red_flags = job.get("red_flags", [])
    typical_qa = job.get("typical_qa", [])

    lines = [
        f"*{title}*",
        f"🏢 {company}",
        f"💰 {salary}",
        f"🌍 Remote",
        f"Score: {score}/100 {score_emoji}",
        "",
        f"_{match_summary}_",
    ]

    if red_flags:
        lines.append("")
        lines.append("*Red flags:*")
        for flag in red_flags[:3]:
            lines.append(f"⚠️ {flag}")

    if typical_qa:
        lines.append("")
        lines.append("*Prep Q&A:*")
        for qa in typical_qa[:3]:
            q = qa.get("question", "")
            a = qa.get("answer", "")
            # Truncate long answers
            if len(a) > 200:
                a = a[:200] + "..."
            lines.append(f"*Q:* {q}")
            lines.append(f"*A:* _{a}_")
            lines.append("")

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


def send_daily_summary(sent: int, skipped_score: int, skipped_junior: int, skipped_dupe: int) -> None:
    """Send a brief daily pipeline summary."""
    text = (
        f"*Daily job search complete*\n"
        f"✅ Sent to you: {sent}\n"
        f"🔕 Low score (<70): {skipped_score}\n"
        f"🚫 Junior/intern filtered: {skipped_junior}\n"
        f"♻️ Duplicates skipped: {skipped_dupe}"
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
