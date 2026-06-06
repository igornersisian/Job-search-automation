"""
Telegram bot — main interface for the job search automation system.

Capabilities:
  - Receive PDF resume → parse with OpenAI → save profile to Supabase
  - /job <url> → scrape job page, analyze vs resume, set as active context
  - Free-text messages → OpenAI assistant with resume + active job context
  - /status → show current profile summary
  - /help → list commands
"""

import os
import sys
import io
import re
import json
import logging
import asyncio
from datetime import datetime, time as dt_time, timezone

import httpx
from openai import OpenAI
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from dotenv import load_dotenv

load_dotenv()

from db import get_supabase as _shared_supabase
from score_job import score_job  # same strict rubric the daily pipeline uses

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Clients (lazy init so module can be imported without env vars set) ---
_openai_client: OpenAI | None = None


def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def get_supabase():
    """Shared Supabase client with the bot's fail-fast timeout (10s) — kept
    tighter than the pipeline's 30s so interactive commands don't hang on a
    slow DB."""
    return _shared_supabase(timeout=10)


# In-memory active job context per chat_id
active_jobs: dict[int, dict] = {}


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def get_job_by_url(url: str) -> dict | None:
    """Look up a job in Supabase by its URL."""
    try:
        result = get_supabase().table("jobs").select("*").eq("url", url).limit(1).execute()
        if result.data:
            row = result.data[0]
            return {
                "id": row.get("id"),
                "title": row.get("title", ""),
                "company": row.get("company", ""),
                "url": row.get("url", ""),
                "salary": row.get("salary_text", ""),
                "description": row.get("description", ""),
                "score": row.get("score"),
                "match_summary": row.get("match_summary", ""),
                "red_flags": json.loads(row.get("red_flags") or "[]"),
                "source": row.get("source", ""),
            }
    except Exception as e:
        logger.error(f"get_job_by_url error: {e}")
    return None


def _extract_job_url_from_message(message) -> str | None:
    """Extract job URL from a job card message.

    Telegram renders Markdown links like [Open job](url) into entities
    of type 'text_link', so reply.text contains just "Open job" without
    the URL.  We check entities first, then fall back to raw text patterns.
    """
    # 1. Check entities for text_link (rendered Markdown links)
    entities = message.entities or []
    text = message.text or ""
    for ent in entities:
        if ent.type == "text_link" and ent.url:
            return ent.url
        if ent.type == "url":
            # Plain URL pasted in text
            url = text[ent.offset : ent.offset + ent.length]
            if url.startswith("http"):
                return url

    # 2. Fallback: raw Markdown syntax (shouldn't happen but just in case)
    m = re.search(r'\[Open job\]\(([^)]+)\)', text)
    if m:
        return m.group(1)

    # 3. Fallback: common job board URLs in plain text
    m = re.search(r'(https?://(?:www\.)?(?:linkedin\.com|glassdoor\.com|indeed\.com|wellfound\.com)\S+)', text)
    if m:
        return m.group(1)
    return None


def get_profile() -> dict | None:
    """Return the latest parsed resume profile from Supabase."""
    try:
        result = get_supabase().table("profile").select("parsed").order("updated_at", desc=True).limit(1).execute()
        if result.data:
            return result.data[0]["parsed"]
    except Exception as e:
        logger.error(f"get_profile error: {e}")
    return None


def save_profile(raw_text: str, parsed: dict) -> None:
    """Upsert resume profile in Supabase (single row, id=1).
    Preserves user-configured settings from the existing profile.
    """
    existing = get_profile()
    if existing:
        for key in ("search_keywords", "custom_red_flags", "score_threshold", "excluded_title_keywords", "wellfound_roles"):
            if key in existing:
                parsed[key] = existing[key]
    get_supabase().table("profile").upsert({
        "id": 1,
        "raw_text": raw_text,
        "parsed": parsed,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).execute()


def get_search_keywords() -> list[str] | None:
    """Return custom search keywords from profile, or None if not set."""
    profile = get_profile()
    if profile:
        return profile.get("search_keywords")
    return None


def _update_profile_field(key: str, value) -> None:
    """Set one field inside the profile's parsed JSONB (single row, id=1).
    No-op (with a log) if no profile exists yet."""
    profile = get_profile()
    if not profile:
        logger.error("No profile found — upload resume first")
        return
    profile[key] = value
    get_supabase().table("profile").update({
        "parsed": profile,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", 1).execute()


def save_search_keywords(keywords: list[str]) -> None:
    """Save search keywords into the profile's parsed JSONB."""
    _update_profile_field("search_keywords", keywords)


WELLFOUND_VALID_ROLES = [
    "Software Engineer", "Mobile Developer", "iOS Developer", "Android Developer",
    "Frontend Engineer", "Backend Engineer", "Full-Stack Engineer", "Software Architect",
    "Embedded Engineer", "Data Engineer", "Security Engineer", "Machine Learning Engineer",
    "Engineering Manager", "QA Engineer", "DevOps", "Data Scientist",
    # AI/automation roles — proven to exist as Wellfound categories (probed 2026-06-06).
    # The generic categories above pull traditional senior-SWE that score ~0 for an
    # AI-automation profile; these surface AI/ML/automation jobs that actually match.
    "AI Engineer", "Artificial Intelligence Engineer", "Automation Engineer",
    "Designer", "User Researcher", "Visual Designer", "Creative Director",
    "Graphic Designer", "Product Designer", "Design Manager",
    "Product Manager",
    "Operations", "Finance/Accounting", "H.R.", "Office Manager", "Recruiter",
    "Customer Service", "Operations Manager", "Chief Of Staff",
    "Sales", "Business Development", "Sales Development Representative",
    "Account Executive", "BD Manager", "Account Manager", "Sales Manager",
    "Customer Success Manager",
    "Marketing", "Growth Hacker", "Marketing Manager",
    "Digital Marketing Manager", "Product Marketing Manager",
]


def get_wellfound_roles() -> list[str] | None:
    """Return Wellfound role filters from profile, or None if not set."""
    profile = get_profile()
    if profile:
        return profile.get("wellfound_roles")
    return None


def save_wellfound_roles(roles: list[str]) -> None:
    """Save Wellfound roles into the profile's parsed JSONB."""
    _update_profile_field("wellfound_roles", roles)


def get_score_threshold() -> int:
    """Return the score threshold from profile, or default 70."""
    profile = get_profile()
    if profile:
        return profile.get("score_threshold") or 70
    return 70


def save_score_threshold(threshold: int) -> None:
    """Save score threshold into the profile's parsed JSONB."""
    _update_profile_field("score_threshold", threshold)


def get_custom_red_flags() -> list[str]:
    """Return custom dealbreakers from profile."""
    profile = get_profile()
    if profile:
        return profile.get("custom_red_flags") or []
    return []


def save_custom_red_flags(flags: list[str]) -> None:
    """Save custom dealbreakers into the profile's parsed JSONB."""
    _update_profile_field("custom_red_flags", flags)


def get_excluded_title_keywords() -> list[str]:
    """Return excluded title keywords from profile, or empty list if not set."""
    profile = get_profile()
    if profile:
        return profile.get("excluded_title_keywords") or []
    return []


def save_excluded_title_keywords(keywords: list[str]) -> None:
    """Save excluded title keywords into the profile's parsed JSONB."""
    _update_profile_field("excluded_title_keywords", keywords)


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using pypdf."""
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def parse_resume_with_openai(raw_text: str) -> dict:
    """Use OpenAI to extract structured profile from resume text."""
    response = get_openai().chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a resume parser. Extract structured information from the resume text "
                    "and return a JSON object with these fields:\n"
                    "- name: string\n"
                    "- title: string (current/desired job title)\n"
                    "- summary: string (2-3 sentence professional summary)\n"
                    "- skills: list of strings (technical skills, tools, platforms)\n"
                    "- experience: list of objects [{company, role, duration, highlights}]\n"
                    "- projects: list of objects [{name, description, tech_stack}]\n"
                    "- languages: list of strings\n"
                    "- education: list of objects [{institution, degree, years}]\n"
                    "Return only valid JSON, no extra text."
                ),
            },
            {"role": "user", "content": raw_text},
        ],
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Job page scraping
# ---------------------------------------------------------------------------

def scrape_job_page(url: str) -> str:
    """Scrape a job posting URL and return text content."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    with httpx.Client(timeout=15, follow_redirects=True) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()

    # Strip HTML tags with a simple approach
    from html.parser import HTMLParser

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.texts = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "nav", "header", "footer"):
                self._skip = True

        def handle_endtag(self, tag):
            if tag in ("script", "style", "nav", "header", "footer"):
                self._skip = False

        def handle_data(self, data):
            if not self._skip and data.strip():
                self.texts.append(data.strip())

    extractor = TextExtractor()
    extractor.feed(resp.text)
    return " ".join(extractor.texts)[:8000]  # cap at 8k chars


def extract_job_prep(job_text: str, profile: dict) -> dict:
    """gpt-5-mini: pull the job's title/company and draft likely application Q&A.

    The fit SCORE is NOT produced here — /job scores through the same score_job
    rubric (gpt-5-mini) the daily pipeline uses, so the number is consistent
    across both entry points. This call only adds the title/company and the
    application-prep answers.
    """
    instructions = (
        "You extract structured info from a job posting and draft application answers. "
        "Return ONLY a JSON object with:\n"
        "- title: string (job title)\n"
        "- company: string\n"
        "- typical_qa: list of [{question, answer}] — 3-5 likely application "
        "questions with concise suggested answers, written in first person as the "
        "candidate and grounded strictly in the candidate profile.\n"
        "Return only valid JSON."
    )
    user_input = (
        f"CANDIDATE PROFILE:\n{json.dumps(profile, indent=2)}\n\n"
        f"JOB POSTING:\n{job_text}"
    )
    resp = get_openai().responses.create(
        model="gpt-5-mini",
        instructions=instructions,
        input=user_input,
        text={"format": {"type": "json_object"}},
        reasoning={"effort": "minimal"},
        store=False,
    )
    return json.loads(resp.output_text)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Job search assistant ready.\n\n"
        "Send me your resume PDF to get started.\n"
        "Then I'll analyze incoming job matches and help you prepare applications.\n\n"
        "/help — see all commands"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*Commands:*\n"
        "/help — this message\n"
        "/status — show your current profile summary\n"
        "/keywords — set search keywords (comma-separated)\n"
        "/wellfound — set Wellfound role filters (separate from keywords)\n"
        "/threshold — set minimum score % to send (default 70)\n"
        "/redflags — set dealbreakers (e.g. requires US work permit)\n"
        "/excluded — set title keywords to skip (e.g. intern, trainee)\n"
        "/stats — all-time job search statistics\n"
        "/fetch\\_jobs — manually run the job search pipeline now\n"
        "/job <url> — analyze a specific job URL and set it as active context\n\n"
        "*How to use:*\n"
        "1. Send your resume PDF — I'll parse and remember it\n"
        "2. Set keywords: /keywords AI workflow, n8n, automation\n"
        "3. Set threshold: /threshold 75\n"
        "4. Every morning I'll send you matching job cards\n"
        "5. After seeing a card, ask me anything\n"
        "6. Use /job <url> to load any job for discussion",
        parse_mode="Markdown",
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    profile = get_profile()
    if not profile:
        await update.message.reply_text(
            "No resume profile found. Send me your PDF resume to set it up."
        )
        return

    skills_preview = ", ".join(profile.get("skills", [])[:10])
    active = active_jobs.get(update.effective_chat.id)
    active_info = f"\n\n*Active job:* {active.get('title')} @ {active.get('company')}" if active else ""

    await update.message.reply_text(
        f"*Profile:* {profile.get('name')} — {profile.get('title')}\n"
        f"*Skills:* {skills_preview}...\n"
        f"*Experience entries:* {len(profile.get('experience', []))}\n"
        f"*Projects:* {len(profile.get('projects', []))}"
        f"{active_info}",
        parse_mode="Markdown",
    )


async def cmd_keywords(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set or show search keywords."""
    if not context.args:
        # Show current keywords
        kw = get_search_keywords()
        if kw:
            await update.message.reply_text(
                f"*Current keywords:*\n{', '.join(kw)}\n\n"
                "To change: /keywords AI workflow, n8n, automation",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                "⚠️ No keywords set. Pipeline won't run without them.\n"
                "Set keywords: /keywords AI workflow, n8n, automation"
            )
        return

    raw_text = " ".join(context.args)
    keywords = [kw.strip() for kw in raw_text.split(",") if kw.strip()]

    if not keywords:
        await update.message.reply_text("Send keywords separated by commas: /keywords AI workflow, n8n, automation")
        return

    save_search_keywords(keywords)
    await update.message.reply_text(
        f"Search keywords saved ({len(keywords)}):\n" + ", ".join(keywords)
    )


async def cmd_wellfound(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set or show Wellfound role filters."""
    if not context.args:
        roles = get_wellfound_roles()
        roles_list = "\n".join(f"• {r}" for r in WELLFOUND_VALID_ROLES)
        if roles:
            current = ", ".join(roles)
            await update.message.reply_text(
                f"*Current Wellfound roles:*\n{current}\n\n"
                f"*Available roles:*\n{roles_list}\n\n"
                "To change: /wellfound Software Engineer, DevOps, Data Scientist\n"
                "To clear: /wellfound clear",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                "⚠️ No Wellfound roles set — Wellfound search is skipped.\n\n"
                f"*Available roles:*\n{roles_list}\n\n"
                "Set roles: /wellfound Software Engineer, DevOps, Data Scientist",
                parse_mode="Markdown",
            )
        return

    raw_text = " ".join(context.args)
    if raw_text.strip().lower() == "clear":
        save_wellfound_roles([])
        await update.message.reply_text("Wellfound roles cleared. Wellfound search will be skipped.")
        return

    roles = [r.strip() for r in raw_text.split(",") if r.strip()]
    if not roles:
        await update.message.reply_text("Send roles separated by commas: /wellfound Software Engineer, DevOps")
        return

    # Validate against known roles
    invalid = [r for r in roles if r not in WELLFOUND_VALID_ROLES]
    if invalid:
        await update.message.reply_text(
            f"⚠️ Unknown roles: {', '.join(invalid)}\n\n"
            "Use /wellfound without arguments to see the full list."
        )
        return

    save_wellfound_roles(roles)
    await update.message.reply_text(
        f"Wellfound roles saved ({len(roles)}):\n" + ", ".join(roles)
    )


async def cmd_redflags(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set, show, or clear custom dealbreakers for job scoring."""
    if not context.args:
        flags = get_custom_red_flags()
        if flags:
            lines = "\n".join(f"• {f}" for f in flags)
            await update.message.reply_text(
                f"*Your dealbreakers:*\n{lines}\n\n"
                "To replace: /redflags requires US work permit, on-site only\n"
                "To clear: /redflags clear",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                "No custom dealbreakers set.\n"
                "To add: /redflags requires US work permit, on-site only"
            )
        return

    raw_text = " ".join(context.args)

    if raw_text.strip().lower() == "clear":
        save_custom_red_flags([])
        await update.message.reply_text("Custom dealbreakers cleared.")
        return

    flags = [f.strip() for f in raw_text.split(",") if f.strip()]
    if not flags:
        await update.message.reply_text(
            "Send dealbreakers separated by commas:\n"
            "/redflags requires US work permit, on-site only, requires clearance"
        )
        return

    save_custom_red_flags(flags)
    lines = "\n".join(f"• {f}" for f in flags)
    await update.message.reply_text(f"Dealbreakers saved ({len(flags)}):\n{lines}")


async def cmd_excluded(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set, show, or clear excluded title keywords (jobs with these words in title are skipped)."""
    if not context.args:
        excluded = get_excluded_title_keywords()
        if excluded:
            lines = "\n".join(f"• {e}" for e in excluded)
            await update.message.reply_text(
                f"*Excluded title keywords:*\n{lines}\n\n"
                "To replace: /excluded intern, trainee, apprentice\n"
                "To clear: /excluded clear",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                "No excluded title keywords set (all jobs pass through).\n"
                "To add: /excluded intern, trainee, apprentice"
            )
        return

    raw_text = " ".join(context.args)

    if raw_text.strip().lower() == "clear":
        save_excluded_title_keywords([])
        await update.message.reply_text("Excluded title keywords cleared. All jobs will pass through.")
        return

    keywords = [kw.strip().lower() for kw in raw_text.split(",") if kw.strip()]
    if not keywords:
        await update.message.reply_text(
            "Send keywords separated by commas:\n"
            "/excluded intern, trainee, apprentice"
        )
        return

    save_excluded_title_keywords(keywords)
    lines = "\n".join(f"• {kw}" for kw in keywords)
    await update.message.reply_text(f"Excluded title keywords saved ({len(keywords)}):\n{lines}")


async def cmd_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set or show the minimum score threshold for sending jobs."""
    if not context.args:
        threshold = get_score_threshold()
        await update.message.reply_text(
            f"*Current threshold:* {threshold}%\n\n"
            "To change: /threshold 75",
            parse_mode="Markdown",
        )
        return

    try:
        value = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Threshold must be a number (0-100).")
        return

    if not 0 <= value <= 100:
        await update.message.reply_text("Threshold must be between 0 and 100.")
        return

    save_score_threshold(value)
    await update.message.reply_text(f"Score threshold set to {value}%")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show all-time job search statistics (server-side counts, not a full scan)."""
    try:
        sb = get_supabase()

        def _count(status=None) -> int:
            # head=True + count=exact → PostgREST returns only the total in
            # Content-Range, no rows. Avoids pulling 24k+ rows (and silently
            # capping at the PostgREST row limit) just to count them.
            q = sb.table("jobs").select("id", count="exact", head=True)
            if isinstance(status, (list, tuple)):
                q = q.in_("status", list(status))
            elif status is not None:
                q = q.eq("status", status)
            return q.execute().count or 0

        total = _count()
        if not total:
            await update.message.reply_text("No jobs processed yet.")
            return

        sent = _count("sent")
        low_score = _count("low_score")
        excluded = _count("filtered_excluded")
        errors = _count(("score_error", "notify_failed"))

        # Per-day averages from the created_at span — two one-row queries instead
        # of fetching every row to count distinct days.
        def _edge(newest: bool):
            r = (sb.table("jobs").select("created_at")
                 .order("created_at", desc=newest).limit(1).execute().data)
            return r[0]["created_at"] if r and r[0].get("created_at") else None

        first, last = _edge(False), _edge(True)
        days_active = 1
        if first and last:
            d0 = datetime.fromisoformat(first[:10])
            d1 = datetime.fromisoformat(last[:10])
            days_active = max((d1 - d0).days + 1, 1)

        pct = (sent / total * 100) if total else 0
        avg_processed = total / days_active
        avg_sent = sent / days_active

        await update.message.reply_text(
            f"*All-time statistics*\n\n"
            f"📊 Total processed: {total}\n"
            f"✅ Sent (suitable): {sent}\n"
            f"🔕 Low score: {low_score}\n"
            f"🚫 Excluded by title: {excluded}\n"
            f"❌ Errors: {errors}\n\n"
            f"📈 Match rate: {pct:.1f}%\n"
            f"📅 Days active: {days_active}\n"
            f"📬 Avg processed/day: {avg_processed:.1f}\n"
            f"✅ Avg sent/day: {avg_sent:.1f}",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.error(f"Stats error: {e}")
        await update.message.reply_text(f"Failed to load stats: {e}")


async def cmd_fetch_jobs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually trigger the daily job pipeline."""
    msg = await update.message.reply_text("Starting job search pipeline... This takes a few minutes.")

    def run_pipeline_sync():
        sys.path.insert(0, os.path.dirname(__file__))
        from process_jobs import run_pipeline
        run_pipeline()

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, run_pipeline_sync)
        await msg.edit_text("Pipeline finished. Check above for job cards and the daily summary.")
    except Exception as e:
        logger.error(f"fetch_jobs error: {e}")
        await msg.edit_text(f"Pipeline failed: {e}")


async def cmd_job(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Load a job URL, analyze it, set as active context."""
    if not context.args:
        await update.message.reply_text("Usage: /job <url>")
        return

    url = context.args[0]
    profile = get_profile()
    if not profile:
        await update.message.reply_text(
            "No resume profile found. Send me your PDF first."
        )
        return

    msg = await update.message.reply_text("Loading job page...")

    try:
        job_text = scrape_job_page(url)
    except Exception as e:
        await msg.edit_text(f"Failed to load page: {e}")
        return

    await msg.edit_text("Analyzing...")

    try:
        prep = extract_job_prep(job_text, profile)
    except Exception as e:
        await msg.edit_text(f"Analysis failed: {e}")
        return

    # Score with the SAME rubric as the daily pipeline (gpt-5-mini) so /job and
    # the scheduled run agree on the number. score_job reads the full description.
    job = {
        "title": prep.get("title", ""),
        "company": prep.get("company", ""),
        "description": job_text,
        "url": url,
        "source": "manual",
    }
    try:
        score_job(job, profile)  # mutates job: score, match_summary, red_flags, score_breakdown
    except Exception as e:
        await msg.edit_text(f"Scoring failed: {e}")
        return

    # Store as active job context (truncate description to the prompt budget)
    job["typical_qa"] = prep.get("typical_qa", [])
    job["description"] = job_text[:3000]
    active_jobs[update.effective_chat.id] = job

    score = job.get("score", 0)
    score_emoji = "✅" if score >= 70 else "⚠️" if score >= 50 else "❌"
    red_flags = job.get("red_flags", [])
    flags_text = "\n".join(f"⚠️ {f}" for f in red_flags) if red_flags else "None"

    qa_preview = ""
    for qa in job.get("typical_qa", [])[:2]:
        qa_preview += f"\n\n*Q:* {qa.get('question')}\n*A:* _{qa.get('answer', '')[:200]}_"

    await msg.edit_text(
        f"*{job.get('title')} @ {job.get('company')}*\n"
        f"Score: {score}/100 {score_emoji}\n\n"
        f"*Match:* {job.get('match_summary')}\n\n"
        f"*Red flags:* {flags_text}"
        f"{qa_preview}\n\n"
        f"_Job set as active context. Ask me anything about it._",
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Document handler (PDF resume)
# ---------------------------------------------------------------------------

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    doc = update.message.document
    if not doc.mime_type == "application/pdf":
        await update.message.reply_text("Please send a PDF file.")
        return

    msg = await update.message.reply_text("Parsing resume...")

    pdf_file = await context.bot.get_file(doc.file_id)
    pdf_bytes = await pdf_file.download_as_bytearray()

    try:
        raw_text = extract_text_from_pdf(bytes(pdf_bytes))
    except Exception as e:
        await msg.edit_text(f"Failed to read PDF: {e}")
        return

    if not raw_text:
        await msg.edit_text("Could not extract text from PDF. Try a text-based (non-scanned) PDF.")
        return

    await msg.edit_text("Extracting profile with AI...")

    try:
        parsed = parse_resume_with_openai(raw_text)
    except Exception as e:
        await msg.edit_text(f"AI parsing failed: {e}")
        return

    save_profile(raw_text, parsed)

    skills_preview = ", ".join(parsed.get("skills", [])[:8])
    await msg.edit_text(
        f"✅ Resume saved.\n\n"
        f"*Name:* {parsed.get('name')}\n"
        f"*Title:* {parsed.get('title')}\n"
        f"*Skills:* {skills_preview}...\n\n"
        f"I'll use this profile for job scoring and application prep.",
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Free-text message handler (assistant chat)
# ---------------------------------------------------------------------------

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_text = update.message.text

    profile = get_profile()

    # If user replies to a job card message, load that job as active context
    reply = update.message.reply_to_message
    if reply and reply.text:
        job_url = _extract_job_url_from_message(reply)
        if job_url:
            job_from_db = get_job_by_url(job_url)
            if job_from_db:
                active_jobs[chat_id] = job_from_db
                logger.info(f"Loaded job from reply: {job_from_db['title']} @ {job_from_db['company']}")

    active_job = active_jobs.get(chat_id)

    system_parts = [
        "You are a job application assistant helping a candidate prepare for job applications. "
        "Be concise and practical. When asked for answers to application questions, "
        "write in first person as the candidate."
    ]

    if profile:
        system_parts.append(
            f"\nCANDIDATE PROFILE:\n{json.dumps(profile, indent=2)}"
        )
    else:
        system_parts.append(
            "\nNo resume profile loaded yet. Remind the user to send their PDF resume."
        )

    if active_job:
        system_parts.append(
            f"\nACTIVE JOB CONTEXT:\n"
            f"Title: {active_job.get('title')} @ {active_job.get('company')}\n"
            f"Score: {active_job.get('score')}/100\n"
            f"Description: {active_job.get('description', '')}"
        )

    system_prompt = "\n".join(system_parts)

    try:
        response = get_openai().chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            max_tokens=600,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"Error: {e}"

    await update.message.reply_text(reply)


# ---------------------------------------------------------------------------
# Scheduled daily pipeline (replaces cron container)
# ---------------------------------------------------------------------------

async def scheduled_pipeline(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Run the job search pipeline for one scheduled slot.

    `context.job.data` carries the slot label ('03:00' etc.) so the pipeline
    can gate low-yield sources (Wellfound) to a single slot.
    """
    slot = getattr(context.job, "data", None)
    logger.info(f"Scheduled pipeline triggered (slot={slot})")

    def run_pipeline_sync():
        sys.path.insert(0, os.path.dirname(__file__))
        from process_jobs import run_pipeline
        run_pipeline(slot=slot)

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, run_pipeline_sync)
        logger.info("Scheduled pipeline finished")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Scheduled pipeline failed: {e}\n{tb}")
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from notify_telegram import _send_chunked
            _send_chunked(
                f"⚠️ Scheduled pipeline crashed:\n\n{type(e).__name__}: {e}\n\n{tb}",
                parse_mode=None,
            )
        except Exception as notify_err:
            logger.error(f"Could not notify Telegram about pipeline crash: {notify_err}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("keywords", cmd_keywords))
    app.add_handler(CommandHandler("wellfound", cmd_wellfound))
    app.add_handler(CommandHandler("threshold", cmd_threshold))
    app.add_handler(CommandHandler("redflags", cmd_redflags))
    app.add_handler(CommandHandler("excluded", cmd_excluded))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("fetch_jobs", cmd_fetch_jobs))
    app.add_handler(CommandHandler("job", cmd_job))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Pipeline 4×/day at 03:00 / 09:00 / 15:00 / 21:00 UTC
    # (= 10:00 / 16:00 / 22:00 / 04:00 GMT+7), every day. Even 6h spacing.
    for hour in (3, 9, 15, 21):
        slot = f"{hour:02d}:00"
        app.job_queue.run_daily(
            scheduled_pipeline,
            time=dt_time(hour=hour, minute=0, tzinfo=timezone.utc),
            days=(0, 1, 2, 3, 4, 5, 6),
            name=f"job_search_{slot}",
            data=slot,
        )
    logger.info("Scheduled pipeline at 03/09/15/21 UTC daily (4×/day)")

    logger.info("Bot started (long-polling)")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
