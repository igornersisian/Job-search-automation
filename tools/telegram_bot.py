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
import io
import json
import logging
import asyncio
from datetime import datetime

import httpx
from openai import OpenAI
from supabase import create_client, Client
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

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Clients (lazy init so module can be imported without env vars set) ---
_openai_client: OpenAI | None = None
_supabase: Client | None = None


def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        )
    return _supabase


# In-memory active job context per chat_id
active_jobs: dict[int, dict] = {}


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def get_profile() -> dict | None:
    """Return the latest parsed resume profile from Supabase."""
    result = get_supabase().table("profile").select("parsed").order("updated_at", desc=True).limit(1).execute()
    if result.data:
        return result.data[0]["parsed"]
    return None


def save_profile(raw_text: str, parsed: dict) -> None:
    """Upsert resume profile in Supabase (single row, id=1)."""
    get_supabase().table("profile").upsert({
        "id": 1,
        "raw_text": raw_text,
        "parsed": parsed,
        "updated_at": datetime.utcnow().isoformat(),
    }).execute()


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
        model="gpt-4o-mini",
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


def analyze_job_with_openai(job_text: str, profile: dict) -> dict:
    """Analyze a job posting against the user's profile."""
    response = get_openai().chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a career advisor. Analyze the job posting against the candidate's profile. "
                    "Return a JSON object with:\n"
                    "- title: string (job title)\n"
                    "- company: string\n"
                    "- score: int 0-100 (fit score)\n"
                    "- match_summary: string (what matches well, 1-2 sentences)\n"
                    "- red_flags: list of strings (concerns or mismatches)\n"
                    "- typical_qa: list of objects [{question, answer}] "
                    "  (3-5 likely application questions with suggested answers based on candidate profile)\n"
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CANDIDATE PROFILE:\n{json.dumps(profile, indent=2)}\n\n"
                    f"JOB POSTING:\n{job_text}"
                ),
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


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
        "/job <url> — analyze a specific job URL and set it as active context\n\n"
        "*How to use:*\n"
        "1. Send your resume PDF — I'll parse and remember it\n"
        "2. Every morning I'll send you matching job cards\n"
        "3. After seeing a card, ask me anything: \"What to write for cover letter?\", "
        "\"What to answer to 'describe your experience with n8n'?\" etc.\n"
        "4. Use /job <url> to load any job for discussion",
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
        analysis = analyze_job_with_openai(job_text, profile)
    except Exception as e:
        await msg.edit_text(f"Analysis failed: {e}")
        return

    # Store as active job context
    analysis["url"] = url
    analysis["description"] = job_text[:3000]
    active_jobs[update.effective_chat.id] = analysis

    score = analysis.get("score", 0)
    score_emoji = "✅" if score >= 70 else "⚠️" if score >= 50 else "❌"
    red_flags = analysis.get("red_flags", [])
    flags_text = "\n".join(f"⚠️ {f}" for f in red_flags) if red_flags else "None"

    qa_preview = ""
    for qa in analysis.get("typical_qa", [])[:2]:
        qa_preview += f"\n\n*Q:* {qa.get('question')}\n*A:* _{qa.get('answer', '')[:200]}_"

    await msg.edit_text(
        f"*{analysis.get('title')} @ {analysis.get('company')}*\n"
        f"Score: {score}/100 {score_emoji}\n\n"
        f"*Match:* {analysis.get('match_summary')}\n\n"
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
            f"Description excerpt: {active_job.get('description', '')[:2000]}"
        )

    system_prompt = "\n".join(system_parts)

    try:
        response = get_openai().chat.completions.create(
            model="gpt-4o",
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("job", cmd_job))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started (long-polling)")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
