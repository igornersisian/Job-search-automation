# Job Application Automation

A self-hosted job search assistant that scrapes 20+ job platforms daily, scores listings against your resume with AI, and delivers the best matches to your Telegram.

## How it works

1. **Send your resume PDF** to the Telegram bot — it parses it with OpenAI and stores a structured profile in Supabase.
2. **Every weekday at 9:00 UTC**, a cron job fetches remote jobs from 6 sources (20+ platforms) in parallel, deduplicates, scores each one against your profile, and sends high-scoring cards to Telegram.
3. **In the bot**, you can ask follow-up questions, analyze any job URL with `/job <url>`, tune scoring with `/threshold` and `/redflags`, or get cover letter help.

## Sources

All scrapers run in parallel via Apify:

| Source | Platforms | Actor |
|--------|-----------|-------|
| LinkedIn | LinkedIn | `cheap_scraper/linkedin-job-scraper` |
| Glassdoor | Glassdoor | `valig/glassdoor-jobs-scraper` |
| Indeed | Indeed | `valig/indeed-jobs-scraper` |
| Wellfound | Wellfound (AngelList) | `clearpath/wellfound-api-ppe` |
| RemoteBoards | RemoteOK, Remotive, WeWorkRemotely | `zyncodltd/JobsFlow` |
| ATS | Greenhouse, Lever, Workday, Ashby, Workable, SmartRecruiters, BambooHR, Rippling, Personio, JazzHR, Breezy HR, Recruitee, Polymer | `jobo.world/ats-jobs-search` |

## Architecture

```
tools/
  telegram_bot.py          — Telegram interface (resume upload, /job, /threshold, /redflags, /stats)
  process_jobs.py          — Daily pipeline orchestrator (fetch → dedup → score → notify)
  run_apify_search.py      — LinkedIn scraper
  run_glassdoor_search.py  — Glassdoor scraper
  run_indeed_search.py     — Indeed scraper
  run_wellfound_search.py  — Wellfound scraper
  run_remoteboards_search.py — RemoteOK + Remotive + WeWorkRemotely (JobsFlow)
  run_ats_search.py        — 13 ATS platforms (Greenhouse, Lever, Workday, etc.)
  score_job.py             — OpenAI scoring + enrichment
  notify_telegram.py       — Telegram job card sender
  setup_db.py              — Auto-create Supabase tables on startup

workflows/
  daily_job_search.md      — Detailed pipeline SOP
  telegram_bot_usage.md    — Bot command reference
```

## Pipeline flow

```
6 scrapers (parallel)
    ↓
Normalise to shared schema
    ↓
Dedup: cross-run (Supabase) → same-run (ID) → fuzzy (title+company+description)
    ↓
Junior/intern filter
    ↓
Quick score (OpenAI, 0-100)
    ↓
Score >= threshold? → Enrich (match_summary, red_flags) → Telegram card
    ↓
Daily summary → Telegram
```

## Prerequisites

- Python 3.11+
- Docker (for deployment)
- Accounts: OpenAI, Telegram bot, Supabase (or self-hosted), Apify

## Environment variables

Create a `.env` file (see `.env.example`):

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | From [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | Your chat ID (the bot will send daily digests here) |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENROUTER_API_KEY` | OpenRouter API key (fallback for scoring) |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key (not the anon key) |
| `APIFY_API_TOKEN` | Apify API token |
| `DATABASE_URL` | *(Optional)* Direct Postgres URL — enables auto table creation on startup |

## Database setup

### Option A — Automatic (with DATABASE_URL)

Add `DATABASE_URL` to your environment. Tables are created automatically when the bot starts.

### Option B — Manual (Supabase SQL editor)

Run the contents of [migrations/001_init.sql](migrations/001_init.sql) in your Supabase SQL editor.

## Running locally

```bash
pip install -r requirements.txt
python tools/telegram_bot.py
```

## Deployment (Dokploy / Docker Compose)

1. Push to GitHub.
2. In Dokploy → **New Service → Compose** → point to this repo, branch `main`, path `./docker-compose.yml`.
3. In **Environment** tab, add all variables from the table above.
4. Click **Deploy**.

The `bot` service starts long-polling immediately. The `cron` service runs `process_jobs.py` every weekday at 9:00 UTC.

## Bot commands

| Command | Description |
|---|---|
| Send PDF | Upload resume to set up profile |
| `/status` | Check current profile |
| `/job <url>` | Analyze any job listing |
| `/threshold <N>` | Set minimum score (default 70) |
| `/redflags` | Configure personal dealbreakers |
| `/stats` | View pipeline statistics |
| Free text | Chat about jobs, get cover letter help |

## Cost estimate

~$0.50/day, ~$10/month (Apify scrapers + OpenAI scoring).
