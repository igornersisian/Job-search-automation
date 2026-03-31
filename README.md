# Job Application Automation

A self-hosted job search assistant that runs on a schedule, scores incoming job listings against your resume, and lets you interact with them via a Telegram bot.

## How it works

1. **Send your resume PDF** to the Telegram bot — it parses it with OpenAI and stores a structured profile in Supabase.
2. **Every weekday at 9:00 UTC**, a cron job fetches jobs from LinkedIn via Apify, scores each one against your profile, and sends high-scoring cards to Telegram.
3. **In the bot**, you can ask follow-up questions, analyze any job URL with `/job <url>`, or get cover letter / application question help.

## Architecture

```
telegram_bot.py   — Telegram interface (resume upload, /job, free chat)
process_jobs.py   — Daily pipeline (Apify → score → notify)
score_job.py      — OpenAI scoring logic
run_apify_search.py — LinkedIn job search via Apify
notify_telegram.py  — Send job cards to Telegram
setup_db.py       — Auto-create Supabase tables on startup
```

## Prerequisites

- Python 3.12+
- Docker (for deployment)
- Accounts: OpenAI, Telegram bot, Supabase (or self-hosted), Apify

## Environment variables

Create a `.env` file (see `.env.example`):

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | From [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | Your chat ID (the bot will send daily digests here) |
| `OPENAI_API_KEY` | OpenAI API key |
| `SUPABASE_URL` | Your Supabase project URL (e.g. `https://your-project.supabase.co`) |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key (not the anon key) |
| `APIFY_API_TOKEN` | Apify API token |
| `DATABASE_URL` | *(Optional)* Direct Postgres URL — enables auto table creation on startup |

### DATABASE_URL format (self-hosted Supabase)

```
postgresql://postgres:<password>@<host>:5432/postgres
```

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

## Usage

| Action | What to do |
|---|---|
| Set up profile | Send your resume PDF to the bot |
| Check profile | `/status` |
| Analyze a job | `/job https://linkedin.com/jobs/view/...` |
| Free chat | Just type — the bot knows your profile and active job context |
| Get daily digests | Happens automatically on weekdays at 9 UTC |
