# Workflow: Telegram Bot Usage

## Objective
Interactive job application assistant. Always running on VPS. Handles resume setup, job analysis, and application question prep.

## Setup (one-time)

1. Create a Telegram bot via @BotFather → get `TELEGRAM_BOT_TOKEN`
2. Get your chat ID: start a chat with @userinfobot → get `TELEGRAM_CHAT_ID`
3. Add both to `.env`
4. Send your PDF resume to the bot → wait for confirmation

## Bot Commands

| Command | What it does |
|---------|-------------|
| `/start` | Welcome message |
| `/help` | List all commands |
| `/status` | Show current profile summary + active job |
| `/job <url>` | Load a LinkedIn job URL, analyze it, set as chat context |

## Resume Loading

Send any PDF file to the bot:
1. Bot extracts text from PDF using pypdf
2. OpenAI parses it into structured JSON (name, skills, experience, projects)
3. Profile saved to `supabase.profile` (overwrites previous)
4. Bot confirms with skill summary

To update your resume: just send a new PDF anytime.

## Daily Job Cards

The daily pipeline sends job cards to this chat automatically. Each card shows:
- Job title, company
- Score (0-100) with color: ✅ ≥80, 🟡 70-79
- Match summary
- Red flags (if any)
- 3 prep Q&As (likely interview/application questions with suggested answers)
- Link to the job

After receiving a card, you can ask the bot anything about that job.
The bot remembers the last job it sent as context.

## Interactive Chat

After a job card arrives (or after `/job <url>`), ask anything:

**Application questions:**
> "What to write for 'describe your automation experience'?"
> "Write a cover letter for this role"
> "Why do I want to work here?"

**Research:**
> "What do you know about this company?"
> "Is this a good company for someone with my background?"

**Iteration:**
> "Make the cover letter shorter"
> "Add more about n8n"
> "More formal tone"

The bot uses your full resume profile + the active job description as context.

## Edge Cases

- **Bot not responding**: check if `telegram_bot.py` process is running on VPS
- **"No resume profile"**: send your PDF to the bot again
- **Job page fails to load** (`/job` command): try copy-pasting the job description directly into chat instead
- **Active job context lost after restart**: set again with `/job <url>` or just paste the job description

## Running the Bot

```bash
python tools/telegram_bot.py
```

On VPS it runs as a Docker service defined in `docker-compose.yml`.
