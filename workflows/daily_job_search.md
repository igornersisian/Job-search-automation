# Workflow: Daily Job Search

## Objective
Run once per weekday morning. Fetch LinkedIn jobs posted in the last 24h, filter and score them against the user's resume, and send qualifying matches to Telegram.

## Prerequisites
- `.env` filled with all required keys (see `.env` file)
- Resume profile saved in Supabase `profile` table (send PDF to Telegram bot)
- Supabase tables created (see **Database Setup** below)
- Apify account with `curious_coder/linkedin-jobs-scraper` actor available

## Database Setup (run once in Supabase SQL editor)
```sql
create table if not exists profile (
  id serial primary key,
  raw_text text,
  parsed jsonb,
  updated_at timestamptz default now()
);

create table if not exists jobs (
  id text primary key,
  source text default 'linkedin',
  title text,
  company text,
  url text,
  salary_text text,
  is_remote boolean default true,
  description text,
  posted_at timestamptz,
  score int,
  match_summary text,
  red_flags text,
  typical_qa text,
  status text,  -- sent / low_score / filtered_junior / score_error / notify_failed
  created_at timestamptz default now()
);
```

## Trigger
Cron: `0 9 * * 1-5` (Mon-Fri 9:00 UTC)
Command: `python tools/process_jobs.py`

## Flow

1. **Load profile** — reads parsed resume from `supabase.profile`
   - If missing: log error, abort (user must send PDF to bot first)

2. **Apify search** (`tools/run_apify_search.py`)
   - Runs `curious_coder/linkedin-jobs-scraper` with queries:
     - "n8n automation", "no-code automation engineer", "AI automation engineer",
       "workflow automation developer", "AI agent developer", "vibe coding developer"
   - Filters: remote=true, datePosted=past24Hours, limit=20 per query
   - Waits up to 5 min for completion, downloads dataset

3. **Per job: normalise** — maps Apify fields to standard schema

4. **Deduplication** — checks `supabase.jobs` by job ID. Skip if already seen.

5. **Junior/intern filter** — skip if title or description contains:
   intern, internship, entry level, entry-level, no experience required,
   0-1 year, fresh graduate, recent graduate, junior developer

6. **Score** (`tools/score_job.py`)
   - OpenAI `gpt-4o-mini` analyzes job vs profile
   - Returns: score (0-100), match_summary, red_flags, typical_qa (3-5 Q&As)
   - If score < 70 → save as `low_score`, skip Telegram

7. **Notify** (`tools/notify_telegram.py`)
   - Sends formatted card: title, company, salary, score, match summary, red flags, prep Q&A
   - Saves job as `sent` in Supabase

8. **Daily summary** — sends count of sent/filtered/duped jobs

## Edge Cases

- **Apify actor changes output schema**: check field names in `normalise_job()` in `process_jobs.py`
- **Profile missing**: pipeline aborts cleanly with error. Fix: send PDF to Telegram bot
- **OpenAI rate limit**: score_job will raise exception → job saved as `score_error`, pipeline continues
- **Telegram send fails**: job saved as `notify_failed`, pipeline continues
- **No new jobs found**: pipeline completes, sends summary with 0 sent

## Cost Estimate (per run)
- Apify: ~120 jobs × $1/1000 = ~$0.12/day
- OpenAI scoring: ~60 jobs × ~$0.002 = ~$0.12/day (gpt-4o-mini is cheap)
- Total: ~$0.25/day, ~$5/month
