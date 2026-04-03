# Workflow: Daily Job Search

## Objective
Run once per weekday morning. Fetch remote jobs posted in the last 24h from 6 sources (20+ platforms), filter and score them against the user's resume, and send qualifying matches to Telegram.

## Prerequisites
- `.env` filled with all required keys (see `.env` file)
- Resume profile saved in Supabase `profile` table (send PDF to Telegram bot)
- Supabase tables created (see **Database Setup** below)
- Apify account with sufficient credits

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

## Sources

All 6 scrapers run **in parallel** via `ThreadPoolExecutor`:

| # | Source | Script | Apify Actor | Platforms | Price/1000 |
|---|--------|--------|-------------|-----------|------------|
| 1 | LinkedIn | `run_apify_search.py` | `cheap_scraper/linkedin-job-scraper` | LinkedIn | ~$1.00 |
| 2 | Glassdoor | `run_glassdoor_search.py` | `valig/glassdoor-jobs-scraper` | Glassdoor | ~$1.00 |
| 3 | Indeed | `run_indeed_search.py` | `valig/indeed-jobs-scraper` | Indeed | ~$1.00 |
| 4 | Wellfound | `run_wellfound_search.py` | `clearpath/wellfound-api-ppe` | Wellfound (AngelList) | ~$1.00 |
| 5 | RemoteBoards | `run_remoteboards_search.py` | `zyncodltd/JobsFlow` | RemoteOK, Remotive, WeWorkRemotely | $0.01 |
| 6 | ATS | `run_ats_search.py` | `jobo.world/ats-jobs-search` | Greenhouse, Lever, Workday, Ashby, Workable, SmartRecruiters, BambooHR, Rippling, Personio, JazzHR, Breezy HR, Recruitee, Polymer | $0.30 |

**Total coverage:** 20+ job platforms in a single pipeline run.

## Flow

1. **Load profile** — reads parsed resume from `supabase.profile`
   - If missing: log error, abort (user must send PDF to bot first)

2. **Parallel search** (`process_jobs.py → _fetch_all_sources`)
   - All 6 scrapers run simultaneously
   - Each scraper normalises output to shared schema: `id, title, company, url, salary, description, location, postedAt, is_remote, source`

3. **Phase 1: Dedup + Junior filter** (fast, sequential, no API calls)
   - Cross-run dedup: checks `supabase.jobs` by job ID
   - Same-run dedup: skips duplicate IDs within current batch
   - Fuzzy dedup: normalises title+company, compares description similarity (Jaccard >= 45%)
   - Junior/intern filter: skips based on title/description keywords

4. **Phase 2: Score + Enrich + Notify** (parallel, 5 workers)
   - `quick_score()` — fast OpenAI scoring (0-100)
   - If score < threshold (default 70) → save as `low_score`, skip
   - `score_job()` — enrichment: match_summary, red_flags
   - `send_job_card()` → Telegram notification
   - Save to Supabase

5. **Daily summary** — sends stats to Telegram

## Edge Cases

- **Apify actor changes output schema**: check normalise function in corresponding `run_*_search.py`
- **Profile missing**: pipeline aborts cleanly with error. Fix: send PDF to Telegram bot
- **OpenAI rate limit**: score_job will raise exception → job saved as `score_error`, pipeline continues
- **Telegram send fails**: job saved as `notify_failed`, pipeline continues
- **No new jobs found**: pipeline completes, sends summary with 0 sent
- **Single source fails**: other sources continue, error logged

## Cost Estimate (per run)
- Apify scrapers: ~$0.30-0.50/run (ATS + RemoteBoards are cheapest)
- OpenAI scoring: ~60 jobs × ~$0.002 = ~$0.12/day (gpt-4o-mini)
- Total: ~$0.50/day, ~$10/month
