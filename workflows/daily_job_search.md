# Workflow: Daily Job Search

## Objective
Run once per weekday morning. Fetch remote jobs posted in the last 24h from 6 sources (20+ platforms), filter and score them against the user's resume, and send qualifying matches to Telegram.

## Prerequisites
- `.env` filled with all required keys (see `.env` file)
- Resume profile saved in Supabase `profile` table (send PDF to Telegram bot)
- Search keywords set via `/keywords` bot command (no defaults — pipeline won't run without them)
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
**4×/day** at 03:00 / 09:00 / 15:00 / 21:00 UTC (= 10/16/22/04 GMT+7), every
day. Registered as one `job_queue.run_daily` per slot in `telegram_bot.py`
(`scheduled_pipeline`), with the slot label passed via `data`.
Manual run: `python tools/process_jobs.py [--lookback SECONDS] [--slot HH:MM]`.

Search window: `LOOKBACK_SECONDS` env (default 25200 = 7h = 6h interval + 1h
buffer). Only LinkedIn (`publishedAt=r{LOOKBACK}`) and ATS (`posted_after`)
actually narrow with it — Indeed/Glassdoor stay 24h (actor minimum), dedup
removes the slot overlap. Wellfound runs only in the 09:00 slot (low yield).

## Sources

All scrapers run on the **unified runner** `apify_client.run_actor_job(...)`,
which handles token rotation, retry-on-credit (per source, on the next token),
the cap-check, `usageTotalUsd` capture and error capture in ONE place. Each
`run_*_search.py` just provides `actor_id` + input builder + `normalise` and
returns a `SourceResult`. The lightweight scrapers run in parallel; ATS runs
after them (Apify per-account memory cap). Indeed/Glassdoor fan out per keyword
in parallel internally.

| # | Source | Script | Apify Actor | Platforms | Price/1000 |
|---|--------|--------|-------------|-----------|------------|
| 1 | LinkedIn | `run_apify_search.py` | `cheap_scraper/linkedin-job-scraper` | LinkedIn | ~$1.00 |
| 2 | Glassdoor | `run_glassdoor_search.py` | `valig/glassdoor-jobs-scraper` | Glassdoor | ~$1.00 |
| 3 | Indeed | `run_indeed_search.py` | `valig/indeed-jobs-scraper` | Indeed | ~$1.00 |
| 4 | Wellfound | `run_wellfound_search.py` | `clearpath/wellfound-api-ppe` | Wellfound (AngelList) | ~$0.15 (pageLimit=1, keywords→slugs) |
| 5 | RemoteBoards | `run_remoteboards_search.py` | `zyncodltd/JobsFlow` | RemoteOK, Remotive, WeWorkRemotely | $0.01 |
| 6 | ATS | `run_ats_search.py` | `jobo.world/ats-jobs-search` | Greenhouse, Lever, Workday, Ashby, Workable, SmartRecruiters, BambooHR, Rippling, Personio, JazzHR, Breezy HR, Recruitee, Polymer | $0.30 |

**Total coverage:** 20+ job platforms in a single pipeline run.

### Wellfound keyword→slug caveat

Wellfound doesn't support free-text search — it uses fixed role-category URLs like `wellfound.com/role/r/automation-engineer`. The pipeline converts `search_keywords` to URL slugs automatically (`_keyword_to_slug()`), but many keywords won't map to real Wellfound categories and will simply return 0 results. This is harmless (no errors, no wasted cost beyond the actor start) but means Wellfound coverage depends on how "role-like" the keywords are. If this becomes a problem, consider a mapping layer from keywords to known Wellfound role slugs.

### ATS query limit

The `jobo.world/ats-jobs-search` actor caps `queries` at **5 items** — sending more returns HTTP 400 `invalid-input: Field input.queries must NOT have more than 5 items`. The pipeline passes `search_keywords[:5]` (see `run_ats_search.py`), so only the first five keywords reach this source. If you want broader ATS coverage, reorder `/keywords` to put the most important ones first, or split into multiple sequential runs.

### Error reporting

When any scraper fails, the daily summary lists `⚠️ Failed sources` and then sends each error as a separate plain-text Telegram message containing the full HTTP status + response body (token redacted, body capped at 1500 chars). Don't dig through server logs — copy the Telegram message back into the conversation and ask Claude to fix.

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
   - `quick_score()` → `score_job()` — a **single** `gpt-5-mini` call per
     candidate that writes match_summary + red_flags, then sub-scores (0-100)
   - If score < threshold (default 70) → `low_score`, skip
   - `send_job_card()` → Telegram notification for qualifying jobs
   - All rows (excluded/low_score/sent/...) are **batch-upserted** to Supabase
     once at the end, not per-job

5. **Daily summary** — sends stats to Telegram

## Edge Cases

- **Apify actor changes output schema**: check normalise function in corresponding `run_*_search.py`
- **Profile missing**: pipeline aborts cleanly with error. Fix: send PDF to Telegram bot
- **Keywords missing**: pipeline aborts, sends Telegram notification. Fix: `/keywords` bot command
- **OpenAI rate limit**: score_job will raise exception → job saved as `score_error`, pipeline continues
- **Telegram send fails**: job saved as `notify_failed`, pipeline continues
- **No new jobs found**: pipeline completes, sends summary with 0 sent
- **Single source fails**: other sources continue, error logged

## Cost Estimate (real, from Apify console + DB)
Per-source $/run (24h window): ATS ~$0.40 (per-result, narrows with window),
LinkedIn ~$0.13 (narrows), Glassdoor ~$0.08, Indeed/RemoteBoards $0, Wellfound
~$0.12. With 4×/day @6h windows + Wellfound 1×: **~$0.96/day ≈ ~$29/mo** Apify.
OpenAI scoring is one `gpt-5-mini` call/candidate (~$0.002) and does NOT grow
with frequency (cross-run dedup → each posting scored once). Actual per-run cost
is captured from each run's `usageTotalUsd` and stored in `pipeline_runs` +
shown in the Telegram summary — prefer those real numbers over this estimate.

## Observability (always queryable)
Every run writes one row to Supabase `pipeline_runs` (`per_source`, `totals`,
`errors`, `ok`) — including fatal crashes (written in `run_pipeline`'s
`finally`). Debug with:
`select started_at, ok, totals, errors from pipeline_runs order by started_at desc limit 5;`
The Telegram daily summary shows a per-source `fetched / new / sent ($cost)`
breakdown + a ⚠️cap flag when a source hit its result cap (possible truncation).
