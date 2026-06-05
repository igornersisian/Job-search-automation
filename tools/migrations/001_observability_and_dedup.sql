-- Migration 001: observability (pipeline_runs) + cross-run dedup (jobs.fingerprint)
-- Run this once in the Supabase SQL editor (self-hosted Studio).
-- Safe to re-run: all statements are IF NOT EXISTS / idempotent.

-- 1) Per-run audit log. One row per pipeline run. Every error lands here so it
--    is always queryable, not just in truncated Telegram messages / ephemeral logs.
create table if not exists pipeline_runs (
  id          uuid primary key default gen_random_uuid(),
  started_at  timestamptz default now(),
  finished_at timestamptz,
  slot_utc    text,        -- '03:00' / '09:00' / '15:00' / '21:00'
  keywords    jsonb,
  per_source  jsonb,       -- {linkedin:{fetched,new,sent,cost_usd,capped,error,ms,attempts}, ...}
  totals      jsonb,       -- {fetched,new,sent,cost_usd}
  errors      jsonb,       -- [{source, stage, message}]
  ok          boolean default true
);

create index if not exists pipeline_runs_started_at_idx
  on pipeline_runs (started_at desc);

-- 2) Cross-run fuzzy dedup key on jobs: norm_title|norm_company.
--    Lets Phase 1 cheaply check a candidate against recently-seen jobs
--    (not just exact id) so the same posting from another source in a later
--    slot is not re-scored / re-sent.
alter table jobs add column if not exists fingerprint text;

create index if not exists jobs_fingerprint_created_idx
  on jobs (fingerprint, created_at desc);

-- Optional backfill for existing rows is unnecessary: the cross-run check only
-- looks back a few days, and all new rows get fingerprint on insert.
