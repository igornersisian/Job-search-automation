-- Run this in your Supabase SQL Editor to set up required tables.

CREATE TABLE IF NOT EXISTS profile (
    id          INTEGER PRIMARY KEY DEFAULT 1,
    raw_text    TEXT,
    parsed      JSONB,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    source        TEXT,
    title         TEXT,
    company       TEXT,
    url           TEXT,
    salary_text   TEXT,
    is_remote     BOOLEAN DEFAULT TRUE,
    description   TEXT,
    posted_at     TIMESTAMPTZ,
    score         INTEGER,
    match_summary TEXT,
    red_flags     TEXT,
    typical_qa    TEXT,
    status        TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
