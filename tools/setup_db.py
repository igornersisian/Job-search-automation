"""
Database setup — creates required tables if they don't exist.

Requires DATABASE_URL env var pointing to the Postgres instance directly.
For self-hosted Supabase: postgresql://postgres:<password>@<host>:5432/postgres

Usage:
    python tools/setup_db.py
Or called automatically at bot startup when DATABASE_URL is set.
"""

import os
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CREATE_PROFILE = """
CREATE TABLE IF NOT EXISTS profile (
    id          INTEGER PRIMARY KEY DEFAULT 1,
    raw_text    TEXT,
    parsed      JSONB,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_JOBS = """
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
"""


def ensure_tables() -> None:
    """Create tables if they don't exist. Requires DATABASE_URL."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        logger.warning("DATABASE_URL not set — skipping auto table creation")
        return

    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(CREATE_PROFILE)
            cur.execute(CREATE_JOBS)
        conn.close()
        logger.info("Database tables verified/created")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ensure_tables()
    print("Done.")
