"""Shared Supabase client factory — one place for connection options.

Cached per timeout so the bot (10s, fail-fast for interactive commands) and the
pipeline (30s, tolerant of batch writes on a flaky self-hosted DB) each reuse
their own client instead of every module re-rolling create_client().
"""

import os

from supabase import create_client, Client, ClientOptions
from dotenv import load_dotenv

load_dotenv()

_clients: dict[int, Client] = {}


def get_supabase(timeout: int = 30) -> Client:
    """Return a cached Supabase client for the given PostgREST timeout."""
    client = _clients.get(timeout)
    if client is None:
        client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
            options=ClientOptions(postgrest_client_timeout=timeout),
        )
        _clients[timeout] = client
    return client
