"""Shared OpenAI client factory + a thin Responses helper.

One place for the client singleton (mirrors db.get_supabase) and for the
default model / reasoning effort, so call sites don't re-roll OpenAI(...) or
repeat the same create() kwargs.
"""

import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gpt-5-mini"

_client: OpenAI | None = None


def get_openai() -> OpenAI:
    """Return a cached OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def respond(
    instructions: str,
    user_input: str,
    *,
    json_mode: bool = False,
    max_output_tokens: int | None = None,
    reasoning_effort: str = "minimal",
    model: str = DEFAULT_MODEL,
) -> str:
    """One-shot Responses API call returning output_text.

    Centralizes the model + reasoning-effort + store=False defaults that the bot
    otherwise repeats per call site. With json_mode=True the response is forced
    to a JSON object (the caller must still include the word "json" in the input,
    as the Responses API requires).
    """
    kwargs: dict = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "reasoning": {"effort": reasoning_effort},
        "store": False,
    }
    if json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    return get_openai().responses.create(**kwargs).output_text
