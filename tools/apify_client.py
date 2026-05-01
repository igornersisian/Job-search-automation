"""
Shared Apify HTTP client with automatic token rotation.

Reads APIFY_API_TOKEN, APIFY_API_TOKEN2, APIFY_API_TOKEN3, ... from env
(stops at the first missing index >= 2). Each token usually represents a
separate free Apify account with $5 of credit.

When a token runs out of credit, the next request returns HTTP 402/403 with
a body mentioning "monthly usage", "limit", "exceeded", "quota", etc. We
detect that, persist the slot as exhausted in .tmp/apify_token_state.json,
and rotate to the next token.

Important: a run started by token A cannot be polled or fetched by token B
(different account, different runs). So `post()` returns the token it used
and the caller must thread it through the wait_for_run / fetch_dataset
calls via `get()`.

To reset (e.g. after monthly credit refresh), delete .tmp/apify_token_state.json.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

STATE_FILE = Path(".tmp") / "apify_token_state.json"

# Substrings that indicate the response failed because the account is out
# of credit / quota — i.e. rotation should help.
_CREDIT_ERROR_SUBSTRINGS = (
    "monthly-usage",
    "monthly usage",
    "usage-hard-limit",
    "usage hard limit",
    "usage limit",
    "exceeded",
    "quota",
    "insufficient",
    "credit",
    "no-available-resources",
    "payment",
    "billing",
)


def _load_tokens() -> list[str]:
    """Return tokens in slot order: APIFY_API_TOKEN, APIFY_API_TOKEN2, ..."""
    tokens: list[str] = []
    primary = os.environ.get("APIFY_API_TOKEN")
    if primary:
        tokens.append(primary)
    i = 2
    while True:
        v = os.environ.get(f"APIFY_API_TOKEN{i}")
        if not v:
            break
        tokens.append(v)
        i += 1
    return tokens


def _fingerprint(token: str) -> str:
    """Last 4 chars of the token, safe to log."""
    if not token:
        return "????"
    return token[-4:] if len(token) >= 4 else "????"


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.warning(f"Could not parse {STATE_FILE}, treating as empty")
    return {"exhausted": []}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _exhausted_set() -> set[int]:
    return set(_load_state().get("exhausted", []))


def get_active_token() -> tuple[int, str]:
    """First non-exhausted token. Raises if none available."""
    tokens = _load_tokens()
    if not tokens:
        raise RuntimeError(
            "No APIFY_API_TOKEN* env variables found. Add at least APIFY_API_TOKEN to .env."
        )
    exhausted = _exhausted_set()
    for idx, tok in enumerate(tokens):
        if idx not in exhausted:
            return idx, tok
    raise RuntimeError(
        f"All {len(tokens)} Apify token slots are marked exhausted. "
        f"Top up an account or delete {STATE_FILE} to retry."
    )


def mark_exhausted(index: int, reason: str = "") -> None:
    state = _load_state()
    exhausted = set(state.get("exhausted", []))
    if index in exhausted:
        return
    exhausted.add(index)
    state["exhausted"] = sorted(exhausted)
    _save_state(state)
    tokens = _load_tokens()
    fp = _fingerprint(tokens[index]) if index < len(tokens) else "????"
    logger.warning(
        f"Apify token slot {index + 1} (...{fp}) marked exhausted. Reason: {reason[:200]}"
    )


def is_credit_error(status_code: int, body: str) -> bool:
    """True if the response looks like an out-of-credit / quota failure."""
    if status_code not in (402, 403):
        return False
    body_lower = (body or "").lower()
    return any(s in body_lower for s in _CREDIT_ERROR_SUBSTRINGS)


def looks_like_credit_failure_message(msg: str) -> bool:
    """For run statusMessage strings on FAILED/ABORTED runs."""
    if not msg:
        return False
    m = msg.lower()
    return any(s in m for s in _CREDIT_ERROR_SUBSTRINGS)


def _token_index(token: str) -> int | None:
    for idx, tok in enumerate(_load_tokens()):
        if tok == token:
            return idx
    return None


def post(
    url: str,
    *,
    json_body: dict,
    timeout: int = 30,
    extra_params: dict | None = None,
) -> tuple[httpx.Response, str]:
    """POST with auto-rotation on credit errors.

    Returns (response, token_used). Caller is responsible for calling
    `response.raise_for_status()` on non-credit errors.

    The returned token MUST be reused for any follow-up GETs that reference
    resources created by this POST (Apify runs are scoped to the account).
    """
    tokens = _load_tokens()
    if not tokens:
        raise RuntimeError("No APIFY_API_TOKEN* env variables found")

    last_response: httpx.Response | None = None
    last_token: str = ""

    # try up to N times where N == number of token slots
    for _ in range(len(tokens) + 1):
        try:
            idx, tok = get_active_token()
        except RuntimeError:
            # all exhausted — return the last response if we have one so the caller sees the actual API error
            if last_response is not None:
                return last_response, last_token
            raise

        params = dict(extra_params or {})
        params["token"] = tok

        try:
            resp = httpx.post(url, json=json_body, params=params, timeout=timeout)
        except httpx.HTTPError as e:
            logger.warning(
                f"Apify POST network error on slot {idx + 1} (...{_fingerprint(tok)}): {e}"
            )
            # network error isn't a credit issue — don't mark exhausted, just retry next slot
            mark_exhausted(idx, f"network error: {e}")
            continue

        if is_credit_error(resp.status_code, resp.text):
            mark_exhausted(
                idx, f"HTTP {resp.status_code}: {resp.text[:200]}"
            )
            last_response = resp
            last_token = tok
            continue

        if resp.status_code < 400:
            logger.info(
                f"Apify POST ok via slot {idx + 1}/{len(tokens)} (...{_fingerprint(tok)})"
            )
        return resp, tok

    # ran out of attempts
    if last_response is not None:
        return last_response, last_token
    raise RuntimeError("Apify POST failed and no response captured")


def get(
    url: str,
    token: str,
    *,
    params: dict | None = None,
    timeout: int = 30,
) -> httpx.Response:
    """GET against Apify with the given token (no rotation — runs are account-scoped)."""
    p = dict(params or {})
    p["token"] = token
    return httpx.get(url, params=p, timeout=timeout)


def report_run_failure(token: str, status: str, status_message: str) -> None:
    """If a run ended FAILED/ABORTED with a credit-related message, mark its token exhausted."""
    if status not in ("FAILED", "ABORTED", "TIMED-OUT"):
        return
    if not looks_like_credit_failure_message(status_message):
        return
    idx = _token_index(token)
    if idx is None:
        return
    mark_exhausted(idx, f"run {status}: {status_message[:200]}")


def active_slot_summary() -> str:
    """Short string for logs: 'slot 2/7 (...wxyz)'."""
    try:
        idx, tok = get_active_token()
    except RuntimeError as e:
        return f"NO ACTIVE TOKEN: {e}"
    return f"slot {idx + 1}/{len(_load_tokens())} (...{_fingerprint(tok)})"
