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
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx
from concurrent.futures import ThreadPoolExecutor
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
    # Free-tier account with $0 remaining: Apify caps the run's max charge at $0
    # and rejects the START with HTTP 400 (not 402/403), e.g.
    # 'max-total-charge-usd-must-be-greater-than-zero'. Same meaning: rotate token.
    "max-total-charge",
    "maximum cost per run",
)

# HTTP-400 error-type substrings that mean "free-tier credit spent" rather than
# "bad input". When an account hits $0, Apify rejects the run START with a 400
# whose type depends on the actor's pricing model (see is_credit_error). We never
# send a zero cap ourselves, so matching these can't be confused with real input
# errors. Kept separate from _CREDIT_ERROR_SUBSTRINGS because those gate 402/403.
_CREDIT_ERROR_400_TYPES = (
    "max-total-charge",                    # max-charge actors
    "max-items-must-be-greater-than-zero", # pay-per-result actors
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


def _mask_token(token: str) -> str:
    """Last 4 chars of the token, safe to log."""
    if not token:
        return "????"
    return token[-4:] if len(token) >= 4 else "????"


def _today() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).date().isoformat()


def _load_state() -> dict:
    """Load state. Exhausted slots reset daily — token errors re-detected within the same run."""
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if state.get("date") != _today():
                n = len(state.get("exhausted", []))
                if n:
                    logger.info(f"New day — clearing {n} exhausted Apify slots")
                return {"date": _today(), "exhausted": []}
            return state
        except Exception:
            logger.warning(f"Could not parse {STATE_FILE}, treating as empty")
    return {"date": _today(), "exhausted": []}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state.setdefault("date", _today())
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _exhausted_set() -> set[int]:
    return set(_load_state().get("exhausted", []))


def get_active_token(skip: set[int] | None = None) -> tuple[int, str]:
    """First token slot that is neither persistently exhausted nor in `skip`.

    `skip` is a per-call set of slots to step over *without* persisting them as
    exhausted — used for transient network errors, which shouldn't burn a token
    for the rest of the day (only real credit/auth failures do that).
    Raises if none available.
    """
    tokens = _load_tokens()
    if not tokens:
        raise RuntimeError(
            "No APIFY_API_TOKEN* env variables found. Add at least APIFY_API_TOKEN to .env."
        )
    unavailable = _exhausted_set() | (skip or set())
    for idx, tok in enumerate(tokens):
        if idx not in unavailable:
            return idx, tok
    raise RuntimeError(
        f"All {len(tokens)} Apify token slots are unavailable "
        f"(exhausted or skipped this call). "
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
    fp = _mask_token(tokens[index]) if index < len(tokens) else "????"
    logger.warning(
        f"Apify token slot {index + 1} (...{fp}) marked exhausted. Reason: {reason[:200]}"
    )


def is_credit_error(status_code: int, body: str) -> bool:
    """True if the response looks like an out-of-credit / quota failure.

    Covers two free-tier HTTP 400 exhaustion shapes — once an account's credit is
    spent, Apify caps the run's affordable charge at $0 and rejects the START with
    400 (not 402/403), so the runner must rotate to the next token instead of
    failing the whole source. The exact error type depends on the actor's pricing:
      - max-charge actors: 'max-total-charge-usd-must-be-greater-than-zero'
      - pay-per-result actors: 'max-items-must-be-greater-than-zero'
        ("Maximum charged results must be greater than zero")
    Other 400s (genuine bad input) are NOT credit errors — we match these specific
    types only, and we never send a zero cap ourselves, so this can't misfire.
    """
    body_lower = (body or "").lower()
    if status_code == 400 and any(s in body_lower for s in _CREDIT_ERROR_400_TYPES):
        return True
    if status_code not in (402, 403):
        return False
    return any(s in body_lower for s in _CREDIT_ERROR_SUBSTRINGS)


def is_invalid_token_error(status_code: int, body: str) -> bool:
    """True if the response indicates the token itself is invalid/revoked.

    Apify returns 401 with type 'user-or-token-not-found' when a token has
    been deleted or mistyped — distinct from a credit error, but rotation
    should still help (the other slots may be valid).
    """
    if status_code != 401:
        return False
    body_lower = (body or "").lower()
    return (
        "user-or-token-not-found" in body_lower
        or "authentication token is not valid" in body_lower
        or "token-not-found" in body_lower
    )


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
    # Slots stepped over for this call only (transient network errors). NOT
    # persisted as exhausted — a DNS/timeout blip must not burn a token for the day.
    tried: set[int] = set()

    # try up to N times where N == number of token slots
    for _ in range(len(tokens) + 1):
        try:
            idx, tok = get_active_token(skip=tried)
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
                f"Apify POST network error on slot {idx + 1} (...{_mask_token(tok)}): {e}"
            )
            # network error isn't a credit issue — skip this slot for THIS call only
            # (no mark_exhausted), so a transient blip doesn't disable the token all day.
            tried.add(idx)
            continue

        if is_credit_error(resp.status_code, resp.text) or is_invalid_token_error(
            resp.status_code, resp.text
        ):
            mark_exhausted(
                idx, f"HTTP {resp.status_code}: {resp.text[:200]}"
            )
            tried.add(idx)  # also skip locally so the loop advances regardless
            last_response = resp
            last_token = tok
            continue

        if resp.status_code < 400:
            logger.info(
                f"Apify POST ok via slot {idx + 1}/{len(tokens)} (...{_mask_token(tok)})"
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
    return f"slot {idx + 1}/{len(_load_tokens())} (...{_mask_token(tok)})"


_TOKEN_PARAM_RE = re.compile(r"(token=)[^&\s'\"]+")


def _redact(text: str) -> str:
    """Strip Apify tokens from any string (URLs, response bodies, etc.)."""
    return _TOKEN_PARAM_RE.sub(r"\1<redacted>", text or "")


def raise_for_status_verbose(resp: httpx.Response, label: str) -> None:
    """Like resp.raise_for_status() but the exception message carries the
    response status, reason, and body — so callers (and Telegram error
    summaries) can see *why* a request failed, not just the URL.

    The Apify token is redacted from the body/URL before inclusion.
    """
    if resp.status_code < 400:
        return
    body = _redact((resp.text or "").strip())
    max_body = 1500
    if len(body) > max_body:
        body = body[:max_body] + f"... (truncated, {len(body)} chars total)"
    raise RuntimeError(
        f"{label}: HTTP {resp.status_code} {resp.reason_phrase or ''} | "
        f"body: {body or '<empty>'}"
    )


# ---------------------------------------------------------------------------
# Unified actor runner — single place for token rotation, retry-on-credit,
# cap-check, cost capture, error capture and timing. Every scraper calls this
# instead of copy-pasting the start/poll/fetch lifecycle.
# ---------------------------------------------------------------------------

@dataclass
class SourceResult:
    """Outcome of one Apify actor job, used uniformly by the pipeline.

    Never carries an exception — operational failures land in `error` so the
    pipeline keeps going and the failure is logged/persisted in one place.
    """
    source: str
    items: list[dict] = field(default_factory=list)   # normalised job dicts
    fetched: int = 0                                   # raw items before normalise
    cost_usd: float = 0.0                              # Apify usageTotalUsd
    capped: bool = False                               # fetched hit the requested cap
    error: str | None = None
    attempts: int = 0
    ms: int = 0


def run_actor_job(
    actor_id: str,
    actor_input: dict,
    *,
    source: str,
    normalise: Callable[[dict], dict],
    cap: int | None = None,
    timeout_seconds: int = 600,
    poll_interval: int = 10,
) -> SourceResult:
    """Run one Apify actor end-to-end. Returns SourceResult (never raises for
    operational failures).

    Handles, in one place:
      - token rotation on start-credit errors (via post())
      - retry on the NEXT token when a *started* run dies with a credit-related
        FAILED/ABORTED (the run can't be resumed cross-account, so we restart
        the whole cycle for this one source)
      - cap-check: capped = fetched >= cap (signals possible truncation)
      - cost capture from the run's usageTotalUsd
      - error + timing capture
    """
    started = time.perf_counter()
    res = SourceResult(source=source)
    runs_url = f"https://api.apify.com/v2/acts/{actor_id}/runs"
    max_attempts = max(1, len(_load_tokens()))

    for attempt in range(1, max_attempts + 1):
        res.attempts = attempt

        # ---- start (post() rotates internally on start-credit errors) ----
        try:
            resp, token = post(runs_url, json_body=actor_input, timeout=30)
        except Exception as e:  # network/all-exhausted
            res.error = f"start failed: {e}"
            break
        if resp.status_code >= 400:
            res.error = f"start HTTP {resp.status_code}: {_redact((resp.text or '')[:400])}"
            break
        try:
            run_id = resp.json()["data"]["id"]
        except Exception as e:
            res.error = f"bad start response: {e}"
            break

        # ---- poll ----
        run_url = f"https://api.apify.com/v2/actor-runs/{run_id}"
        deadline = time.time() + timeout_seconds
        status: str | None = None
        run_data: dict = {}
        poll_error: str | None = None
        while time.time() < deadline:
            try:
                pr = get(run_url, token, timeout=15)
                if pr.status_code >= 400:
                    poll_error = f"poll HTTP {pr.status_code}: {_redact((pr.text or '')[:300])}"
                    break
                run_data = pr.json()["data"]
            except Exception as e:
                # Transient network/parse blip — the run is still alive on Apify.
                # Retry until the deadline instead of dropping the whole source on
                # one flaky round-trip (the self-hosted host flaps getaddrinfo).
                logger.warning(f"[{source}] transient poll error (retrying): {e}")
                time.sleep(poll_interval)
                continue
            status = run_data.get("status")
            if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                break
            time.sleep(poll_interval)

        if poll_error:
            res.error = poll_error
            break

        # ---- success ----
        if status == "SUCCEEDED":
            res.cost_usd = float(run_data.get("usageTotalUsd") or 0.0)
            dataset_id = run_data.get("defaultDatasetId")
            try:
                ds = get(
                    f"https://api.apify.com/v2/datasets/{dataset_id}/items",
                    token, params={"format": "json", "clean": "true"}, timeout=60,
                )
            except Exception as e:
                res.error = f"dataset fetch failed: {e}"
                break
            if ds.status_code >= 400:
                res.error = f"dataset HTTP {ds.status_code}: {_redact((ds.text or '')[:300])}"
                break
            raw = ds.json()
            res.fetched = len(raw)
            res.capped = cap is not None and len(raw) >= cap
            if res.capped:
                logger.warning(f"[{source}] hit cap {cap} (fetched {len(raw)}) — possible truncation")
            out: list[dict] = []
            for item in raw:
                try:
                    out.append(normalise(item))
                except Exception as e:
                    logger.warning(f"[{source}] normalise failed for one item: {e}")
            res.items = out
            res.error = None
            logger.info(
                f"[{source}] ok: {len(out)} jobs, ${res.cost_usd:.4f}, "
                f"capped={res.capped}, attempt {attempt}/{max_attempts}"
            )
            break

        # ---- terminal non-success (FAILED/ABORTED/TIMED-OUT) ----
        status_msg = run_data.get("statusMessage") or ""
        report_run_failure(token, status or "", status_msg)
        if looks_like_credit_failure_message(status_msg) and attempt < max_attempts:
            logger.warning(
                f"[{source}] credit-fail mid-run on slot {_mask_token(token)} "
                f"(attempt {attempt}) — retrying on next token"
            )
            res.error = f"run {status} (credit): {status_msg[:200]}"
            continue  # restart whole cycle on next token
        res.error = f"run {status}: {status_msg[:300]}" if status else "no terminal status (timeout)"
        break

    res.ms = int((time.perf_counter() - started) * 1000)
    if res.error:
        logger.error(f"[{source}] FAILED after {res.attempts} attempt(s): {res.error}")
    return res


def fan_out_keywords(
    actor_id: str,
    keywords: list[str],
    build_input: Callable[[str], dict],
    *,
    source: str,
    normalise: Callable[[dict], dict],
    cap: int | None,
    max_workers: int = 5,
) -> SourceResult:
    """Run one actor job per keyword in parallel and merge the results.

    Shared by per-keyword sources (Glassdoor, Indeed, RemoteBoards) that fan a
    single search out into one actor run per keyword. `build_input(keyword)`
    returns the actor input for that keyword; empties are dropped and an empty
    keyword list yields an empty SourceResult.
    """
    kws = [k for k in (keywords or []) if k]
    if not kws:
        return SourceResult(source=source)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(kws))) as ex:
        results = list(ex.map(
            lambda kw: run_actor_job(
                actor_id, build_input(kw),
                source=source, normalise=normalise, cap=cap,
            ),
            kws,
        ))
    return merge_results(source, results)


def merge_results(source: str, results: list[SourceResult]) -> SourceResult:
    """Aggregate several SourceResults (e.g. per-keyword runs) into one.

    Items are de-duplicated by id; cost/fetched summed; capped OR-ed; errors
    joined. Used by per-keyword sources (Indeed, Glassdoor) that fan out into
    one actor run per keyword.
    """
    merged = SourceResult(source=source)
    seen: set[str] = set()
    errors: list[str] = []
    for r in results:
        merged.fetched += r.fetched
        merged.cost_usd += r.cost_usd
        merged.capped = merged.capped or r.capped
        merged.attempts = max(merged.attempts, r.attempts)
        merged.ms = max(merged.ms, r.ms)
        if r.error:
            errors.append(r.error)
        for item in r.items:
            jid = item.get("id")
            if jid and jid in seen:
                continue
            if jid:
                seen.add(jid)
            merged.items.append(item)
    merged.error = "; ".join(errors) if errors else None
    return merged
