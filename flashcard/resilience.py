"""Resilient session wrapper: retry 429/5xx with exponential backoff.

Seeded from the sibling `dejavu` project. Transient detection covers OpenAI
connection/timeout errors and HTTP 429/500/502/503/504 on the OpenAI generation
and embedding calls (both surface `status_code`).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

_TRANSIENT_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_BACKOFF_CAP: float = 60.0


def _is_transient(exc: BaseException) -> bool:
    """True for connection/timeout errors and HTTP 429/500/502/503/504.

    OpenAI connection/timeout exceptions are matched by type; any other error
    exposing a transient `status_code` is also retried.
    """
    try:
        import openai  # present in production; may be absent in isolated test envs

        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return True
    except ImportError:
        pass
    status = getattr(exc, "status_code", None)
    return status in _TRANSIENT_STATUSES if status is not None else False


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff (2^attempt seconds) capped at _BACKOFF_CAP."""
    return min(2.0 ** min(attempt, 63), _BACKOFF_CAP)


def _notify(on_note: Callable | None, msg: Any) -> None:
    if on_note is not None:
        on_note(msg)


def _clear_note(on_note: Callable | None, note_active: list) -> None:
    if note_active[0]:
        _notify(on_note, None)
        note_active[0] = False


def _gate(on_token: Callable, cursor: int) -> Callable:
    """Return an on_token wrapper that suppresses events for turns < cursor."""
    if cursor == 0:
        return on_token

    def _cb(event: Any) -> Any:
        idx = getattr(event, "index", None)
        if idx is not None and idx < cursor:
            return None
        return on_token(event)

    return _cb


async def _iter_turns(
    factory: Callable,
    cfg: Any,
    on_token: Callable,
    on_note: Callable | None,
    cursor: list,
    note_active: list,
):
    """Yield turns from factory, skipping already-yielded ones; manage note state."""
    i = 0
    async for turn in factory(cfg, on_token=on_token):
        _clear_note(on_note, note_active)
        if i >= cursor[0]:
            cursor[0] += 1
            yield turn
        i += 1
    _clear_note(on_note, note_active)


def _default_factory() -> Callable:
    from flashcard.runner import (
        run_session,
    )  # lazy — avoids circular import at module load

    return run_session


async def resilient_session(
    cfg: Any,
    *,
    on_token: Callable,
    session_factory: Callable | None = None,
    on_note: Callable | None = None,
    sleep: Callable = asyncio.sleep,
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    """Wrap session_factory with retry + exponential backoff on transient errors.

    On a transient failure, emits on_note("retrying…"), sleeps, restarts the
    session, and skips turns already surfaced to the caller (tracked by cursor).
    Clears the note with on_note(None) on the first turn of a successful recovery.
    Non-transient errors and exhausted retries propagate unchanged.

    Trade-off: restart-with-cursor re-issues earlier turns' API calls on recovery.
    For CAG those calls are cheap (cache-hit only); for RAG the retrieved context
    is small, so a restart stays inexpensive. Per-turn retry would require modifying
    the runner loop, which is out of scope here.
    """
    factory = session_factory if session_factory is not None else _default_factory()
    cursor, note_active = [0], [False]
    for attempt in range(max_retries + 1):
        try:
            gated = _gate(on_token, cursor[0])
            async for turn in _iter_turns(
                factory, cfg, gated, on_note, cursor, note_active
            ):
                yield turn
            return
        except BaseException as exc:
            if not _is_transient(exc) or attempt >= max_retries:
                raise
            _notify(on_note, "retrying…")
            note_active[0] = True
            await sleep(base_delay * _backoff_delay(attempt))
