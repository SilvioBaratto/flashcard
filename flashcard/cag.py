"""CAG pipeline — whole-document stable prefix → AnswerCAG → TurnMetrics.

Formats the full document as a stable prefix before the question so OpenAI's
automatic prompt caching keys on the (system + document) prefix across
questions.  No cache_control marker (OpenAI caches automatically).
No retrieval, no embedding — the model sees the whole document every turn.

Q1 pays the full prompt at base rate; Q2+ read the cached prefix at the
discounted rate.  Pricing is the runner's responsibility: this module exposes
raw TurnMetrics and leaves cost calculation to flashcard.pricing.price_cag_turn.

Callback contract: on_token receives each raw partial str so the runner can
wrap it into the appropriate event type (avoids a circular import with runner.py
if runner imports this module).
"""
from __future__ import annotations

import inspect
from typing import Any, Callable

import baml_py

from flashcard.metrics import TurnMetrics, turn_metrics

# Lazy default — resolved at call time to avoid import-time side-effects.
_SENTINEL = object()


async def answer_cag(
    document: str,
    question: str,
    *,
    on_token: Callable[[str], Any] | None = None,
    b: Any = _SENTINEL,
    collector_factory: Callable[[], Any] = baml_py.Collector,
) -> tuple[str, TurnMetrics]:
    """Stream AnswerCAG and return the final text plus a TurnMetrics snapshot.

    Args:
        document: Full knowledge source passed as a stable prefix.
        question: The question to answer.
        on_token: Optional; called with each raw partial str during streaming.
        b: Injectable BAML async client.  Defaults to baml_client.async_client.b.
        collector_factory: Factory producing a Collector (injectable for tests).

    Returns:
        ``(final_text, metrics)`` — pricing is the caller's responsibility.
    """
    client = _resolve_client(b)
    collector = collector_factory()
    raw = client.stream.AnswerCAG(
        document=document,
        question=question,
        baml_options={"collector": collector},
    )
    stream = await raw if inspect.iscoroutine(raw) else raw
    async for partial in stream:
        await _emit(on_token, partial)
    raw_text = stream.get_final_response()
    text = await raw_text if inspect.iscoroutine(raw_text) else (raw_text or "")
    return str(text), turn_metrics(collector)


def _resolve_client(b: Any) -> Any:
    if b is _SENTINEL:
        from baml_client.async_client import b as _b  # noqa: PLC0415

        return _b
    return b


async def _emit(callback: Callable[[str], Any] | None, partial: str) -> None:
    if callback is None:
        return
    result = callback(partial)
    if inspect.isawaitable(result):
        await result


# Public alias: runner.py patches ``cag.answer``; cag.py avoids importing runner
# to prevent a circular dependency (runner → cag → runner).
answer = answer_cag
