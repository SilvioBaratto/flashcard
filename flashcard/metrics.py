"""Per-turn metrics parsed from a BAML Collector (OpenAI usage).

BAML's typed usage gives input/output tokens, cached input tokens (cache reads),
and call timing. OpenAI reports ``prompt_tokens`` INCLUSIVE of cached tokens, so
fresh tokens are ``input_tokens − cached_input_tokens``.  The ``fresh_tokens``
helper here is the single source of truth — the runner's local copy is replaced
by it.  OpenAI has no separate cache-write count, so none is tracked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flashcard.tokens import count_tokens  # noqa: F401 (re-exported)



@dataclass(frozen=True)
class TurnMetrics:
    """Immutable snapshot of one turn's token and latency usage."""

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    duration_ms: float


@dataclass(frozen=True)
class RetrievalMetrics:
    """Per-turn RAG retrieval metadata (display-only; does not feed pricing).

    ``context_token_count`` approximates the billed context but the Collector's
    ``input_tokens`` is authoritative for pricing (it includes the system prompt
    and prompt-template scaffolding that ``context_token_count`` does not see).
    """

    chunk_ids: list[int]
    similarity_scores: list[float]
    context_token_count: int
    query_embed_tokens: int


def fresh_tokens(metrics: Any) -> int:
    """Return ``max(input_tokens − cached_input_tokens, 0)``.

    OpenAI reports ``prompt_tokens`` inclusive of cached tokens, so this is the
    count billed at the base (uncached) input rate.  The ``max(…, 0)`` clamp
    guards against the edge case where cached > input (which OpenAI should not
    produce, but the API spec does not formally forbid).
    """
    return max((metrics.input_tokens or 0) - (metrics.cached_input_tokens or 0), 0)


def retrieval_metrics_from_retriever(retriever: Any) -> RetrievalMetrics:
    """Build a ``RetrievalMetrics`` from an ``EmbeddingRetriever`` after ``retrieve()``.

    Reads:
    * ``retriever.last_scores``        — list of ``(chunk_id, score)`` tuples
    * ``retriever.last_query_tokens``  — int (embed token count for the query)
    * ``retriever.last_context_tokens``— int (tiktoken count of retrieved texts)
    """
    chunk_ids = [cid for cid, _ in retriever.last_scores]
    scores = [score for _, score in retriever.last_scores]
    context_tokens = getattr(retriever, "last_context_tokens", 0)
    return RetrievalMetrics(
        chunk_ids=chunk_ids,
        similarity_scores=scores,
        context_token_count=context_tokens,
        query_embed_tokens=retriever.last_query_tokens,
    )


def turn_metrics(collector: Any) -> TurnMetrics:
    """Parse a BAML Collector into a frozen TurnMetrics record."""
    usage = collector.usage
    log = collector.last
    return TurnMetrics(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        duration_ms=_duration_ms(log),
    )


def _duration_ms(log: Any | None) -> float:
    if log is None:
        return 0.0
    return log.timing.duration_ms
