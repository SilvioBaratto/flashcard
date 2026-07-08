"""OpenAI price table (+ embedding rate) and per-call cost calculators.

OpenAI list prices, USD per 1M tokens (fetched 2026-07-08):
  gpt-5.2                — input $1.75, cached input $0.175, output $14.00  (default)
  gpt-5-mini             — input $0.25, cached input $0.025, output $2.00   (judge)
  text-embedding-3-small — $0.02 (RAG embeddings)

Cached input is a 90% discount on input. OpenAI caches automatically — no explicit
marker and no separate cache-write price — so there is no write term in the CAG
cost. Verify these figures against current list pricing at build time.
Sources: https://developers.openai.com/api/docs/models/gpt-5.2
         https://developers.openai.com/api/docs/models/gpt-5-mini
         https://developers.openai.com/api/docs/models/text-embedding-3-small
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

MTOK: int = 1_000_000

# OpenAI text-embedding-3-small — USD per MTok. Verify at build time.
EMBED_USD_PER_MTOK: float = 0.02


class Model(str, Enum):
    """Model identifiers; values double as `--model` CLI keys."""

    GPT_52 = "gpt-5.2"  # default generation model
    GPT_5_MINI = "gpt-5-mini"  # optional CompareAnswers judge


@dataclass(frozen=True)
class Price:
    """USD per MTok pricing row for one model."""

    base: float  # input
    cached: float  # cached input (cache read)
    output: float  # output


PRICES: dict[Model, Price] = {
    Model.GPT_52: Price(base=1.75, cached=0.175, output=14.00),
    Model.GPT_5_MINI: Price(base=0.25, cached=0.025, output=2.00),
}


def _per_mtok(tokens: int, usd_per_mtok: float) -> float:
    """Convert a raw token count to USD using per-MTok rate."""
    return tokens / MTOK * usd_per_mtok


def get_price(model: Model) -> Price:
    """Return the Price row for *model*."""
    return PRICES[model]


def embed_cost(tokens: int, usd_per_mtok: float = EMBED_USD_PER_MTOK) -> float:
    """OpenAI embedding cost (per-MTok) for *tokens* embedded (query or corpus)."""
    return _per_mtok(tokens, usd_per_mtok)


def price_rag_turn(
    *,
    model: Model,
    input_tokens: int,
    output_tokens: int,
    query_embed_tokens: int = 0,
    embed_usd_per_mtok: float = EMBED_USD_PER_MTOK,
) -> float:
    """Cost of one RAG turn (per-MTok).

    query embed + (top-k chunks + question)×base + output×output_rate. Context is
    small and fresh every turn, so no cached tokens are involved.
    """
    p = get_price(model)
    return (
        embed_cost(query_embed_tokens, embed_usd_per_mtok)
        + _per_mtok(input_tokens, p.base)
        + _per_mtok(output_tokens, p.output)
    )


def price_cag_turn(
    *,
    model: Model,
    cache_read_tokens: int,
    fresh_question_tokens: int,
    output_tokens: int,
) -> float:
    """Cost of one CAG turn (per-MTok).

    cached-read + fresh + output. On Q1 nothing is cached yet, so `fresh` is the
    whole prompt at base rate (the expensive first call); on Q2+ the document is a
    cached read and only the question is fresh. OpenAI has no cache-write premium,
    so there is no write term.
    """
    p = get_price(model)
    return (
        _per_mtok(cache_read_tokens, p.cached)
        + _per_mtok(fresh_question_tokens, p.base)
        + _per_mtok(output_tokens, p.output)
    )
