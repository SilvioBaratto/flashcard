"""Pure worst-case spend estimator for the flashcard cost-guard.

CAG worst-case: every turn uncached (cache cold / miss). OpenAI's prefix cache
is best-effort — a cold or evicted prefix means that CAG turn re-pays the full
document at base rate. The correct upper bound is therefore every CAG turn at
base rate with zero cache hits.

Two figures are computed:
  cag_total — all N turns at base rate (worst case; gates the confirm prompt)
  cag_q1    — CAG Q1 cost surfaced explicitly (same formula, first question)
"""

from __future__ import annotations

from dataclasses import dataclass

import flashcard.pricing as _p
from flashcard.tokens import count_tokens  # encoding: o200k_base (gpt-5.x/gpt-4o)


@dataclass(frozen=True)
class Estimate:
    """Worst-case spend estimate across both pipelines × all questions."""

    rag_total: float  # RAG total across all questions
    cag_total: float  # CAG worst-case total (all turns uncached)
    cag_q1: float  # CAG Q1 cost — whole doc at base rate, nothing cached
    combined: float  # rag_total + cag_total (worst-case combined)


def estimate_worst_case(
    doc: str,
    questions: list[str],
    max_tokens: int,
    model: str,
    *,
    top_k: int = 3,
    chunk_size: int = 800,
) -> Estimate:
    """Compute worst-case spend for both pipelines × all questions.

    RAG: up to chunk_size*top_k context tokens per turn (capped at doc size).
    CAG: all turns uncached — no prefix cache hits assumed.
    """
    m = _p.Model(model)
    doc_tok = count_tokens(doc)
    rag_context = min(chunk_size * top_k, doc_tok)

    rag_costs = [
        _p.price_rag_turn(
            model=m,
            input_tokens=rag_context,
            output_tokens=max_tokens,
            query_embed_tokens=count_tokens(q),
        )
        for q in questions
    ]
    cag_costs = [
        _p.price_cag_turn(
            model=m,
            cache_read_tokens=0,
            fresh_question_tokens=doc_tok + count_tokens(q),
            output_tokens=max_tokens,
        )
        for q in questions
    ]

    rag_total = sum(rag_costs)
    cag_total = sum(cag_costs)
    return Estimate(
        rag_total=rag_total,
        cag_total=cag_total,
        cag_q1=cag_costs[0],
        combined=rag_total + cag_total,
    )
