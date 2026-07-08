"""TUI cumulative state layer — Rich-free aggregator + frozen view models.

Isolates cost/cache/miss-recall math from the Rich renderer so all arithmetic
is unit-testable without a terminal. No ``rich`` import here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flashcard.runner import Side


# ---------------------------------------------------------------------------
# Pure formatting helpers
# ---------------------------------------------------------------------------


def format_cost(usd: float) -> str:
    """Return a USD cost as '$0.0123'."""
    return f"${usd:.4f}"


def format_multiple(x: float) -> str:
    """Return a cost multiple as '4.2x'."""
    return f"{x:.1f}x"


# ---------------------------------------------------------------------------
# Frozen view-model dataclasses (immutable snapshots)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RagHeaderVM:
    """Per-turn snapshot for the RAG panel header."""

    turn_number: int
    chunk_ids: tuple[int, ...]
    context_tokens: int
    cumulative_cost: float


@dataclass(frozen=True)
class CagHeaderVM:
    """Per-turn snapshot for the CAG panel header."""

    turn_number: int
    cache_read_tokens: int
    context_size: int  # = cag.input_tokens (context_token_count is always 0 for CAG)
    cumulative_cost: float


@dataclass(frozen=True)
class FooterVM:
    """Cumulative snapshot for the footer/metrics strip."""

    rag_cost: float
    cag_cost: float
    cheaper_multiple: float | None  # None when undefined or divide-by-zero
    cheaper_side: Side | None       # which side is cheaper; None when equal/undefined
    rag_miss_count: int
    cag_full_recall_count: int
    rag_cache_read_total: int
    cag_cache_read_total: int


# ---------------------------------------------------------------------------
# Module-level helpers (pure, testable, single-responsibility)
# ---------------------------------------------------------------------------


def verdict_to_markers(verdict: Any) -> tuple[bool, bool]:
    """Translate a Verdict's real fields into (rag_missed, cag_full_recall).

    Truth table (per issue #20 comment):
      more_complete=="cag" and not agree → (True, True)
      more_complete=="cag" and agree → (False, True)
      anything else / None → (False, False)
    """
    if verdict is None:
        return False, False
    cag_wins = getattr(verdict, "more_complete", "") == "cag"
    agree = bool(getattr(verdict, "agree", True))
    return cag_wins and not agree, cag_wins


def _read_verdict(verdict: Any) -> tuple[int, int]:
    """Return (rag_missed, cag_full_recall) as ints for the TuiState accumulator."""
    missed, recall = verdict_to_markers(verdict)
    return int(missed), int(recall)


def _rag_vm(turn: Any, cumulative_cost: float) -> RagHeaderVM:
    return RagHeaderVM(
        turn_number=turn.turn_number,
        chunk_ids=tuple(turn.rag.retrieved),
        context_tokens=turn.rag.context_token_count,
        cumulative_cost=cumulative_cost,
    )


def _cag_vm(turn: Any, cumulative_cost: float) -> CagHeaderVM:
    return CagHeaderVM(
        turn_number=turn.turn_number,
        cache_read_tokens=int(turn.cag.cached_input_tokens or 0),
        context_size=int(turn.cag.input_tokens or 0),
        cumulative_cost=cumulative_cost,
    )


def _build_headers(
    turn: Any, rag_cost: float, cag_cost: float
) -> tuple[RagHeaderVM, CagHeaderVM]:
    return _rag_vm(turn, rag_cost), _cag_vm(turn, cag_cost)


def _compute_multiple(
    rag_cost: float, cag_cost: float, turns: int
) -> tuple[float | None, Side | None]:
    """Return (multiple, cheaper_side); multiple is None when undefined/DZ-unsafe."""
    if turns == 0 or rag_cost == cag_cost:
        return None, None
    if rag_cost > cag_cost:
        return (None, Side.CAG) if cag_cost == 0.0 else (rag_cost / cag_cost, Side.CAG)
    return (None, Side.RAG) if rag_cost == 0.0 else (cag_cost / rag_cost, Side.RAG)


# ---------------------------------------------------------------------------
# TuiState — main accumulator
# ---------------------------------------------------------------------------


class TuiState:
    """Folds TurnResults into per-side running totals and emits frozen view models."""

    def __init__(self) -> None:
        self._turns = 0
        self._rag_cost = 0.0
        self._cag_cost = 0.0
        self._rag_cache = 0
        self._cag_cache = 0
        self._rag_miss = 0
        self._cag_recall = 0
        self._rag_hdr: RagHeaderVM | None = None
        self._cag_hdr: CagHeaderVM | None = None

    def apply(self, turn: Any) -> None:
        """Fold one TurnResult into running totals and refresh snapshots."""
        self._turns += 1
        self._rag_cost += turn.rag.cost_usd
        self._cag_cost += turn.cag.cost_usd
        self._rag_cache += int(turn.rag.cached_input_tokens or 0)
        self._cag_cache += int(turn.cag.cached_input_tokens or 0)
        miss, recall = _read_verdict(turn.verdict)
        self._rag_miss += miss
        self._cag_recall += recall
        self._rag_hdr, self._cag_hdr = _build_headers(
            turn, self._rag_cost, self._cag_cost
        )

    @property
    def rag_header(self) -> RagHeaderVM:
        assert self._rag_hdr is not None, "Call apply() before accessing snapshots"
        return self._rag_hdr

    @property
    def cag_header(self) -> CagHeaderVM:
        assert self._cag_hdr is not None, "Call apply() before accessing snapshots"
        return self._cag_hdr

    @property
    def footer(self) -> FooterVM:
        multiple, side = _compute_multiple(self._rag_cost, self._cag_cost, self._turns)
        return FooterVM(
            rag_cost=self._rag_cost,
            cag_cost=self._cag_cost,
            cheaper_multiple=multiple,
            cheaper_side=side,
            rag_miss_count=self._rag_miss,
            cag_full_recall_count=self._cag_recall,
            rag_cache_read_total=self._rag_cache,
            cag_cache_read_total=self._cag_cache,
        )
