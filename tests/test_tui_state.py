"""Source-blind example tests for flashcard/tui_state.py (issue #15).

Tests are authored from the acceptance criteria only; no implementation source
was read. The acceptance criteria specify:

    TuiState              — aggregator: apply(turn) + immutable snapshot properties
    RagHeaderVM           — per-turn view model: cumulative cost, turn_number,
                            chunk_ids, context_tokens
    CagHeaderVM           — per-turn view model: cumulative cost, turn_number,
                            cache_read_tokens, context_size (= cag.input_tokens)
    FooterVM              — cumulative cost per side, cheaper_multiple (DZ-safe),
                            rag_miss_count, cag_full_recall_count,
                            rag_cache_read_total, cag_cache_read_total
    format_cost(usd)      — pure formatter, returns "$0.0123"-style string
    format_multiple(x)    — pure formatter, returns "4.2x"-style string

Data contracts are derived from the issue body's "Data contracts to consume" section
and the requirements.md TUI spec — no implementation files were opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st

# ---------------------------------------------------------------------------
# Stub data contracts — built from criterion text; no production code read
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TurnMetrics:
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None
    duration_ms: float


@dataclass(frozen=True)
class _SideResult:
    text: str
    metrics: _TurnMetrics
    cost_usd: float
    retrieved: tuple[int, ...] = ()
    similarity_scores: tuple[float, ...] = ()
    context_token_count: int = 0

    @property
    def cost(self) -> float:
        return self.cost_usd

    @property
    def input_tokens(self) -> int | None:
        return self.metrics.input_tokens

    @property
    def output_tokens(self) -> int | None:
        return self.metrics.output_tokens

    @property
    def cached_input_tokens(self) -> int | None:
        return self.metrics.cached_input_tokens

    @property
    def duration_ms(self) -> float:
        return self.metrics.duration_ms


@dataclass(frozen=True)
class _TurnResult:
    index: int
    question: str
    rag: _SideResult
    cag: _SideResult
    verdict: Any = None

    @property
    def turn_number(self) -> int:
        return self.index + 1


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _metrics(
    input_tokens: int | None = 400,
    output_tokens: int | None = 80,
    cached_input_tokens: int | None = 0,
    duration_ms: float = 300.0,
) -> _TurnMetrics:
    return _TurnMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        duration_ms=duration_ms,
    )


def _rag(
    cost_usd: float = 0.010,
    retrieved: tuple[int, ...] = (0, 1, 2),
    context_token_count: int = 300,
    cached_input_tokens: int | None = 0,
    input_tokens: int | None = 350,
) -> _SideResult:
    return _SideResult(
        text="RAG answer",
        metrics=_metrics(
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
        ),
        cost_usd=cost_usd,
        retrieved=retrieved,
        context_token_count=context_token_count,
    )


def _cag(
    cost_usd: float = 0.005,
    cached_input_tokens: int | None = 800,
    input_tokens: int | None = 900,
) -> _SideResult:
    return _SideResult(
        text="CAG answer",
        metrics=_metrics(
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
        ),
        cost_usd=cost_usd,
        retrieved=(),
        context_token_count=0,
    )


def _turn(
    index: int = 0,
    question: str = "What is X?",
    rag_cost: float = 0.010,
    cag_cost: float = 0.005,
    verdict: Any = None,
    rag_retrieved: tuple[int, ...] = (0, 1, 2),
    rag_ctx_tokens: int = 300,
    cag_cached_tokens: int | None = 800,
    cag_input_tokens: int | None = 900,
    rag_cached_tokens: int | None = 0,
    rag_input_tokens: int | None = 350,
) -> _TurnResult:
    return _TurnResult(
        index=index,
        question=question,
        rag=_rag(
            cost_usd=rag_cost,
            retrieved=rag_retrieved,
            context_token_count=rag_ctx_tokens,
            cached_input_tokens=rag_cached_tokens,
            input_tokens=rag_input_tokens,
        ),
        cag=_cag(
            cost_usd=cag_cost,
            cached_input_tokens=cag_cached_tokens,
            input_tokens=cag_input_tokens,
        ),
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from flashcard.tui_state import (  # noqa: E402
    TuiState,
    format_cost,
    format_multiple,
)


# ---------------------------------------------------------------------------
# TuiState: apply() exists and returns immutable snapshots
# ---------------------------------------------------------------------------


def test_when_turn_applied_then_rag_header_snapshot_is_accessible():
    """TuiState.apply() causes rag_header to be accessible (not None)."""
    state = TuiState()
    state.apply(_turn())
    assert state.rag_header is not None


def test_when_turn_applied_then_cag_header_snapshot_is_accessible():
    state = TuiState()
    state.apply(_turn())
    assert state.cag_header is not None


def test_when_turn_applied_then_footer_snapshot_is_accessible():
    state = TuiState()
    state.apply(_turn())
    assert state.footer is not None


def test_when_rag_header_mutated_then_frozen_error_is_raised():
    """Snapshots are immutable; assigning any field must raise an error."""
    state = TuiState()
    state.apply(_turn())
    vm = state.rag_header
    with pytest.raises((AttributeError, TypeError)):
        vm.turn_number = 99  # type: ignore[misc]


def test_when_cag_header_mutated_then_frozen_error_is_raised():
    state = TuiState()
    state.apply(_turn())
    vm = state.cag_header
    with pytest.raises((AttributeError, TypeError)):
        vm.turn_number = 99  # type: ignore[misc]


def test_when_footer_mutated_then_frozen_error_is_raised():
    state = TuiState()
    state.apply(_turn())
    vm = state.footer
    with pytest.raises((AttributeError, TypeError)):
        vm.rag_cost = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RAG header per-turn view model fields
# ---------------------------------------------------------------------------


def test_when_first_turn_applied_then_rag_header_turn_number_is_one():
    """turn_number is 1-based (index=0 → turn_number=1)."""
    state = TuiState()
    state.apply(_turn(index=0))
    assert state.rag_header.turn_number == 1


def test_when_third_turn_applied_then_rag_header_turn_number_is_three():
    state = TuiState()
    for i in range(3):
        state.apply(_turn(index=i))
    assert state.rag_header.turn_number == 3


def test_when_turn_applied_then_rag_header_exposes_retrieved_chunk_ids():
    state = TuiState()
    state.apply(_turn(rag_retrieved=(3, 7, 11)))
    assert tuple(state.rag_header.chunk_ids) == (3, 7, 11)


def test_when_turn_applied_then_rag_header_exposes_context_token_count():
    state = TuiState()
    state.apply(_turn(rag_ctx_tokens=512))
    assert state.rag_header.context_tokens == 512


def test_when_turn_applied_then_rag_header_exposes_cumulative_cost():
    """RAG header shows running cumulative cost (spec: 'running $ cost')."""
    state = TuiState()
    state.apply(_turn(index=0, rag_cost=0.010))
    state.apply(_turn(index=1, rag_cost=0.015))
    assert abs(state.rag_header.cumulative_cost - 0.025) < 1e-9


# ---------------------------------------------------------------------------
# CAG header per-turn view model fields
# ---------------------------------------------------------------------------


def test_when_first_turn_applied_then_cag_header_turn_number_is_one():
    state = TuiState()
    state.apply(_turn(index=0))
    assert state.cag_header.turn_number == 1


def test_when_turn_applied_then_cag_header_exposes_cache_read_tokens():
    """CAG header: this-turn cache-read tokens = cached_input_tokens."""
    state = TuiState()
    state.apply(_turn(cag_cached_tokens=1024))
    assert state.cag_header.cache_read_tokens == 1024


def test_when_turn_applied_then_cag_header_context_size_equals_cag_input_tokens():
    """CAG context size is cag.input_tokens (cag.context_token_count is always 0)."""
    state = TuiState()
    state.apply(_turn(cag_input_tokens=2048))
    assert state.cag_header.context_size == 2048


def test_when_turn_applied_then_cag_header_exposes_cumulative_cost():
    state = TuiState()
    state.apply(_turn(index=0, cag_cost=0.005))
    state.apply(_turn(index=1, cag_cost=0.003))
    assert abs(state.cag_header.cumulative_cost - 0.008) < 1e-9


# ---------------------------------------------------------------------------
# Cumulative cost accumulation (footer)
# ---------------------------------------------------------------------------


def test_when_two_turns_applied_then_footer_rag_cost_is_sum():
    state = TuiState()
    state.apply(_turn(index=0, rag_cost=0.010))
    state.apply(_turn(index=1, rag_cost=0.015))
    assert abs(state.footer.rag_cost - 0.025) < 1e-9


def test_when_two_turns_applied_then_footer_cag_cost_is_sum():
    state = TuiState()
    state.apply(_turn(index=0, cag_cost=0.005))
    state.apply(_turn(index=1, cag_cost=0.003))
    assert abs(state.footer.cag_cost - 0.008) < 1e-9


def test_when_single_turn_applied_then_footer_costs_match_that_turn():
    state = TuiState()
    state.apply(_turn(rag_cost=0.012, cag_cost=0.004))
    assert abs(state.footer.rag_cost - 0.012) < 1e-9
    assert abs(state.footer.cag_cost - 0.004) < 1e-9


# ---------------------------------------------------------------------------
# Per-side cache-read totals (footer)
# ---------------------------------------------------------------------------


def test_when_two_turns_applied_then_cag_cache_read_total_accumulates():
    state = TuiState()
    state.apply(_turn(index=0, cag_cached_tokens=500))
    state.apply(_turn(index=1, cag_cached_tokens=700))
    assert state.footer.cag_cache_read_total == 1200


def test_when_two_turns_applied_then_rag_cache_read_total_accumulates():
    state = TuiState()
    state.apply(_turn(index=0, rag_cached_tokens=50))
    state.apply(_turn(index=1, rag_cached_tokens=30))
    assert state.footer.rag_cache_read_total == 80


def test_when_first_turn_has_no_cache_reads_then_totals_are_zero():
    state = TuiState()
    state.apply(_turn(index=0, rag_cached_tokens=0, cag_cached_tokens=0))
    assert state.footer.rag_cache_read_total == 0
    assert state.footer.cag_cache_read_total == 0


# ---------------------------------------------------------------------------
# Nx cheaper multiple (divide-by-zero safe)
# ---------------------------------------------------------------------------


def test_when_rag_costs_more_than_cag_then_multiple_is_rag_over_cag():
    """Multiple = mean of more-expensive side ÷ mean of cheaper side (≥ 1)."""
    state = TuiState()
    state.apply(_turn(index=0, rag_cost=0.020, cag_cost=0.005))
    state.apply(_turn(index=1, rag_cost=0.020, cag_cost=0.005))
    # mean RAG = 0.020, mean CAG = 0.005 → 0.020 / 0.005 = 4.0
    assert state.footer.cheaper_multiple is not None
    assert abs(state.footer.cheaper_multiple - 4.0) < 0.01


def test_when_cag_costs_more_than_rag_then_multiple_is_still_at_least_one():
    """Multiple is always ≥ 1 regardless of which side is cheaper."""
    state = TuiState()
    state.apply(_turn(index=0, rag_cost=0.002, cag_cost=0.020))
    m = state.footer.cheaper_multiple
    assert m is None or m >= 1.0


def test_when_both_sides_zero_cost_then_multiple_is_safe_sentinel():
    """Divide-by-zero: both sides zero → must not raise; returns sentinel."""
    state = TuiState()
    state.apply(_turn(rag_cost=0.0, cag_cost=0.0))
    # Must not raise; sentinel is None or inf (not a finite positive number)
    m = state.footer.cheaper_multiple
    assert m is None or (isinstance(m, float) and not (0.0 < m < float("inf")))


def test_when_cheaper_side_has_zero_mean_cost_then_no_exception_raised():
    """Even if only the cheaper side's mean is zero, no ZeroDivisionError."""
    state = TuiState()
    state.apply(_turn(rag_cost=0.010, cag_cost=0.0))
    _ = state.footer.cheaper_multiple  # must not raise


# ---------------------------------------------------------------------------
# Miss / recall counters
# ---------------------------------------------------------------------------


def test_when_verdict_is_none_then_miss_and_recall_counts_are_zero():
    """verdict=None must leave both counters at zero."""
    state = TuiState()
    state.apply(_turn(verdict=None))
    assert state.footer.rag_miss_count == 0
    assert state.footer.cag_full_recall_count == 0


def test_when_verdict_none_applied_twice_then_counters_remain_zero():
    state = TuiState()
    state.apply(_turn(index=0, verdict=None))
    state.apply(_turn(index=1, verdict=None))
    assert state.footer.rag_miss_count == 0
    assert state.footer.cag_full_recall_count == 0


def test_when_verdict_rag_missed_true_then_rag_miss_count_increments():
    """rag_missed=True on verdict increments rag_miss_count by 1."""

    class _V:
        rag_missed = True
        cag_full_recall = False

    state = TuiState()
    state.apply(_turn(verdict=_V()))
    assert state.footer.rag_miss_count == 1
    assert state.footer.cag_full_recall_count == 0


def test_when_verdict_cag_full_recall_true_then_cag_count_increments():
    """cag_full_recall=True on verdict increments cag_full_recall_count by 1."""

    class _V:
        rag_missed = False
        cag_full_recall = True

    state = TuiState()
    state.apply(_turn(verdict=_V()))
    assert state.footer.cag_full_recall_count == 1
    assert state.footer.rag_miss_count == 0


def test_when_both_verdict_flags_true_then_both_counters_increment():
    class _V:
        rag_missed = True
        cag_full_recall = True

    state = TuiState()
    state.apply(_turn(verdict=_V()))
    assert state.footer.rag_miss_count == 1
    assert state.footer.cag_full_recall_count == 1


def test_when_multiple_verdicts_miss_then_miss_count_accumulates():
    """Miss count accumulates across multiple turns with rag_missed=True."""

    class _V:
        rag_missed = True
        cag_full_recall = True

    state = TuiState()
    state.apply(_turn(index=0, verdict=_V()))
    state.apply(_turn(index=1, verdict=_V()))
    assert state.footer.rag_miss_count == 2
    assert state.footer.cag_full_recall_count == 2


def test_when_verdict_has_unknown_shape_then_no_exception_is_raised():
    """Verdict adapter is defensive: unexpected attribute shape must not raise."""

    class _WeirdVerdict:
        # intentionally no rag_missed / cag_full_recall
        divergent = True

    state = TuiState()
    state.apply(_turn(verdict=_WeirdVerdict()))
    # Counters default to 0 for unknown verdict shapes
    assert state.footer.rag_miss_count == 0
    assert state.footer.cag_full_recall_count == 0


def test_when_verdict_rag_missed_false_then_miss_count_does_not_increment():
    class _V:
        rag_missed = False
        cag_full_recall = False

    state = TuiState()
    state.apply(_turn(verdict=_V()))
    assert state.footer.rag_miss_count == 0


# ---------------------------------------------------------------------------
# None token fields — guard against None in metric properties
# ---------------------------------------------------------------------------


def test_when_cag_cached_input_tokens_is_none_then_cache_total_treats_as_zero():
    """cached_input_tokens=None must be coerced to 0, not raise TypeError."""
    cag_side = _SideResult(
        text="a",
        metrics=_TurnMetrics(
            input_tokens=None,
            output_tokens=50,
            cached_input_tokens=None,
            duration_ms=200.0,
        ),
        cost_usd=0.005,
        retrieved=(),
        context_token_count=0,
    )
    state = TuiState()
    state.apply(_TurnResult(0, "q", _rag(), cag_side))
    assert state.footer.cag_cache_read_total == 0


def test_when_cag_input_tokens_is_none_then_context_size_is_zero():
    """cag.input_tokens=None → context_size=0 (no crash)."""
    cag_side = _SideResult(
        text="a",
        metrics=_TurnMetrics(
            input_tokens=None,
            output_tokens=50,
            cached_input_tokens=0,
            duration_ms=200.0,
        ),
        cost_usd=0.005,
        retrieved=(),
        context_token_count=0,
    )
    state = TuiState()
    state.apply(_TurnResult(0, "q", _rag(), cag_side))
    assert state.cag_header.context_size == 0


def test_when_rag_cached_input_tokens_is_none_then_rag_cache_total_treats_as_zero():
    rag_side = _SideResult(
        text="a",
        metrics=_TurnMetrics(
            input_tokens=350,
            output_tokens=80,
            cached_input_tokens=None,
            duration_ms=300.0,
        ),
        cost_usd=0.010,
        retrieved=(0,),
        context_token_count=100,
    )
    state = TuiState()
    state.apply(_TurnResult(0, "q", rag_side, _cag()))
    assert state.footer.rag_cache_read_total == 0


# ---------------------------------------------------------------------------
# format_cost — pure formatting helper
# ---------------------------------------------------------------------------


def test_when_cost_formatted_then_result_starts_with_dollar_sign():
    assert format_cost(0.0123).startswith("$")


def test_when_cost_formatted_then_at_least_four_decimal_places_shown():
    """spec example '$0.0123' implies four decimal digits."""
    result = format_cost(0.0123)
    numeric = result.lstrip("$")
    assert "." in numeric
    assert len(numeric.split(".")[1]) >= 4


def test_when_cost_is_zero_then_format_cost_returns_dollar_string():
    assert format_cost(0.0).startswith("$")


def test_when_cost_is_large_then_format_cost_does_not_raise():
    result = format_cost(9999.99)
    assert result.startswith("$")


# ---------------------------------------------------------------------------
# format_multiple — pure formatting helper
# ---------------------------------------------------------------------------


def test_when_multiple_formatted_then_result_ends_with_x():
    """spec example '4.2x' implies trailing 'x'."""
    assert format_multiple(4.2).endswith("x")


def test_when_multiple_is_integer_then_format_multiple_includes_digit_and_x():
    result = format_multiple(3.0)
    assert "3" in result
    assert result.endswith("x")


def test_when_multiple_formatted_then_numeric_part_contains_decimal():
    """spec example '4.2x' implies at least one decimal digit."""
    result = format_multiple(4.2)
    numeric = result.rstrip("x")
    assert "." in numeric


def test_format_multiple_of_one_contains_one():
    assert "1" in format_multiple(1.0)


# ---------------------------------------------------------------------------
# Property-based tests (invariants from the criteria)
# ---------------------------------------------------------------------------


@given(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
def test_format_cost_never_raises_for_non_negative_floats(cost: float) -> None:
    """format_cost accepts any non-negative float without raising."""
    result = format_cost(cost)
    assert result.startswith("$")


@given(
    st.floats(min_value=1.0, max_value=1_000.0, allow_nan=False, allow_infinity=False)
)
def test_format_multiple_never_raises_for_positive_floats(m: float) -> None:
    """format_multiple accepts any positive float without raising."""
    result = format_multiple(m)
    assert result.endswith("x")


@given(
    costs=st.lists(
        st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    )
)
@settings(max_examples=50)
def test_cumulative_rag_cost_equals_sum_of_applied_costs(costs: list[float]) -> None:
    """Cumulative RAG cost = Σ cost_usd over all applied turns (monotone accumulation)."""
    state = TuiState()
    for i, c in enumerate(costs):
        state.apply(_turn(index=i, rag_cost=c, cag_cost=0.001))
    assert abs(state.footer.rag_cost - sum(costs)) < 1e-9


@given(
    costs=st.lists(
        st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    )
)
@settings(max_examples=50)
def test_cumulative_cag_cost_equals_sum_of_applied_costs(costs: list[float]) -> None:
    """Cumulative CAG cost = Σ cost_usd over all applied turns (monotone accumulation)."""
    state = TuiState()
    for i, c in enumerate(costs):
        state.apply(_turn(index=i, rag_cost=0.001, cag_cost=c))
    assert abs(state.footer.cag_cost - sum(costs)) < 1e-9


@given(
    counts=st.lists(
        st.integers(min_value=0, max_value=5_000),
        min_size=1,
        max_size=8,
    )
)
@settings(max_examples=50)
def test_cag_cache_read_total_equals_sum_of_cached_input_tokens(
    counts: list[int],
) -> None:
    """cag_cache_read_total = Σ cached_input_tokens over all turns."""
    state = TuiState()
    for i, cnt in enumerate(counts):
        state.apply(_turn(index=i, cag_cached_tokens=cnt))
    assert state.footer.cag_cache_read_total == sum(counts)


@given(
    n_misses=st.integers(min_value=0, max_value=10),
    n_clean=st.integers(min_value=0, max_value=10),
)
@settings(max_examples=50)
def test_rag_miss_count_equals_number_of_turns_with_rag_missed_true(
    n_misses: int, n_clean: int
) -> None:
    """rag_miss_count increments once per turn with rag_missed=True, never for None."""

    class _Miss:
        rag_missed = True
        cag_full_recall = False

    state = TuiState()
    for i in range(n_misses):
        state.apply(_turn(index=i, verdict=_Miss()))
    for j in range(n_clean):
        state.apply(_turn(index=n_misses + j, verdict=None))
    assert state.footer.rag_miss_count == n_misses
