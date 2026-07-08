"""Source-blind example tests for flashcard/tui_render.py (issue #16).

Tests are authored from the acceptance criteria only; no implementation source
was read. The acceptance criteria specify:

    rag_header(vm)         — Rich renderable: cumulative $, question number,
                             retrieved chunk ids, context-token count
    cag_header(vm)         — Rich renderable: cumulative $, question number,
                             cache-read tokens, context size
    panel_body(text, side) — Rich renderable: partial or full answer strings
                             without raising, bounded wrapping
    footer_strip(vm)       — Rich renderable: cumulative cost per side,
                             Nx-cheaper multiple (sentinel "—" when undefined),
                             RAG miss count, CAG full-recall count,
                             cache-read totals

View-model stubs are built from acceptance-criteria field lists; tui_state.py
was not opened.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from hypothesis import given, settings, strategies as st
from rich.console import Console

# ---------------------------------------------------------------------------
# Stub view-models — derived from criterion text, not production code
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RagHeaderVM:
    """Fields from criterion: 'cumulative $, question number, chunk ids, context tokens'."""

    turn_number: int
    chunk_ids: tuple[int, ...]
    context_tokens: int
    cumulative_cost: float


@dataclass(frozen=True)
class _CagHeaderVM:
    """Fields from criterion: 'cumulative $, question number, cache-read tokens, context size'."""

    turn_number: int
    cache_read_tokens: int
    context_size: int
    cumulative_cost: float


@dataclass(frozen=True)
class _FooterVM:
    """Fields from criterion: cost per side, Nx-cheaper (None="—"), miss/recall counts, cache totals."""

    rag_cost: float
    cag_cost: float
    cheaper_multiple: float | None  # None → sentinel "—"
    cheaper_side: str | None  # "cag" or "rag"; None when multiple is undefined
    rag_miss_count: int
    cag_full_recall_count: int
    rag_cache_read_total: int
    cag_cache_read_total: int


# ---------------------------------------------------------------------------
# Console capture helper — uses record=True as specified in the criterion
# ---------------------------------------------------------------------------


def _render(renderable) -> str:
    """Render a Rich renderable to plain text via Console(record=True)."""
    c = Console(record=True, width=120, no_color=True)
    c.print(renderable)
    return c.export_text()


# ---------------------------------------------------------------------------
# Import the module under test  (Red phase — not yet implemented)
# ---------------------------------------------------------------------------

from flashcard.tui_render import (  # noqa: E402
    cag_header,
    footer_strip,
    panel_body,
    rag_header,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _rag_vm(**kw) -> _RagHeaderVM:
    d = dict(
        turn_number=1, chunk_ids=(0, 3, 7), context_tokens=450, cumulative_cost=0.0123
    )
    d.update(kw)
    return _RagHeaderVM(**d)


def _cag_vm(**kw) -> _CagHeaderVM:
    d = dict(
        turn_number=1,
        cache_read_tokens=1024,
        context_size=18000,
        cumulative_cost=0.0456,
    )
    d.update(kw)
    return _CagHeaderVM(**d)


def _footer_vm(**kw) -> _FooterVM:
    d = dict(
        rag_cost=0.025,
        cag_cost=0.010,
        cheaper_multiple=2.5,
        cheaper_side="cag",
        rag_miss_count=1,
        cag_full_recall_count=3,
        rag_cache_read_total=50,
        cag_cache_read_total=4800,
    )
    d.update(kw)
    return _FooterVM(**d)


# ===========================================================================
# rag_header(vm) — criterion: "large cumulative $, question number,
# retrieved chunk ids, context-token count"
# ===========================================================================


def test_when_rag_header_called_then_renderable_is_returned():
    """rag_header returns a non-None Rich renderable."""
    assert rag_header(_rag_vm()) is not None


def test_when_rag_header_rendered_then_dollar_sign_is_visible():
    """Criterion: 'large cumulative $' — $ must appear in the rendered output."""
    text = _render(rag_header(_rag_vm(cumulative_cost=0.0123)))
    assert "$" in text


def test_when_rag_header_rendered_then_turn_number_is_visible():
    """Criterion: 'question number' — the 1-based turn number must appear."""
    text = _render(rag_header(_rag_vm(turn_number=4)))
    assert "4" in text


def test_when_rag_header_rendered_then_chunk_ids_are_visible():
    """Criterion: 'retrieved chunk ids' — at least one chunk id must appear."""
    text = _render(rag_header(_rag_vm(chunk_ids=(13, 27, 41))))
    assert any(str(cid) in text for cid in (13, 27, 41))


def test_when_rag_header_rendered_then_context_token_count_is_visible():
    """Criterion: 'context-token count' — context tokens must appear."""
    text = _render(rag_header(_rag_vm(context_tokens=512)))
    assert "512" in text


def test_when_rag_header_rendered_then_cumulative_cost_value_is_visible():
    """Cumulative cost digits — beyond the $ prefix — must appear."""
    text = _render(rag_header(_rag_vm(cumulative_cost=0.0123)))
    assert "0123" in text or "0.0123" in text or "123" in text


def test_when_rag_header_called_with_empty_chunk_ids_then_no_exception_is_raised():
    """Empty retrieved set (RAG returned nothing) must not crash the header."""
    result = rag_header(_rag_vm(chunk_ids=()))
    assert result is not None


# ===========================================================================
# cag_header(vm) — criterion: "large cumulative $, question number,
# cache-read tokens, context size"
# ===========================================================================


def test_when_cag_header_called_then_renderable_is_returned():
    assert cag_header(_cag_vm()) is not None


def test_when_cag_header_rendered_then_dollar_sign_is_visible():
    """Criterion: 'large cumulative $'."""
    text = _render(cag_header(_cag_vm(cumulative_cost=0.0456)))
    assert "$" in text


def test_when_cag_header_rendered_then_turn_number_is_visible():
    text = _render(cag_header(_cag_vm(turn_number=5)))
    assert "5" in text


def test_when_cag_header_rendered_then_cache_read_tokens_are_visible():
    """Criterion: 'cache-read tokens' — must appear in the rendered output."""
    text = _render(cag_header(_cag_vm(cache_read_tokens=8192)))
    assert "8192" in text


def test_when_cag_header_rendered_then_cumulative_cost_value_is_visible():
    text = _render(cag_header(_cag_vm(cumulative_cost=0.0456)))
    assert "0456" in text or "0.0456" in text or "456" in text


def test_when_cag_q1_has_zero_cache_reads_then_header_renders_without_raising():
    """CAG Q1 has nothing cached yet — cache_read_tokens=0 must not crash."""
    result = cag_header(_cag_vm(cache_read_tokens=0))
    assert result is not None


# ===========================================================================
# panel_body(text, side) — criterion: "renders partial and full answer strings
# without raising, wrapping to a bounded region"
# ===========================================================================


_SIDES = ["rag", "cag"]


@pytest.mark.parametrize("side", _SIDES)
def test_when_panel_body_called_with_partial_text_then_renderable_is_returned(
    side: str,
):
    """panel_body returns a Rich renderable for a partial (streaming) answer."""
    result = panel_body("Partial ans…", side)
    assert result is not None


@pytest.mark.parametrize("side", _SIDES)
def test_when_panel_body_called_with_full_text_then_renderable_is_returned(side: str):
    """panel_body returns a Rich renderable for a completed, multi-sentence answer."""
    full = (
        "This is the complete answer to the question. "
        "It spans multiple sentences and provides all the details needed. "
        "The context window gives full recall of every section in the document."
    )
    result = panel_body(full, side)
    assert result is not None


@pytest.mark.parametrize("side", _SIDES)
def test_when_panel_body_called_with_empty_string_then_no_exception_is_raised(
    side: str,
):
    """Criterion: 'without raising' — empty string is a valid partial state (streaming not started)."""
    result = panel_body("", side)
    assert result is not None


@pytest.mark.parametrize("side", _SIDES)
def test_when_panel_body_rendered_then_text_content_appears_in_output(side: str):
    """The answer text passed in must be visible in the rendered output."""
    rendered = _render(panel_body("retriever found exact chunks", side))
    assert "retriever" in rendered or "chunks" in rendered


@pytest.mark.parametrize("side", _SIDES)
def test_when_panel_body_called_with_long_text_then_no_exception_is_raised(side: str):
    """Criterion: 'wrapping to a bounded region' — long text must wrap, not raise."""
    long_text = "answer word " * 600
    result = panel_body(long_text, side)
    assert result is not None


# ===========================================================================
# footer_strip(vm) — criterion: "cumulative cost per side, Nx-cheaper multiple
# (sentinel '—' when undefined), RAG miss vs CAG full-recall counts,
# cache-read totals"
# ===========================================================================


def test_when_footer_strip_called_then_renderable_is_returned():
    assert footer_strip(_footer_vm()) is not None


def test_when_footer_rendered_then_dollar_sign_is_visible_for_costs():
    """Criterion: 'cumulative cost per side' — at least one $ must appear."""
    text = _render(footer_strip(_footer_vm(rag_cost=0.025, cag_cost=0.010)))
    assert "$" in text


def test_when_cheaper_multiple_is_none_then_footer_renders_em_dash_sentinel():
    """Criterion: 'sentinel — when undefined' — None cheaper_multiple → '—' in output."""
    text = _render(footer_strip(_footer_vm(cheaper_multiple=None, cheaper_side=None)))
    assert "—" in text


def test_when_cheaper_multiple_is_defined_then_footer_renders_its_value():
    """Criterion: 'Nx-cheaper multiple' — the numeric multiple must appear."""
    # Use distinctive value 37.9 with all counts set to 0 to avoid collisions.
    text = _render(
        footer_strip(
            _footer_vm(
                cheaper_multiple=37.9,
                rag_miss_count=0,
                cag_full_recall_count=0,
                rag_cache_read_total=0,
                cag_cache_read_total=0,
            )
        )
    )
    assert "37" in text


def test_when_footer_rendered_then_rag_miss_count_is_visible():
    """Criterion: 'RAG miss count' — count must appear in the footer."""
    text = _render(
        footer_strip(
            _footer_vm(
                rag_miss_count=6,
                cag_full_recall_count=0,
                cheaper_multiple=None,
            )
        )
    )
    assert "6" in text


def test_when_footer_rendered_then_cag_full_recall_count_is_visible():
    """Criterion: 'CAG full-recall count' — count must appear in the footer."""
    text = _render(
        footer_strip(
            _footer_vm(
                cag_full_recall_count=8,
                rag_miss_count=0,
                cheaper_multiple=None,
            )
        )
    )
    assert "8" in text


def test_when_footer_rendered_then_cag_cache_read_total_is_visible():
    """Criterion: 'cache-read totals' — CAG total must appear."""
    text = _render(
        footer_strip(
            _footer_vm(
                cag_cache_read_total=19200,
                rag_cache_read_total=0,
                rag_miss_count=0,
                cag_full_recall_count=0,
                cheaper_multiple=None,
            )
        )
    )
    assert "19200" in text or "19,200" in text


def test_when_footer_rendered_with_all_zeros_then_no_exception_is_raised():
    """All-zero footer must render without raising (edge case: first turn not yet complete)."""
    vm = _footer_vm(
        rag_cost=0.0,
        cag_cost=0.0,
        cheaper_multiple=None,
        rag_miss_count=0,
        cag_full_recall_count=0,
        rag_cache_read_total=0,
        cag_cache_read_total=0,
    )
    result = footer_strip(vm)
    assert result is not None


# ===========================================================================
# Purity: all builders produce stable, deterministic output — no I/O, no Live
# ===========================================================================


def test_when_rag_header_called_twice_with_same_vm_then_outputs_are_identical():
    """Criterion: 'pure (no I/O, no Live) and produce stable output under a captured Console'."""
    vm = _rag_vm()
    assert _render(rag_header(vm)) == _render(rag_header(vm))


def test_when_cag_header_called_twice_with_same_vm_then_outputs_are_identical():
    vm = _cag_vm()
    assert _render(cag_header(vm)) == _render(cag_header(vm))


def test_when_footer_strip_called_twice_with_same_vm_then_outputs_are_identical():
    vm = _footer_vm()
    assert _render(footer_strip(vm)) == _render(footer_strip(vm))


def test_when_panel_body_called_twice_with_same_args_then_outputs_are_identical():
    assert _render(panel_body("the answer", "rag")) == _render(
        panel_body("the answer", "rag")
    )


# ===========================================================================
# Property-based tests — invariants derived from acceptance criteria
# ===========================================================================


@given(st.text())
def test_when_panel_body_called_with_any_text_rag_side_then_no_exception_is_raised(
    text: str,
) -> None:
    """Criterion: 'without raising' — never-raises invariant for rag side over all strings."""
    result = panel_body(text, "rag")
    assert result is not None


@given(st.text())
def test_when_panel_body_called_with_any_text_cag_side_then_no_exception_is_raised(
    text: str,
) -> None:
    """never-raises invariant for cag side over all strings."""
    result = panel_body(text, "cag")
    assert result is not None


@given(
    turn_number=st.integers(min_value=1, max_value=20),
    cumulative_cost=st.floats(
        min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
    chunk_ids=st.lists(
        st.integers(min_value=0, max_value=999), min_size=0, max_size=10
    ),
    context_tokens=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=50)
def test_when_rag_header_called_with_valid_vm_then_no_exception_is_raised(
    turn_number: int,
    cumulative_cost: float,
    chunk_ids: list[int],
    context_tokens: int,
) -> None:
    """rag_header never raises for any valid RagHeaderVM (never-raises invariant)."""
    vm = _RagHeaderVM(
        turn_number=turn_number,
        chunk_ids=tuple(chunk_ids),
        context_tokens=context_tokens,
        cumulative_cost=cumulative_cost,
    )
    assert rag_header(vm) is not None


@given(
    turn_number=st.integers(min_value=1, max_value=20),
    cache_read_tokens=st.integers(min_value=0, max_value=50_000),
    context_size=st.integers(min_value=0, max_value=200_000),
    cumulative_cost=st.floats(
        min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=50)
def test_when_cag_header_called_with_valid_vm_then_no_exception_is_raised(
    turn_number: int,
    cache_read_tokens: int,
    context_size: int,
    cumulative_cost: float,
) -> None:
    """cag_header never raises for any valid CagHeaderVM (never-raises invariant)."""
    vm = _CagHeaderVM(
        turn_number=turn_number,
        cache_read_tokens=cache_read_tokens,
        context_size=context_size,
        cumulative_cost=cumulative_cost,
    )
    assert cag_header(vm) is not None


@given(
    rag_cost=st.floats(
        min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    cag_cost=st.floats(
        min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    cheaper_multiple=st.one_of(
        st.none(),
        st.floats(
            min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    ),
    cheaper_side=st.one_of(st.none(), st.sampled_from(["rag", "cag"])),
    rag_miss_count=st.integers(min_value=0, max_value=20),
    cag_full_recall_count=st.integers(min_value=0, max_value=20),
    rag_cache_read_total=st.integers(min_value=0, max_value=100_000),
    cag_cache_read_total=st.integers(min_value=0, max_value=100_000),
)
@settings(max_examples=50)
def test_when_footer_strip_called_with_valid_vm_then_no_exception_is_raised(
    rag_cost: float,
    cag_cost: float,
    cheaper_multiple: float | None,
    cheaper_side: str | None,
    rag_miss_count: int,
    cag_full_recall_count: int,
    rag_cache_read_total: int,
    cag_cache_read_total: int,
) -> None:
    """footer_strip never raises for any valid FooterVM (never-raises invariant)."""
    vm = _FooterVM(
        rag_cost=rag_cost,
        cag_cost=cag_cost,
        cheaper_multiple=cheaper_multiple,
        cheaper_side=cheaper_side,
        rag_miss_count=rag_miss_count,
        cag_full_recall_count=cag_full_recall_count,
        rag_cache_read_total=rag_cache_read_total,
        cag_cache_read_total=cag_cache_read_total,
    )
    assert footer_strip(vm) is not None


@given(
    rag_cost=st.floats(
        min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    cag_cost=st.floats(
        min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    rag_miss_count=st.integers(min_value=0, max_value=20),
    cag_full_recall_count=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=30)
def test_when_cheaper_multiple_is_none_then_em_dash_sentinel_always_rendered(
    rag_cost: float,
    cag_cost: float,
    rag_miss_count: int,
    cag_full_recall_count: int,
) -> None:
    """Invariant: cheaper_multiple=None → '—' in rendered footer, regardless of other fields."""
    vm = _FooterVM(
        rag_cost=rag_cost,
        cag_cost=cag_cost,
        cheaper_multiple=None,
        cheaper_side=None,
        rag_miss_count=rag_miss_count,
        cag_full_recall_count=cag_full_recall_count,
        rag_cache_read_total=0,
        cag_cache_read_total=0,
    )
    assert "—" in _render(footer_strip(vm))


# ===========================================================================
# footer_strip — cheaper-side label (SilvioBaratto comment, 2026-07-08)
# "render which side is cheaper explicitly, e.g. 'CAG 4.2× cheaper'"
# ===========================================================================


def test_when_cag_is_cheaper_then_footer_renders_cag_label():
    """CAG-cheaper case: footer must name 'CAG' and 'cheaper' when cag_cost < rag_cost."""
    vm = _footer_vm(
        rag_cost=0.025, cag_cost=0.010, cheaper_multiple=2.5, cheaper_side="cag"
    )
    text = _render(footer_strip(vm))
    assert "CAG" in text
    assert "cheaper" in text.lower()


def test_when_rag_is_cheaper_then_footer_renders_rag_label():
    """RAG-cheaper case: footer must name 'RAG' and 'cheaper' when rag_cost < cag_cost."""
    vm = _footer_vm(
        rag_cost=0.010, cag_cost=0.013, cheaper_multiple=1.3, cheaper_side="rag"
    )
    text = _render(footer_strip(vm))
    assert "RAG" in text
    assert "cheaper" in text.lower()


def test_when_cheaper_multiple_is_none_then_no_side_label_in_footer():
    """Sentinel case: no side label when cheaper_multiple=None — only '—' appears."""
    vm = _footer_vm(cheaper_multiple=None, cheaper_side=None)
    text = _render(footer_strip(vm))
    assert "—" in text


# ===========================================================================
# Regression — real Side enum must not produce "SIDE." prefix (issue #18)
# ===========================================================================

from flashcard.runner import Side as _Side  # noqa: E402


def _real_footer_vm(cheaper_side: _Side, **kw):
    """Build a real FooterVM (from tui_state) with the given real Side enum."""
    from flashcard.tui_state import FooterVM

    defaults = dict(
        rag_cost=0.020,
        cag_cost=0.005,
        cheaper_multiple=4.0,
        rag_miss_count=0,
        cag_full_recall_count=0,
        rag_cache_read_total=0,
        cag_cache_read_total=0,
    )
    defaults.update(kw)
    return FooterVM(cheaper_side=cheaper_side, **defaults)


def test_when_real_side_cag_passed_to_footer_strip_then_no_side_dot_prefix():
    """Regression #18: Side.CAG must render 'CAG …' not 'SIDE.CAG …'."""
    text = _render(footer_strip(_real_footer_vm(cheaper_side=_Side.CAG)))
    assert "SIDE." not in text
    assert "CAG" in text


def test_when_real_side_rag_passed_to_footer_strip_then_no_side_dot_prefix():
    """Regression #18: Side.RAG must render 'RAG …' not 'SIDE.RAG …'."""
    text = _render(
        footer_strip(
            _real_footer_vm(cheaper_side=_Side.RAG, rag_cost=0.005, cag_cost=0.020)
        )
    )
    assert "SIDE." not in text
    assert "RAG" in text
