"""Regression tests for issue #18 — footer side-label must never include 'SIDE.' prefix.

Root cause pinned by these tests:
  `str(Side.CAG).upper()` → `"SIDE.CAG"` on Python ≤ 3.10 str-mixin enum behaviour.
  The fix must produce `"CAG N.N× cheaper"` / `"RAG N.N× cheaper"`, not `"SIDE.CAG …"`.

Criteria covered:
  [UNIT] Criterion 9 — regression test passes the real Side enum (not a string stub)
         and asserts absence of "SIDE.".
  [T3]   Criterion 8 — footer renders correctly when driven by a real TuiState.footer
         FooterVM (end-to-end through state → render, no terminal, no API call).
  [UNIT] Criterion 3 — .env.example is shipped; OPENAI_API_KEY read from env only.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

from hypothesis import given, settings, strategies as st
from rich.console import Console

from flashcard.runner import Side
from flashcard.tui_render import footer_strip
from flashcard.tui_state import FooterVM, TuiState


# ---------------------------------------------------------------------------
# Rendering helper
# ---------------------------------------------------------------------------


def _render(renderable: Any) -> str:
    """Render a Rich renderable to plain text without a live terminal."""
    c = Console(record=True, width=120, no_color=True)
    c.print(renderable)
    return c.export_text()


# ---------------------------------------------------------------------------
# Duck-typed TurnResult stub — built from criterion text, no source read
# (TuiState.apply needs rag/cag sides with cost, cache tokens, retrieval info)
# ---------------------------------------------------------------------------


@dataclass
class _Side:
    text: str
    cost_usd: float
    cached_input_tokens: int
    input_tokens: int
    retrieved: tuple
    context_token_count: int

    @property
    def cost(self) -> float:
        return self.cost_usd


@dataclass
class _Turn:
    index: int
    rag: _Side
    cag: _Side
    question: str = "Q?"
    verdict: Any = None

    @property
    def turn_number(self) -> int:
        return self.index + 1


def _turn(rag_cost: float = 0.020, cag_cost: float = 0.005) -> _Turn:
    return _Turn(
        index=0,
        rag=_Side(
            text="RAG answer",
            cost_usd=rag_cost,
            cached_input_tokens=0,
            input_tokens=400,
            retrieved=(0, 1),
            context_token_count=300,
        ),
        cag=_Side(
            text="CAG answer",
            cost_usd=cag_cost,
            cached_input_tokens=800,
            input_tokens=900,
            retrieved=(),
            context_token_count=0,
        ),
    )


# ===========================================================================
# Criterion 9 [UNIT]: real Side enum — regression: "SIDE." must not appear
# ===========================================================================


def test_when_footer_vm_cheaper_side_is_real_side_cag_then_no_side_dot_prefix():
    """Regression: Side.CAG must render as 'CAG …' not 'SIDE.CAG …'."""
    vm = FooterVM(
        rag_cost=0.020,
        cag_cost=0.005,
        cheaper_multiple=4.0,
        cheaper_side=Side.CAG,
        rag_miss_count=0,
        cag_full_recall_count=0,
        rag_cache_read_total=0,
        cag_cache_read_total=0,
    )
    text = _render(footer_strip(vm))
    assert "SIDE." not in text


def test_when_footer_vm_cheaper_side_is_real_side_cag_then_cag_label_appears():
    """Regression: rendered footer must contain 'CAG' when Side.CAG is cheaper."""
    vm = FooterVM(
        rag_cost=0.020,
        cag_cost=0.005,
        cheaper_multiple=4.0,
        cheaper_side=Side.CAG,
        rag_miss_count=0,
        cag_full_recall_count=0,
        rag_cache_read_total=0,
        cag_cache_read_total=0,
    )
    text = _render(footer_strip(vm))
    assert "CAG" in text
    assert "cheaper" in text.lower()


def test_when_footer_vm_cheaper_side_is_real_side_rag_then_no_side_dot_prefix():
    """Regression: Side.RAG must render as 'RAG …' not 'SIDE.RAG …'."""
    vm = FooterVM(
        rag_cost=0.005,
        cag_cost=0.020,
        cheaper_multiple=4.0,
        cheaper_side=Side.RAG,
        rag_miss_count=0,
        cag_full_recall_count=0,
        rag_cache_read_total=0,
        cag_cache_read_total=0,
    )
    text = _render(footer_strip(vm))
    assert "SIDE." not in text


def test_when_footer_vm_cheaper_side_is_real_side_rag_then_rag_label_appears():
    """Regression: rendered footer must contain 'RAG' when Side.RAG is cheaper."""
    vm = FooterVM(
        rag_cost=0.005,
        cag_cost=0.020,
        cheaper_multiple=4.0,
        cheaper_side=Side.RAG,
        rag_miss_count=0,
        cag_full_recall_count=0,
        rag_cache_read_total=0,
        cag_cache_read_total=0,
    )
    text = _render(footer_strip(vm))
    assert "RAG" in text
    assert "cheaper" in text.lower()


def test_when_footer_vm_cheaper_side_is_none_then_no_side_dot_prefix():
    """Edge case: cheaper_side=None must never produce 'SIDE.' either."""
    vm = FooterVM(
        rag_cost=0.010,
        cag_cost=0.010,
        cheaper_multiple=None,
        cheaper_side=None,
        rag_miss_count=0,
        cag_full_recall_count=0,
        rag_cache_read_total=0,
        cag_cache_read_total=0,
    )
    text = _render(footer_strip(vm))
    assert "SIDE." not in text


# ===========================================================================
# Criterion 8 [T3]: through real TuiState → FooterVM → footer_strip
# ===========================================================================


def test_when_tui_state_applied_cag_cheaper_then_footer_renders_cag_not_side_cag():
    """Criterion 8: end-to-end TuiState.footer → footer_strip produces 'CAG … cheaper'."""
    state = TuiState()
    state.apply(_turn(rag_cost=0.020, cag_cost=0.005))
    text = _render(footer_strip(state.footer))
    assert "SIDE." not in text
    assert "CAG" in text
    assert "cheaper" in text.lower()


def test_when_tui_state_applied_rag_cheaper_then_footer_renders_rag_not_side_rag():
    """Criterion 8: end-to-end TuiState.footer → footer_strip produces 'RAG … cheaper'."""
    state = TuiState()
    state.apply(_turn(rag_cost=0.005, cag_cost=0.020))
    text = _render(footer_strip(state.footer))
    assert "SIDE." not in text
    assert "RAG" in text
    assert "cheaper" in text.lower()


def test_when_tui_state_applied_multiple_turns_then_footer_never_shows_side_prefix():
    """Criterion 8: after N turns, footer label is still a bare 'CAG'/'RAG', never 'SIDE.*'."""
    state = TuiState()
    for i in range(3):
        state.apply(_turn_indexed(i, rag_cost=0.020, cag_cost=0.005))
    text = _render(footer_strip(state.footer))
    assert "SIDE." not in text


# fix: _turn doesn't accept index kwarg — provide a helper that does
def _turn_indexed(
    index: int, rag_cost: float = 0.020, cag_cost: float = 0.005
) -> _Turn:
    t = _turn(rag_cost=rag_cost, cag_cost=cag_cost)
    return _Turn(
        index=index,
        rag=t.rag,
        cag=t.cag,
        question=t.question,
        verdict=t.verdict,
    )


def test_when_tui_state_applied_three_turns_then_footer_cag_cheaper_label_is_clean():
    """After three turns with CAG consistently cheaper, label is 'CAG N.N× cheaper'."""
    state = TuiState()
    for i in range(3):
        state.apply(_turn_indexed(i, rag_cost=0.020, cag_cost=0.005))
    text = _render(footer_strip(state.footer))
    assert "SIDE." not in text
    assert "CAG" in text


# ===========================================================================
# Criterion 3 [UNIT]: .env.example shipped; OPENAI_API_KEY from env only
# ===========================================================================


def test_when_project_root_inspected_then_env_example_file_exists():
    """Criterion: '.env.example is shipped' — file must exist at repo root."""
    root = pathlib.Path(__file__).parent.parent
    assert (root / ".env.example").exists(), ".env.example not found in repo root"


def test_when_env_example_read_then_openai_api_key_is_referenced():
    """Criterion: key read from OPENAI_API_KEY — .env.example must name it."""
    root = pathlib.Path(__file__).parent.parent
    content = (root / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in content


def test_when_env_example_read_then_it_contains_only_a_placeholder_not_a_real_key():
    """Criterion: no secrets in source — .env.example value must be a placeholder."""
    root = pathlib.Path(__file__).parent.parent
    content = (root / ".env.example").read_text(encoding="utf-8")
    # A real key starts with 'sk-' followed by many non-placeholder chars.
    # A placeholder looks like 'sk-...' or 'your-key-here' or 'REPLACE_ME'.
    lines = [ln.strip() for ln in content.splitlines() if "OPENAI_API_KEY" in ln]
    assert lines, "OPENAI_API_KEY line not found in .env.example"
    for line in lines:
        if "=" in line:
            value = line.split("=", 1)[1].strip()
            # Real key: starts with 'sk-' and ≥ 30 real chars with no placeholder markers.
            is_real_key = (
                value.startswith("sk-")
                and len(value) >= 30
                and "." not in value  # placeholders like sk-... use dots
                and "<" not in value
                and "your" not in value.lower()
                and "replace" not in value.lower()
            )
            assert not is_real_key, f"Real API key found in .env.example: {value[:8]}…"


# ===========================================================================
# Property-based tests — invariant: SIDE. never appears for any real Side
# ===========================================================================


@given(side=st.sampled_from([Side.CAG, Side.RAG]))
def test_when_footer_vm_has_any_real_side_then_side_dot_never_appears_in_output(
    side: Side,
) -> None:
    """Never-raises invariant: for any real Side value, rendered footer never contains 'SIDE.'."""
    rag_cost = 0.020 if side is Side.CAG else 0.005
    cag_cost = 0.005 if side is Side.CAG else 0.020
    vm = FooterVM(
        rag_cost=rag_cost,
        cag_cost=cag_cost,
        cheaper_multiple=4.0,
        cheaper_side=side,
        rag_miss_count=0,
        cag_full_recall_count=0,
        rag_cache_read_total=0,
        cag_cache_read_total=0,
    )
    assert "SIDE." not in _render(footer_strip(vm))


@given(
    multiple=st.floats(
        min_value=1.0, max_value=1_000.0, allow_nan=False, allow_infinity=False
    ),
    side=st.sampled_from([Side.CAG, Side.RAG]),
)
@settings(max_examples=40)
def test_when_footer_vm_has_real_side_and_any_valid_multiple_then_no_side_dot_prefix(
    multiple: float,
    side: Side,
) -> None:
    """Invariant: any positive multiple × any real Side enum → 'SIDE.' never in output."""
    vm = FooterVM(
        rag_cost=0.020,
        cag_cost=0.005,
        cheaper_multiple=multiple,
        cheaper_side=side,
        rag_miss_count=0,
        cag_full_recall_count=0,
        rag_cache_read_total=0,
        cag_cache_read_total=0,
    )
    assert "SIDE." not in _render(footer_strip(vm))
