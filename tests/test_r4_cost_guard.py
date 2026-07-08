"""
tests/test_r4_cost_guard.py

Source-blind example + property tests for Issue #4:
"feat: add cost-guard skeleton to the CLI with worst-case spend estimate + confirmation"

Oracle-classified [UNIT] criteria covered:
  C1 – both pipelines share model + max_tokens in the cost estimate
  C2 – no secrets in source; .env.example ships OPENAI_API_KEY placeholder
  C3 – declining confirmation exits cleanly without calling any pipeline
  C4 – token counts derived from tiktoken, capped by --max-tokens for output
  C5 – estimate is computed via pricing.py (monkeypatch -> known dollar figure)
  C6 – CAG Q1 (whole doc, base rate, nothing cached) is the largest single-question cost

[T3] and [NOT VERIFIABLE] criteria are intentionally excluded.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import settings as h_settings
from hypothesis import strategies as st
from typer.testing import CliRunner


# ---------------------------------------------------------------------------
# Shared constants — derived from criterion text, not from source
# ---------------------------------------------------------------------------

_DOC = (
    "This handbook covers the return policy, password reset procedure, "
    "service centre locations, and extended warranty terms in full. " * 100
)  # ~1 500 words; tiktoken count differs meaningfully from whitespace count

_QUESTIONS = [
    "What is the return policy?",
    "How do I reset my password?",
    "Where are the service centres?",
]

_MODEL = "gpt-5.2"
_MAX_TOKENS = 400


def _g(estimate, key: str):
    """Access estimate as dict or dataclass without knowing the concrete type."""
    return estimate[key] if isinstance(estimate, dict) else getattr(estimate, key)


# ---------------------------------------------------------------------------
# C5 — Estimate delegates arithmetic to pricing.py
# ---------------------------------------------------------------------------


def test_when_price_rag_monkeypatched_then_rag_total_equals_fixed_times_n_questions(
    monkeypatch,
):
    """when price_rag_turn always returns 0.07, then rag_total = 0.07 * n_questions"""
    import flashcard.pricing as pricing
    from flashcard.cli import estimate_worst_case

    monkeypatch.setattr(pricing, "price_rag_turn", lambda *a, **kw: 0.07)
    monkeypatch.setattr(pricing, "price_cag_turn", lambda *a, **kw: 0.00)

    est = estimate_worst_case(_DOC, _QUESTIONS, _MAX_TOKENS, _MODEL)

    assert abs(_g(est, "rag_total") - 0.07 * len(_QUESTIONS)) < 1e-9


def test_when_price_cag_monkeypatched_then_cag_total_equals_fixed_times_n_questions(
    monkeypatch,
):
    """when price_cag_turn always returns 0.05, then cag_total = 0.05 * n_questions"""
    import flashcard.pricing as pricing
    from flashcard.cli import estimate_worst_case

    monkeypatch.setattr(pricing, "price_rag_turn", lambda *a, **kw: 0.00)
    monkeypatch.setattr(pricing, "price_cag_turn", lambda *a, **kw: 0.05)

    est = estimate_worst_case(_DOC, _QUESTIONS, _MAX_TOKENS, _MODEL)

    assert abs(_g(est, "cag_total") - 0.05 * len(_QUESTIONS)) < 1e-9


def test_when_both_prices_monkeypatched_then_combined_equals_rag_plus_cag(monkeypatch):
    """when both price functions are fixed, then combined = rag_total + cag_total"""
    import flashcard.pricing as pricing
    from flashcard.cli import estimate_worst_case

    monkeypatch.setattr(pricing, "price_rag_turn", lambda *a, **kw: 0.03)
    monkeypatch.setattr(pricing, "price_cag_turn", lambda *a, **kw: 0.08)

    est = estimate_worst_case(_DOC, _QUESTIONS, _MAX_TOKENS, _MODEL)

    rag = _g(est, "rag_total")
    cag = _g(est, "cag_total")
    combined = _g(est, "combined")
    assert abs(combined - (rag + cag)) < 1e-9


# ---------------------------------------------------------------------------
# C6 — CAG Q1 (whole doc, base rate, nothing cached) dominates
# ---------------------------------------------------------------------------


def test_when_estimate_returned_then_cag_q1_is_an_explicit_field():
    """when estimate_worst_case returns, then cag_q1 is surfaced as its own explicit field"""
    from flashcard.cli import estimate_worst_case

    est = estimate_worst_case(_DOC, _QUESTIONS, _MAX_TOKENS, _MODEL)

    if isinstance(est, dict):
        assert "cag_q1" in est, "estimate dict must contain key 'cag_q1'"
    else:
        assert hasattr(est, "cag_q1"), "estimate object must expose attribute 'cag_q1'"


def test_when_estimate_computed_then_cag_q1_exceeds_average_rag_question_cost():
    """when doc tokens >> chunk_size * top_k, then CAG Q1 cost > per-question RAG cost"""
    from flashcard.cli import estimate_worst_case

    est = estimate_worst_case(_DOC, _QUESTIONS, _MAX_TOKENS, _MODEL)

    cag_q1 = _g(est, "cag_q1")
    rag_per_q = _g(est, "rag_total") / len(_QUESTIONS)

    assert cag_q1 > 0, "CAG Q1 must have positive cost"
    assert cag_q1 > rag_per_q, (
        f"CAG Q1 ${cag_q1:.6f} must exceed per-question RAG cost ${rag_per_q:.6f}; "
        "cost gap must be attributable to retrieved-slice vs whole-doc alone"
    )


# ---------------------------------------------------------------------------
# C3 — Declining confirmation exits cleanly without calling any pipeline
# ---------------------------------------------------------------------------


def test_when_user_inputs_n_then_run_turn_is_not_called(tmp_path):
    """when 'n' is entered at the confirmation prompt, then flashcard.runner.run_turn is never invoked"""
    from flashcard.cli import app

    doc_file = tmp_path / "doc.txt"
    doc_file.write_text(_DOC)

    runner = CliRunner()
    with patch("flashcard.runner.run_turn") as mock_run:
        runner.invoke(app, ["--doc", str(doc_file)], input="n\n")

    mock_run.assert_not_called()


def test_when_user_inputs_n_then_cli_exits_with_code_zero(tmp_path):
    """when 'n' is entered at the confirmation prompt, then the CLI exits with code 0"""
    from flashcard.cli import app

    doc_file = tmp_path / "doc.txt"
    doc_file.write_text(_DOC)

    runner = CliRunner()
    with patch(
        "flashcard.runner.run_turn",
        side_effect=AssertionError("pipeline must not fire"),
    ):
        result = runner.invoke(app, ["--doc", str(doc_file)], input="n\n")

    assert result.exit_code == 0, (
        f"Declining must produce exit code 0; got {result.exit_code}.\n{result.output}"
    )


# ---------------------------------------------------------------------------
# C4 — Token counts use tiktoken; output is capped by --max-tokens
# ---------------------------------------------------------------------------


def test_when_known_text_counted_then_result_matches_tiktoken_o200k_base():
    """when count_tokens is called with a known string, then it matches tiktoken o200k_base"""
    import tiktoken
    from flashcard.cli import count_tokens

    text = "The quick brown fox jumps over the lazy dog."
    enc = tiktoken.get_encoding("o200k_base")
    assert count_tokens(text) == len(enc.encode(text))


def test_when_max_tokens_is_one_then_combined_estimate_stays_below_one_dollar():
    """when max_tokens=1, then the combined estimate is well below $1 (output cost is capped)"""
    from flashcard.cli import estimate_worst_case

    est = estimate_worst_case(_DOC, _QUESTIONS[:1], max_tokens=1, model=_MODEL)
    # gpt-5.2 output = $14.00/MTok; 1 token x 2 pipelines x 1 question = $0.000028
    assert _g(est, "combined") < 1.0


@given(st.integers(min_value=1, max_value=2048))
@h_settings(max_examples=40)
def test_when_max_tokens_doubled_then_combined_estimate_is_non_decreasing(
    max_tokens: int,
):
    """for any max_tokens in [1, 2048], doubling it must not decrease the combined estimate"""
    from flashcard.cli import estimate_worst_case

    est_a = estimate_worst_case(
        _DOC, _QUESTIONS[:1], max_tokens=max_tokens, model=_MODEL
    )
    est_b = estimate_worst_case(
        _DOC, _QUESTIONS[:1], max_tokens=min(max_tokens * 2, 4096), model=_MODEL
    )

    a = _g(est_a, "combined")
    b = _g(est_b, "combined")
    assert a <= b + 1e-9, (
        f"estimate(max_tokens={max_tokens})={a:.8f} must not exceed "
        f"estimate(max_tokens={min(max_tokens * 2, 4096)})={b:.8f}"
    )


# ---------------------------------------------------------------------------
# C2 — No secrets in source; .env.example ships OPENAI_API_KEY placeholder
# ---------------------------------------------------------------------------


def test_when_env_example_read_then_openai_api_key_var_is_present():
    """when .env.example is read, then it references OPENAI_API_KEY"""
    root = Path(__file__).parent.parent
    env_example = root / ".env.example"

    assert env_example.is_file(), ".env.example must exist at the project root"
    assert "OPENAI_API_KEY" in env_example.read_text(), (
        ".env.example must reference OPENAI_API_KEY so users know what to configure"
    )


def test_when_env_example_read_then_no_real_key_value_appears():
    """when .env.example is read, then no real 'sk-...' key appears as a value"""
    root = Path(__file__).parent.parent
    env_example = root / ".env.example"

    if not env_example.is_file():
        pytest.skip(".env.example not present — covered by the existence test")

    for line in env_example.read_text().splitlines():
        if "OPENAI_API_KEY" in line and "=" in line:
            value = line.partition("=")[2].strip().strip('"').strip("'")
            assert not value.startswith("sk-"), (
                "A real OpenAI key must not appear in .env.example"
            )


def test_when_pricing_source_inspected_then_no_hardcoded_openai_key():
    """when flashcard.pricing source is inspected at runtime, then no 'sk-' prefix is present"""
    import flashcard.pricing as pricing

    src = inspect.getsource(pricing)
    assert "sk-" not in src, (
        "flashcard/pricing.py contains 'sk-', the OpenAI key prefix — "
        "secrets must not appear in source"
    )


# ---------------------------------------------------------------------------
# C1 — Both pipelines share model and max_tokens in the estimate
# ---------------------------------------------------------------------------


def test_when_max_tokens_increases_then_both_rag_and_cag_totals_are_non_decreasing():
    """when max_tokens increases from 50 to 400, both rag_total and cag_total must not decrease"""
    from flashcard.cli import estimate_worst_case

    est_low = estimate_worst_case(_DOC, _QUESTIONS, max_tokens=50, model=_MODEL)
    est_high = estimate_worst_case(_DOC, _QUESTIONS, max_tokens=400, model=_MODEL)

    assert _g(est_high, "rag_total") >= _g(est_low, "rag_total"), (
        "RAG total must not decrease when max_tokens grows — "
        "both sides must cap output at the same max_tokens"
    )
    assert _g(est_high, "cag_total") >= _g(est_low, "cag_total"), (
        "CAG total must not decrease when max_tokens grows — "
        "both sides must cap output at the same max_tokens"
    )
