"""
Source-blind tests for Issue #5: o200k_base encoding alignment for cost_guard.

Oracle-classified [UNIT] scope criteria covered:
  C1 – cost_guard tokenizes with o200k_base, matching corpus.py /
       test_issue_2_corpus_loader.py
  C2 – the cost-guard token-count test asserts o200k_base (this file IS that test)
  C3 – corpus.py and cost_guard share the same encoding (same-encoding invariant)

[T3] and [NOT VERIFIABLE] criteria are intentionally excluded.

All tests are authored from the acceptance criteria only.  No implementation
source was read.  These tests currently fail because cost_guard uses
cl100k_base; they will pass once the implementation switches to o200k_base.
"""
from __future__ import annotations

import inspect

import tiktoken
from hypothesis import given
from hypothesis import settings as h_settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# C1 + C2 — count_tokens uses o200k_base, not cl100k_base
# ---------------------------------------------------------------------------


def test_when_count_tokens_called_then_result_matches_o200k_base():
    """when count_tokens is called on a known string, then it matches o200k_base"""
    from flashcard.cli import count_tokens

    text = "The quick brown fox jumps over the lazy dog."
    enc = tiktoken.get_encoding("o200k_base")
    assert count_tokens(text) == len(enc.encode(text))


@given(st.text())
@h_settings(max_examples=50)
def test_when_any_text_counted_then_result_always_matches_o200k_base(text: str):
    """for any text, count_tokens must equal the o200k_base token count (invariant)"""
    from flashcard.cli import count_tokens

    enc = tiktoken.get_encoding("o200k_base")
    assert count_tokens(text) == len(enc.encode(text))


# ---------------------------------------------------------------------------
# C1 — cost_guard module names o200k_base, not cl100k_base
# ---------------------------------------------------------------------------


def test_when_cost_guard_source_inspected_then_o200k_base_is_the_encoding():
    """when cost_guard is inspected at runtime, then o200k_base appears and cl100k_base does not

    Chosen behaviour: the module constant _ENCODING (or equivalent) must be
    'o200k_base'.  cl100k_base is the old encoding used for gpt-3.5/gpt-4;
    gpt-5.2 (o200k family) requires o200k_base so that pre-flight token counts
    match the model's actual tokenisation.
    """
    import flashcard.cost_guard as cg

    src = inspect.getsource(cg)
    assert "o200k_base" in src, (
        "flashcard/cost_guard.py must reference 'o200k_base' — "
        "this is the tiktoken encoding for the gpt-5.x / gpt-4o model family"
    )
    assert "cl100k_base" not in src, (
        "flashcard/cost_guard.py must NOT reference 'cl100k_base' — "
        "the old encoding must be replaced with 'o200k_base'"
    )


# ---------------------------------------------------------------------------
# C3 — corpus.py and cost_guard share the same encoding
# ---------------------------------------------------------------------------


def test_when_corpus_counted_by_cost_guard_then_result_matches_direct_o200k_base():
    """when the built-in corpus is counted via cost_guard.count_tokens, then it
    equals a direct tiktoken o200k_base encode — the two modules share one encoding"""
    from flashcard.corpus import load_document
    from flashcard.cost_guard import count_tokens

    corpus = str(load_document())
    enc = tiktoken.get_encoding("o200k_base")
    direct = len(enc.encode(corpus))
    guard = count_tokens(corpus)

    assert guard == direct, (
        f"cost_guard.count_tokens({guard}) diverges from o200k_base direct count "
        f"({direct}); corpus.py and cost_guard must use the same encoding"
    )
