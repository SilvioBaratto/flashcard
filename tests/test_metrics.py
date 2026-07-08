"""
Source-blind tests for flashcard/metrics.py — issue #11.

Every test is derived directly from the acceptance criteria text and
requirements.md; no implementation source was opened during authorship.
Fakes are built from criterion language only (Red phase of TDD).

Criteria covered (oracle-classified as [UNIT]):
  C8  — RetrievalMetrics record: chunk_ids, similarity_scores,
         context_token_count, query_embed_tokens.
  C9  — fresh_tokens(metrics) → max(input − cached, 0) with clamp.
  C10 — count_tokens(text, encoding="o200k_base") via tiktoken.
  C11 — Building RetrievalMetrics from EmbeddingRetriever.last_scores /
         last_query_tokens; covering fresh-token clamp + context-token
         counting in the unit-test suite.

Criteria intentionally excluded:
  C1, C3 — already covered in test_r4_cost_guard.py / test_env.py.
  C2, C4 — oracle [T3]; require live API calls, excluded from unit phase.
  NOT VERIFIABLE criteria — never emit a proxy assertion for these.
"""

from __future__ import annotations

from typing import List, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Fakes — constructed from acceptance-criteria text only, not from source
# ---------------------------------------------------------------------------


class _TokenCounts:
    """
    Minimal protocol for fresh_tokens(metrics).

    The criterion says the helper accepts any object with input_tokens and
    cached_input_tokens; no concrete TurnMetrics import is required here.
    """

    def __init__(self, input_tokens: int, cached_input_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.cached_input_tokens = cached_input_tokens


class _FakeRetriever:
    """
    Fake EmbeddingRetriever exposing the two attributes named in criterion C11:
    last_scores and last_query_tokens.

    last_scores  — list of (chunk_id: int, similarity_score: float) pairs,
                   recording which chunks were retrieved and how similar they were.
    last_query_tokens — int, the token count consumed by the embedded query.
    last_context_tokens — int, token count of the concatenated retrieved context;
                          present here so the factory can read it if it tracks it
                          on the retriever rather than computing it inline.
    """

    def __init__(
        self,
        last_scores: List[Tuple[int, float]],
        last_query_tokens: int,
        last_context_tokens: int = 0,
    ) -> None:
        self.last_scores = last_scores
        self.last_query_tokens = last_query_tokens
        self.last_context_tokens = last_context_tokens


# ===========================================================================
# C8 — RetrievalMetrics record
# ===========================================================================


def test_when_retrieval_metrics_created_then_chunk_ids_are_accessible() -> None:
    """RetrievalMetrics stores chunk_ids exactly as supplied."""
    from flashcard.metrics import RetrievalMetrics

    m = RetrievalMetrics(
        chunk_ids=[0, 1, 2],
        similarity_scores=[0.95, 0.82, 0.71],
        context_token_count=150,
        query_embed_tokens=8,
    )
    assert m.chunk_ids == [0, 1, 2]


def test_when_retrieval_metrics_created_then_similarity_scores_are_accessible() -> None:
    """RetrievalMetrics stores similarity_scores exactly as supplied."""
    from flashcard.metrics import RetrievalMetrics

    m = RetrievalMetrics(
        chunk_ids=[0, 1],
        similarity_scores=[0.90, 0.80],
        context_token_count=100,
        query_embed_tokens=6,
    )
    assert m.similarity_scores == [0.90, 0.80]


def test_when_retrieval_metrics_created_then_context_token_count_is_accessible() -> (
    None
):
    """RetrievalMetrics stores context_token_count exactly as supplied."""
    from flashcard.metrics import RetrievalMetrics

    m = RetrievalMetrics(
        chunk_ids=[3],
        similarity_scores=[0.99],
        context_token_count=247,
        query_embed_tokens=5,
    )
    assert m.context_token_count == 247


def test_when_retrieval_metrics_created_then_query_embed_tokens_are_accessible() -> (
    None
):
    """RetrievalMetrics stores query_embed_tokens exactly as supplied."""
    from flashcard.metrics import RetrievalMetrics

    m = RetrievalMetrics(
        chunk_ids=[0, 2, 4],
        similarity_scores=[0.88, 0.76, 0.65],
        context_token_count=312,
        query_embed_tokens=11,
    )
    assert m.query_embed_tokens == 11


def test_when_retrieval_metrics_created_then_chunk_ids_and_scores_have_equal_length() -> (
    None
):
    """chunk_ids and similarity_scores are parallel sequences — one entry per chunk."""
    from flashcard.metrics import RetrievalMetrics

    ids = [7, 3, 12]
    scores = [0.91, 0.85, 0.78]
    m = RetrievalMetrics(
        chunk_ids=ids,
        similarity_scores=scores,
        context_token_count=400,
        query_embed_tokens=9,
    )
    assert len(m.chunk_ids) == len(m.similarity_scores) == 3


def test_when_retrieval_metrics_created_with_empty_ids_then_zero_chunks_recorded() -> (
    None
):
    """An empty retrieval (no chunks selected) is representable with empty lists."""
    from flashcard.metrics import RetrievalMetrics

    m = RetrievalMetrics(
        chunk_ids=[],
        similarity_scores=[],
        context_token_count=0,
        query_embed_tokens=4,
    )
    assert m.chunk_ids == []
    assert m.similarity_scores == []
    assert m.context_token_count == 0


# ===========================================================================
# C9 — fresh_tokens(metrics) helper: max(input − cached, 0)
# ===========================================================================


def test_when_cached_less_than_input_then_fresh_tokens_returns_difference() -> None:
    """fresh_tokens returns input − cached when cached < input (normal warm-cache case)."""
    from flashcard.metrics import fresh_tokens

    m = _TokenCounts(input_tokens=200, cached_input_tokens=50)
    assert fresh_tokens(m) == 150


def test_when_cached_exceeds_input_then_fresh_tokens_is_clamped_to_zero() -> None:
    """
    When cached > input the subtraction would be negative; the max(…, 0) clamp
    must return 0 rather than a negative token count.

    Criterion: 'max(input − cached, 0)' — the clamp is explicit and must be tested.
    """
    from flashcard.metrics import fresh_tokens

    m = _TokenCounts(input_tokens=100, cached_input_tokens=150)
    assert fresh_tokens(m) == 0


def test_when_cached_equals_input_then_fresh_tokens_returns_zero() -> None:
    """All input tokens served from cache → zero fresh tokens billed at base rate."""
    from flashcard.metrics import fresh_tokens

    m = _TokenCounts(input_tokens=300, cached_input_tokens=300)
    assert fresh_tokens(m) == 0


def test_when_cache_is_cold_then_fresh_tokens_equals_all_input_tokens() -> None:
    """Q1 has nothing cached: cached=0, so every input token is fresh (full base-rate)."""
    from flashcard.metrics import fresh_tokens

    m = _TokenCounts(input_tokens=500, cached_input_tokens=0)
    assert fresh_tokens(m) == 500


def test_when_fresh_tokens_called_then_result_is_an_integer() -> None:
    """fresh_tokens always returns an int, never a float."""
    from flashcard.metrics import fresh_tokens

    m = _TokenCounts(input_tokens=100, cached_input_tokens=40)
    assert isinstance(fresh_tokens(m), int)


def test_when_fresh_tokens_called_then_result_does_not_exceed_input_tokens() -> None:
    """fresh_tokens ≤ input_tokens for all valid usage."""
    from flashcard.metrics import fresh_tokens

    m = _TokenCounts(input_tokens=400, cached_input_tokens=100)
    assert fresh_tokens(m) <= m.input_tokens


# --- Property: non-negativity for all (input, cached) pairs ---------------


@settings(max_examples=300)
@given(
    input_tokens=st.integers(min_value=0, max_value=100_000),
    cached_input_tokens=st.integers(min_value=0, max_value=100_000),
)
def test_when_any_token_counts_given_then_fresh_tokens_is_non_negative(
    input_tokens: int,
    cached_input_tokens: int,
) -> None:
    """
    Invariant (non-negativity): fresh_tokens ≥ 0 for every possible (input, cached)
    pair, including the invalid case where cached > input.

    The criterion explicitly calls this out: 'including the cached ≥ input clamp'.
    """
    from flashcard.metrics import fresh_tokens

    m = _TokenCounts(input_tokens=input_tokens, cached_input_tokens=cached_input_tokens)
    assert fresh_tokens(m) >= 0


# --- Property: cold-cache identity ----------------------------------------


@settings(max_examples=150)
@given(input_tokens=st.integers(min_value=0, max_value=100_000))
def test_when_cached_is_zero_then_fresh_tokens_equals_input_for_any_input(
    input_tokens: int,
) -> None:
    """Invariant (cold-cache identity): cached=0 → fresh = input for all input values."""
    from flashcard.metrics import fresh_tokens

    m = _TokenCounts(input_tokens=input_tokens, cached_input_tokens=0)
    assert fresh_tokens(m) == input_tokens


# ===========================================================================
# C10 — count_tokens(text, encoding="o200k_base") via tiktoken
# ===========================================================================


def test_when_empty_string_given_then_count_tokens_returns_zero() -> None:
    """count_tokens('') == 0; an empty string has no tokens."""
    from flashcard.metrics import count_tokens

    assert count_tokens("") == 0


def test_when_known_text_counted_then_result_matches_tiktoken_o200k_base() -> None:
    """count_tokens must delegate to tiktoken o200k_base and return an exact match."""
    import tiktoken

    from flashcard.metrics import count_tokens

    text = "The quick brown fox jumps over the lazy dog."
    enc = tiktoken.get_encoding("o200k_base")
    expected = len(enc.encode(text))
    assert count_tokens(text) == expected


def test_when_nonempty_text_given_then_count_tokens_returns_positive_integer() -> None:
    """Any non-empty text yields at least one token."""
    from flashcard.metrics import count_tokens

    result = count_tokens("hello world")
    assert isinstance(result, int)
    assert result >= 1


def test_when_count_tokens_called_with_explicit_o200k_base_then_matches_default() -> (
    None
):
    """Passing encoding='o200k_base' explicitly returns the same value as the default."""
    from flashcard.metrics import count_tokens

    text = "context token counting with explicit encoding"
    assert count_tokens(text) == count_tokens(text, encoding="o200k_base")


def test_when_longer_text_given_then_count_tokens_is_at_least_as_large_as_shorter() -> (
    None
):
    """Monotonicity: a strictly longer string has at least as many tokens."""
    from flashcard.metrics import count_tokens

    short = "hello"
    longer = "hello world this sentence adds more words"
    assert count_tokens(longer) >= count_tokens(short)


def test_when_same_text_counted_twice_then_result_is_identical() -> None:
    """Determinism: count_tokens is a pure function; successive calls return the same int."""
    from flashcard.metrics import count_tokens

    text = "deterministic tokenisation check — repeated call must agree"
    assert count_tokens(text) == count_tokens(text)


# --- Property: totality (defined for all strings, returns ≥ 0) ------------


@settings(max_examples=80)
@given(text=st.text())
def test_when_any_text_given_then_count_tokens_returns_non_negative_integer(
    text: str,
) -> None:
    """Invariant (totality): count_tokens is defined for every string and returns int ≥ 0."""
    from flashcard.metrics import count_tokens

    result = count_tokens(text)
    assert isinstance(result, int)
    assert result >= 0


# --- Property: idempotence (same text → same count) -----------------------


@settings(max_examples=80)
@given(text=st.text())
def test_when_count_tokens_called_twice_with_same_text_then_result_is_unchanged(
    text: str,
) -> None:
    """Idempotence: count_tokens(text) == count_tokens(text) for all strings."""
    from flashcard.metrics import count_tokens

    assert count_tokens(text) == count_tokens(text)


# ===========================================================================
# C11 — Building RetrievalMetrics from EmbeddingRetriever.last_scores /
#        last_query_tokens
# ===========================================================================


def test_when_retriever_has_last_scores_then_chunk_ids_are_extracted() -> None:
    """
    Factory reads chunk ids from the first element of each (chunk_id, score) pair
    in retriever.last_scores.

    Criterion: 'building RetrievalMetrics from an EmbeddingRetriever's last_scores'.
    """
    from flashcard.metrics import retrieval_metrics_from_retriever

    retriever = _FakeRetriever(
        last_scores=[(0, 0.95), (2, 0.82), (5, 0.71)],
        last_query_tokens=7,
        last_context_tokens=180,
    )
    m = retrieval_metrics_from_retriever(retriever)
    assert m.chunk_ids == [0, 2, 5]


def test_when_retriever_has_last_scores_then_similarity_scores_are_extracted() -> None:
    """Factory reads scores from the second element of each pair in last_scores."""
    from flashcard.metrics import retrieval_metrics_from_retriever

    retriever = _FakeRetriever(
        last_scores=[(1, 0.90), (3, 0.77)],
        last_query_tokens=10,
        last_context_tokens=200,
    )
    m = retrieval_metrics_from_retriever(retriever)
    assert m.similarity_scores == [0.90, 0.77]


def test_when_retriever_has_last_query_tokens_then_query_embed_tokens_match() -> None:
    """Factory maps retriever.last_query_tokens → RetrievalMetrics.query_embed_tokens."""
    from flashcard.metrics import retrieval_metrics_from_retriever

    retriever = _FakeRetriever(
        last_scores=[(0, 0.88)],
        last_query_tokens=13,
        last_context_tokens=120,
    )
    m = retrieval_metrics_from_retriever(retriever)
    assert m.query_embed_tokens == 13


def test_when_retriever_has_no_scores_then_empty_lists_returned() -> None:
    """Empty retriever (nothing retrieved) yields empty chunk_ids and scores."""
    from flashcard.metrics import retrieval_metrics_from_retriever

    retriever = _FakeRetriever(
        last_scores=[],
        last_query_tokens=5,
        last_context_tokens=0,
    )
    m = retrieval_metrics_from_retriever(retriever)
    assert m.chunk_ids == []
    assert m.similarity_scores == []


def test_when_retrieval_metrics_built_from_retriever_then_context_token_count_is_non_negative() -> (
    None
):
    """context_token_count in the returned RetrievalMetrics is always a non-negative int."""
    from flashcard.metrics import retrieval_metrics_from_retriever

    retriever = _FakeRetriever(
        last_scores=[(0, 0.95), (1, 0.80)],
        last_query_tokens=9,
        last_context_tokens=350,
    )
    m = retrieval_metrics_from_retriever(retriever)
    assert isinstance(m.context_token_count, int)
    assert m.context_token_count >= 0


def test_when_retrieval_metrics_built_then_chunk_ids_and_scores_lengths_match() -> None:
    """Built RetrievalMetrics preserves the 1-to-1 correspondence of ids to scores."""
    from flashcard.metrics import retrieval_metrics_from_retriever

    retriever = _FakeRetriever(
        last_scores=[(4, 0.91), (7, 0.84), (1, 0.73)],
        last_query_tokens=6,
        last_context_tokens=270,
    )
    m = retrieval_metrics_from_retriever(retriever)
    assert len(m.chunk_ids) == len(m.similarity_scores) == 3


def test_when_retrieval_metrics_built_then_query_embed_tokens_is_non_negative() -> None:
    """query_embed_tokens in the built record is a non-negative int."""
    from flashcard.metrics import retrieval_metrics_from_retriever

    retriever = _FakeRetriever(
        last_scores=[(2, 0.79)],
        last_query_tokens=8,
        last_context_tokens=90,
    )
    m = retrieval_metrics_from_retriever(retriever)
    assert isinstance(m.query_embed_tokens, int)
    assert m.query_embed_tokens >= 0
