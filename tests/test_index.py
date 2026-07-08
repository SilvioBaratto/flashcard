"""
Source-blind tests for flashcard.index (issue #8).

All tests are derived from the acceptance criteria for:
  feat: in-memory numpy cosine index + top-k retrieval (index.py)

No implementation file was read. Tests are written against the spec contract only.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

from flashcard.index import VectorIndex

# ---------------------------------------------------------------------------
# Helpers (built from criteria, not from production code)
# ---------------------------------------------------------------------------


def _index_from_rows(rows: list[list[float]], start_id: int = 0) -> VectorIndex:
    """Build a VectorIndex from a list of equal-length float rows."""
    vectors = np.array(rows, dtype=float)
    n = len(rows)
    ids = list(range(start_id, start_id + n))
    texts = [f"text_{i}" for i in ids]
    idx = VectorIndex()
    idx.add(vectors, ids, texts)
    return idx


# ---------------------------------------------------------------------------
# Criterion: add() raises ValueError when lengths are mismatched
# ---------------------------------------------------------------------------


def test_when_chunk_ids_count_differs_from_vectors_count_then_value_error_is_raised():
    idx = VectorIndex()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 vectors
    with pytest.raises(ValueError):
        idx.add(vectors, chunk_ids=[0], texts=["a", "b"])  # 1 id, 2 texts


def test_when_texts_count_differs_from_vectors_count_then_value_error_is_raised():
    idx = VectorIndex()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 vectors
    with pytest.raises(ValueError):
        idx.add(vectors, chunk_ids=[0, 1], texts=["only one"])  # 2 ids, 1 text


def test_when_all_three_lengths_match_then_add_does_not_raise():
    idx = VectorIndex()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    idx.add(vectors, chunk_ids=[10, 20], texts=["alpha", "beta"])  # must not raise


# ---------------------------------------------------------------------------
# Criterion: search() returns Match(chunk_id, text, score) sorted descending
# ---------------------------------------------------------------------------


def test_when_searching_then_each_result_carries_chunk_id_text_and_score():
    idx = VectorIndex()
    idx.add(
        np.array([[1.0, 0.0]]),
        chunk_ids=[42],
        texts=["hello world"],
    )
    results = idx.search(np.array([1.0, 0.0]), k=1)
    assert len(results) == 1
    m = results[0]
    assert m.chunk_id == 42
    assert m.text == "hello world"
    assert isinstance(m.score, float)


def test_when_searching_then_results_are_sorted_by_descending_score():
    # Three vectors; query = [1, 0] → expected cosines: v0≈1.0, v2≈0.707, v1≈0.0
    idx = VectorIndex()
    vectors = np.array(
        [
            [1.0, 0.0],  # cosine 1.0 with [1,0]
            [0.0, 1.0],  # cosine 0.0
            [0.707, 0.707],  # cosine ≈ 0.707
        ]
    )
    idx.add(vectors, chunk_ids=[0, 1, 2], texts=["a", "b", "c"])
    results = idx.search(np.array([1.0, 0.0]), k=3)
    scores = [m.score for m in results]
    assert scores == sorted(scores, reverse=True)


def test_when_k_is_two_and_corpus_has_three_then_exactly_two_results_are_returned():
    idx = _index_from_rows([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    results = idx.search(np.array([1.0, 0.0]), k=2)
    assert len(results) == 2


def test_when_searching_then_returned_chunk_ids_belong_to_stored_ids():
    # Choice: stored ids are 100, 200, 300 — results must only reference those.
    idx = VectorIndex()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    idx.add(vectors, chunk_ids=[100, 200, 300], texts=["x", "y", "z"])
    results = idx.search(np.array([1.0, 0.0]), k=3)
    returned_ids = {m.chunk_id for m in results}
    assert returned_ids <= {100, 200, 300}


# ---------------------------------------------------------------------------
# Criterion: cosine via L2-normalized dot product; zero-norm guard
# ---------------------------------------------------------------------------


def test_when_query_equals_stored_vector_exactly_then_score_is_approximately_one():
    # Any non-unit vector: cosine with itself must be 1.0 after L2 normalisation.
    idx = VectorIndex()
    v = np.array([3.0, 4.0])  # deliberately non-unit; ‖v‖=5
    idx.add(np.array([v]), chunk_ids=[0], texts=["self"])
    results = idx.search(v, k=1)
    assert abs(results[0].score - 1.0) < 1e-6


def test_when_query_is_opposite_to_stored_vector_then_score_is_approximately_minus_one():
    idx = VectorIndex()
    v = np.array([1.0, 0.0])
    idx.add(np.array([v]), chunk_ids=[0], texts=["x"])
    results = idx.search(np.array([-1.0, 0.0]), k=1)
    assert abs(results[0].score - (-1.0)) < 1e-6


def test_when_orthogonal_vectors_are_stored_then_lower_scored_result_is_near_zero():
    # v0=[1,0], v1=[0,1]; query=[1,0] → v0 should score ≈1.0 and v1 ≈0.0.
    idx = _index_from_rows([[1.0, 0.0], [0.0, 1.0]])
    results = idx.search(np.array([1.0, 0.0]), k=2)
    scores = {m.chunk_id: m.score for m in results}
    assert abs(scores[0] - 1.0) < 1e-6
    assert abs(scores[1] - 0.0) < 1e-6


def test_when_zero_norm_vector_is_added_then_no_error_is_raised():
    """Zero-norm guard: the index must not crash on an all-zero vector."""
    idx = VectorIndex()
    # Must not raise; score produced for the zero vector is implementation-defined (e.g. 0.0).
    idx.add(np.array([[0.0, 0.0]]), chunk_ids=[0], texts=["zero"])


# ---------------------------------------------------------------------------
# Criterion: k > corpus → all returned; k <= 0 → empty list
# ---------------------------------------------------------------------------


def test_when_k_exceeds_corpus_size_then_all_corpus_chunks_are_returned():
    idx = _index_from_rows([[1.0, 0.0], [0.0, 1.0]])  # 2 vectors
    results = idx.search(np.array([1.0, 0.0]), k=999)
    assert len(results) == 2


def test_when_k_is_zero_then_empty_list_is_returned():
    idx = _index_from_rows([[1.0, 0.0]])
    results = idx.search(np.array([1.0, 0.0]), k=0)
    assert results == []


def test_when_k_is_negative_then_empty_list_is_returned():
    idx = _index_from_rows([[1.0, 0.0]])
    results = idx.search(np.array([1.0, 0.0]), k=-5)
    assert results == []


# ---------------------------------------------------------------------------
# Criterion: pure numpy — no OpenAI calls
# ---------------------------------------------------------------------------


def test_when_vector_index_is_used_without_openai_key_then_no_error_is_raised(
    monkeypatch,
):
    """VectorIndex must operate without OPENAI_API_KEY — pure numpy, no network call."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    idx = VectorIndex()
    idx.add(np.array([[1.0, 0.0]]), chunk_ids=[0], texts=["pure"])
    results = idx.search(np.array([1.0, 0.0]), k=1)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Property-based tests
# Invariants derived from UNIT criteria (ordering, score bounds, self-query).
# ---------------------------------------------------------------------------

_SAFE_FLOAT = st.floats(
    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
)
_DIM = st.integers(min_value=1, max_value=8)
_N = st.integers(min_value=1, max_value=15)


def _rows(n: int, d: int, data) -> list[list[float]]:
    return [data.draw(st.lists(_SAFE_FLOAT, min_size=d, max_size=d)) for _ in range(n)]


@given(n=_N, d=_DIM, data=st.data())
@settings(max_examples=60)
def test_when_searching_any_corpus_then_scores_are_returned_in_descending_order(
    n, d, data
):
    """Ordering invariant: search always returns results sorted by descending cosine score."""
    rows = _rows(n, d, data)
    query_row = data.draw(st.lists(_SAFE_FLOAT, min_size=d, max_size=d))
    idx = _index_from_rows(rows)
    results = idx.search(np.array(query_row, dtype=float), k=n)
    scores = [m.score for m in results]
    # Non-increasing within float tolerance: search() tie-breaks near-equal
    # cosine scores by chunk_id, so raw .score values can differ by ~1 ULP across
    # platforms (arm64 vs x64). Exact `== sorted(...)` is too strict for near-ties.
    assert all(a >= b - 1e-9 for a, b in zip(scores, scores[1:])), (
        f"scores not descending within tolerance: {scores}"
    )


@given(n=_N, d=_DIM, data=st.data())
@settings(max_examples=60)
def test_when_searching_any_corpus_then_all_scores_are_within_minus_one_and_one(
    n, d, data
):
    """Cosine scores are bounded to [-1, 1] for any valid input."""
    rows = _rows(n, d, data)
    # Skip the degenerate all-zero-vector case only when every row is zero.
    query_row = data.draw(st.lists(_SAFE_FLOAT, min_size=d, max_size=d))
    idx = _index_from_rows(rows)
    results = idx.search(np.array(query_row, dtype=float), k=n)
    for m in results:
        assert -1.0 - 1e-9 <= m.score <= 1.0 + 1e-9, (
            f"score {m.score} out of [-1, 1] for chunk_id={m.chunk_id}"
        )


@given(d=_DIM, extra=st.integers(min_value=0, max_value=8), data=st.data())
@settings(max_examples=60)
def test_when_query_equals_stored_vector_then_it_ranks_first_with_score_near_one(
    d, extra, data
):
    """Self-query invariant: the stored vector closest to the query scores ≈ 1.0 and ranks first."""
    target_row = data.draw(st.lists(_SAFE_FLOAT, min_size=d, max_size=d))
    target = np.array(target_row, dtype=float)
    assume(np.linalg.norm(target) > 1e-9)  # skip zero-norm edge case

    other_rows = _rows(extra, d, data) if extra > 0 else []
    all_rows = [target_row] + other_rows
    idx = _index_from_rows(all_rows)

    results = idx.search(target, k=len(all_rows))
    assert len(results) >= 1
    # The target vector (chunk_id=0, the first row) must be returned first.
    assert results[0].chunk_id == 0
    assert abs(results[0].score - 1.0) < 1e-5
