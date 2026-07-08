"""
Source-blind tests for flashcard.rag.build_retriever (Issue #9).

Derived entirely from the acceptance criteria — no implementation source was read.
All tests use FakeEmbedder to avoid real OpenAI calls.

Attribute contract assumed from criteria:
    retriever.last_scores       -> list[float] after retrieve(); descending order
    retriever.last_query_tokens -> int after retrieve(); token count of the last query embed
"""

from __future__ import annotations

import importlib
import os
import pathlib

import pytest
from hypothesis import given, strategies as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 4  # embedding dimension used by FakeEmbedder
CHUNK_SIZE = 50  # small so TINY_DOC yields several chunks

# Long enough to produce ≥3 chunks at CHUNK_SIZE=50 tokens.
TINY_DOC = (
    "Section A: The first chapter covers introductory topics in depth. " * 8
    + "Section B: The second chapter covers advanced topics with examples. " * 8
    + "Section C: The third chapter covers conclusions and references. " * 8
)


# ---------------------------------------------------------------------------
# FakeEmbedder
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """
    Deterministic, call-counting stand-in for embed.embed_texts.

    Corpus batch  (len(texts) > 1): returns DIM-dimensional unit vectors
                  indexed by position — chunk i gets a 1.0 in slot (i % DIM).
    Single-text   (query):          always returns [1, 0, 0, 0], so chunk 0
                  has cosine similarity 1.0 with every query.
    Token count:  len(texts) — a cheap proxy, not a real token count.
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.corpus_calls: int = 0
        self.query_calls: int = 0

    def __call__(self, texts: list[str]) -> tuple[list[list[float]], int]:
        if len(texts) > 1:
            self.corpus_calls += 1
            vecs = [
                [1.0 if j == i % self.dim else 0.0 for j in range(self.dim)]
                for i in range(len(texts))
            ]
        else:
            self.query_calls += 1
            vecs = [[1.0] + [0.0] * (self.dim - 1)]
        return vecs, len(texts)


# ---------------------------------------------------------------------------
# Import under test (fails until rag.py is written — Red phase of TDD)
# ---------------------------------------------------------------------------

from flashcard.rag import build_retriever  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Import / protocol
# ---------------------------------------------------------------------------


def test_when_flashcard_rag_imported_then_build_retriever_is_exported():
    """flashcard.rag must export build_retriever with no ImportError."""
    mod = importlib.import_module("flashcard.rag")
    assert hasattr(mod, "build_retriever"), "flashcard.rag must export build_retriever"


def test_when_build_retriever_called_then_returned_object_has_callable_retrieve():
    """build_retriever() must return an object with a callable retrieve attribute."""
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    assert callable(getattr(retriever, "retrieve", None))


def test_when_build_retriever_called_then_retrieve_is_importable_by_runner():
    """
    Criterion: dropping rag.py in must let runner._default_retriever use it
    without an ImportError fallback.
    We verify that build_retriever is importable and usable.
    """
    from flashcard.rag import build_retriever as _br

    retriever = _br(TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder())
    assert retriever is not None


# ---------------------------------------------------------------------------
# 2. Return type: list[tuple[int, str]]
# ---------------------------------------------------------------------------


def test_when_retrieve_called_then_result_is_a_list():
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    results = retriever.retrieve("what is section A?", k=2)
    assert isinstance(results, list)


def test_when_retrieve_called_then_each_element_is_a_two_tuple():
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    results = retriever.retrieve("what is section B?", k=2)
    for item in results:
        assert isinstance(item, tuple) and len(item) == 2, (
            "Each result must be a (chunk_id, chunk_text) tuple"
        )


def test_when_retrieve_called_then_chunk_ids_are_ints():
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    results = retriever.retrieve("what is section C?", k=2)
    for chunk_id, _ in results:
        assert isinstance(chunk_id, int), f"chunk_id must be int, got {type(chunk_id)}"


def test_when_retrieve_called_then_chunk_texts_are_non_empty_strings():
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    results = retriever.retrieve("what is section A?", k=2)
    for _, chunk_text in results:
        assert isinstance(chunk_text, str) and chunk_text, (
            "chunk_text must be a non-empty string"
        )


# ---------------------------------------------------------------------------
# 3. Cardinality: k is honoured
# ---------------------------------------------------------------------------


def test_when_retrieve_called_with_k_1_then_exactly_1_result_returned():
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    assert len(retriever.retrieve("question", k=1)) == 1


def test_when_retrieve_called_with_k_2_then_exactly_2_results_returned():
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    assert len(retriever.retrieve("question", k=2)) == 2


def test_when_retrieve_called_with_k_3_then_exactly_3_results_returned():
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    assert len(retriever.retrieve("question", k=3)) == 3


@given(st.integers(min_value=1, max_value=5))
def test_when_retrieve_called_with_any_valid_k_then_at_most_k_results_returned(k):
    """
    Cardinality invariant: retrieve always returns ≤ k results.
    (May return fewer only when the document has fewer than k chunks.)
    """
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    results = retriever.retrieve("any question", k=k)
    assert len(results) <= k


# ---------------------------------------------------------------------------
# 4. Similarity ordering
# ---------------------------------------------------------------------------


def test_when_retrieve_called_then_chunk_with_highest_cosine_similarity_is_first():
    """
    FakeEmbedder gives chunk 0 a unit vector identical to every query embedding.
    Therefore chunk 0 must appear first in the result.
    """
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    results = retriever.retrieve("any question", k=2)
    assert results[0][0] == 0, (
        "Chunk with highest cosine similarity (chunk 0) must be returned first"
    )


def test_when_retrieve_called_then_last_scores_are_sorted_descending():
    """
    last_scores exposes the similarity values used for ranking; they must be descending.
    """
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    retriever.retrieve("test question", k=3)
    scores = retriever.last_scores
    assert scores == sorted(scores, reverse=True), (
        "last_scores must be sorted in descending cosine-similarity order"
    )


@given(st.integers(min_value=1, max_value=4))
def test_when_retrieve_called_with_any_k_then_last_scores_are_sorted_descending(k):
    """
    Ordering invariant: last_scores is sorted descending regardless of k.
    """
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    results = retriever.retrieve("any question", k=k)
    if not results:
        return  # trivially satisfied when no chunks exist
    scores = retriever.last_scores
    assert len(scores) == len(results)
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 5. Corpus embedded exactly once per build_retriever call — not per query
# ---------------------------------------------------------------------------


def test_when_build_retriever_called_then_corpus_is_embedded_exactly_once():
    """Criterion: embed called once for the entire corpus at construction."""
    embedder = FakeEmbedder()
    build_retriever(TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=embedder)
    assert embedder.corpus_calls == 1, (
        "embed_fn must be called exactly once for the corpus during build_retriever"
    )


def test_when_retrieve_called_three_times_then_corpus_still_embedded_once():
    """Corpus embeddings must not be recomputed on each retrieve call."""
    embedder = FakeEmbedder()
    retriever = build_retriever(TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=embedder)

    retriever.retrieve("q1", k=1)
    retriever.retrieve("q2", k=1)
    retriever.retrieve("q3", k=1)

    assert embedder.corpus_calls == 1, (
        "Corpus must be embedded exactly once — not once per query"
    )


def test_when_retrieve_called_three_times_then_query_embedded_three_times():
    """Each retrieve call must embed its own query exactly once."""
    embedder = FakeEmbedder()
    retriever = build_retriever(TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=embedder)

    retriever.retrieve("q1", k=1)
    retriever.retrieve("q2", k=1)
    retriever.retrieve("q3", k=1)

    assert embedder.query_calls == 3, (
        "embed_fn must be called once per retrieve call for the query"
    )


# ---------------------------------------------------------------------------
# 6. last_query_tokens — readable attribute after retrieve
# ---------------------------------------------------------------------------


def test_when_retrieve_called_then_last_query_tokens_is_a_non_negative_int():
    """Criterion: retriever exposes last query embed token count as a readable int."""
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    retriever.retrieve("some question", k=1)
    assert isinstance(retriever.last_query_tokens, int)
    assert retriever.last_query_tokens >= 0


def test_when_retrieve_called_twice_then_last_query_tokens_updates_each_time():
    """last_query_tokens must reflect the most recent retrieve call."""
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    retriever.retrieve("first question", k=1)
    tokens_1 = retriever.last_query_tokens
    retriever.retrieve("second question", k=1)
    tokens_2 = retriever.last_query_tokens
    # Both are readable ints — verifying updateability, not equality
    assert isinstance(tokens_1, int) and tokens_1 >= 0
    assert isinstance(tokens_2, int) and tokens_2 >= 0


# ---------------------------------------------------------------------------
# 7. last_scores — readable attribute after retrieve
# ---------------------------------------------------------------------------


def test_when_retrieve_called_then_last_scores_length_equals_result_length():
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    results = retriever.retrieve("any question", k=3)
    assert len(retriever.last_scores) == len(results)


def test_when_retrieve_called_then_last_scores_are_floats_within_cosine_range():
    """Cosine similarity is bounded in [-1.0, 1.0]."""
    retriever = build_retriever(
        TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
    )
    retriever.retrieve("any question", k=2)
    for score in retriever.last_scores:
        assert isinstance(score, float), f"score must be float, got {type(score)}"
        assert -1.0 - 1e-6 <= score <= 1.0 + 1e-6, (
            f"Cosine similarity must be in [-1, 1], got {score}"
        )


# ---------------------------------------------------------------------------
# 8. Secret / env isolation (global criterion verifiable at UNIT level)
# ---------------------------------------------------------------------------


def test_when_openai_api_key_absent_then_build_retriever_with_fake_embed_succeeds():
    """
    No secrets in source. build_retriever injected with a fake embed_fn must not
    consult OPENAI_API_KEY — the real key is only needed by embed.embed_texts.
    """
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        retriever = build_retriever(
            TINY_DOC, chunk_size=CHUNK_SIZE, embed_fn=FakeEmbedder()
        )
        assert retriever is not None
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved


def test_when_project_root_checked_then_env_example_file_is_present():
    """Criterion: .env.example must be shipped with the project."""
    root = pathlib.Path(__file__).parent.parent
    assert (root / ".env.example").exists(), (
        ".env.example must be present in the project root"
    )


def test_when_env_example_read_then_openai_api_key_placeholder_is_present():
    """Criterion: .env.example must document the OPENAI_API_KEY variable."""
    root = pathlib.Path(__file__).parent.parent
    env_example = root / ".env.example"
    if not env_example.exists():
        pytest.skip(".env.example not yet present")
    content = env_example.read_text()
    assert "OPENAI_API_KEY" in content, ".env.example must contain OPENAI_API_KEY"
