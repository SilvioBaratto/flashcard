"""
Tests for flashcard.embed — Issue #7: OpenAI text-embedding-3-small wrapper.

Source-blind: written against acceptance-criteria text only; no implementation
file has been read. All tests are in the Red phase — they will fail until a
conforming implementation is shipped.

Assumptions recorded here (derived from criterion text, not source):
  - flashcard.embed exposes embed_texts, embed_query, and EmbedResult.
  - EmbedResult has .vectors (numpy.ndarray) and .token_count (int).
  - Both functions accept an injectable `client` keyword argument (satisfying
    "constructor/param" — the client has a .embeddings.create(...) method).
  - response.usage.total_tokens carries the aggregated token count.
  - response.data is a list of objects with .embedding (list[float]) and
    .index (int) matching the OpenAI embeddings response shape.
  - text-embedding-3-small default output dimension is 1536.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Module under test — import fails in Red; that is correct.
# ---------------------------------------------------------------------------
from flashcard.embed import embed_query, embed_texts  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_DIM = 1536  # text-embedding-3-small default dimensionality
MODEL_ID = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Fake OpenAI client infrastructure
# ---------------------------------------------------------------------------


def _make_embedding_obj(idx: int, dim: int) -> MagicMock:
    """Return a fake OpenAI Embedding datum.

    Each position gets a unique sentinel value (float(idx + 1)) so
    ordering tests can detect transpositions without reading source.
    """
    obj = MagicMock()
    obj.embedding = [float(idx + 1)] * dim
    obj.index = idx
    return obj


class _FakeEmbeddings:
    """Mimics client.embeddings — the namespace the real SDK exposes."""

    def __init__(self, dim: int = EMBED_DIM, total_tokens: int = 42) -> None:
        self.dim = dim
        self.total_tokens = total_tokens
        self.call_count = 0
        self.last_model: str = ""
        self.last_input: list[str] = []

    def create(self, *, model: str, input: list[str]) -> MagicMock:  # noqa: A002
        self.call_count += 1
        self.last_model = model
        self.last_input = list(input)
        n = len(input)
        response = MagicMock()
        response.data = [_make_embedding_obj(i, self.dim) for i in range(n)]
        usage = MagicMock()
        # Both fields mirror the real OpenAI embeddings usage shape where
        # total_tokens == prompt_tokens (no output tokens for embeddings).
        usage.total_tokens = self.total_tokens
        usage.prompt_tokens = self.total_tokens
        response.usage = usage
        return response


class _FakeClient:
    """Injectable stand-in for openai.OpenAI(); no network, no key."""

    def __init__(self, dim: int = EMBED_DIM, total_tokens: int = 42) -> None:
        self.embeddings = _FakeEmbeddings(dim=dim, total_tokens=total_tokens)


# ---------------------------------------------------------------------------
# Example tests — embed_texts
# ---------------------------------------------------------------------------


def test_when_texts_are_embedded_then_model_is_text_embedding_3_small():
    client = _FakeClient()
    embed_texts(["hello", "world"], client=client)
    assert client.embeddings.last_model == MODEL_ID


def test_when_two_texts_are_embedded_then_matrix_shape_is_n_by_d():
    client = _FakeClient(dim=EMBED_DIM)
    result = embed_texts(["foo", "bar"], client=client)
    assert result.vectors.shape == (2, EMBED_DIM)


def test_when_texts_are_embedded_then_matrix_dtype_is_float32():
    client = _FakeClient()
    result = embed_texts(["a", "b"], client=client)
    assert result.vectors.dtype == np.float32


def test_when_texts_are_embedded_then_token_count_equals_usage_total_tokens():
    client = _FakeClient(total_tokens=99)
    result = embed_texts(["one text"], client=client)
    assert result.token_count == 99


def test_when_multiple_texts_are_embedded_then_single_api_call_is_made():
    """Criterion: batches into one … client.embeddings.create(…) call."""
    client = _FakeClient()
    embed_texts(["a", "b", "c", "d"], client=client)
    assert client.embeddings.call_count == 1


def test_when_texts_are_embedded_then_vectors_are_in_input_order():
    """Row i must correspond to texts[i].

    The fake assigns sentinel value float(i+1) per position, so row[0]
    is all 1.0, row[1] is all 2.0, row[2] is all 3.0.  Any reordering
    of the embeddings by the implementation will flip these values and
    cause the assertion to fail.
    """
    client = _FakeClient(dim=EMBED_DIM)
    result = embed_texts(["first", "second", "third"], client=client)
    assert result.vectors[0, 0] == pytest.approx(1.0)
    assert result.vectors[1, 0] == pytest.approx(2.0)
    assert result.vectors[2, 0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Example tests — empty input (no API call, zero-row matrix, zero tokens)
# ---------------------------------------------------------------------------


def test_when_texts_is_empty_then_no_api_call_is_made():
    """Criterion: Empty texts … without an API call."""
    client = _FakeClient()
    embed_texts([], client=client)
    assert client.embeddings.call_count == 0


def test_when_texts_is_empty_then_matrix_first_dimension_is_zero():
    """Criterion: returns an empty (0, d) matrix."""
    client = _FakeClient()
    result = embed_texts([], client=client)
    assert result.vectors.ndim == 2
    assert result.vectors.shape[0] == 0


def test_when_texts_is_empty_then_token_count_is_zero():
    """Criterion: token_count == 0."""
    client = _FakeClient()
    result = embed_texts([], client=client)
    assert result.token_count == 0


# ---------------------------------------------------------------------------
# Example tests — embed_query
# ---------------------------------------------------------------------------


def test_when_query_is_embedded_then_model_is_text_embedding_3_small():
    client = _FakeClient()
    embed_query("what is RAG?", client=client)
    assert client.embeddings.last_model == MODEL_ID


def test_when_query_is_embedded_then_vector_is_one_dimensional():
    """Criterion: returns its (d,) vector — a 1-D array, not a 2-D matrix."""
    client = _FakeClient(dim=EMBED_DIM)
    result = embed_query("what is CAG?", client=client)
    assert result.vectors.ndim == 1
    assert result.vectors.shape == (EMBED_DIM,)


def test_when_query_is_embedded_then_dtype_is_float32():
    client = _FakeClient()
    result = embed_query("test query", client=client)
    assert result.vectors.dtype == np.float32


def test_when_query_is_embedded_then_token_count_equals_usage_total_tokens():
    client = _FakeClient(total_tokens=7)
    result = embed_query("hello", client=client)
    assert result.token_count == 7


# ---------------------------------------------------------------------------
# Secret isolation — injectable client must not require OPENAI_API_KEY
# ---------------------------------------------------------------------------


def test_when_client_is_injected_then_openai_api_key_env_var_is_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Criterion: no secrets in source; key is read only from OPENAI_API_KEY env.

    When a client is injected the wrapper must not touch the env var, proving
    the key is never hard-coded and unit tests need no credentials.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = _FakeClient()
    result = embed_texts(["hello"], client=client)
    assert result is not None


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(st.lists(st.text(min_size=1), min_size=1, max_size=20))
def test_when_n_texts_are_embedded_then_output_row_count_equals_n(
    texts: list[str],
) -> None:
    """Ordering invariant: for all non-empty text lists, output row count == len(texts)."""
    client = _FakeClient()
    result = embed_texts(texts, client=client)
    assert result.vectors.shape[0] == len(texts)


@given(st.lists(st.text(min_size=1), min_size=1, max_size=10))
def test_when_any_texts_are_embedded_then_dtype_is_always_float32(
    texts: list[str],
) -> None:
    """dtype invariant: float32 regardless of text content or list length."""
    client = _FakeClient()
    result = embed_texts(texts, client=client)
    assert result.vectors.dtype == np.float32


@given(st.text(min_size=1))
def test_when_any_valid_query_is_embedded_then_result_is_1d_float32_with_non_negative_tokens(
    text: str,
) -> None:
    """Never-raises invariant: embed_query accepts any non-empty string and
    returns a 1-D float32 vector with a non-negative token count."""
    client = _FakeClient()
    result = embed_query(text, client=client)
    assert result.vectors.ndim == 1
    assert result.vectors.dtype == np.float32
    assert result.token_count >= 0


@given(st.lists(st.text(min_size=1), min_size=2, max_size=15))
@settings(max_examples=50)
def test_when_n_texts_are_embedded_then_output_matrix_is_2d(
    texts: list[str],
) -> None:
    """Structural invariant: embed_texts always returns a 2-D matrix."""
    client = _FakeClient()
    result = embed_texts(texts, client=client)
    assert result.vectors.ndim == 2


# ---------------------------------------------------------------------------
# Out-of-order API response test (requested in issue comment)
# ---------------------------------------------------------------------------


class _FakeEmbeddingsOutOfOrder(_FakeEmbeddings):
    """Fake that returns response.data in *reverse* index order to prove the
    implementation sorts by .index before stacking — not by wire order."""

    def create(self, *, model: str, input: list[str]) -> MagicMock:  # noqa: A002
        response = super().create(model=model, input=input)
        # Reverse the wire order; .index attributes remain 0, 1, 2, …
        response.data = list(reversed(response.data))
        return response


class _FakeClientOutOfOrder(_FakeClient):
    def __init__(self, dim: int = EMBED_DIM, total_tokens: int = 42) -> None:
        self.embeddings = _FakeEmbeddingsOutOfOrder(dim=dim, total_tokens=total_tokens)


def test_when_api_returns_embeddings_out_of_order_then_vectors_match_input_order():
    """Criterion: vectors are returned in input order (sort by response .index).

    The fake deliberately reverses the wire order so the implementation *must*
    sort by .index to pass.  Row 0 of the output must still equal the sentinel
    value for position 0 (1.0), not for the last position.
    """
    client = _FakeClientOutOfOrder(dim=EMBED_DIM)
    result = embed_texts(["first", "second", "third"], client=client)
    assert result.vectors[0, 0] == pytest.approx(1.0)
    assert result.vectors[1, 0] == pytest.approx(2.0)
    assert result.vectors[2, 0] == pytest.approx(3.0)
