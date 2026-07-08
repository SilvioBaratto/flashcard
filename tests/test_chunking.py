"""Tests for flashcard.chunking (issue #6).

Written source-blind against the acceptance criteria only. The implementation
does not exist at authoring time — all fixtures are derived from the criterion
text, never from production source.

Criteria covered:
  AC1  chunk_document splits on token boundaries using tiktoken o200k_base
  AC2  each Chunk carries chunk_id (0-based), text, and n_tokens; contiguous overlap
  AC3  invalid overlap (< 0 or >= chunk_size) raises ValueError
  AC4  empty/whitespace → empty list; document shorter than chunk_size → one chunk
  AC5  (property) no chunk exceeds chunk_size; non-overlapping spans reconstruct token stream
"""

from __future__ import annotations

import pytest
import tiktoken
from hypothesis import given, settings, strategies as st

from flashcard.chunking import chunk_document

_ENC = tiktoken.get_encoding("o200k_base")


def _encode(text: str) -> list[int]:
    return _ENC.encode(text)


# ---------------------------------------------------------------------------
# AC1 — splits on token boundaries using tiktoken o200k_base
# ---------------------------------------------------------------------------


def test_when_chunk_document_is_called_then_each_chunk_n_tokens_matches_o200k_base():
    """n_tokens on each Chunk equals len(o200k_base.encode(chunk.text))."""
    text = "The quick brown fox jumps over the lazy dog. " * 60
    chunks = chunk_document(text, chunk_size=50, overlap=5)
    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert chunk.n_tokens == len(_encode(chunk.text))


# ---------------------------------------------------------------------------
# AC2 — chunk_id is 0-based and sequential; text and n_tokens are present
# ---------------------------------------------------------------------------


def test_when_chunk_document_is_called_then_chunk_ids_are_zero_based_and_sequential():
    """chunk_id is 0 for the first chunk and increments by 1 across the list."""
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa. " * 80
    chunks = chunk_document(text, chunk_size=30, overlap=3)
    assert len(chunks) > 1, "need multiple chunks for this test"
    for expected_id, chunk in enumerate(chunks):
        assert chunk.chunk_id == expected_id


def test_when_chunk_document_is_called_then_every_chunk_has_non_empty_text_and_positive_n_tokens():
    """Each Chunk exposes a non-empty text str and a positive integer n_tokens."""
    text = "word " * 200
    chunks = chunk_document(text, chunk_size=40, overlap=4)
    for chunk in chunks:
        assert isinstance(chunk.text, str)
        assert len(chunk.text) > 0
        assert isinstance(chunk.n_tokens, int)
        assert chunk.n_tokens > 0


def test_when_chunk_document_is_called_then_consecutive_chunks_share_overlap_tokens():
    """The last `overlap` token ids of chunk[i] equal the first `overlap` ids of chunk[i+1]."""
    text = "word " * 400
    chunk_size = 50
    overlap = 10
    chunks = chunk_document(text, chunk_size=chunk_size, overlap=overlap)
    assert len(chunks) > 1, "need multiple chunks for overlap check"
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_ids = _encode(prev.text)
        next_ids = _encode(nxt.text)
        assert prev_ids[-overlap:] == next_ids[:overlap], (
            f"chunk {prev.chunk_id}→{nxt.chunk_id}: trailing {overlap} ids "
            f"of prev do not match leading {overlap} ids of next"
        )


# ---------------------------------------------------------------------------
# AC3 — invalid overlap raises ValueError
# ---------------------------------------------------------------------------


def test_when_overlap_is_negative_then_value_error_is_raised():
    with pytest.raises(ValueError):
        chunk_document("some text here", chunk_size=100, overlap=-1)


def test_when_overlap_equals_chunk_size_then_value_error_is_raised():
    with pytest.raises(ValueError):
        chunk_document("some text here", chunk_size=100, overlap=100)


def test_when_overlap_exceeds_chunk_size_then_value_error_is_raised():
    with pytest.raises(ValueError):
        chunk_document("some text here", chunk_size=100, overlap=200)


def test_when_overlap_is_zero_then_no_error_is_raised():
    """overlap=0 is valid (lower bound of 0 <= overlap < chunk_size)."""
    text = "word " * 100
    chunks = chunk_document(text, chunk_size=40, overlap=0)
    assert isinstance(chunks, list)


def test_when_overlap_is_one_less_than_chunk_size_then_no_error_is_raised():
    """overlap = chunk_size - 1 is the maximum valid overlap."""
    text = "word " * 200
    chunks = chunk_document(text, chunk_size=20, overlap=19)
    assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# AC4 — empty/whitespace → empty list; short doc → exactly one chunk
# ---------------------------------------------------------------------------


def test_when_input_is_empty_string_then_empty_list_is_returned():
    result = chunk_document("", chunk_size=800, overlap=80)
    assert result == []


def test_when_input_is_whitespace_only_then_empty_list_is_returned():
    result = chunk_document("   \n\t\n  ", chunk_size=800, overlap=80)
    assert result == []


def test_when_document_has_fewer_tokens_than_chunk_size_then_exactly_one_chunk_is_returned():
    """A short document (<<chunk_size tokens) yields a single Chunk with chunk_id=0."""
    short = "This is a very short sentence."
    # Verify assumption: well under 800 tokens
    assert len(_encode(short)) < 800, "test fixture must be shorter than chunk_size"
    chunks = chunk_document(short, chunk_size=800, overlap=80)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == 0


def test_when_single_chunk_document_then_chunk_text_contains_the_source_content():
    """The single chunk's text must include the source content."""
    short = "unique_marker_phrase_here"
    chunks = chunk_document(short, chunk_size=800, overlap=80)
    assert len(chunks) == 1
    assert "unique_marker_phrase_here" in chunks[0].text


# ---------------------------------------------------------------------------
# AC5 — property: no chunk exceeds chunk_size; reconstruction from non-overlapping spans
# ---------------------------------------------------------------------------

# Safe strategy: ASCII text with no zero-width or surrogate characters avoids
# the rare case where tiktoken cannot round-trip a decoded window.
_SAFE_TEXT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?",
    min_size=1,
).filter(lambda t: len(_encode(t)) >= 1)

_CHUNK_SIZE = st.integers(min_value=10, max_value=150)


@given(_SAFE_TEXT, _CHUNK_SIZE)
@settings(max_examples=60)
def test_when_any_valid_input_is_chunked_then_no_chunk_exceeds_chunk_size_tokens(
    text: str, chunk_size: int
) -> None:
    """Invariant: for all valid inputs, chunk.n_tokens <= chunk_size for every chunk."""
    chunks = chunk_document(text, chunk_size=chunk_size, overlap=0)
    for chunk in chunks:
        assert chunk.n_tokens <= chunk_size, (
            f"chunk {chunk.chunk_id} has {chunk.n_tokens} tokens, exceeds chunk_size={chunk_size}"
        )


@given(_SAFE_TEXT, _CHUNK_SIZE)
@settings(max_examples=60)
def test_when_non_overlapping_spans_are_concatenated_then_original_token_stream_is_reconstructed(
    text: str, chunk_size: int
) -> None:
    """Invariant (overlap=0): concatenating all chunks' token ids reconstructs the original stream.

    With overlap=0, each chunk is fully non-overlapping. Their token ids concatenated
    must equal tiktoken.o200k_base.encode(original_text).
    """
    original_ids = _encode(text)
    if not original_ids:
        return

    chunks = chunk_document(text, chunk_size=chunk_size, overlap=0)
    if not chunks:
        return  # consistent with empty/whitespace criterion — not this property's domain

    reconstructed: list[int] = []
    for chunk in chunks:
        reconstructed.extend(_encode(chunk.text))

    assert reconstructed == original_ids, (
        f"reconstruction mismatch: original has {len(original_ids)} token ids, "
        f"reconstructed has {len(reconstructed)} token ids"
    )
