"""Token-based document chunker for the flashcard RAG pipeline.

Splits a text into overlapping windows of ``chunk_size`` tokens using
tiktoken's o200k_base encoding — matching the cost-guard tokenizer so
chunk-size math is consistent across the application.
"""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

_ENCODING = "o200k_base"
_DEFAULT_OVERLAP = 80


@dataclass(frozen=True)
class Chunk:
    """Immutable token-bounded slice of a source document."""

    chunk_id: int
    text: str
    n_tokens: int


def chunk_document(
    text: str,
    *,
    chunk_size: int = 800,
    overlap: int = _DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split *text* into token-bounded chunks with a sliding overlap.

    Args:
        text: Source document text.
        chunk_size: Maximum tokens per chunk (default 800).
        overlap: Tokens shared between consecutive chunks.
                 Must satisfy ``0 <= overlap < chunk_size``.

    Returns:
        A list of :class:`Chunk` objects, empty when *text* is empty or
        whitespace-only.

    Raises:
        ValueError: When *overlap* is outside ``[0, chunk_size)``.
    """
    _validate(chunk_size, overlap)
    if not text or not text.strip():
        return []
    enc = tiktoken.get_encoding(_ENCODING)
    ids = enc.encode(text)
    return _build_chunks(enc, ids, chunk_size, overlap)


def _validate(chunk_size: int, overlap: int) -> None:
    if not (0 <= overlap < chunk_size):
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < chunk_size; "
            f"got overlap={overlap}, chunk_size={chunk_size}"
        )


def _build_chunks(
    enc: tiktoken.Encoding,
    ids: list[int],
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    stride = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < len(ids):
        window = ids[start : start + chunk_size]
        # decode_bytes avoids U+FFFD replacement for tokens that straddle
        # multi-byte UTF-8 boundaries; n_tokens uses the id count directly.
        text = enc.decode_bytes(window).decode("utf-8", errors="replace")
        chunks.append(Chunk(chunk_id=len(chunks), text=text, n_tokens=len(window)))
        # Stop when the current window covers all remaining tokens — the next
        # chunk's content would be entirely within this chunk's overlap region.
        if len(ids) <= start + chunk_size:
            break
        start += stride
    return chunks
