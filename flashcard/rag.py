"""RAG retriever assembly — offline index build + online cosine retrieval.

Offline phase (``build_retriever``): chunk the document with
``chunking.chunk_document``, batch-embed the corpus once via the injectable
``embed_fn``, and persist the resulting matrix in a ``VectorIndex``.

Online phase (``EmbeddingRetriever.retrieve``): embed the question with the
same function, cosine-rank the index, return the top-k ``(chunk_id, text)``
pairs, and stash the similarity scores and query token count so the cost
engine and TUI can consume them.

Raising ``ImportError`` when no API key and no ``embed_fn`` is provided keeps
``runner._default_retriever`` able to degrade gracefully to the naive retriever
without a runner.py change (mirrors ``_safe_build_registry``).
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

from flashcard.chunking import chunk_document
from flashcard.embed import Embedder
from flashcard.index import VectorIndex

# Default overlap: mirrors chunking._DEFAULT_OVERLAP but clamped below chunk_size.
_DEFAULT_OVERLAP = 80

EmbedFn = Callable[[list[str]], tuple[Any, int]]


class EmbeddingRetriever:
    """Cosine retriever backed by a pre-built ``VectorIndex``.

    Attributes:
        last_scores:       Similarity scores from the last ``retrieve`` call,
                           sorted descending.
        last_query_tokens: Token count reported by ``embed_fn`` for the last
                           query.
    """

    def __init__(self, index: VectorIndex, embed_fn: EmbedFn) -> None:
        self._index = index
        self._embed_fn = embed_fn
        self.last_scores: list[float] = []
        self.last_query_tokens: int = 0

    def retrieve(self, question: str, k: int) -> list[tuple[int, str]]:
        """Embed *question*, cosine-rank the index, return top-*k* pairs."""
        vecs, token_count = self._embed_fn([question])
        self.last_query_tokens = token_count
        query_vec = np.array(vecs[0], dtype=np.float64)
        matches = self._index.search(query_vec, k)
        self.last_scores = [m.score for m in matches]
        return [(m.chunk_id, m.text) for m in matches]


def build_retriever(
    document: str,
    chunk_size: int,
    *,
    embed_fn: EmbedFn | None = None,
) -> EmbeddingRetriever:
    """Build an ``EmbeddingRetriever`` from *document* (offline phase).

    Chunks the document, batch-embeds the corpus **once**, and stores the
    resulting vectors in a ``VectorIndex``.  Subsequent ``retrieve`` calls
    only embed the query — never the corpus.

    Args:
        document:  Source document text.
        chunk_size: Maximum tokens per chunk (forwarded to ``chunk_document``).
        embed_fn:  Callable ``(texts: list[str]) -> (vectors, token_count)``
                   where *vectors* is an array-like of shape ``(n, d)`` and
                   *token_count* is an ``int``.  Defaults to the real OpenAI
                   ``text-embedding-3-small`` embedder.  When ``None`` and
                   ``OPENAI_API_KEY`` is absent, raises ``ImportError`` so
                   ``runner._default_retriever`` falls back to the naive
                   retriever gracefully.

    Returns:
        A ready-to-use ``EmbeddingRetriever`` with a built ``VectorIndex``.

    Raises:
        ImportError: When ``OPENAI_API_KEY`` is unset and no ``embed_fn`` is
                     injected.  This lets ``_default_retriever`` degrade to
                     ``_NaiveRetriever`` without modifying ``runner.py``.
    """
    if embed_fn is None:
        embed_fn = _real_embed_fn()
    chunks = chunk_document(
        document, chunk_size=chunk_size, overlap=_safe_overlap(chunk_size)
    )
    index = _build_index(chunks, embed_fn)
    return EmbeddingRetriever(index, embed_fn)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _real_embed_fn() -> EmbedFn:
    """Return the default OpenAI embed function, guarded by key presence."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ImportError(
            "flashcard.rag: OPENAI_API_KEY is not set. "
            "Provide the environment variable or inject embed_fn= for testing. "
            "runner._default_retriever will degrade to the naive retriever."
        )
    embedder = Embedder()

    def fn(texts: list[str]) -> tuple[np.ndarray, int]:
        result = embedder.embed_texts(texts)
        return result.vectors, result.token_count

    return fn


def _safe_overlap(chunk_size: int) -> int:
    """Largest overlap that satisfies ``0 <= overlap < chunk_size``."""
    return min(_DEFAULT_OVERLAP, max(0, chunk_size // 10))


def _build_index(chunks: list, embed_fn: EmbedFn) -> VectorIndex:
    """Embed *chunks* once and return a populated ``VectorIndex``."""
    index = VectorIndex()
    if not chunks:
        return index
    texts = [c.text for c in chunks]
    vecs, _ = embed_fn(texts)
    matrix = np.array(vecs, dtype=np.float64)
    chunk_ids = [c.chunk_id for c in chunks]
    index.add(matrix, chunk_ids, texts)
    return index
