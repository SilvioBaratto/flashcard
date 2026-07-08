"""OpenAI text-embedding-3-small wrapper — OpenAI-only, $0.02/MTok.

Provides batch corpus embedding and single-query embedding, both returning
float32 numpy vectors plus the token count the API reports (for exact cost
billing via ``flashcard.pricing.embed_cost``).

No cosine search here — that is ``index.py``'s job.
No API key hard-coded — pass a real ``openai.OpenAI`` client or inject a fake.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

EMBED_MODEL: str = "text-embedding-3-small"
EMBED_DIM: int = 1536  # default output dimension; used for the empty-input shape


@dataclass(frozen=True)
class EmbedResult:
    """Embedding output: a float32 numpy array plus the token count."""

    vectors: np.ndarray  # shape (n, d) for batch; (d,) for single query
    token_count: int


@runtime_checkable
class _EmbeddingData(Protocol):
    """Structural interface for an embedding datum."""

    index: int
    embedding: list[float]


@runtime_checkable
class _EmbeddingUsage(Protocol):
    """Structural interface for embedding response usage."""

    prompt_tokens: int


@runtime_checkable
class _EmbeddingResponse(Protocol):
    """Structural interface for embedding API response."""

    data: list[_EmbeddingData]
    usage: _EmbeddingUsage


@runtime_checkable
class _EmbeddingsClient(Protocol):
    """Structural interface for ``client.embeddings`` — accepts fakes in tests."""

    def create(self, *, model: str, input: list[str]) -> _EmbeddingResponse: ...  # noqa: A002


@runtime_checkable
class _OpenAIClient(Protocol):
    """Structural interface for OpenAI client."""

    embeddings: _EmbeddingsClient


class Embedder:
    """Thin wrapper around ``client.embeddings.create`` for ``EMBED_MODEL``.

    Args:
        client: An ``openai.OpenAI``-shaped object (or a test fake).  When
                ``None`` a real client is created using ``OPENAI_API_KEY``.
    """

    def __init__(self, client: _OpenAIClient | None = None) -> None:
        self._client = client or _default_client()

    def embed_texts(self, texts: list[str]) -> EmbedResult:
        """Return an ``(n, d)`` float32 matrix — one row per input text."""
        if not texts:
            return EmbedResult(
                vectors=np.empty((0, EMBED_DIM), dtype=np.float32),
                token_count=0,
            )
        response = self._client.embeddings.create(model=EMBED_MODEL, input=texts)
        vectors = np.array(
            [d.embedding for d in sorted(response.data, key=lambda d: d.index)],
            dtype=np.float32,
        )
        return EmbedResult(vectors=vectors, token_count=response.usage.prompt_tokens)

    def embed_query(self, text: str) -> EmbedResult:
        """Return a ``(d,)`` float32 vector for a single query string."""
        result = self.embed_texts([text])
        return EmbedResult(vectors=result.vectors[0], token_count=result.token_count)


# ---------------------------------------------------------------------------
# Module-level convenience functions (injectable via the ``client`` kwarg)
# ---------------------------------------------------------------------------


def embed_texts(
    texts: list[str], *, client: _OpenAIClient | None = None
) -> EmbedResult:
    """Batch-embed *texts*; see :class:`Embedder` for full docs."""
    return Embedder(client=client).embed_texts(texts)


def embed_query(
    text: str, *, client: _OpenAIClient | None = None
) -> EmbedResult:
    """Embed a single query *text*; see :class:`Embedder` for full docs."""
    return Embedder(client=client).embed_query(text)


# ---------------------------------------------------------------------------
# Default-client factory — key sourced from env, never hard-coded
# ---------------------------------------------------------------------------


def _default_client() -> _OpenAIClient:
    import openai  # noqa: PLC0415 (lazy import keeps tests free of openai)

    return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # type: ignore[return-value]
