"""In-process numpy vector index: cosine top-k retrieval.

No external vector DB — a single L2-normalised numpy matrix and a
vectorised dot-product rank. Pure numpy; no OpenAI calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Match:
    """A ranked retrieval result."""

    chunk_id: int
    text: str
    score: float


class VectorIndex:
    """In-memory cosine-similarity index backed by an (n, d) numpy matrix.

    Vectors are L2-normalised at add-time so search is a single matmul.
    Zero-norm vectors are kept as zero (score 0.0 for any query).
    Scores are clipped to [-1, 1] to absorb float rounding.
    """

    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None
        self._chunk_ids: list[int] = []
        self._texts: list[str] = []

    def add(self, vectors: np.ndarray, chunk_ids: list[int], texts: list[str]) -> None:
        """Store normalised vectors; raise ValueError on length mismatch."""
        n = len(vectors)
        if len(chunk_ids) != n or len(texts) != n:
            raise ValueError(
                f"lengths differ: vectors={n}, chunk_ids={len(chunk_ids)}, "
                f"texts={len(texts)}"
            )
        self._matrix = _normalise_rows(vectors.astype(np.float64))
        self._chunk_ids = list(chunk_ids)
        self._texts = list(texts)

    def search(self, query: np.ndarray, k: int) -> list[Match]:
        """Return up to *k* Matches sorted by descending cosine similarity."""
        if k <= 0 or self._matrix is None:
            return []
        scores = self._cosine_scores(query)
        indices = self._top_k_indices(scores, k)
        return [
            Match(self._chunk_ids[i], self._texts[i], float(scores[i]))
            for i in indices
        ]

    def _cosine_scores(self, query: np.ndarray) -> np.ndarray:
        q_norm = _normalise_rows(query.reshape(1, -1).astype(np.float64))[0]
        return np.clip(self._matrix @ q_norm, -1.0, 1.0)

    def _top_k_indices(self, scores: np.ndarray, k: int) -> np.ndarray:
        top_k = min(k, len(scores))
        return np.argsort(-scores, kind="stable")[:top_k]


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row; zero-norm rows remain at zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)
