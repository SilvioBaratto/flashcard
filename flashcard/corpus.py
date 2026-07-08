"""Built-in English knowledge source for the flashcard RAG-vs-CAG demo.

The bundled document is the *Meridian Cloud Platform Technical Reference Manual*
(fictional, ~19k tokens with o200k_base encoding, 15 cross-referenced sections).

SPANNING ANSWERS (for the question set to target — require combining two sections):

1. "What happens when an Enterprise customer exceeds their 5 TB storage quota,
   including the grace period behavior and the per-GB monthly overage rate?"
   → Section 3.2 (quota limit, notification schedule, 72-hour grace period)
     + Section 8.2 (Enterprise overage rate: $0.019/GB for first 10 TB,
       $0.016/GB beyond, with day-level proration and grace-period interaction).
   RAG's top-3 keyhole will retrieve one section but miss the other.

2. "What SLA credit percentage applies to an MCP-OBJ Standard outage lasting
   between 1 and 4 hours, and what is the process for submitting a credit claim?"
   → Section 3.7 (establishes the Regional Storage SLA and refers to Section 9.2)
     + Section 9.2 (credit table: 1–4 h outage ≈ 99.77–99.86% uptime → 10% credit;
       plus the full claim procedure with 30-day submission deadline).
"""

from __future__ import annotations

import importlib.resources
import pathlib


def load_corpus(path: str | None = None) -> str:
    """Return the knowledge-source text as a plain string.

    Args:
        path: When given, read the UTF-8 text file at this filesystem path.
              When ``None`` (default), return the built-in bundled corpus.

    Returns:
        The document text.  Both call forms always return ``str``.
    """
    if path is not None:
        return pathlib.Path(path).read_text(encoding="utf-8")
    return _load_built_in()


def _load_built_in() -> str:
    """Read the bundled corpus.txt via importlib.resources."""
    ref = importlib.resources.files("flashcard") / "data" / "corpus.txt"
    return ref.read_text(encoding="utf-8")


# Alias: tests and implementation notes may use the name ``load_document``.
# The runner contract uses ``load_corpus``; both names are exported.
load_document = load_corpus
