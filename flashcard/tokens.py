"""Shared tiktoken helper — single encoding source for the flashcard package.

Caches the encoding object with ``lru_cache`` so the expensive first call
(which loads the full BPE vocabulary) is paid once per process, not once per
turn.  All callers that need to tokenise text should import from here to
prevent silent encoding drift if the family changes in the future.
"""

from __future__ import annotations

import functools

import tiktoken

DEFAULT_ENCODING = "o200k_base"  # gpt-5.x / gpt-4o family


@functools.lru_cache(maxsize=4)
def _enc(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding: str = DEFAULT_ENCODING) -> int:
    """Count BPE tokens in *text* using the given tiktoken encoding."""
    return len(_enc(encoding).encode(text))
