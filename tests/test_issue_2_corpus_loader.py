"""
Source-blind tests for issue #2: built-in English corpus + --doc loader.

Criteria exercised (oracle tier in brackets):
  [UNIT] A loader returns the built-in document by default and loads the file
         at --doc PATH when provided (path override), returning the same object
         type either way.
  [UNIT] Unit tests cover: built-in doc is non-empty and roughly within the
         token band (use tiktoken), the --doc override reads an external file,
         and the cross-referenced span exists.

All tests are authored from the acceptance criteria only (TDD red phase).
No implementation source was read — flashcard/corpus.py does not exist yet;
these tests all fail with ImportError until it is created.

No network calls are made anywhere in this module.
"""

from __future__ import annotations

import os
import tempfile

from hypothesis import given, settings, strategies as st

# Primary gate: fails with ImportError until flashcard/corpus.py is created.
# The function name `load_document` is derived from the spec phrase:
#   "A loader returns the built-in document by default and loads the file
#    at --doc PATH when provided."
# The CLI flag is --doc, so the keyword parameter is `path`.
from flashcard.corpus import load_document  # noqa: E402


# ── built-in document: non-empty ──────────────────────────────────────────────


def test_when_built_in_doc_loaded_then_it_is_non_empty():
    """
    Criterion: 'built-in doc is non-empty'.
    load_document() with no arguments must return a non-empty, truthy value.
    """
    doc = load_document()
    assert doc, "load_document() returned an empty or falsy value"
    assert len(str(doc)) > 0, "str(load_document()) must contain at least one character"


# ── built-in document: token band ─────────────────────────────────────────────


def test_when_built_in_doc_loaded_then_token_count_is_within_band():
    """
    Criterion: 'roughly 15–25k tokens'.
    We apply a ±20 % slack (12k–30k) to tolerate minor encoder variations.

    Uses o200k_base — the tiktoken encoding for the gpt-5.x / gpt-4o model
    family (tiktoken may not yet know the literal model id "gpt-5.2", so we
    pin the encoding explicitly as noted in the issue comment).  The cost
    estimate in the cost guard uses the same encoding, so the token count
    here and the estimate always agree.
    """
    import tiktoken

    doc = load_document()
    enc = tiktoken.get_encoding("o200k_base")
    n_tokens = len(enc.encode(str(doc)))
    assert n_tokens >= 12_000, (
        f"Built-in document is too short: {n_tokens} tokens. "
        "Criterion requires roughly 15–25k tokens (floor set at 12k with ±20 % slack)."
    )
    assert n_tokens <= 30_000, (
        f"Built-in document is too long: {n_tokens} tokens. "
        "Criterion requires roughly 15–25k tokens (ceiling set at 30k with ±20 % slack)."
    )


# ── built-in document: cross-referenced span ──────────────────────────────────


def test_when_built_in_doc_loaded_then_cross_referenced_span_exists():
    """
    Criterion: 'at least one answer spans two distant sections (documented in a
    short comment so the question set can target it)'.

    Observable proxy at the text level: the document must contain at least one
    explicit cross-reference phrase linking two named sections (e.g.
    'see section', 'refer to section', 'as described in section').
    Such phrasing is the canonical marker in product manuals / service agreements
    that signals an intentional spanning answer.
    """
    doc = load_document()
    text = str(doc).lower()
    cross_ref_markers = [
        "see section",
        "refer to section",
        "as described in section",
        "see also section",
        "described in section",
        "detailed in section",
        "outlined in section",
    ]
    assert any(marker in text for marker in cross_ref_markers), (
        "Built-in document must contain at least one cross-reference phrase between "
        "sections (e.g. 'see section X'). This is the intentional spanning-answer "
        "teaching beat required by the acceptance criteria."
    )


# ── --doc PATH override: reads external file ──────────────────────────────────


def test_when_doc_path_provided_then_external_file_content_is_returned():
    """
    Criterion: 'the --doc override reads an external file'.
    A unique sentinel written to a temp file must appear in the returned document.
    """
    sentinel = "EXTERNAL_DOC_SENTINEL_7A3F_ISSUE2"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(sentinel + "\nExternal document body for loader test.")
        path = tmp.name
    try:
        doc = load_document(path=path)
        assert sentinel in str(doc), (
            "load_document(path=...) must return the content of the file at that path; "
            f"expected sentinel '{sentinel}' not found in returned document."
        )
    finally:
        os.unlink(path)


def test_when_doc_path_provided_then_built_in_doc_is_not_returned():
    """
    The path override must NOT silently return the built-in document.
    We write content that cannot appear in the built-in corpus and verify it
    is present, which also guarantees the built-in was not substituted.
    """
    exclusive_content = "OVERRIDE_EXCLUSIVE_CONTENT_BB99_NOTINBUILTIN"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(exclusive_content)
        path = tmp.name
    try:
        doc = load_document(path=path)
        assert exclusive_content in str(doc), (
            "load_document(path=...) must return the external file's content, "
            "not the built-in document."
        )
    finally:
        os.unlink(path)


# ── return-type parity ────────────────────────────────────────────────────────


def test_when_doc_loaded_by_path_then_type_matches_built_in_type():
    """
    Criterion: 'returning the same object type either way'.
    Both call forms — load_document() and load_document(path=...) — must return
    the same Python type, so downstream callers need no conditional handling.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write("Any document text for type-parity test.")
        path = tmp.name
    try:
        built_in = load_document()
        from_path = load_document(path=path)
        assert type(from_path) is type(built_in), (
            f"load_document() returns {type(built_in).__name__!r} but "
            f"load_document(path=...) returns {type(from_path).__name__!r}. "
            "Both must return the same type."
        )
    finally:
        os.unlink(path)


# ── property: type invariant holds for all valid path inputs ──────────────────

_SAFE_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?\n-"
)


@given(
    content=st.text(min_size=1, max_size=3000, alphabet=_SAFE_ALPHABET).filter(
        lambda s: s.strip()
    )
)
@settings(max_examples=25)
def test_when_any_text_file_loaded_via_path_then_type_matches_built_in(
    content: str,
) -> None:
    """
    Invariant: load_document(path=P) always returns the same type as
    load_document(), regardless of what text the file contains.

    Derived from criterion: 'returning the same object type either way'.
    Input strategy: any non-empty printable text up to 3000 chars.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        path = tmp.name
    try:
        built_in = load_document()
        from_path = load_document(path=path)
        assert type(from_path) is type(built_in), (
            f"Type mismatch: load_document() → {type(built_in).__name__!r}, "
            f"load_document(path=...) → {type(from_path).__name__!r}"
        )
    finally:
        os.unlink(path)


# ── same-encoding invariant (Issue #5 scope criterion) ───────────────────────


def test_when_corpus_counted_by_cost_guard_then_result_matches_o200k_base():
    """Same-encoding invariant: corpus.py and cost_guard must both use o200k_base.

    This test makes the invariant documented in the token-band test's docstring
    above ('the cost estimate in the cost guard uses the same encoding, so the
    token count here and the estimate always agree') explicitly runtime-verifiable.

    Chosen behaviour: cost_guard.count_tokens(corpus) must equal
    tiktoken.get_encoding('o200k_base').encode(corpus) in length.  If cost_guard
    uses cl100k_base or any other encoding, this assertion fails.
    """
    import tiktoken

    from flashcard.cost_guard import count_tokens

    corpus = str(load_document())
    enc = tiktoken.get_encoding("o200k_base")
    direct = len(enc.encode(corpus))
    guard = count_tokens(corpus)

    assert guard == direct, (
        f"cost_guard.count_tokens returned {guard} but direct o200k_base count is "
        f"{direct}.  corpus.py (via test_when_built_in_doc_loaded_then_token_count_is"
        f"_within_band) and cost_guard.count_tokens must use the same encoding "
        f"(o200k_base) so that cost estimates match the measured document size."
    )
