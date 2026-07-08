"""
Smoke and behavioral tests for the generated baml_client package (issue #1).

Tests derived from acceptance criteria — authored before implementation exists.
All tests in this module fail with ImportError until `baml-cli generate` is run.

API notes (BAML 0.222.0):
  b.request.AnswerCAG(...)  →  baml_py.HTTPRequest
  req.body.text()           →  JSON string with {"messages": [...], ...}

BAML merges consecutive same-role turns into one message with a content array,
so the CAG prompt renders as:
  msg[0] role=system  — stable instructions
  msg[1] role=user    — content[0]: Document text (stable prefix)
                        content[1]: Question text (varies per turn)

No network calls are made anywhere in this module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hypothesis import given, settings, strategies as st

# Ensure the local flashcard baml_client is used (not from another project)
_local_project_dir = Path(__file__).parent.parent
if str(_local_project_dir) not in sys.path:
    sys.path.insert(0, str(_local_project_dir))

from baml_client import b  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────


def _cag_body(document: str, question: str) -> dict:
    """Return parsed JSON body for AnswerCAG (no network call)."""
    req = b.request.AnswerCAG(document=document, question=question)
    return json.loads(req.body.text())


def _cag_body_text(document: str, question: str) -> str:
    """Return raw JSON body string for AnswerCAG (no network call)."""
    req = b.request.AnswerCAG(document=document, question=question)
    return req.body.text()


# ── import / smoke ────────────────────────────────────────────────────────────


def test_when_baml_client_imported_then_b_is_accessible():
    """baml_client/ is generated and `from baml_client import b` succeeds."""
    assert b is not None


def test_when_baml_client_imported_then_answer_rag_is_exposed():
    """The generated client exposes a callable AnswerRAG."""
    assert hasattr(b, "AnswerRAG"), "b must expose AnswerRAG"
    assert callable(b.AnswerRAG), "b.AnswerRAG must be callable"


def test_when_baml_client_imported_then_answer_cag_is_exposed():
    """The generated client exposes a callable AnswerCAG."""
    assert hasattr(b, "AnswerCAG"), "b must expose AnswerCAG"
    assert callable(b.AnswerCAG), "b.AnswerCAG must be callable"


# ── AnswerCAG prompt-structure (no live call) ─────────────────────────────────


def test_when_answer_cag_request_built_then_document_appears_before_question():
    """
    AnswerCAG places the document before the question in the rendered prompt.

    BAML renders the two _.role("user") turns into one user message whose
    content array has document at index 0 and question at index 1. The JSON
    body therefore serializes document text before question text — that ordering
    is the observable proxy for "document precedes question".
    """
    doc_anchor = "UNIQUE_DOC_ANCHOR_AABBCC_1234"
    q_anchor = "UNIQUE_Q_ANCHOR_XXYYZZ_5678"

    body_text = _cag_body_text(document=doc_anchor, question=q_anchor)

    doc_pos = body_text.find(doc_anchor)
    q_pos = body_text.find(q_anchor)

    assert doc_pos != -1, "document text must appear in the serialized request body"
    assert q_pos != -1, "question text must appear in the serialized request body"
    assert doc_pos < q_pos, (
        f"document (position {doc_pos}) must appear before "
        f"question (position {q_pos}) in the AnswerCAG request body"
    )


def test_when_answer_cag_request_built_then_document_content_precedes_question_in_user_message():
    """
    In the content array of the user message, the document content item
    appears at a lower index than the question content item.
    """
    doc_anchor = "DOCBLOCK_ANCHOR_MMNN"
    q_anchor = "QBLOCK_ANCHOR_PPQQ"

    body = _cag_body(document=doc_anchor, question=q_anchor)
    messages = body["messages"]

    user_messages = [m for m in messages if m["role"] == "user"]
    assert user_messages, "AnswerCAG request must contain at least one user message"

    # Find which content block holds the document and which holds the question
    doc_item_idx: int | None = None
    q_item_idx: int | None = None

    for msg in user_messages:
        content = msg["content"]
        if isinstance(content, str):
            # Single-string content: check if both anchors present (rare)
            if doc_anchor in content:
                doc_item_idx = 0
            if q_anchor in content:
                q_item_idx = 0
        elif isinstance(content, list):
            for idx, part in enumerate(content):
                text = part.get("text", "")
                if doc_anchor in text and doc_item_idx is None:
                    doc_item_idx = idx
                if q_anchor in text and q_item_idx is None:
                    q_item_idx = idx

    assert doc_item_idx is not None, "document anchor must appear in some content item"
    assert q_item_idx is not None, "question anchor must appear in some content item"
    assert doc_item_idx < q_item_idx, (
        f"document content (index {doc_item_idx}) must precede "
        f"question content (index {q_item_idx}) within the user message"
    )


def test_when_answer_cag_request_built_then_no_cache_control_marker_is_present():
    """
    AnswerCAG carries no cache_control marker — OpenAI caches the prefix
    automatically; there is no explicit marker or off-switch.
    """
    body_text = _cag_body_text(
        document="some document text", question="some question text"
    )
    assert "cache_control" not in body_text.lower(), (
        "AnswerCAG must not contain a cache_control marker in the request body"
    )


# ── property: ordering holds for all valid inputs ────────────────────────────


_SAFE_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-"


@given(
    document=st.text(min_size=6, max_size=300, alphabet=_SAFE_CHARS).filter(
        lambda s: "PDOC_" not in s and "PQ_" not in s and s.strip()
    ),
    question=st.text(min_size=3, max_size=80, alphabet=_SAFE_CHARS).filter(
        lambda s: "PDOC_" not in s and "PQ_" not in s and s.strip()
    ),
)
@settings(max_examples=25)
def test_when_answer_cag_built_with_any_valid_inputs_then_document_always_precedes_question(
    document: str,
    question: str,
) -> None:
    """
    Ordering invariant: for all valid (document, question) pairs the document
    text always appears before the question text in the serialized request body.

    Sentinel prefixes are chosen to be absent from generated text (filtered above)
    to prevent false-positive position matches.
    """
    doc_sentinel = f"PDOC_{document}_END"
    q_sentinel = f"PQ_{question}_END"

    body_text = _cag_body_text(document=doc_sentinel, question=q_sentinel)

    doc_pos = body_text.find(doc_sentinel)
    q_pos = body_text.find(q_sentinel)

    assert doc_pos != -1, "document sentinel must appear in the request body"
    assert q_pos != -1, "question sentinel must appear in the request body"
    assert doc_pos < q_pos, (
        f"document (pos {doc_pos}) must always precede question (pos {q_pos})"
    )
