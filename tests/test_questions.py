"""
Source-blind tests for flashcard.questions — issue #3.

Criteria covered (UNIT-verifiable per oracle):
  - A loader returns the built-in set by default and parses a JSON list at
    --questions-file PATH when provided, validating it is a non-empty list
    of strings.
  - Malformed JSON / wrong shape raises a clear, actionable error (not a raw
    traceback).

Criteria NOT covered (oracle: NOT VERIFIABLE):
  - "8–10 English" mixing narrow + spanning — subjective content check.
  - "At least one spanning question targets cross-referenced answer" — content.
  - TUI legibility, SOLID code quality — no concrete runtime assertion.
"""

import json
import os
import tempfile

import pytest
from hypothesis import given, settings, strategies as st

# The contract we expect from flashcard/questions.py:
#   BUILT_IN_QUESTIONS: list[str]   — the bundled set (8–10 items)
#   load_questions(path=None) -> list[str]   — returns BUILT_IN_QUESTIONS when
#                                              path is None; parses JSON otherwise
from flashcard.questions import BUILT_IN_QUESTIONS, load_questions


# ---------------------------------------------------------------------------
# Default built-in set
# ---------------------------------------------------------------------------


def test_when_no_path_is_given_built_in_questions_are_returned():
    assert load_questions() == BUILT_IN_QUESTIONS


def test_when_no_path_is_given_question_count_is_between_8_and_10():
    """Spec mandates exactly 8–10 bundled questions (issue #3 scope criterion)."""
    result = load_questions()
    assert 8 <= len(result) <= 10, (
        f"Expected 8–10 built-in questions, got {len(result)}"
    )


def test_when_no_path_is_given_every_question_is_a_non_empty_string():
    result = load_questions()
    for i, q in enumerate(result):
        assert isinstance(q, str), f"Question {i} is not a string: {q!r}"
        assert q.strip(), f"Question {i} is empty or whitespace-only"


def test_when_no_path_is_given_built_in_constant_matches_loader_output():
    """BUILT_IN_QUESTIONS and load_questions() must be the same object/value."""
    assert load_questions() == BUILT_IN_QUESTIONS


# ---------------------------------------------------------------------------
# JSON override — happy path
# ---------------------------------------------------------------------------


def test_when_valid_json_list_of_strings_is_given_those_questions_are_returned(
    tmp_path,
):
    questions = ["What is RAG?", "What is CAG?", "How does caching work?"]
    p = tmp_path / "questions.json"
    p.write_text(json.dumps(questions), encoding="utf-8")
    assert load_questions(p) == questions


def test_when_json_file_has_single_question_it_is_returned(tmp_path):
    """A single-element list is a valid non-empty list of strings."""
    p = tmp_path / "q.json"
    p.write_text(json.dumps(["Only one question?"]), encoding="utf-8")
    assert load_questions(p) == ["Only one question?"]


def test_when_questions_file_path_is_a_string_it_is_accepted(tmp_path):
    """--questions-file receives a CLI string; load_questions must accept str."""
    questions = ["Does a string path work?"]
    p = tmp_path / "questions.json"
    p.write_text(json.dumps(questions), encoding="utf-8")
    assert load_questions(str(p)) == questions


def test_when_json_file_has_many_questions_all_are_returned(tmp_path):
    """The loader must not silently truncate a large override list."""
    questions = [f"Question number {i}?" for i in range(20)]
    p = tmp_path / "big.json"
    p.write_text(json.dumps(questions), encoding="utf-8")
    assert load_questions(p) == questions


# ---------------------------------------------------------------------------
# Validation: empty list → error
# ---------------------------------------------------------------------------


def test_when_json_file_contains_empty_list_then_error_is_raised(tmp_path):
    """An empty list is not a valid non-empty question set."""
    p = tmp_path / "empty.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises((ValueError, SystemExit)):
        load_questions(p)


# ---------------------------------------------------------------------------
# Validation: wrong shape → error
# ---------------------------------------------------------------------------


def test_when_json_file_contains_a_dict_then_error_is_raised(tmp_path):
    """A JSON object (not a list) is the wrong shape."""
    p = tmp_path / "wrong.json"
    p.write_text('{"question": "What is RAG?"}', encoding="utf-8")
    with pytest.raises((ValueError, SystemExit)):
        load_questions(p)


def test_when_json_file_contains_list_of_integers_then_error_is_raised(tmp_path):
    """A list of non-strings is the wrong shape."""
    p = tmp_path / "nonstr.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises((ValueError, SystemExit)):
        load_questions(p)


def test_when_json_file_contains_list_with_null_item_then_error_is_raised(tmp_path):
    """A list mixing strings and null is the wrong shape."""
    p = tmp_path / "mixed.json"
    p.write_text('["valid question", null, "another"]', encoding="utf-8")
    with pytest.raises((ValueError, SystemExit)):
        load_questions(p)


def test_when_json_file_contains_bare_string_then_error_is_raised(tmp_path):
    """A JSON string (not a list) is the wrong shape."""
    p = tmp_path / "bare.json"
    p.write_text('"just a string"', encoding="utf-8")
    with pytest.raises((ValueError, SystemExit)):
        load_questions(p)


def test_when_json_file_contains_nested_list_of_lists_then_error_is_raised(tmp_path):
    """A list of lists (wrong element type) is the wrong shape."""
    p = tmp_path / "nested.json"
    p.write_text('[["a", "b"], ["c"]]', encoding="utf-8")
    with pytest.raises((ValueError, SystemExit)):
        load_questions(p)


# ---------------------------------------------------------------------------
# Malformed JSON → clear actionable error (not a raw traceback)
# ---------------------------------------------------------------------------


def test_when_json_file_is_malformed_then_an_error_is_raised(tmp_path):
    """Malformed JSON must raise rather than silently return garbage."""
    p = tmp_path / "bad.json"
    p.write_text("{not valid json at all", encoding="utf-8")
    with pytest.raises((ValueError, SystemExit)):
        load_questions(p)


def test_when_json_file_is_malformed_error_is_not_a_raw_json_decode_error(tmp_path):
    """The error for malformed JSON must be wrapped, not a raw JSONDecodeError.

    Rationale: criterion says 'a clear, actionable error (not a raw traceback)'.
    json.JSONDecodeError is a subclass of ValueError but leaks internal detail;
    load_questions must wrap it with a user-readable message.
    """
    p = tmp_path / "bad.json"
    p.write_text("[this is not json}", encoding="utf-8")
    try:
        load_questions(p)
    except json.JSONDecodeError:
        pytest.fail(
            "load_questions raised a raw json.JSONDecodeError — "
            "it must wrap it in a ValueError with an actionable message"
        )
    except (ValueError, SystemExit):
        pass  # correct: a wrapped, user-facing error
    else:
        pytest.fail("Expected an error for malformed JSON but none was raised")


def test_when_json_file_is_malformed_error_message_is_non_empty(tmp_path):
    """Error message for bad JSON must be a non-empty human-readable string."""
    p = tmp_path / "bad.json"
    p.write_text("{{{{invalid", encoding="utf-8")
    with pytest.raises((ValueError, SystemExit)) as exc_info:
        load_questions(p)
    assert str(exc_info.value).strip(), "Error message must not be empty"


def test_when_json_file_is_truncated_then_error_is_raised(tmp_path):
    """A partially-written (truncated) JSON file must raise."""
    p = tmp_path / "truncated.json"
    p.write_text('["question one", "question tw', encoding="utf-8")
    with pytest.raises((ValueError, SystemExit)):
        load_questions(p)


# ---------------------------------------------------------------------------
# Property-based: round-trip invariant
# ---------------------------------------------------------------------------


@given(st.lists(st.text(min_size=1), min_size=1, max_size=50))
@settings(max_examples=100)
def test_when_any_valid_list_of_strings_is_written_as_json_load_questions_returns_it(
    questions,
):
    """Round-trip invariant: any non-empty list of non-empty strings written as
    JSON and read back by load_questions must be returned unchanged.

    Strategy derived from the criterion's stated domain: 'non-empty list of strings'.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as fh:
        json.dump(questions, fh)
        path = fh.name
    try:
        result = load_questions(path)
        assert result == questions
    finally:
        os.unlink(path)
