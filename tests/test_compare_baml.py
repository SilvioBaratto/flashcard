"""
Source-blind example tests for issue #13, criteria 8 and 9.

Criterion 8: baml_src/compare.baml defines CompareAnswers(question, rag_answer, cag_answer)
on Gpt5Mini, returning a structured verdict
  { agree: bool, more_complete: "rag" | "cag" | "tie", note: string }.

Criterion 9: baml_client/ is regenerated via baml-cli generate; generated files must not be
hand-edited (i.e. they carry an auto-generation marker).

Tests are authored from the acceptance-criteria text and requirements.md only.
No implementation file was read during authoring.
"""

import re
from pathlib import Path


ROOT = Path(__file__).parent.parent
COMPARE_BAML = ROOT / "baml_src" / "compare.baml"
BAML_CLIENT_DIR = ROOT / "baml_client"


# ---------------------------------------------------------------------------
# Criterion 8 — CompareAnswers function definition in compare.baml
# ---------------------------------------------------------------------------


def _baml_text() -> str:
    """Return the content of compare.baml, failing clearly if the file is absent."""
    assert COMPARE_BAML.exists(), (
        "baml_src/compare.baml must exist; create it with the CompareAnswers function."
    )
    return COMPARE_BAML.read_text()


def _compare_answers_params(text: str) -> str:
    """Extract the raw parameter list of CompareAnswers, or return ''."""
    m = re.search(r"function\s+CompareAnswers\s*\(([^)]+)\)", text, re.DOTALL)
    return m.group(1) if m else ""


def test_when_compare_baml_exists_then_compare_answers_function_is_declared():
    """compare.baml must declare a function named CompareAnswers."""
    text = _baml_text()
    assert re.search(r"function\s+CompareAnswers\s*\(", text), (
        "compare.baml must contain: function CompareAnswers(...)"
    )


def test_when_compare_answers_is_declared_then_question_parameter_is_present():
    """CompareAnswers must accept a 'question' parameter."""
    text = _baml_text()
    params = _compare_answers_params(text)
    assert params, (
        "compare.baml: CompareAnswers must have a parenthesised parameter list"
    )
    assert "question" in params, (
        "compare.baml: CompareAnswers must include a 'question' parameter"
    )


def test_when_compare_answers_is_declared_then_rag_answer_parameter_is_present():
    """CompareAnswers must accept a 'rag_answer' parameter."""
    text = _baml_text()
    params = _compare_answers_params(text)
    assert params, (
        "compare.baml: CompareAnswers must have a parenthesised parameter list"
    )
    assert "rag_answer" in params, (
        "compare.baml: CompareAnswers must include a 'rag_answer' parameter"
    )


def test_when_compare_answers_is_declared_then_cag_answer_parameter_is_present():
    """CompareAnswers must accept a 'cag_answer' parameter."""
    text = _baml_text()
    params = _compare_answers_params(text)
    assert params, (
        "compare.baml: CompareAnswers must have a parenthesised parameter list"
    )
    assert "cag_answer" in params, (
        "compare.baml: CompareAnswers must include a 'cag_answer' parameter"
    )


def test_when_compare_answers_is_declared_then_it_targets_gpt5_mini_client():
    """CompareAnswers must reference the Gpt5Mini client."""
    text = _baml_text()
    assert "Gpt5Mini" in text, (
        "compare.baml: CompareAnswers must reference the Gpt5Mini client "
        "(gpt-5-mini is cheaper than the generation model)"
    )


def test_when_compare_answers_return_type_is_defined_then_agree_field_is_present():
    """The return type must contain an 'agree' field."""
    text = _baml_text()
    assert re.search(r"\bagree\b", text), (
        "compare.baml: the return type must include an 'agree' field (bool)"
    )


def test_when_compare_answers_return_type_is_defined_then_more_complete_field_is_present():
    """The return type must contain a 'more_complete' field."""
    text = _baml_text()
    assert re.search(r"\bmore_complete\b", text), (
        "compare.baml: the return type must include a 'more_complete' field"
    )


def test_when_more_complete_is_defined_then_rag_is_a_valid_enum_value():
    """'rag' must be listed as a valid value for more_complete."""
    text = _baml_text()
    assert '"rag"' in text or "'rag'" in text, (
        "compare.baml: 'rag' must appear as a valid more_complete value"
    )


def test_when_more_complete_is_defined_then_cag_is_a_valid_enum_value():
    """'cag' must be listed as a valid value for more_complete."""
    text = _baml_text()
    assert '"cag"' in text or "'cag'" in text, (
        "compare.baml: 'cag' must appear as a valid more_complete value"
    )


def test_when_more_complete_is_defined_then_tie_is_a_valid_enum_value():
    """'tie' must be listed as a valid value for more_complete."""
    text = _baml_text()
    assert '"tie"' in text or "'tie'" in text, (
        "compare.baml: 'tie' must appear as a valid more_complete value"
    )


def test_when_compare_answers_return_type_is_defined_then_note_field_is_present():
    """The return type must contain a 'note' field."""
    text = _baml_text()
    assert re.search(r"\bnote\b", text), (
        "compare.baml: the return type must include a 'note' field (string)"
    )


# ---------------------------------------------------------------------------
# Criterion 9 — baml_client/ is auto-generated (not hand-edited)
# ---------------------------------------------------------------------------


def test_when_baml_client_dir_exists_then_it_contains_python_files():
    """baml_client/ must exist and contain at least one .py file."""
    assert BAML_CLIENT_DIR.exists(), (
        "baml_client/ directory must exist; run: baml-cli generate"
    )
    py_files = list(BAML_CLIENT_DIR.rglob("*.py"))
    assert len(py_files) > 0, (
        "baml_client/ must contain generated .py files; run: baml-cli generate"
    )


def test_when_baml_client_python_files_are_read_then_each_carries_a_generation_marker():
    """Every .py file inside baml_client/ must carry an auto-generation marker."""
    _MARKERS = frozenset(
        (
            "do not edit",
            "do not modify",
            "generated by",
            "autogenerated",
            "auto-generated",
        )
    )
    py_files = list(BAML_CLIENT_DIR.rglob("*.py"))
    assert py_files, "baml_client/ must contain .py files (run: baml-cli generate)"
    for path in py_files:
        content = path.read_text(errors="ignore").lower()
        has_marker = any(marker in content for marker in _MARKERS)
        assert has_marker, (
            f"baml_client/{path.name}: must carry an auto-generation marker; "
            "hand-editing this directory is prohibited"
        )
