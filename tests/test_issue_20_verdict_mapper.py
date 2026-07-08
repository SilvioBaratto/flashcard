"""Tests for issue #20: gate the CompareAnswers judge and map its Verdict
to the recall markers.

Source-blind authoring: all tests derived exclusively from acceptance criteria
and requirements.md. No implementation files were read.

Criteria covered (per oracle classification):
  [UNIT] Both pipelines share the same model and max_tokens — apples-to-apples.
  [UNIT] No secrets in source; OPENAI_API_KEY from env only; .env.example shipped.
  [UNIT] A pure mapper turns a Verdict (agree / more_complete / note) into
         (rag_missed: bool, cag_full_recall: bool).
  [UNIT] With --judge on, more_complete=="cag" lights ⚠ missed on RAG and
         ✓ full recall on CAG.
  [UNIT] _read_verdict (or its replacement) reads real Verdict fields, not the
         non-existent rag_missed / cag_full_recall.

Criteria skipped (oracle: NOT VERIFIABLE or T3):
  Network-hiccup resilience, TUI legibility, price verification at build time,
  --judge-off display, real-API numbers, cost-guard confirmation.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Minimal Verdict stand-in — derived from the spec, not from baml_client.
#
# Spec: "a Verdict (agree / more_complete / note)".
# Deliberately has NO rag_missed or cag_full_recall — those fields do not exist
# on the real BAML-generated Verdict type; criterion 5 pins this down.
# ---------------------------------------------------------------------------


@dataclass
class _Verdict:
    agree: bool
    more_complete: str  # "rag" | "cag" | "equal"
    note: str


# ---------------------------------------------------------------------------
# Import helpers — probe likely locations; fail loudly in the Red phase.
# ---------------------------------------------------------------------------


def _import_mapper():
    """Return the pure Verdict → (rag_missed: bool, cag_full_recall: bool) mapper."""
    import importlib

    for module_path, fn_name in [
        ("flashcard.tui", "verdict_to_markers"),
        ("flashcard.tui", "_read_verdict"),
        ("flashcard.runner", "verdict_to_markers"),
    ]:
        try:
            mod = importlib.import_module(module_path)
            fn = getattr(mod, fn_name, None)
            if fn is not None:
                return fn
        except ImportError:
            pass
    pytest.fail(
        "No pure Verdict mapper found. "
        "Implement flashcard.tui.verdict_to_markers(verdict) → (rag_missed: bool, cag_full_recall: bool). "
        "Alternatively, expose _read_verdict with the same signature."
    )


def _import_marker_formatter():
    """Return a function (rag_missed, cag_full_recall) → (rag_marker: str, cag_marker: str)."""
    import importlib

    for module_path in ("flashcard.tui",):
        try:
            mod = importlib.import_module(module_path)
            for fn_name in (
                "format_verdict_markers",
                "verdict_panel_markers",
                "recall_markers",
            ):
                fn = getattr(mod, fn_name, None)
                if fn is not None:
                    return fn
        except ImportError:
            pass
    pytest.fail(
        "No marker-formatter found in flashcard.tui. "
        "Implement one of: format_verdict_markers, verdict_panel_markers, recall_markers. "
        "Signature: (rag_missed: bool, cag_full_recall: bool) → (rag_marker: str, cag_marker: str)."
    )


# ---------------------------------------------------------------------------
# Criterion: both pipelines share ONE model and ONE max_tokens (apples-to-apples)
# ---------------------------------------------------------------------------


class TestRunConfigApplesToApples:
    """RunConfig carries a single model and max_tokens used by both pipelines."""

    def test_when_run_config_created_then_model_attribute_is_accessible(self):
        from flashcard.runner import RunConfig

        cfg = RunConfig(model="gpt-5.2", max_tokens=400)
        assert cfg.model is not None

    def test_when_run_config_created_then_max_tokens_attribute_equals_given_value(self):
        from flashcard.runner import RunConfig

        cfg = RunConfig(model="gpt-5.2", max_tokens=400)
        assert cfg.max_tokens == 400

    def test_when_run_config_created_then_no_side_specific_model_fields_exist(self):
        """One model field, not two — no rag_model / cag_model — enforces apples-to-apples."""
        from flashcard.runner import RunConfig

        cfg = RunConfig(model="gpt-5.2", max_tokens=400)
        assert not hasattr(cfg, "rag_model"), (
            "RunConfig.rag_model must not exist; a single .model field is shared by both pipelines"
        )
        assert not hasattr(cfg, "cag_model"), (
            "RunConfig.cag_model must not exist; a single .model field is shared by both pipelines"
        )

    def test_when_run_config_created_then_no_side_specific_max_tokens_fields_exist(
        self,
    ):
        """One max_tokens field, not two — both sides are capped identically."""
        from flashcard.runner import RunConfig

        cfg = RunConfig(model="gpt-5.2", max_tokens=400)
        assert not hasattr(cfg, "rag_max_tokens"), (
            "RunConfig.rag_max_tokens must not exist; .max_tokens is shared by both sides"
        )
        assert not hasattr(cfg, "cag_max_tokens"), (
            "RunConfig.cag_max_tokens must not exist; .max_tokens is shared by both sides"
        )


# ---------------------------------------------------------------------------
# Criterion: no secrets in source; OPENAI_API_KEY from env; .env.example present
# ---------------------------------------------------------------------------


class TestSecretsPolicy:
    def test_when_project_root_inspected_then_env_example_file_is_present(self):
        """A .env.example file must be shipped alongside the project."""
        root = pathlib.Path(__file__).parent.parent
        assert (root / ".env.example").exists(), (
            ".env.example not found at project root — it must be shipped so users can "
            "configure OPENAI_API_KEY safely without exposing their real key."
        )

    def test_when_python_source_files_scanned_then_no_hardcoded_openai_key_pattern_found(
        self,
    ):
        """No sk-... key literal may appear in any .py source file."""
        key_pattern = re.compile(r"sk-[A-Za-z0-9\-_]{20,}")
        src_root = pathlib.Path(__file__).parent.parent / "flashcard"
        violations: list[str] = []
        for py_file in src_root.rglob("*.py"):
            text = py_file.read_text(encoding="utf-8")
            if key_pattern.search(text):
                violations.append(str(py_file))
        assert not violations, (
            f"Hardcoded API key pattern (sk-...) found in: {violations}. "
            "Read the key only from os.environ['OPENAI_API_KEY'] or via python-dotenv."
        )


# ---------------------------------------------------------------------------
# Criterion: pure mapper Verdict → (rag_missed: bool, cag_full_recall: bool)
# ---------------------------------------------------------------------------


class TestVerdictMapper:
    """The mapper is a pure function that converts a Verdict to two booleans."""

    def test_when_more_complete_is_cag_then_rag_missed_is_true(self):
        mapper = _import_mapper()
        verdict = _Verdict(
            agree=False, more_complete="cag", note="CAG answered more fully"
        )
        rag_missed, _ = mapper(verdict)
        assert rag_missed is True, (
            "more_complete=='cag' means RAG's keyhole missed part of the answer; "
            "rag_missed must be True"
        )

    def test_when_more_complete_is_cag_then_cag_full_recall_is_true(self):
        mapper = _import_mapper()
        verdict = _Verdict(
            agree=False, more_complete="cag", note="CAG answered more fully"
        )
        _, cag_full_recall = mapper(verdict)
        assert cag_full_recall is True, (
            "more_complete=='cag' means CAG saw the full document and answered completely; "
            "cag_full_recall must be True"
        )

    def test_when_verdict_is_mapped_then_result_is_a_two_element_sequence(self):
        mapper = _import_mapper()
        verdict = _Verdict(agree=True, more_complete="equal", note="")
        result = mapper(verdict)
        assert len(result) == 2, (
            f"Mapper must return a 2-element (rag_missed, cag_full_recall) tuple, "
            f"got length {len(result)}"
        )

    def test_when_verdict_is_mapped_then_rag_missed_is_a_bool(self):
        mapper = _import_mapper()
        verdict = _Verdict(agree=False, more_complete="cag", note="test")
        rag_missed, _ = mapper(verdict)
        assert isinstance(rag_missed, bool), (
            f"rag_missed must be bool, got {type(rag_missed).__name__}"
        )

    def test_when_verdict_is_mapped_then_cag_full_recall_is_a_bool(self):
        mapper = _import_mapper()
        verdict = _Verdict(agree=False, more_complete="cag", note="test")
        _, cag_full_recall = mapper(verdict)
        assert isinstance(cag_full_recall, bool), (
            f"cag_full_recall must be bool, got {type(cag_full_recall).__name__}"
        )

    def test_when_same_verdict_mapped_twice_then_results_are_identical(self):
        """Purity: identical inputs must always produce identical outputs."""
        mapper = _import_mapper()
        verdict = _Verdict(
            agree=False, more_complete="cag", note="same input each time"
        )
        assert mapper(verdict) == mapper(verdict), (
            "Mapper is not pure — same Verdict produced different outputs on two calls"
        )


# ---------------------------------------------------------------------------
# Criterion: _read_verdict uses real Verdict fields, not rag_missed/cag_full_recall
# ---------------------------------------------------------------------------


class TestReadVerdictUsesRealFields:
    """The mapper must read verdict.agree / .more_complete / .note — not non-existent attrs."""

    def test_when_verdict_lacks_rag_missed_attr_then_mapper_does_not_raise_attribute_error(
        self,
    ):
        """Old bug: verdict.rag_missed → AttributeError because that field never existed."""
        mapper = _import_mapper()
        verdict = _Verdict(agree=False, more_complete="cag", note="")
        # _Verdict has no .rag_missed — a buggy mapper raises AttributeError here.
        try:
            mapper(verdict)
        except AttributeError as exc:
            pytest.fail(
                f"Mapper raised AttributeError: {exc}. "
                "Do not access verdict.rag_missed — it does not exist on the real Verdict type. "
                "Derive rag_missed from verdict.more_complete instead."
            )

    def test_when_verdict_lacks_cag_full_recall_attr_then_mapper_does_not_raise_attribute_error(
        self,
    ):
        """Old bug: verdict.cag_full_recall → AttributeError because that field never existed."""
        mapper = _import_mapper()
        verdict = _Verdict(agree=True, more_complete="equal", note="")
        try:
            mapper(verdict)
        except AttributeError as exc:
            pytest.fail(
                f"Mapper raised AttributeError: {exc}. "
                "Do not access verdict.cag_full_recall — it does not exist on the real Verdict type. "
                "Derive cag_full_recall from verdict.more_complete instead."
            )


# ---------------------------------------------------------------------------
# Criterion: --judge on + more_complete=="cag" → ⚠ missed on RAG, ✓ full recall on CAG
# ---------------------------------------------------------------------------


class TestJudgeMarkerStrings:
    """Marker formatter produces the correct display strings for the TUI panels."""

    def test_when_rag_missed_is_true_then_rag_marker_contains_warning_or_missed_text(
        self,
    ):
        formatter = _import_marker_formatter()
        rag_marker, _ = formatter(rag_missed=True, cag_full_recall=True)
        has_warning = "⚠" in rag_marker or "missed" in rag_marker.lower()
        assert has_warning, (
            f"RAG marker must contain '⚠' or 'missed' when rag_missed=True, got: {rag_marker!r}"
        )

    def test_when_cag_full_recall_is_true_then_cag_marker_contains_checkmark_or_full_recall_text(
        self,
    ):
        formatter = _import_marker_formatter()
        _, cag_marker = formatter(rag_missed=True, cag_full_recall=True)
        has_recall = "✓" in cag_marker or "full recall" in cag_marker.lower()
        assert has_recall, (
            f"CAG marker must contain '✓' or 'full recall' when cag_full_recall=True, "
            f"got: {cag_marker!r}"
        )

    def test_when_more_complete_is_cag_then_rag_marker_shows_missed_end_to_end(self):
        """End-to-end: Verdict(more_complete='cag') → mapper → formatter → RAG shows ⚠ missed."""
        mapper = _import_mapper()
        formatter = _import_marker_formatter()
        verdict = _Verdict(agree=False, more_complete="cag", note="CAG more complete")
        rag_missed, cag_full_recall = mapper(verdict)
        rag_marker, _ = formatter(
            rag_missed=rag_missed, cag_full_recall=cag_full_recall
        )
        has_warning = "⚠" in rag_marker or "missed" in rag_marker.lower()
        assert has_warning, (
            f"Verdict(more_complete='cag') must ultimately produce a RAG '⚠ missed …' marker, "
            f"got: {rag_marker!r}"
        )

    def test_when_more_complete_is_cag_then_cag_marker_shows_full_recall_end_to_end(
        self,
    ):
        """End-to-end: Verdict(more_complete='cag') → mapper → formatter → CAG shows ✓ full recall."""
        mapper = _import_mapper()
        formatter = _import_marker_formatter()
        verdict = _Verdict(agree=False, more_complete="cag", note="CAG more complete")
        rag_missed, cag_full_recall = mapper(verdict)
        _, cag_marker = formatter(
            rag_missed=rag_missed, cag_full_recall=cag_full_recall
        )
        has_recall = "✓" in cag_marker or "full recall" in cag_marker.lower()
        assert has_recall, (
            f"Verdict(more_complete='cag') must ultimately produce a CAG '✓ full recall' marker, "
            f"got: {cag_marker!r}"
        )


# ---------------------------------------------------------------------------
# Property-based tests (hypothesis)
# ---------------------------------------------------------------------------

_MORE_COMPLETE_STRATEGY = st.sampled_from(["rag", "cag", "equal"])


@given(
    agree=st.booleans(),
    more_complete=_MORE_COMPLETE_STRATEGY,
    note=st.text(),
)
@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
def test_when_any_valid_verdict_is_mapped_then_two_booleans_are_returned_property(
    agree: bool, more_complete: str, note: str
) -> None:
    """Invariant: the mapper is a total function over the full Verdict domain.

    For any combination of valid Verdict fields, it must return (bool, bool) without raising.
    """
    mapper = _import_mapper()
    verdict = _Verdict(agree=agree, more_complete=more_complete, note=note)
    result = mapper(verdict)
    assert len(result) == 2
    rag_missed, cag_full_recall = result
    assert isinstance(rag_missed, bool), (
        f"rag_missed must be bool, got {type(rag_missed).__name__}"
    )
    assert isinstance(cag_full_recall, bool), (
        f"cag_full_recall must be bool, got {type(cag_full_recall).__name__}"
    )


@given(
    agree=st.booleans(),
    more_complete=_MORE_COMPLETE_STRATEGY,
    note=st.text(),
)
@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
def test_when_same_verdict_mapped_twice_then_result_is_identical_property(
    agree: bool, more_complete: str, note: str
) -> None:
    """Idempotence/purity: the mapper is referentially transparent.

    Applying it twice to the same Verdict must yield the same (rag_missed, cag_full_recall).
    """
    mapper = _import_mapper()
    verdict = _Verdict(agree=agree, more_complete=more_complete, note=note)
    assert mapper(verdict) == mapper(verdict), (
        "Mapper is not pure — calling it twice with the same Verdict gave different results."
    )
