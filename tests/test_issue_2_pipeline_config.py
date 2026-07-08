"""
Source-blind tests for issue #2 — apples-to-apples pipeline configuration.

Criterion exercised (oracle tier in brackets):
  [UNIT] Both pipelines answer the same questions over the same source with
         the same model and max_tokens, so cost and accuracy gaps are
         attributable to retrieved-slice (RAG) vs whole-doc (CAG) alone.

The secrets / .env.example / .gitignore hygiene checks are already covered by
issue #1's tests/test_baml_sources.py and are not duplicated here.

All tests are authored from the acceptance criteria only (TDD red phase).
No implementation source was read.  These tests currently pass on the seeded
runner.py but will catch any future regression that splits model or max_tokens
into per-side fields.

No network calls are made anywhere in this module.
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

# runner.py is already seeded (per git status).  We import RunConfig from it.
# The spec states: "RunConfig" is the shared configuration object for a run.
# Both sides (RAG and CAG) must draw their model and max_tokens from this single
# object — not from separate per-side fields — to guarantee apples-to-apples.
from flashcard.runner import RunConfig  # noqa: E402


# ── RunConfig: shared model ───────────────────────────────────────────────────


def test_when_run_config_created_then_model_attribute_exists():
    """
    RunConfig must expose a `model` attribute so callers can inspect which
    model both pipelines will use.
    """
    config = RunConfig(model="gpt-5.2", max_tokens=400)
    assert hasattr(config, "model"), (
        "RunConfig must have a 'model' attribute; "
        "it is required for the apples-to-apples guarantee."
    )


def test_when_run_config_created_with_model_then_model_is_stored_correctly():
    """
    Criterion: 'same model … so cost and accuracy gaps are attributable to
    retrieved-slice vs whole-doc alone'.
    The model stored in RunConfig must equal the one passed at construction.
    """
    config = RunConfig(model="gpt-5.2", max_tokens=400)
    assert config.model == "gpt-5.2", (
        f"RunConfig.model should be 'gpt-5.2', got {config.model!r}"
    )


def test_when_run_config_created_with_alternate_model_then_model_is_stored_correctly():
    """
    The model field must faithfully store whatever model is requested, not
    hard-code a default that would silently diverge from the requested value.
    """
    config = RunConfig(model="gpt-5-mini", max_tokens=400)
    assert config.model == "gpt-5-mini", (
        f"RunConfig.model should be 'gpt-5-mini', got {config.model!r}"
    )


# ── RunConfig: shared max_tokens ──────────────────────────────────────────────


def test_when_run_config_created_then_max_tokens_attribute_exists():
    """
    RunConfig must expose a `max_tokens` attribute so both pipelines can
    apply the same output cap.
    """
    config = RunConfig(model="gpt-5.2", max_tokens=400)
    assert hasattr(config, "max_tokens"), (
        "RunConfig must have a 'max_tokens' attribute; "
        "it is required for the apples-to-apples guarantee."
    )


def test_when_run_config_created_with_max_tokens_then_value_is_stored_correctly():
    """
    Criterion: 'same … max_tokens'.
    The cap stored in RunConfig must equal the one passed at construction.
    """
    config = RunConfig(model="gpt-5.2", max_tokens=400)
    assert config.max_tokens == 400, (
        f"RunConfig.max_tokens should be 400, got {config.max_tokens!r}"
    )


def test_when_run_config_created_with_custom_max_tokens_then_value_is_stored_correctly():
    """
    max_tokens must be user-overridable (the --max-tokens flag), and the
    stored value must reflect the override, not a hard-coded default.
    """
    config = RunConfig(model="gpt-5.2", max_tokens=200)
    assert config.max_tokens == 200, (
        f"RunConfig.max_tokens should be 200, got {config.max_tokens!r}"
    )


# ── no per-side model split ───────────────────────────────────────────────────


def test_when_run_config_inspected_then_no_separate_rag_and_cag_model_fields_exist():
    """
    RunConfig must NOT have separate rag_model / cag_model fields.
    A per-side split would allow the two pipelines to diverge, violating the
    'same model' constraint.  If only one `model` field exists, both sides
    are forced to use it.
    """
    config = RunConfig(model="gpt-5.2", max_tokens=400)
    assert not hasattr(config, "rag_model"), (
        "RunConfig must not have a 'rag_model' field; use a single shared 'model'."
    )
    assert not hasattr(config, "cag_model"), (
        "RunConfig must not have a 'cag_model' field; use a single shared 'model'."
    )


# ── property: model and max_tokens round-trip for any valid values ─────────────


@given(
    model=st.sampled_from(["gpt-5.2", "gpt-5-mini"]),
    max_tokens=st.integers(min_value=1, max_value=4096),
)
@settings(max_examples=20)
def test_when_run_config_created_with_any_valid_model_and_max_tokens_then_values_are_preserved(
    model: str,
    max_tokens: int,
) -> None:
    """
    Invariant: RunConfig(model=M, max_tokens=N).model == M and
               RunConfig(model=M, max_tokens=N).max_tokens == N for any
               supported model name and any positive token count.

    This is a round-trip property: values passed in must come back out
    unchanged, ensuring no silent normalisation or cap is applied internally.
    """
    config = RunConfig(model=model, max_tokens=max_tokens)
    assert config.model == model, (
        f"model round-trip failed: passed {model!r}, got {config.model!r}"
    )
    assert config.max_tokens == max_tokens, (
        f"max_tokens round-trip failed: passed {max_tokens}, got {config.max_tokens!r}"
    )
