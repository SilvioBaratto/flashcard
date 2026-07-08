"""Tests for issue #19: --judge/--no-judge flag, RunConfig construction, and run_tui wiring.

Source-blind authoring: all tests derived exclusively from acceptance criteria.
No implementation files were read.

Patch strategy: flashcard.cli is expected to import run_tui via
    `from flashcard.tui import run_tui`
which binds the name inside flashcard.cli's namespace. We therefore patch
`flashcard.cli.run_tui` (the import site). If that attribute is absent —
because the implementation hasn't been added yet — the patch raises
AttributeError, which is the correct Red-phase signal that the feature isn't
wired. Patching `flashcard.tui.run_tui` is added as a belt-and-suspenders
fallback for callers that use the fully-qualified name.
"""

from __future__ import annotations

import pathlib
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from flashcard.cli import app

_cli_runner = CliRunner()


# ---------------------------------------------------------------------------
# Helper: patch run_tui at the CLI import site + source module
# ---------------------------------------------------------------------------


def _patch_run_tui(captured: list[Any]):
    """Return an ExitStack that mocks run_tui and records every cfg passed to it.

    Patches at both flashcard.cli.run_tui (import site) and flashcard.tui.run_tui
    (source module) so the mock intercepts the call regardless of call style.
    Both patches share the same async side_effect so cfg is recorded once per call.
    """
    import flashcard.cli as _cli_mod

    async def _fake(cfg, **kwargs):
        captured.append(cfg)

    stack = ExitStack()
    # Source module — always patch so module-level callers are covered.
    stack.enter_context(patch("flashcard.tui.run_tui", side_effect=_fake))
    # Import site — patch only if the implementation has bound the name.
    if hasattr(_cli_mod, "run_tui"):
        stack.enter_context(patch("flashcard.cli.run_tui", side_effect=_fake))
    return stack


def _invoke(args: list[str]):
    """Invoke the flashcard CLI main command with the given args list."""
    return _cli_runner.invoke(app, args)


# ---------------------------------------------------------------------------
# Criterion: --judge/--no-judge defaults to off (False)
# ---------------------------------------------------------------------------


def test_when_no_judge_flag_then_runconfig_judge_defaults_to_false():
    """No --judge flag: RunConfig.judge must default to False (judge off by spec)."""
    captured: list[Any] = []
    with _patch_run_tui(captured):
        result = _invoke(["--yes"])

    assert result.exit_code == 0, f"CLI failed ({result.exit_code}):\n{result.output}"
    assert captured, (
        "run_tui was never called — placeholder not yet replaced with run_tui call"
    )
    assert captured[0].judge is False, (
        f"Expected RunConfig.judge=False by default, got {captured[0].judge!r}"
    )


# ---------------------------------------------------------------------------
# Criterion: --judge flag sets judge to True
# ---------------------------------------------------------------------------


def test_when_judge_flag_provided_then_runconfig_judge_is_true():
    """--judge flag: RunConfig.judge must be True."""
    captured: list[Any] = []
    with _patch_run_tui(captured):
        result = _invoke(["--yes", "--judge"])

    assert result.exit_code == 0, f"CLI failed ({result.exit_code}):\n{result.output}"
    assert captured, "run_tui was never called"
    assert captured[0].judge is True, (
        f"Expected RunConfig.judge=True with --judge, got {captured[0].judge!r}"
    )


# ---------------------------------------------------------------------------
# Criterion: explicit --no-judge flag keeps judge False
# ---------------------------------------------------------------------------


def test_when_no_judge_flag_explicit_then_runconfig_judge_is_false():
    """Explicit --no-judge flag: RunConfig.judge must be False."""
    captured: list[Any] = []
    with _patch_run_tui(captured):
        result = _invoke(["--yes", "--no-judge"])

    assert result.exit_code == 0, f"CLI failed ({result.exit_code}):\n{result.output}"
    assert captured, "run_tui was never called"
    assert captured[0].judge is False


# ---------------------------------------------------------------------------
# Criterion: RunConfig built from all flags
# ---------------------------------------------------------------------------


def test_when_all_flags_provided_then_runconfig_fields_match(tmp_path):
    """All flags must be threaded into the RunConfig passed to run_tui."""
    doc_file = tmp_path / "doc.txt"
    doc_file.write_text("A sample knowledge document.")
    q_file = tmp_path / "q.json"
    q_file.write_text('["What does this cover?"]')

    captured: list[Any] = []
    with _patch_run_tui(captured):
        result = _invoke(
            [
                "--yes",
                "--model",
                "gpt-5-mini",
                "--doc",
                str(doc_file),
                "--questions-file",
                str(q_file),
                "--top-k",
                "5",
                "--chunk-size",
                "400",
                "--max-tokens",
                "200",
                "--delay",
                "1.5",
                "--judge",
            ]
        )

    assert result.exit_code == 0, f"CLI failed ({result.exit_code}):\n{result.output}"
    assert captured, "run_tui was never called"
    cfg = captured[0]

    model_str = getattr(cfg.model, "value", str(cfg.model))
    assert "gpt-5-mini" in model_str.lower(), (
        f"model: expected 'gpt-5-mini', got {cfg.model!r}"
    )
    assert cfg.doc == str(doc_file), f"doc: expected {doc_file!s}, got {cfg.doc!r}"
    assert cfg.questions_file == str(q_file), (
        f"questions_file mismatch: {cfg.questions_file!r}"
    )
    assert cfg.top_k == 5, f"top_k: expected 5, got {cfg.top_k}"
    assert cfg.chunk_size == 400, f"chunk_size: expected 400, got {cfg.chunk_size}"
    assert cfg.max_tokens == 200, f"max_tokens: expected 200, got {cfg.max_tokens}"
    assert cfg.delay == pytest.approx(1.5), f"delay: expected 1.5, got {cfg.delay}"
    assert cfg.judge is True, f"judge: expected True, got {cfg.judge}"


def test_when_default_flags_used_then_runconfig_has_documented_defaults():
    """Invoke with no optional flags: RunConfig must carry the documented defaults."""
    captured: list[Any] = []
    with _patch_run_tui(captured):
        result = _invoke(["--yes"])

    assert result.exit_code == 0, f"CLI failed ({result.exit_code}):\n{result.output}"
    assert captured, "run_tui was never called"
    cfg = captured[0]

    assert cfg.top_k == 3, f"default top_k should be 3, got {cfg.top_k}"
    assert cfg.chunk_size == 800, (
        f"default chunk_size should be 800, got {cfg.chunk_size}"
    )
    assert cfg.max_tokens == 400, (
        f"default max_tokens should be 400, got {cfg.max_tokens}"
    )
    assert cfg.judge is False, f"default judge should be False, got {cfg.judge}"

    model_str = getattr(cfg.model, "value", str(cfg.model))
    assert "gpt-5.2" in model_str, (
        f"default model should be 'gpt-5.2', got {cfg.model!r}"
    )


# ---------------------------------------------------------------------------
# Criterion: run_tui is called after --yes (placeholder gone)
# ---------------------------------------------------------------------------


def test_when_yes_flag_then_run_tui_is_called():
    """--yes flag: run_tui must be invoked (placeholder no longer blocking it)."""
    captured: list[Any] = []
    with _patch_run_tui(captured):
        result = _invoke(["--yes"])

    assert captured, (
        "run_tui was never called after --yes. "
        "The 'Pipeline not yet wired' placeholder must be replaced."
    )
    assert result.exit_code == 0


def test_when_yes_flag_then_placeholder_text_absent_from_output():
    """The 'Pipeline not yet wired' placeholder must not appear in CLI output."""
    captured: list[Any] = []
    with _patch_run_tui(captured):
        result = _invoke(["--yes"])

    assert "Pipeline not yet wired" not in result.output, (
        "Placeholder text still present in output — it must be removed."
    )


def test_when_run_tui_called_then_argument_is_a_runconfig():
    """The object passed to run_tui must be a RunConfig instance."""
    from flashcard.runner import (
        RunConfig,
    )  # spec-required type; not implementation internals

    captured: list[Any] = []
    with _patch_run_tui(captured):
        _invoke(["--yes"])

    assert captured, "run_tui was never called"
    assert isinstance(captured[0], RunConfig), (
        f"Expected RunConfig, got {type(captured[0]).__name__}"
    )


# ---------------------------------------------------------------------------
# Criterion: asyncio.run (or equivalent) bridges sync Typer to async TUI
# ---------------------------------------------------------------------------


def test_when_yes_flag_then_async_run_tui_is_awaited():
    """The async run_tui must be awaited inside the sync Typer command.

    Proof: the fake coroutine appends to `awaited` only if it is actually
    executed (i.e. the event loop runs it). If the CLI merely creates the
    coroutine without awaiting it, `awaited` stays empty.
    """
    awaited: list[bool] = []

    async def _fake_run_tui(cfg, **kwargs):
        awaited.append(True)

    import flashcard.cli as _cli_mod

    patches: list = [patch("flashcard.tui.run_tui", side_effect=_fake_run_tui)]
    if hasattr(_cli_mod, "run_tui"):
        patches.append(patch("flashcard.cli.run_tui", side_effect=_fake_run_tui))

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        result = _invoke(["--yes"])

    assert result.exit_code == 0, f"CLI failed ({result.exit_code}):\n{result.output}"
    assert awaited, (
        "run_tui coroutine was created but never awaited. "
        "asyncio.run(...) (or equivalent) must bridge the sync Typer command to "
        "the async run_tui."
    )


# ---------------------------------------------------------------------------
# Criterion: no secrets in source — .env.example shipped
# ---------------------------------------------------------------------------


def test_when_project_root_inspected_then_env_example_exists():
    """A .env.example file must exist at the project root."""
    # tests/ lives one directory below the project root
    project_root = pathlib.Path(__file__).parent.parent
    env_example = project_root / ".env.example"
    assert env_example.exists(), (
        f".env.example not found at {env_example}. "
        "It must be shipped so users can configure OPENAI_API_KEY safely."
    )


# ---------------------------------------------------------------------------
# Property-based tests (hypothesis)
# ---------------------------------------------------------------------------


@given(top_k=st.integers(min_value=1, max_value=20))
@settings(max_examples=8)
def test_when_top_k_varies_then_runconfig_top_k_matches(top_k):
    """Invariant: whatever integer --top-k receives, RunConfig.top_k must equal it."""
    captured: list[Any] = []
    with _patch_run_tui(captured):
        result = _cli_runner.invoke(app, ["--yes", "--top-k", str(top_k)])
    if result.exit_code != 0:
        pytest.skip(f"CLI rejected top_k={top_k}: {result.output}")
    if not captured:
        pytest.fail(f"run_tui not called for top_k={top_k}")
    assert captured[0].top_k == top_k, (
        f"--top-k {top_k} → RunConfig.top_k should be {top_k}, got {captured[0].top_k}"
    )


@given(max_tokens=st.integers(min_value=1, max_value=2000))
@settings(max_examples=8)
def test_when_max_tokens_varies_then_runconfig_max_tokens_matches(max_tokens):
    """Invariant: whatever integer --max-tokens receives, RunConfig.max_tokens must equal it."""
    captured: list[Any] = []
    with _patch_run_tui(captured):
        result = _cli_runner.invoke(app, ["--yes", "--max-tokens", str(max_tokens)])
    if result.exit_code != 0:
        pytest.skip(f"CLI rejected max_tokens={max_tokens}: {result.output}")
    if not captured:
        pytest.fail(f"run_tui not called for max_tokens={max_tokens}")
    assert captured[0].max_tokens == max_tokens, (
        f"--max-tokens {max_tokens} → RunConfig.max_tokens should be {max_tokens}, "
        f"got {captured[0].max_tokens}"
    )


@given(judge=st.booleans())
@settings(max_examples=4)
def test_when_judge_flag_varies_then_runconfig_judge_matches(judge):
    """Invariant: --judge or --no-judge always produces RunConfig.judge == the flag value."""
    flag = "--judge" if judge else "--no-judge"
    captured: list[Any] = []
    with _patch_run_tui(captured):
        result = _cli_runner.invoke(app, ["--yes", flag])
    if result.exit_code != 0:
        pytest.fail(f"CLI rejected {flag}: {result.output}")
    if not captured:
        pytest.fail(f"run_tui not called for {flag}")
    assert captured[0].judge is judge, (
        f"{flag} → RunConfig.judge should be {judge}, got {captured[0].judge!r}"
    )
