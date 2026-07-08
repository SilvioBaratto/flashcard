"""Tests for issue #21: run_tui integrates resilient_session with on_token/on_note.

Source-blind authoring: all tests derived exclusively from acceptance criteria
and requirements.md. No implementation files were read.

Criteria covered (per oracle classification):
  [UNIT] run_tui drives resilient_session (not run_session directly),
         passing on_token and an on_note callback.
  [UNIT] The TUI shows a transient "retrying…" note on failure and clears it
         (on_note(None)) on recovery, without altering the two-panel layout.

Criteria skipped (oracle: NOT VERIFIABLE):
  429/5xx retry-with-backoff logic and take continuation (no live API in CI).
  Non-transient error propagation (no live API in CI).
  --delay pacing under cache TTL (no live session in CI).
  TUI legibility on phone-recorded terminal (visual / subjective).
  Price verification at build time (NOT VERIFIABLE prose).
  All-tests-pass gate, SOLID/clean-code (boilerplate / subjective).

Patch strategy for run_tui:
  run_tui is imported from flashcard.tui and is async.
  resilient_session is expected in flashcard.resilience (per requirements.md:
  "flashcard/resilience.py: 429/5xx retry+backoff").  We patch at both the
  source module and at flashcard.tui's likely import site so the mock
  intercepts regardless of import style.
  Rich Live is suppressed to prevent terminal I/O during tests.
"""

from __future__ import annotations

import asyncio
import importlib
from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Helpers: import probing
# ---------------------------------------------------------------------------


def _find_resilient_session_path() -> str:
    """Return the dotted module path that exports resilient_session.

    Per requirements.md §Constraints "resilience.py: 429/5xx retry+backoff, OpenAI"
    the primary location is flashcard.resilience.  Probe flashcard.runner as a
    secondary location in case the implementation hoists it there.

    Fails loudly in the Red phase so the error message guides implementation.
    """
    for mod_path in ("flashcard.resilience", "flashcard.runner"):
        try:
            mod = importlib.import_module(mod_path)
            if hasattr(mod, "resilient_session"):
                return mod_path
        except ImportError:
            pass
    pytest.fail(
        "resilient_session not found in flashcard.resilience or flashcard.runner. "
        "Implement flashcard.resilience.resilient_session (429/5xx retry + backoff) "
        "as specified in requirements.md."
    )


def _get_run_tui():
    """Import and return the run_tui coroutine from flashcard.tui."""
    try:
        from flashcard.tui import run_tui

        return run_tui
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"Cannot import flashcard.tui.run_tui: {exc}")


def _make_cfg():
    """Construct a minimal valid RunConfig (model + max_tokens defaults)."""
    from flashcard.runner import RunConfig

    return RunConfig(model="gpt-5.2", max_tokens=400)


def _run(coro):
    """Execute a coroutine synchronously in a fresh event loop."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Minimal Rich Live stub — prevents terminal I/O during tests.
# ---------------------------------------------------------------------------


class _FakeLive:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def update(self, *_):
        pass

    def refresh(self):
        pass

    def start(self, *_):
        pass

    def stop(self):
        pass


# ---------------------------------------------------------------------------
# Context-manager helper: patch resilient_session + suppress TUI I/O.
# ---------------------------------------------------------------------------


def _patch_env(fake_resilient_fn, *, fake_run_session_fn=None) -> ExitStack:
    """Return an ExitStack that wires up all necessary patches for unit tests.

    Patches applied:
    - resilient_session at its source module and at flashcard.tui's import site.
    - Rich Live context manager (suppresses terminal rendering).
    - asyncio.sleep (removes pacing delays so tests run instantly).
    - Optionally run_session at flashcard.runner and flashcard.tui (to track
      direct calls).
    """
    rs_path = _find_resilient_session_path()
    stack = ExitStack()

    stack.enter_context(
        patch(f"{rs_path}.resilient_session", side_effect=fake_resilient_fn)
    )
    # Belt-and-suspenders: also patch at the import site inside flashcard.tui.
    stack.enter_context(
        patch(
            "flashcard.tui.resilient_session",
            side_effect=fake_resilient_fn,
            create=True,
        )
    )

    # Suppress Rich terminal rendering.
    stack.enter_context(patch("rich.live.Live", return_value=_FakeLive()))
    stack.enter_context(
        patch("flashcard.tui.Live", return_value=_FakeLive(), create=True)
    )

    # Remove asyncio.sleep delays (pacing) so tests run without hanging.
    stack.enter_context(patch("asyncio.sleep", new=AsyncMock(return_value=None)))

    if fake_run_session_fn is not None:
        stack.enter_context(
            patch(
                "flashcard.runner.run_session",
                side_effect=fake_run_session_fn,
                create=True,
            )
        )
        stack.enter_context(
            patch(
                "flashcard.tui.run_session",
                side_effect=fake_run_session_fn,
                create=True,
            )
        )

    return stack


# ---------------------------------------------------------------------------
# Criterion: run_tui drives resilient_session, NOT run_session directly.
# ---------------------------------------------------------------------------


class TestRunTuiDrivesResilientSession:
    """run_tui must delegate each turn to resilient_session, never to run_session."""

    def test_when_run_tui_invoked_then_resilient_session_is_called(self):
        """run_tui must call resilient_session at least once (one call per turn).

        Per spec: 'run_tui drives resilient_session (not run_session directly)'.
        In the Red phase this test fails because resilient_session is not yet called.
        """
        calls: list[dict] = []

        async def _fake(*_args, **kwargs):
            calls.append(kwargs)

        run_tui = _get_run_tui()
        cfg = _make_cfg()
        with _patch_env(_fake):
            try:
                _run(run_tui(cfg))
            except Exception:
                pass  # We assert only that the call happened before any crash.

        assert calls, (
            "run_tui never called resilient_session. "
            "Per spec: 'run_tui drives resilient_session (not run_session directly)'. "
            "Replace any direct run_session call with resilient_session."
        )

    def test_when_run_tui_invoked_then_run_session_is_not_called_directly(self):
        """run_tui must NOT bypass resilient_session and call run_session directly.

        Spec: 'run_tui drives resilient_session (not run_session directly)'.
        We intercept resilient_session so it returns immediately; if run_session
        is still called, the implementation violates the criterion.
        """
        run_session_calls: list = []

        async def _fake_rs(*_args, **_kwargs):
            pass  # resilient_session intercepted; prevents run_session from running

        async def _fake_run_sess(*_args, **_kwargs):
            run_session_calls.append(True)

        run_tui = _get_run_tui()
        cfg = _make_cfg()
        with _patch_env(_fake_rs, fake_run_session_fn=_fake_run_sess):
            try:
                _run(run_tui(cfg))
            except Exception:
                pass

        assert not run_session_calls, (
            f"run_tui called run_session directly {len(run_session_calls)} time(s). "
            "Per spec: run_tui must drive resilient_session, not run_session directly. "
            "Remove the direct run_session call and delegate through resilient_session."
        )


# ---------------------------------------------------------------------------
# Criterion: run_tui passes on_token and on_note to resilient_session.
# ---------------------------------------------------------------------------


def _capture_resilient_session_calls() -> tuple[list[dict], ExitStack]:
    """Return (call_list, ExitStack) that captures every resilient_session kwarg dict."""
    calls: list[dict] = []

    async def _fake(*_args, **kwargs):
        calls.append(dict(kwargs))

    return calls, _patch_env(_fake)


class TestRunTuiPassesCallbacks:
    """Both on_token and on_note must be passed as kwargs to resilient_session."""

    def test_when_run_tui_invoked_then_on_token_is_passed_to_resilient_session(self):
        """resilient_session must receive an on_token keyword argument.

        Per spec: 'run_tui drives resilient_session … passing on_token … callback'.
        """
        calls, stack = _capture_resilient_session_calls()
        run_tui = _get_run_tui()
        cfg = _make_cfg()
        with stack:
            try:
                _run(run_tui(cfg))
            except Exception:
                pass

        assert calls, "resilient_session was never called — see above test."
        assert any("on_token" in kw for kw in calls), (
            f"resilient_session was not passed on_token. "
            f"kwargs seen across all calls: {[sorted(kw) for kw in calls]}. "
            "Add on_token=<callback> when calling resilient_session inside run_tui."
        )

    def test_when_run_tui_invoked_then_on_note_is_passed_to_resilient_session(self):
        """resilient_session must receive an on_note keyword argument.

        Per spec: 'run_tui drives resilient_session … passing … an on_note callback'.
        """
        calls, stack = _capture_resilient_session_calls()
        run_tui = _get_run_tui()
        cfg = _make_cfg()
        with stack:
            try:
                _run(run_tui(cfg))
            except Exception:
                pass

        assert calls, "resilient_session was never called — see above test."
        assert any("on_note" in kw for kw in calls), (
            f"resilient_session was not passed on_note. "
            f"kwargs seen across all calls: {[sorted(kw) for kw in calls]}. "
            "Add on_note=<callback> when calling resilient_session inside run_tui."
        )

    def test_when_run_tui_invoked_then_on_token_value_is_callable(self):
        """The on_token value passed to resilient_session must be callable."""
        calls, stack = _capture_resilient_session_calls()
        run_tui = _get_run_tui()
        cfg = _make_cfg()
        with stack:
            try:
                _run(run_tui(cfg))
            except Exception:
                pass

        assert calls, "resilient_session was never called."
        on_token = next((kw["on_token"] for kw in calls if "on_token" in kw), None)
        assert on_token is not None, (
            "on_token not present in any resilient_session call."
        )
        assert callable(on_token), (
            f"on_token passed to resilient_session must be callable, "
            f"got {type(on_token).__name__!r}."
        )

    def test_when_run_tui_invoked_then_on_note_value_is_callable(self):
        """The on_note value passed to resilient_session must be callable."""
        calls, stack = _capture_resilient_session_calls()
        run_tui = _get_run_tui()
        cfg = _make_cfg()
        with stack:
            try:
                _run(run_tui(cfg))
            except Exception:
                pass

        assert calls, "resilient_session was never called."
        on_note = next((kw["on_note"] for kw in calls if "on_note" in kw), None)
        assert on_note is not None, "on_note not present in any resilient_session call."
        assert callable(on_note), (
            f"on_note passed to resilient_session must be callable, "
            f"got {type(on_note).__name__!r}."
        )


# ---------------------------------------------------------------------------
# Helper: extract the on_note callback from a captured run_tui call.
# ---------------------------------------------------------------------------


def _extract_on_note() -> Any:
    """Run run_tui under mocks and extract the on_note callable.

    Returns the on_note callback, or calls pytest.skip() in the Red phase
    (when resilient_session is not yet called or on_note is not yet passed).
    """
    calls: list[dict] = []

    async def _fake(*_args, **kwargs):
        calls.append(dict(kwargs))

    run_tui = _get_run_tui()
    cfg = _make_cfg()
    with _patch_env(_fake):
        try:
            _run(run_tui(cfg))
        except Exception:
            pass

    if not calls:
        pytest.skip(
            "resilient_session not yet called — Red phase. "
            "Implement the resilient_session call inside run_tui first."
        )
    on_note = next((kw["on_note"] for kw in calls if "on_note" in kw), None)
    if on_note is None:
        pytest.skip(
            "on_note not yet passed to resilient_session — Red phase. "
            "Add on_note=<callback> to the resilient_session call inside run_tui."
        )
    return on_note


# ---------------------------------------------------------------------------
# Criterion: on_note shows a transient note and on_note(None) clears it.
# ---------------------------------------------------------------------------


class TestOnNoteCallback:
    """The on_note callback must accept a note string (show) and None (clear)."""

    def test_when_on_note_called_with_retrying_string_then_does_not_raise(self):
        """on_note('retrying…') must execute without raising.

        Per spec: 'the TUI shows a transient "retrying" note … rather than crashing'.
        Calling on_note with a non-None string is the 'show note' signal; it must
        not crash the TUI loop.
        """
        on_note = _extract_on_note()
        try:
            on_note("retrying…")
        except Exception as exc:
            pytest.fail(
                f"on_note('retrying…') raised {type(exc).__name__}: {exc}. "
                "The note callback must handle transient strings without error."
            )

    def test_when_on_note_called_with_none_then_does_not_raise(self):
        """on_note(None) must execute without raising.

        Per spec: 'clears it (on_note(None)) on recovery'.
        None is the recovery/clear signal; it must not crash.
        """
        on_note = _extract_on_note()
        try:
            on_note(None)
        except Exception as exc:
            pytest.fail(
                f"on_note(None) raised {type(exc).__name__}: {exc}. "
                "The None signal clears the note on recovery; it must not crash."
            )

    def test_when_on_note_called_with_none_after_string_then_does_not_raise(self):
        """Sequence on_note(str) → on_note(None) must complete without error.

        Per spec: 'shows a transient "retrying" note … clears it (on_note(None))'.
        This is the show-then-clear lifecycle.
        """
        on_note = _extract_on_note()
        on_note("retrying…")
        on_note(None)  # clear — must not raise

    def test_when_on_note_called_with_none_before_any_string_then_does_not_raise(self):
        """on_note(None) before any string must be a safe no-op.

        The recovery signal may arrive even if no note was ever shown (e.g. first
        call succeeds — no retry needed). A clear with nothing to clear must not crash.
        """
        on_note = _extract_on_note()
        try:
            on_note(None)
        except Exception as exc:
            pytest.fail(
                f"on_note(None) with no prior note raised {type(exc).__name__}: {exc}. "
                "Clearing a note that was never set must be a safe no-op."
            )

    def test_when_on_note_called_with_none_twice_then_does_not_raise(self):
        """Calling on_note(None) twice in a row must not raise (idempotent clear).

        Per spec: multiple recovery signals must not crash the TUI.
        """
        on_note = _extract_on_note()
        on_note(None)
        on_note(None)  # second clear — idempotent

    def test_when_on_note_called_multiple_times_then_does_not_raise(self):
        """Multiple note updates followed by a clear must not raise.

        Reflects a real retry sequence: the session retries twice, then succeeds.
        """
        on_note = _extract_on_note()
        on_note("retrying… (attempt 1)")
        on_note("retrying… (attempt 2)")
        on_note(None)  # recovery


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(note=st.text(min_size=1))
@settings(max_examples=12, suppress_health_check=[HealthCheck.too_slow])
def test_when_on_note_called_with_any_non_empty_string_then_never_raises(
    note: str,
) -> None:
    """Invariant: on_note is a total function over non-empty strings.

    Per spec: on_note is invoked by resilient_session with transient messages
    such as 'retrying…'; it must never raise for any such string.
    This criterion implies a never-raises-for-valid-input invariant.
    """
    calls: list[dict] = []

    async def _fake(*_args, **kwargs):
        calls.append(dict(kwargs))

    run_tui = _get_run_tui()
    cfg = _make_cfg()
    with _patch_env(_fake):
        try:
            _run(run_tui(cfg))
        except Exception:
            pass

    if not calls:
        pytest.skip("resilient_session not called — Red phase.")
    on_note = next((kw["on_note"] for kw in calls if "on_note" in kw), None)
    if on_note is None:
        pytest.skip("on_note not yet wired — Red phase.")

    try:
        on_note(note)
    except Exception as exc:
        pytest.fail(
            f"on_note({note!r}) raised {type(exc).__name__}: {exc}. "
            "on_note must accept any non-empty string without raising."
        )


@given(
    steps=st.lists(st.one_of(st.text(min_size=1), st.none()), min_size=1, max_size=8)
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
def test_when_on_note_called_in_any_str_none_sequence_then_never_raises(
    steps: list[str | None],
) -> None:
    """Invariant: on_note is a total function over any interleaving of str and None.

    The retrying note may be shown and cleared in any order — fast retries,
    consecutive clears, etc. The callback must never raise regardless of sequence.
    """
    calls: list[dict] = []

    async def _fake(*_args, **kwargs):
        calls.append(dict(kwargs))

    run_tui = _get_run_tui()
    cfg = _make_cfg()
    with _patch_env(_fake):
        try:
            _run(run_tui(cfg))
        except Exception:
            pass

    if not calls:
        pytest.skip("resilient_session not called — Red phase.")
    on_note = next((kw["on_note"] for kw in calls if "on_note" in kw), None)
    if on_note is None:
        pytest.skip("on_note not yet wired — Red phase.")

    for step in steps:
        try:
            on_note(step)
        except Exception as exc:
            pytest.fail(
                f"on_note({step!r}) raised {type(exc).__name__}: {exc} "
                f"during sequence {steps!r}. "
                "on_note must handle any interleaving of str and None."
            )
