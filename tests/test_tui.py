"""Source-blind example tests for flashcard/tui.py (issue #17).

Tests are authored from the acceptance criteria only; no implementation source
was read.  The criteria specify:

    TuiDriver (class)
        - builds a Rich Live + Layout with RAG left column, CAG right column,
          a header over a body per column, and a center/footer strip
        - on_token(side, partial) appends streamed partials to the correct
          side's body buffer (rag_body / cag_body), and must not touch the
          other side's buffer
        - finalize(turn) sets each side's body from turn.rag.text /
          turn.cag.text, correct both with and without prior on_token calls

    run_tui(cfg, *, retriever=None)   [async entrypoint]
        - iterates run_session(...) exactly once, forwarding cfg unchanged
        - passes the driver's on_token callable as the on_token kwarg to
          run_session so streaming partials reach the correct body buffer
        - forwards the retriever kwarg to run_session
        - calls finalize() exactly once per yielded TurnResult
        - creates a Rich Live with a positive refresh_per_second value

All stubs are derived from criterion text; tui.py, tui_state.py, tui_render.py
were not opened.  The same TurnResult/SideResult shape used in test_tui_state.py
is reproduced here to keep the test file fully self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st
from rich.layout import Layout

# ---------------------------------------------------------------------------
# Stub data contracts — derived from criterion text, NOT from production source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TurnMetrics:
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None
    duration_ms: float


@dataclass(frozen=True)
class _SideResult:
    text: str
    metrics: _TurnMetrics
    cost_usd: float
    retrieved: tuple[int, ...] = ()
    context_token_count: int = 0

    @property
    def cost(self) -> float:
        return self.cost_usd

    @property
    def input_tokens(self) -> int | None:
        return self.metrics.input_tokens

    @property
    def output_tokens(self) -> int | None:
        return self.metrics.output_tokens

    @property
    def cached_input_tokens(self) -> int | None:
        return self.metrics.cached_input_tokens

    @property
    def duration_ms(self) -> float:
        return self.metrics.duration_ms


@dataclass(frozen=True)
class _TurnResult:
    index: int
    question: str
    rag: _SideResult
    cag: _SideResult
    verdict: Any = None

    @property
    def turn_number(self) -> int:
        return self.index + 1


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _metrics(**kw: Any) -> _TurnMetrics:
    d: dict[str, Any] = dict(
        input_tokens=400, output_tokens=80, cached_input_tokens=0, duration_ms=300.0
    )
    d.update(kw)
    return _TurnMetrics(**d)


def _rag_side(text: str = "RAG answer", cost: float = 0.010) -> _SideResult:
    return _SideResult(
        text=text,
        metrics=_metrics(),
        cost_usd=cost,
        retrieved=(0, 1, 2),
        context_token_count=300,
    )


def _cag_side(text: str = "CAG answer", cost: float = 0.005) -> _SideResult:
    return _SideResult(
        text=text,
        metrics=_metrics(cached_input_tokens=800, input_tokens=900),
        cost_usd=cost,
    )


def _turn(
    index: int = 0,
    rag_text: str = "RAG answer",
    cag_text: str = "CAG answer",
) -> _TurnResult:
    return _TurnResult(
        index=index,
        question="What is X?",
        rag=_rag_side(text=rag_text),
        cag=_cag_side(text=cag_text),
    )


def _fake_session(turns: list[_TurnResult]):
    """Return an async-generator that yields synthetic TurnResults."""

    async def _gen(*args: Any, **kwargs: Any):
        for t in turns:
            yield t

    return _gen


async def _empty_session(*args: Any, **kwargs: Any):
    """Async generator that yields nothing."""
    if False:
        yield  # makes this an async generator


# ---------------------------------------------------------------------------
# Import the module under test  (Red phase — flashcard/tui.py not yet written)
# ---------------------------------------------------------------------------

from flashcard.tui import TuiDriver, run_tui  # noqa: E402


# ===========================================================================
# Criterion 8 [UNIT]: TuiDriver builds a Rich Live + Layout
# "split into two columns (RAG left, CAG right), each with a header over a
#  body, plus a center/footer strip"
# ===========================================================================


def test_when_tui_driver_created_then_layout_attribute_is_a_rich_layout():
    """TuiDriver.layout must be a Rich Layout instance."""
    driver = TuiDriver()
    assert isinstance(driver.layout, Layout)


def test_when_tui_driver_layout_inspected_then_rag_column_exists():
    """Criterion: 'RAG left' — a named region 'rag' must be accessible."""
    driver = TuiDriver()
    # Rich Layout.__getitem__ raises KeyError when the name is absent.
    rag = driver.layout["rag"]
    assert rag is not None


def test_when_tui_driver_layout_inspected_then_cag_column_exists():
    """Criterion: 'CAG right' — a named region 'cag' must be accessible."""
    driver = TuiDriver()
    cag = driver.layout["cag"]
    assert cag is not None


def test_when_tui_driver_layout_inspected_then_footer_region_exists():
    """Criterion: 'plus a center/footer strip'."""
    driver = TuiDriver()
    layout = driver.layout
    found = False
    for candidate in ("footer", "center", "footer_strip"):
        try:
            layout[candidate]
            found = True
            break
        except (KeyError, ValueError):
            pass
    assert found, "Expected a footer/center/footer_strip region in the layout"


# ===========================================================================
# Criterion 10 [UNIT]: driver-owned token sink (on_token)
# "appends streamed partials to the correct side's body buffer and refreshes"
# ===========================================================================


def test_when_on_token_called_with_rag_side_then_rag_body_contains_partial():
    """Criterion: 'appends … to the correct side's body buffer'."""
    driver = TuiDriver()
    driver.on_token("rag", "streamed piece")
    assert "streamed piece" in driver.rag_body


def test_when_on_token_called_with_cag_side_then_cag_body_contains_partial():
    """Token sink routes cag-side partials to the CAG body buffer."""
    driver = TuiDriver()
    driver.on_token("cag", "cached piece")
    assert "cached piece" in driver.cag_body


def test_when_on_token_called_multiple_times_for_rag_then_all_partials_are_present():
    """Criterion: 'appends' — multiple calls accumulate, not overwrite."""
    driver = TuiDriver()
    driver.on_token("rag", "alpha ")
    driver.on_token("rag", "beta ")
    driver.on_token("rag", "gamma")
    assert "alpha" in driver.rag_body
    assert "beta" in driver.rag_body
    assert "gamma" in driver.rag_body


def test_when_on_token_called_for_rag_side_then_cag_body_is_not_modified():
    """Criterion: 'correct side' — rag token must not bleed into cag buffer."""
    driver = TuiDriver()
    driver.on_token("rag", "rag-only content")
    assert "rag-only content" not in driver.cag_body


def test_when_on_token_called_for_cag_side_then_rag_body_is_not_modified():
    """Criterion: 'correct side' — cag token must not bleed into rag buffer."""
    driver = TuiDriver()
    driver.on_token("cag", "cag-only content")
    assert "cag-only content" not in driver.rag_body


# ===========================================================================
# Criterion 10 [UNIT]: TurnResult finalization from rag.text / cag.text
# "each TurnResult also finalizes its panels from rag.text/cag.text
#  (correct with or without live token pushes)"
# ===========================================================================


def test_when_turn_finalized_then_rag_body_reflects_rag_text():
    """Criterion: 'finalizes its panels from rag.text'."""
    driver = TuiDriver()
    driver.finalize(_turn(rag_text="definitive RAG answer"))
    assert "definitive RAG answer" in driver.rag_body


def test_when_turn_finalized_then_cag_body_reflects_cag_text():
    """Criterion: 'finalizes its panels from … cag.text'."""
    driver = TuiDriver()
    driver.finalize(_turn(cag_text="definitive CAG answer"))
    assert "definitive CAG answer" in driver.cag_body


def test_when_turn_finalized_after_on_token_pushes_then_rag_body_reflects_final_text():
    """Criterion: 'correct with … live token pushes' — finalize must win over partials."""
    driver = TuiDriver()
    driver.on_token("rag", "early partial...")
    driver.finalize(_turn(rag_text="authoritative final text"))
    assert "authoritative final text" in driver.rag_body


def test_when_turn_finalized_without_prior_on_token_then_cag_body_is_still_set():
    """Criterion: 'correct … without live token pushes' — cold finalize works."""
    driver = TuiDriver()
    # No on_token call before finalize — body must still be populated.
    driver.finalize(_turn(cag_text="cold-finalized CAG text"))
    assert "cold-finalized CAG text" in driver.cag_body


def test_when_second_turn_finalized_then_rag_body_reflects_latest_turn():
    """Finalize for turn N replaces (or supersedes) turn N-1 in the rag body."""
    driver = TuiDriver()
    driver.finalize(_turn(index=0, rag_text="first answer"))
    driver.finalize(_turn(index=1, rag_text="second answer"))
    assert "second answer" in driver.rag_body


def test_when_second_turn_finalized_then_cag_body_reflects_latest_turn():
    driver = TuiDriver()
    driver.finalize(_turn(index=0, cag_text="first cag"))
    driver.finalize(_turn(index=1, cag_text="second cag"))
    assert "second cag" in driver.cag_body


# ===========================================================================
# Criterion 9 [T3→unit with monkeypatch]: run_tui iterates run_session
# "async entrypoint … iterates run_session(…) and updates display once per
#  TurnResult"
# ===========================================================================


def _patch_live():
    """Context-manager that replaces rich.live.Live with a no-op mock."""
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=MagicMock())
    mock.__exit__ = MagicMock(return_value=False)
    return patch("flashcard.tui.Live", return_value=mock)


@pytest.mark.asyncio
async def test_when_run_tui_called_then_run_session_is_invoked_once(monkeypatch):
    """Criterion: 'iterates run_session(...)' — exactly one call to run_session."""
    call_log: list[int] = []

    async def spy_session(*args: Any, **kwargs: Any):
        call_log.append(1)
        if False:
            yield  # makes this an async generator

    monkeypatch.setattr("flashcard.tui.run_session", spy_session)
    with _patch_live():
        await run_tui(object())

    assert len(call_log) == 1


@pytest.mark.asyncio
async def test_when_run_tui_receives_cfg_then_run_session_gets_same_cfg(monkeypatch):
    """Criterion: 'same model and max_tokens' — cfg forwarded unchanged to run_session."""
    received_cfg: list[Any] = []

    async def capturing_session(cfg: Any, **kwargs: Any):
        received_cfg.append(cfg)
        if False:
            yield

    monkeypatch.setattr("flashcard.tui.run_session", capturing_session)
    sentinel_cfg = object()
    with _patch_live():
        await run_tui(sentinel_cfg)

    assert received_cfg and received_cfg[0] is sentinel_cfg


@pytest.mark.asyncio
async def test_when_run_tui_called_with_two_turns_then_finalize_called_twice(
    monkeypatch,
):
    """Criterion: 'updates the display once per TurnResult'."""
    turns = [_turn(0), _turn(1)]
    finalize_calls: list[int] = []

    monkeypatch.setattr("flashcard.tui.run_session", _fake_session(turns))

    original_finalize = TuiDriver.finalize

    def spy_finalize(self: TuiDriver, turn: Any) -> None:
        finalize_calls.append(turn.index)
        original_finalize(self, turn)

    monkeypatch.setattr(TuiDriver, "finalize", spy_finalize)
    with _patch_live():
        await run_tui(object())

    assert len(finalize_calls) == 2
    assert finalize_calls == [0, 1]


@pytest.mark.asyncio
async def test_when_run_tui_called_then_on_token_kwarg_is_passed_to_run_session(
    monkeypatch,
):
    """Criterion: 'driver-owned token sink (on_token)' — run_session must receive it."""
    received_kwargs: dict[str, Any] = {}

    async def capturing_session(*args: Any, **kwargs: Any):
        received_kwargs.update(kwargs)
        if False:
            yield

    monkeypatch.setattr("flashcard.tui.run_session", capturing_session)
    with _patch_live():
        await run_tui(object())

    assert "on_token" in received_kwargs
    assert callable(received_kwargs["on_token"])


@pytest.mark.asyncio
async def test_when_run_tui_called_with_retriever_then_retriever_forwarded(monkeypatch):
    """Criterion: 'run_tui(cfg, *, retriever=None)' — retriever kwarg reaches run_session."""
    received_kwargs: dict[str, Any] = {}

    async def capturing_session(*args: Any, **kwargs: Any):
        received_kwargs.update(kwargs)
        if False:
            yield

    monkeypatch.setattr("flashcard.tui.run_session", capturing_session)
    sentinel_retriever = object()
    with _patch_live():
        await run_tui(object(), retriever=sentinel_retriever)

    assert received_kwargs.get("retriever") is sentinel_retriever


@pytest.mark.asyncio
async def test_when_run_tui_called_with_zero_turns_then_no_exception_raised(
    monkeypatch,
):
    """Edge case: empty session (no questions) must not raise."""
    monkeypatch.setattr("flashcard.tui.run_session", _empty_session)
    with _patch_live():
        await run_tui(object())  # must not raise


@pytest.mark.asyncio
async def test_when_run_tui_processes_turn_then_rag_body_is_updated_from_rag_text(
    monkeypatch,
):
    """End-to-end: run_tui must produce a driver whose rag_body reflects the last turn's text.

    This test drives run_tui with a real TuiDriver (not mocked) to verify the
    token sink + finalize wiring without a real terminal.
    """
    final_turn = _turn(0, rag_text="expected RAG text", cag_text="expected CAG text")
    monkeypatch.setattr("flashcard.tui.run_session", _fake_session([final_turn]))

    captured_driver: list[TuiDriver] = []
    original_init = TuiDriver.__init__

    def spy_init(self: TuiDriver, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        captured_driver.append(self)

    monkeypatch.setattr(TuiDriver, "__init__", spy_init)
    with _patch_live():
        await run_tui(object())

    assert captured_driver, "TuiDriver was not instantiated by run_tui"
    driver = captured_driver[0]
    assert "expected RAG text" in driver.rag_body
    assert "expected CAG text" in driver.cag_body


# ===========================================================================
# Criterion 12 [UNIT]: smooth updates — refresh_per_second is positive
# "tuned refresh_per_second, bounded body regions so the layout does not jump"
# ===========================================================================


@pytest.mark.asyncio
async def test_when_run_tui_creates_live_then_refresh_per_second_is_positive(
    monkeypatch,
):
    """Criterion: 'tuned refresh_per_second' — Live must be created with rps > 0."""
    monkeypatch.setattr("flashcard.tui.run_session", _empty_session)

    with patch("flashcard.tui.Live") as MockLive:
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=MagicMock())
        mock_instance.__exit__ = MagicMock(return_value=False)
        MockLive.return_value = mock_instance
        await run_tui(object())

    assert MockLive.called, "Live was never constructed"
    _, kwargs = MockLive.call_args
    rps = kwargs.get("refresh_per_second")
    assert rps is not None and rps > 0, (
        f"Expected positive refresh_per_second, got {rps!r}"
    )


# ===========================================================================
# Property-based tests — invariants derived from acceptance criteria
# ===========================================================================


@given(st.text())
def test_when_on_token_called_with_any_text_for_rag_then_no_exception_raised(
    partial: str,
) -> None:
    """Never-raises invariant: on_token accepts any string for 'rag' side."""
    driver = TuiDriver()
    driver.on_token("rag", partial)  # must not raise


@given(st.text())
def test_when_on_token_called_with_any_text_for_cag_then_no_exception_raised(
    partial: str,
) -> None:
    """Never-raises invariant: on_token accepts any string for 'cag' side."""
    driver = TuiDriver()
    driver.on_token("cag", partial)  # must not raise


@given(rag_text=st.text(min_size=1), cag_text=st.text(min_size=1))
@settings(max_examples=50)
def test_when_finalize_called_with_any_texts_then_no_exception_raised(
    rag_text: str, cag_text: str
) -> None:
    """Never-raises invariant: finalize accepts any non-empty text content."""
    driver = TuiDriver()
    driver.finalize(_turn(rag_text=rag_text, cag_text=cag_text))  # must not raise


@given(
    partials=st.lists(st.text(), min_size=1, max_size=20),
)
@settings(max_examples=40)
def test_when_rag_partials_are_appended_then_rag_body_is_monotonically_growing(
    partials: list[str],
) -> None:
    """Ordering invariant: appending never shrinks the body buffer length."""
    driver = TuiDriver()
    prev_len = len(driver.rag_body)
    for p in partials:
        driver.on_token("rag", p)
        assert len(driver.rag_body) >= prev_len
        prev_len = len(driver.rag_body)


@given(n_turns=st.integers(min_value=1, max_value=8))
@settings(max_examples=30)
def test_when_n_turns_are_finalized_then_last_turns_text_is_in_rag_body(
    n_turns: int,
) -> None:
    """Ordering invariant: the final finalize call's rag_text is always present."""
    driver = TuiDriver()
    for i in range(n_turns):
        driver.finalize(_turn(index=i, rag_text=f"turn-{i}-rag"))
    assert f"turn-{n_turns - 1}-rag" in driver.rag_body


@given(n_turns=st.integers(min_value=1, max_value=8))
@settings(max_examples=30)
def test_when_n_turns_are_finalized_then_last_turns_text_is_in_cag_body(
    n_turns: int,
) -> None:
    """Ordering invariant: the final finalize call's cag_text is always present."""
    driver = TuiDriver()
    for i in range(n_turns):
        driver.finalize(_turn(index=i, cag_text=f"turn-{i}-cag"))
    assert f"turn-{n_turns - 1}-cag" in driver.cag_body
