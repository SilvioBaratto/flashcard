"""Rich Live two-panel TUI driver for RAG-vs-CAG side-by-side display.

Wires TuiState (#15) and renderables (#16) to the runner's async stream.
Two update paths — on_token (streaming partials) and finalize (per-TurnResult) —
both owned by TuiDriver and funnelled through a single refresh() call.

Issue #21: run_tui drives resilient_session (not run_session directly), passing
on_token + on_note. A thin note region shows the transient 'retrying…' message
on 429/5xx and clears on recovery.
"""
from __future__ import annotations

from typing import Any, Callable

from rich.layout import Layout
from rich.live import Live

from flashcard.resilience import resilient_session as _resilient_session_impl
from flashcard.runner import run_session
from flashcard.tui_render import cag_header, footer_strip, panel_body, rag_header
from flashcard.tui_state import TuiState, verdict_to_markers  # noqa: F401 (re-exported)

_REFRESH_PER_SECOND = 10
_HEADER_SIZE = 7
_FOOTER_SIZE = 7


# ---------------------------------------------------------------------------
# Verdict display helpers (pure; used by the TUI and by external callers)
# ---------------------------------------------------------------------------


def format_verdict_markers(
    rag_missed: bool = False, cag_full_recall: bool = False
) -> tuple[str, str]:
    """Return (rag_marker, cag_marker) display strings from verdict booleans."""
    rag_marker = "⚠ missed …" if rag_missed else ""
    cag_marker = "✓ full recall" if cag_full_recall else ""
    return rag_marker, cag_marker


# ---------------------------------------------------------------------------
# resilient_session coroutine bridge
#
# Declared as a plain coroutine (no yield) so unittest.mock.patch() detects
# it as a coroutine function and creates AsyncMock.  AsyncMock awaits the
# side_effect, which lets test helpers record kwargs before any iteration error.
# In production the returned object is the real async generator from resilience.
# ---------------------------------------------------------------------------


async def resilient_session(cfg: Any, **kwargs: Any) -> Any:
    """Return the resilient_session async generator for the given config."""
    return _resilient_session_impl(cfg, **kwargs)


# ---------------------------------------------------------------------------
# Layout builder (pure; returns a named tree with rag / cag / note / footer)
# ---------------------------------------------------------------------------


def _build_layout() -> Layout:
    root = Layout()
    root.split_column(
        Layout(name="columns"),
        Layout(name="note", size=1),
        Layout(name="footer", size=_FOOTER_SIZE),
    )
    root["columns"].split_row(Layout(name="rag"), Layout(name="cag"))
    root["rag"].split_column(
        Layout(name="rag_header", size=_HEADER_SIZE), Layout(name="rag_body")
    )
    root["cag"].split_column(
        Layout(name="cag_header", size=_HEADER_SIZE), Layout(name="cag_body")
    )
    return root


# ---------------------------------------------------------------------------
# Token adapter: single-arg TokenEvent → driver.on_token(side, partial)
# ---------------------------------------------------------------------------


def _token_adapter(driver: "TuiDriver") -> Callable:
    def _cb(event: Any) -> None:
        driver.on_token(getattr(event, "side", ""), getattr(event, "partial", ""))
    return _cb


# ---------------------------------------------------------------------------
# TuiDriver — owns state, body buffers, note buffer, and the one refresh path
# ---------------------------------------------------------------------------


class TuiDriver:
    """Owns Layout, body buffers, note buffer, and TuiState; drives live display.

    Update paths:
    * on_token(side, partial) — appends streamed partial to the side buffer
    * on_note(msg)            — sets or clears the transient retry note
    * finalize(turn)          — overwrites buffers from TurnResult, applies state
    All paths render through refresh(live).
    """

    def __init__(self) -> None:
        self._rag_body = ""
        self._cag_body = ""
        self._note: str | None = None
        self._state = TuiState()
        self._layout = _build_layout()

    @property
    def layout(self) -> Layout:
        return self._layout

    @property
    def rag_body(self) -> str:
        return self._rag_body

    @property
    def cag_body(self) -> str:
        return self._cag_body

    def on_token(self, side: str, partial: str) -> None:
        """Append a streamed partial to the correct side's body buffer."""
        if side == "rag":
            self._rag_body += partial
        else:
            self._cag_body += partial

    def on_note(self, msg: str | None) -> None:
        """Set or clear the transient retry note; updates the layout immediately."""
        self._note = msg
        self._layout["note"].update(msg or "")

    def finalize(self, turn: Any) -> None:
        """Overwrite body buffers from TurnResult text and fold turn into state."""
        self._rag_body = turn.rag.text
        self._cag_body = turn.cag.text
        self._state.apply(turn)

    def refresh(self, live: Any) -> None:
        """Rebuild all layout regions from current state + body buffers."""
        self._layout["rag_header"].update(rag_header(self._state.rag_header))
        self._layout["rag_body"].update(panel_body(self._rag_body, "rag"))
        self._layout["cag_header"].update(cag_header(self._state.cag_header))
        self._layout["cag_body"].update(panel_body(self._cag_body, "cag"))
        self._layout["footer"].update(footer_strip(self._state.footer))
        self._layout["note"].update(self._note or "")
        live.refresh()


# ---------------------------------------------------------------------------
# Public async entrypoint
# ---------------------------------------------------------------------------


async def run_tui(cfg: Any, *, retriever: Any = None) -> None:
    """Drive resilient_session and update the TUI once per turn."""
    driver = TuiDriver()

    def factory(c, *, on_token):
        return run_session(c, on_token=on_token, retriever=retriever)

    with Live(driver.layout, refresh_per_second=_REFRESH_PER_SECOND,
              screen=False) as live:
        gen = await resilient_session(
            cfg, on_token=_token_adapter(driver), on_note=driver.on_note,
            session_factory=factory
        )
        async for turn in gen:
            driver.finalize(turn)
            driver.refresh(live)
