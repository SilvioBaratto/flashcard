"""Rich Live two-panel TUI driver for RAG-vs-CAG side-by-side display (issue #17).

Wires TuiState (#15) and renderables (#16) to the runner's async stream.
Two update paths — on_token (streaming partials) and finalize (per-TurnResult) —
both owned by TuiDriver and funnelled through a single refresh() call.

Scope: layout construction and display only. No CLI flags, cost-guard,
judge, or resilience wiring (those belong to Cycle 5).
"""
from __future__ import annotations

from typing import Any

from rich.layout import Layout
from rich.live import Live

from flashcard.runner import run_session
from flashcard.tui_render import cag_header, footer_strip, panel_body, rag_header
from flashcard.tui_state import TuiState

_REFRESH_PER_SECOND = 10
_HEADER_SIZE = 7
_FOOTER_SIZE = 7


# ---------------------------------------------------------------------------
# Layout builder (pure; returns a named tree with rag / cag / footer leaves)
# ---------------------------------------------------------------------------


def _build_layout() -> Layout:
    root = Layout()
    root.split_column(
        Layout(name="columns"),
        Layout(name="footer", size=_FOOTER_SIZE),
    )
    root["columns"].split_row(
        Layout(name="rag"),
        Layout(name="cag"),
    )
    root["rag"].split_column(
        Layout(name="rag_header", size=_HEADER_SIZE),
        Layout(name="rag_body"),
    )
    root["cag"].split_column(
        Layout(name="cag_header", size=_HEADER_SIZE),
        Layout(name="cag_body"),
    )
    return root


# ---------------------------------------------------------------------------
# TuiDriver — owns state, body buffers, and the one refresh path
# ---------------------------------------------------------------------------


class TuiDriver:
    """Owns the Layout, body buffers, and TuiState; drives the live display.

    Two update paths:
    * on_token(side, partial) — appends a streamed partial to the side buffer
    * finalize(turn)          — overwrites buffers from TurnResult, applies state
    Both paths are wired through refresh(live) which rebuilds all layout regions.
    """

    def __init__(self) -> None:
        self._rag_body = ""
        self._cag_body = ""
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
        live.refresh()


# ---------------------------------------------------------------------------
# Public async entrypoint (Cycle 5's cli.py will call this)
# ---------------------------------------------------------------------------


async def run_tui(cfg: Any, *, retriever: Any = None) -> None:
    """Stream TurnResults from run_session and update the TUI once per turn."""
    driver = TuiDriver()
    with Live(
        driver.layout, refresh_per_second=_REFRESH_PER_SECOND, screen=False
    ) as live:
        async for turn in run_session(
            cfg, on_token=driver.on_token, retriever=retriever
        ):
            driver.finalize(turn)
            driver.refresh(live)
