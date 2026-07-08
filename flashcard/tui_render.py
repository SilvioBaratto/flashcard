"""Pure Rich renderable builders for the RAG-vs-CAG side-by-side TUI (issue #16).

All four functions are side-effect free: same VM → same output, no I/O, no Live.
"""

from __future__ import annotations

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from flashcard.tui_state import CagHeaderVM, FooterVM, RagHeaderVM, format_cost

_RAG_COLOR = "cyan"
_CAG_COLOR = "green"


def rag_header(vm: RagHeaderVM) -> RenderableType:
    chunk_str = ", ".join(str(c) for c in vm.chunk_ids) if vm.chunk_ids else "—"
    grid = Table.grid(padding=(0, 1))
    grid.add_column()
    grid.add_column()
    grid.add_row(
        Text(f"Q{vm.turn_number}", style=f"bold {_RAG_COLOR}"),
        Text(format_cost(vm.cumulative_cost), style="bold"),
    )
    grid.add_row(
        Text(f"chunks: {chunk_str}", style=_RAG_COLOR),
        Text(f"ctx: {vm.context_tokens} tok"),
    )
    return Panel(grid, title="[bold cyan]RAG[/]", border_style=_RAG_COLOR)


def cag_header(vm: CagHeaderVM) -> RenderableType:
    grid = Table.grid(padding=(0, 1))
    grid.add_column()
    grid.add_column()
    grid.add_row(
        Text(f"Q{vm.turn_number}", style=f"bold {_CAG_COLOR}"),
        Text(format_cost(vm.cumulative_cost), style="bold"),
    )
    grid.add_row(
        Text(f"cache-read: {vm.cache_read_tokens} tok", style=_CAG_COLOR),
        Text(f"ctx: {vm.context_size} tok"),
    )
    return Panel(grid, title="[bold green]CAG[/]", border_style=_CAG_COLOR)


def panel_body(text: str, side: str) -> RenderableType:
    color = _RAG_COLOR if side == "rag" else _CAG_COLOR
    return Panel(Text(text), border_style=color)


def footer_strip(vm: FooterVM) -> RenderableType:
    if vm.cheaper_multiple is not None and vm.cheaper_side is not None:
        side_label = str(getattr(vm.cheaper_side, "value", vm.cheaper_side)).upper()
        cheaper_cell = Text(
            f"{side_label} {vm.cheaper_multiple:.1f}× cheaper", style="bold yellow"
        )
    else:
        cheaper_cell = Text("—", style="bold yellow")

    grid = Table.grid(padding=(0, 2))
    grid.add_column()
    grid.add_column()
    grid.add_column()
    grid.add_column()
    grid.add_row(
        Text(f"RAG: {format_cost(vm.rag_cost)}", style=f"bold {_RAG_COLOR}"),
        Text(f"CAG: {format_cost(vm.cag_cost)}", style=f"bold {_CAG_COLOR}"),
        cheaper_cell,
        Text(f"miss: {vm.rag_miss_count}  recall: {vm.cag_full_recall_count}"),
    )
    grid.add_row(
        Text(f"RAG cache: {vm.rag_cache_read_total}"),
        Text(f"CAG cache: {vm.cag_cache_read_total}"),
        Text(""),
        Text(""),
    )
    return Panel(grid, title="[bold]Metrics[/]")
