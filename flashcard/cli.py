"""flashcard CLI entrypoint.

Runs the same questions through a RAG pipeline and a CAG pipeline and renders
both side by side. The cost-guard prints a worst-case spend estimate and
requires confirmation before any real API call fires.
"""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from flashcard.corpus import load_corpus
from flashcard.cost_guard import Estimate, count_tokens, estimate_worst_case
from flashcard.pricing import Model
from flashcard.questions import load_questions
from flashcard.runner import RunConfig
from flashcard.tui import run_tui

app = typer.Typer(add_completion=False, help="RAG vs CAG, side by side.")
console = Console()

# Re-export for tests that import from flashcard.cli
__all__ = ["app", "count_tokens", "estimate_worst_case"]


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    model: str = typer.Option("gpt-5.2", help="Generation model (OpenAI)."),
    doc: str | None = typer.Option(None, help="Path to the knowledge source."),
    questions_file: str | None = typer.Option(
        None, help="Path to a questions JSON file."
    ),
    top_k: int = typer.Option(3, help="RAG: chunks retrieved per question."),
    chunk_size: int = typer.Option(800, help="RAG: chunk length in tokens."),
    max_tokens: int = typer.Option(400, help="Max output tokens per answer."),
    delay: float = typer.Option(0.0, help="Pacing delay between questions (seconds)."),
    yes: bool = typer.Option(
        False,
        "--yes/--no-yes",
        help="Skip confirmation (non-interactive / recording runs).",
    ),
    judge: bool = typer.Option(
        False,
        "--judge/--no-judge",
        help="Run the optional CompareAnswers divergence judge (gpt-5-mini).",
    ),
) -> None:
    """Answer each question twice — via RAG and via CAG — and compare live."""
    if ctx.invoked_subcommand is not None:
        return
    document = load_corpus(doc)
    questions = load_questions(questions_file)

    est = estimate_worst_case(
        document, questions, max_tokens, model, top_k=top_k, chunk_size=chunk_size
    )
    _display_estimate(est, len(questions), model)

    if not yes and not typer.confirm("\nProceed with real API calls?", default=False):
        raise typer.Exit()

    cfg = _build_config(
        model, doc, questions_file, top_k, chunk_size, max_tokens, delay, judge
    )
    asyncio.run(run_tui(cfg))


@app.command("game")
def game_cmd() -> None:
    """Play the RAG-or-CAG scenario quiz (pure local, no API)."""
    from flashcard.game import play  # noqa: PLC0415

    play()


def _build_config(
    model: str,
    doc: str | None,
    questions_file: str | None,
    top_k: int,
    chunk_size: int,
    max_tokens: int,
    delay: float,
    judge: bool,
) -> RunConfig:
    return RunConfig(
        model=Model(model),
        doc=doc,
        questions_file=questions_file,
        top_k=top_k,
        chunk_size=chunk_size,
        max_tokens=max_tokens,
        delay=delay,
        judge=judge,
    )


def _display_estimate(est: Estimate, n_questions: int, model: str) -> None:
    """Render the worst-case spend estimate as a Rich table."""
    console.print()
    console.print(
        f"[bold cyan]flashcard[/bold cyan] — cost-guard estimate "
        f"({n_questions} questions, model=[bold]{model}[/bold])"
    )

    table = Table(show_header=True, header_style="bold magenta", min_width=52)
    table.add_column("Line item", style="cyan")
    table.add_column("USD (worst case)", justify="right", style="bold yellow")

    table.add_row("RAG  — all questions", f"${est.rag_total:.4f}")
    table.add_row("CAG  — Q1 (full doc, base rate)", f"${est.cag_q1:.4f}")
    table.add_row(
        f"CAG  — all {n_questions} questions (cache cold)", f"${est.cag_total:.4f}"
    )
    table.add_row(
        "[bold]Combined worst case[/bold]", f"[bold]${est.combined:.4f}[/bold]"
    )

    console.print(table)
    console.print(
        "[dim]Worst case: every CAG turn re-pays the full document at base rate "
        "(cache cold / TTL miss). Actual spend is likely lower once the prefix is "
        "warm.[/dim]"
    )


if __name__ == "__main__":
    app()
