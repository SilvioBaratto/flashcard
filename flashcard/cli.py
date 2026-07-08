"""flashcard CLI entrypoint.

Runs the same questions through a RAG pipeline and a CAG pipeline and renders
both side by side. This is a scaffold — the pipelines are not implemented yet.
"""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(add_completion=False, help="RAG vs CAG, side by side.")
console = Console()


@app.command()
def main(
    model: str = typer.Option("sonnet-4.6", help="Generation model."),
    doc: str | None = typer.Option(None, help="Path to the knowledge source."),
    questions_file: str | None = typer.Option(None, help="Path to a questions JSON file."),
    top_k: int = typer.Option(3, help="RAG: chunks retrieved per question."),
    chunk_size: int = typer.Option(800, help="RAG: chunk length in tokens."),
    ttl: str = typer.Option("5m", help="CAG cache TTL: 5m or 1h."),
    max_tokens: int = typer.Option(400, help="Max output tokens per answer."),
    delay: float = typer.Option(0.0, help="Pacing delay between questions (seconds)."),
) -> None:
    """Answer each question twice — via RAG and via CAG — and compare live."""
    console.print("[bold]flashcard[/bold] — RAG vs CAG, side by side")
    console.print("[yellow]Pipelines not implemented yet.[/yellow] This is a scaffold.")
    console.print(
        f"config: model={model} doc={doc or '(built-in)'} "
        f"top_k={top_k} chunk_size={chunk_size} ttl={ttl} max_tokens={max_tokens}"
    )


if __name__ == "__main__":
    app()
