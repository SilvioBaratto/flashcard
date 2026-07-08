"""flashcard game — RAG-or-CAG scenario quiz (pure local, no API)."""
from __future__ import annotations

from dataclasses import dataclass

import typer
from rich.console import Console
from rich.panel import Panel

_CHOICES: tuple[str, ...] = ("RAG", "CAG", "Both")


@dataclass(frozen=True)
class Rationale:
    corpus_size: str
    freshness: str
    latency: str
    citation_need: str


@dataclass(frozen=True)
class Scenario:
    description: str
    correct_choice: str
    rationale: Rationale
    valid_choices: tuple[str, ...] = _CHOICES

    def accept_choice(self, choice: str) -> bool:
        """Return True when the choice matches the correct answer."""
        return choice == self.correct_choice


_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        description=(
            "Scenario 1 — IT help-desk bot\n"
            "A 200-page product manual, updated only a few times per year, "
            "powers a help-desk bot that answers employee questions."
        ),
        correct_choice="CAG",
        rationale=Rationale(
            corpus_size=(
                "~200 pages — fits comfortably inside most LLM context windows."
            ),
            freshness="Updated rarely — the cached prefix stays valid for months.",
            latency="Single forward pass once cached — sub-second query responses.",
            citation_need=(
                "Users need concise answers, not citations to specific manual pages."
            ),
        ),
    ),
    Scenario(
        description=(
            "Scenario 2 — Legal research assistant\n"
            "A law firm needs to search across thousands of cases, constantly "
            "updated with new rulings and amendments. Lawyers require accurate "
            "citations to the relevant legal documents."
        ),
        correct_choice="RAG",
        rationale=Rationale(
            corpus_size=(
                "Thousands of cases — far exceeds any context window; "
                "retrieval fetches only the relevant chunks per query."
            ),
            freshness=(
                "Constantly updated — incremental vector-index updates "
                "require no full cache recomputation."
            ),
            latency=(
                "Retrieval overhead is acceptable given the corpus size "
                "and the need for precision."
            ),
            citation_need=(
                "RAG surfaces the source document for each retrieved chunk, "
                "enabling precise legal citations."
            ),
        ),
    ),
    Scenario(
        description=(
            "Scenario 3 — Clinical decision support\n"
            "Doctors query patient records, treatment guidelines, and drug "
            "interactions. Answers must be highly accurate; doctors frequently "
            "ask complex follow-up questions."
        ),
        correct_choice="Both",
        rationale=Rationale(
            corpus_size=(
                "Huge corpus (records + guidelines) — RAG narrows it to a "
                "manageable slice; CAG holds that slice in working context."
            ),
            freshness=(
                "Records update continuously — RAG handles dynamic retrieval; "
                "the retrieved slice is held by CAG for the session."
            ),
            latency=(
                "Follow-ups benefit from CAG: the retrieved slice stays in "
                "context across turns without re-querying the database."
            ),
            citation_need=(
                "RAG supplies traceable source chunks; CAG ensures coherent "
                "reasoning across multi-turn follow-up questions."
            ),
        ),
    ),
)


def get_scenarios() -> tuple[Scenario, ...]:
    """Return the three script_6.md scenarios in canonical order."""
    return _SCENARIOS


def _verdict(scenario: Scenario, choice: str) -> str:
    if scenario.accept_choice(choice):
        return "[bold green]✓ Correct![/bold green]"
    return (
        f"[bold red]✗ Not quite — "
        f"the answer is {scenario.correct_choice}.[/bold red]"
    )


def _build_reveal(scenario: Scenario, choice: str) -> str:
    """Return Rich-markup reveal text for all four reasoning axes."""
    r = scenario.rationale
    return (
        f"{_verdict(scenario, choice)}\n\n"
        f"[bold]Answer:[/bold] {scenario.correct_choice}\n\n"
        f"[cyan]Corpus size:[/cyan]   {r.corpus_size}\n"
        f"[cyan]Freshness:[/cyan]     {r.freshness}\n"
        f"[cyan]Latency:[/cyan]       {r.latency}\n"
        f"[cyan]Citation need:[/cyan] {r.citation_need}"
    )


def play(console: Console | None = None) -> None:
    """Run the RAG-or-CAG quiz through all three scenarios."""
    con = console or Console()
    for scenario in get_scenarios():
        con.print(f"\n[bold]{scenario.description}[/bold]")
        choice = typer.prompt("\nYour answer (RAG / CAG / Both)")
        con.print(Panel(_build_reveal(scenario, choice), border_style="bright_blue"))
    con.print("\n[bold green]Game complete — all three scenarios covered![/bold green]")
