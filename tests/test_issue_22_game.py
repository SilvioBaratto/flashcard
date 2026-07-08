"""Tests for issue #22: flashcard game scenario quiz (flashcard/game.py).

Source-blind authoring: all tests derived exclusively from acceptance criteria
and requirements.md. No implementation files were read.

Criteria covered (per oracle classification):
  [UNIT] `flashcard game` runs the three script_6.md scenarios in order.
  [UNIT] Each scenario prompts for RAG / CAG / Both and accepts the choice.

Criteria skipped (oracle: NOT VERIFIABLE):
  After each answer, the reveal shows the correct choice and reasoning
    across corpus size, freshness, latency, and citation need (subjective/visual).
  Correct answers: scenario 1 → CAG, scenario 2 → RAG, scenario 3 → Both
    (no concrete runtime assertion inferable without reading implementation).
  No network / API calls anywhere in the game path
    (no injectable sentinel or call-counter observable from spec alone).
  All tests pass (boilerplate suite gate; no per-criterion assertion).
  SOLID, clean code — methods < 10 lines, etc. (subjective prose).

Interface contract derived from criteria
(the implementation must satisfy this, not the other way around):
  flashcard.game.get_scenarios() -> Iterable[Scenario]
    Returns exactly three Scenario objects in a fixed, deterministic order.

  Scenario.description: str
    Non-empty text describing the situation shown to the user as the prompt.

  Scenario.valid_choices: Collection[str]
    Must contain at least "RAG", "CAG", and "Both".

  Scenario.accept_choice(choice: str) -> Any
    Must not raise for any choice in {"RAG", "CAG", "Both"}.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from flashcard.cli import app

_cli_runner = CliRunner()

_VALID_CHOICES = ["RAG", "CAG", "Both"]


# ---------------------------------------------------------------------------
# Helper: import get_scenarios with a clear Red-phase failure message.
# ---------------------------------------------------------------------------


def _get_scenarios():
    """Import and call get_scenarios(), failing loudly if the module is absent."""
    try:
        from flashcard.game import get_scenarios  # noqa: PLC0415
    except ImportError as exc:
        pytest.fail(
            f"Cannot import flashcard.game.get_scenarios: {exc}. "
            "Create flashcard/game.py and implement get_scenarios() returning "
            "the three script_6.md scenarios."
        )
    return list(get_scenarios())


# ---------------------------------------------------------------------------
# Criterion: `flashcard game` runs the three script_6.md scenarios in order.
# ---------------------------------------------------------------------------


def test_when_game_subcommand_invoked_with_help_then_it_is_recognised():
    """`flashcard game` must be a registered Typer subcommand.

    `flashcard game --help` exiting 0 proves the command is wired into the CLI.
    'No such command' in the output is the canonical Typer signal that it is not.
    """
    result = _cli_runner.invoke(app, ["game", "--help"])
    assert "No such command" not in result.output, (
        "The `game` subcommand is not registered on the CLI. "
        "Add it in flashcard/cli.py (e.g. app.add_typer or @app.command)."
    )
    assert result.exit_code == 0, (
        f"`flashcard game --help` exited {result.exit_code}:\n{result.output}"
    )


def test_when_scenarios_are_requested_then_exactly_three_are_returned():
    """The game must expose exactly three scenarios.

    Criterion: '`flashcard game` runs the three script_6.md scenarios in order'.
    """
    scenarios = _get_scenarios()
    assert len(scenarios) == 3, (
        f"Expected exactly 3 scenarios (one per script_6.md situation), "
        f"got {len(scenarios)}. "
        "Implement the three scenarios: IT help-desk manual, updated legal cases, "
        "clinical decision support."
    )


def test_when_scenarios_are_requested_twice_then_order_is_stable():
    """The scenario sequence must be deterministic across repeated calls.

    Criterion: 'runs … in order' — the ordered sequence is fixed, not random.
    """
    from flashcard.game import get_scenarios  # noqa: PLC0415

    first_titles = [s.description for s in get_scenarios()]
    second_titles = [s.description for s in get_scenarios()]
    assert first_titles == second_titles, (
        "Scenario order changed between calls. "
        "get_scenarios() must return the same fixed sequence every time."
    )


@pytest.mark.parametrize("scenario_index", [0, 1, 2])
def test_when_scenario_is_inspected_then_description_is_a_non_empty_string(
    scenario_index: int,
) -> None:
    """Every scenario must carry a non-empty description that can be shown to the user.

    Criterion: 'each scenario prompts …' — the prompt text must exist and be legible.
    """
    scenario = _get_scenarios()[scenario_index]
    assert hasattr(scenario, "description"), (
        f"Scenario {scenario_index + 1} has no 'description' attribute. "
        "Each Scenario must expose description: str."
    )
    assert isinstance(scenario.description, str) and scenario.description.strip(), (
        f"Scenario {scenario_index + 1} description must be a non-empty string, "
        f"got {scenario.description!r}."
    )


# ---------------------------------------------------------------------------
# Criterion: Each scenario prompts for RAG / CAG / Both and accepts the choice.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_index", [0, 1, 2])
def test_when_scenario_is_inspected_then_rag_is_a_valid_choice(
    scenario_index: int,
) -> None:
    """'RAG' must be listed as a selectable choice for every scenario.

    Criterion: 'each scenario prompts for RAG / CAG / Both'.
    """
    scenario = _get_scenarios()[scenario_index]
    assert hasattr(scenario, "valid_choices"), (
        f"Scenario {scenario_index + 1} has no 'valid_choices' attribute."
    )
    assert "RAG" in scenario.valid_choices, (
        f"'RAG' not in scenario {scenario_index + 1}.valid_choices "
        f"({scenario.valid_choices!r}). All three choices must be offered."
    )


@pytest.mark.parametrize("scenario_index", [0, 1, 2])
def test_when_scenario_is_inspected_then_cag_is_a_valid_choice(
    scenario_index: int,
) -> None:
    """'CAG' must be listed as a selectable choice for every scenario."""
    scenario = _get_scenarios()[scenario_index]
    assert "CAG" in scenario.valid_choices, (
        f"'CAG' not in scenario {scenario_index + 1}.valid_choices "
        f"({scenario.valid_choices!r})."
    )


@pytest.mark.parametrize("scenario_index", [0, 1, 2])
def test_when_scenario_is_inspected_then_both_is_a_valid_choice(
    scenario_index: int,
) -> None:
    """'Both' must be listed as a selectable choice for every scenario."""
    scenario = _get_scenarios()[scenario_index]
    assert "Both" in scenario.valid_choices, (
        f"'Both' not in scenario {scenario_index + 1}.valid_choices "
        f"({scenario.valid_choices!r})."
    )


@pytest.mark.parametrize("scenario_index", [0, 1, 2])
@pytest.mark.parametrize("choice", _VALID_CHOICES)
def test_when_valid_choice_is_submitted_then_scenario_does_not_raise(
    scenario_index: int,
    choice: str,
) -> None:
    """Every scenario must accept any of the three valid choices without raising.

    Criterion: 'each scenario … accepts the choice'.
    """
    scenario = _get_scenarios()[scenario_index]
    assert hasattr(scenario, "accept_choice"), (
        f"Scenario {scenario_index + 1} has no 'accept_choice' method. "
        "Each Scenario must expose accept_choice(choice: str)."
    )
    try:
        scenario.accept_choice(choice)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"scenario[{scenario_index}].accept_choice({choice!r}) raised "
            f"{type(exc).__name__}: {exc}. "
            f"accept_choice must not raise for any choice in {_VALID_CHOICES}."
        )


def test_when_game_played_with_one_valid_choice_per_scenario_then_cli_exits_successfully():
    """`flashcard game` must complete without error given one valid choice per scenario.

    Simulates a user responding to each of the three scenario prompts by feeding
    three newline-separated inputs to the CLI's stdin.
    The criterion 'runs the three scenarios in order' implies the command must
    finish processing all three and exit cleanly.
    """
    user_input = "RAG\nCAG\nBoth\n"
    result = _cli_runner.invoke(app, ["game"], input=user_input)
    assert result.exit_code == 0, (
        f"`flashcard game` exited {result.exit_code} when given 'RAG / CAG / Both':\n"
        f"{result.output}"
    )


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(choice=st.sampled_from(_VALID_CHOICES))
@settings(max_examples=6)
def test_when_any_valid_choice_is_given_then_every_scenario_accepts_it_without_raising(
    choice: str,
) -> None:
    """Never-raises invariant: accept_choice is a total function over its stated domain.

    Criterion: 'each scenario prompts for RAG / CAG / Both and accepts the choice'.
    For every scenario × every valid choice the method must not raise.
    This is a never-raises-for-valid-input invariant.
    """
    from flashcard.game import get_scenarios  # noqa: PLC0415

    for idx, scenario in enumerate(get_scenarios()):
        try:
            scenario.accept_choice(choice)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(
                f"scenario[{idx}].accept_choice({choice!r}) raised "
                f"{type(exc).__name__}: {exc}. "
                f"accept_choice must be total over {_VALID_CHOICES}."
            )


@given(
    choices=st.lists(
        st.sampled_from(_VALID_CHOICES),
        min_size=3,
        max_size=3,
    )
)
@settings(max_examples=8)
def test_when_any_combination_of_valid_choices_given_then_cli_exits_successfully(
    choices: list[str],
) -> None:
    """Ordering invariant: the CLI exits 0 for every permutation of valid choices.

    Criterion: 'accepts the choice' — the game must not hard-code which choice
    ends it; any combination of three valid answers must complete cleanly.
    """
    user_input = "\n".join(choices) + "\n"
    result = _cli_runner.invoke(app, ["game"], input=user_input)
    assert result.exit_code == 0, (
        f"`flashcard game` exited {result.exit_code} with choices {choices!r}:\n"
        f"{result.output}"
    )
