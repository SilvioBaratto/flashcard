"""Built-in English question set for the flashcard RAG-vs-CAG demo.

Questions are deliberately mixed so the retrieval tradeoff is visible:
  [narrow]   The full answer lives in one or two adjacent paragraphs; a single
             top-k chunk suffices — RAG retrieves it cheaply and exactly.
  [spanning] The answer requires combining two distant sections; RAG's top-k
             keyhole misses one half while CAG (whole document in context)
             answers completely.

Cross-referenced spanning targets (documented in flashcard/corpus.py):
  Q7 → Section 3.2 (quota policy / grace period) +
        Section 8.2 (per-GB overage rates with day-level proration)
  Q8 → Section 3.7 (Regional Storage SLA definition) +
        Section 9.2 (credit percentage table + 30-day claim procedure)
"""

from __future__ import annotations

import json
import pathlib

# fmt: off
BUILT_IN_QUESTIONS: list[str] = [
    # [narrow] single section (1.3)
    "How many Availability Zones does Meridian Cloud Platform operate,"
    " and how are they grouped into geographic Regions?",
    # [narrow] single section (1.4)
    "What is the monthly base price for the Professional tier,"
    " and what object-storage quota does it include?",
    # [narrow] single section (2.1)
    "What is the maximum number of IAM sub-accounts allowed on the Enterprise tier?",
    # [narrow] single section (2.2)
    "How often must Service Account API keys be rotated according to"
    " Meridian's security guidelines?",
    # [narrow] single section (2.2)
    "What TOTP code validity window does MCP accept to accommodate clock drift?",
    # [narrow] single section (1.4)
    "What compute resources (vCPUs and RAM) are included in the Starter tier?",
    # [spanning] Section 3.2 (grace period) + Section 8.2 (per-GB overage rate)
    "What happens when an Enterprise customer exceeds their 5 TB storage quota,"
    " including the grace period behaviour and the per-GB monthly overage rate?",
    # [spanning] Section 3.7 (SLA definition) + Section 9.2 (credit % + claim procedure)
    "What SLA credit percentage applies to an MCP-OBJ Standard outage lasting"
    " between 1 and 4 hours, and what is the complete process for submitting"
    " a credit claim?",
    # [narrow] single section (7.x)
    "What encryption standard does Meridian use for data at rest in MCP-OBJ storage?",
]
# fmt: on


def load_questions(path: str | pathlib.Path | None = None) -> list[str]:
    """Return the question list to play each turn.

    Args:
        path: JSON file containing a non-empty list of question strings.
              When ``None`` (default), return :data:`BUILT_IN_QUESTIONS`.

    Returns:
        A non-empty list of question strings.

    Raises:
        ValueError: If *path* is given but the file contains malformed JSON,
            an empty list, or elements that are not all strings.
    """
    if path is None:
        return BUILT_IN_QUESTIONS
    return _load_json_file(path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_json_file(path: str | pathlib.Path) -> list[str]:
    p = pathlib.Path(path)
    raw = _parse_json(p)
    return _validate(raw, p)


def _parse_json(path: pathlib.Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"--questions-file {path}: invalid JSON — {exc.msg}"
            f" (line {exc.lineno}, col {exc.colno}).\n"
            f'The file must contain a JSON array of strings, e.g. ["Q1?", "Q2?"].'
        ) from exc


def _validate(value: object, path: pathlib.Path) -> list[str]:
    _require_list(value, path)
    questions: list = value  # type: ignore[assignment]
    _require_non_empty(questions, path)
    _require_all_strings(questions, path)
    return questions  # type: ignore[return-value]


def _require_list(value: object, path: pathlib.Path) -> None:
    if not isinstance(value, list):
        raise ValueError(
            f"--questions-file {path}: expected a JSON array,"
            f" got {type(value).__name__}.\n"
            f'The file must contain a JSON array of strings, e.g. ["Q1?", "Q2?"].'
        )


def _require_non_empty(questions: list, path: pathlib.Path) -> None:
    if not questions:
        raise ValueError(
            f"--questions-file {path}: the question list is empty.\n"
            f"Provide at least one question string."
        )


def _require_all_strings(questions: list, path: pathlib.Path) -> None:
    bad = [i for i, q in enumerate(questions) if not isinstance(q, str)]
    if not bad:
        return
    raise ValueError(
        f"--questions-file {path}: items at indices {bad} are not strings.\n"
        f"Every element in the JSON array must be a question string."
    )
