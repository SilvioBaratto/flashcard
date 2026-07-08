"""
Tests that verify the BAML source files in baml_src/ match the required configuration
and that the project's secrets hygiene is in order (issue #1).

Reads baml_src/*.baml as plain text (they are the configuration spec, not Python
implementation). Reads .env.example and .gitignore for secrets-hygiene checks.

No network calls and no baml_client import — these tests run before code generation.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).parents[1]
_BAML_SRC = _ROOT / "baml_src"


def _read_baml(name: str) -> str:
    return (_BAML_SRC / name).read_text()


def _read_all_baml() -> str:
    return "\n".join(p.read_text() for p in sorted(_BAML_SRC.glob("*.baml")))


# ── clients.baml ──────────────────────────────────────────────────────────────


def test_when_clients_baml_read_then_gpt52_client_is_defined():
    """clients.baml must define a client named Gpt52."""
    content = _read_baml("clients.baml")
    assert re.search(r"client<llm>\s+Gpt52\b", content), (
        "Gpt52 client must be declared in clients.baml"
    )


def test_when_clients_baml_read_then_gpt52_uses_openai_provider():
    """
    The Gpt52 client block must specify provider openai.

    Checks for 'provider openai' as a bare keyword (not inside a string value)
    within the clients.baml file.
    """
    content = _read_baml("clients.baml")
    assert re.search(r"\bprovider\s+openai\b", content), (
        "Gpt52 must declare 'provider openai' in clients.baml"
    )


def test_when_clients_baml_read_then_gpt52_model_is_gpt_52():
    """The Gpt52 client must use model 'gpt-5.2'."""
    content = _read_baml("clients.baml")
    assert '"gpt-5.2"' in content, (
        'clients.baml must specify model "gpt-5.2" for the Gpt52 client'
    )


def test_when_clients_baml_read_then_gpt5mini_client_is_defined():
    """clients.baml must define a client named Gpt5Mini."""
    content = _read_baml("clients.baml")
    assert re.search(r"client<llm>\s+Gpt5Mini\b", content), (
        "Gpt5Mini client must be declared in clients.baml"
    )


def test_when_clients_baml_read_then_gpt5mini_model_is_gpt5_mini():
    """The Gpt5Mini client must use model 'gpt-5-mini'."""
    content = _read_baml("clients.baml")
    assert '"gpt-5-mini"' in content, (
        'clients.baml must specify model "gpt-5-mini" for the Gpt5Mini client'
    )


# ── no allowed_role_metadata anywhere ────────────────────────────────────────


def test_when_all_baml_sources_scanned_then_no_allowed_role_metadata_found():
    """
    No baml source file must contain 'allowed_role_metadata'.

    OpenAI caches the prefix automatically (no explicit cache_control marker and
    no off-switch), so there is no role metadata to configure.
    """
    all_content = _read_all_baml()
    assert "allowed_role_metadata" not in all_content, (
        "'allowed_role_metadata' must not appear in any baml source; "
        "OpenAI caching is automatic with no marker needed"
    )


# ── answer.baml — apples-to-apples model parity ──────────────────────────────


def test_when_answer_baml_read_then_answer_rag_function_is_defined():
    """answer.baml must define the AnswerRAG function."""
    content = _read_baml("answer.baml")
    assert re.search(r"\bfunction\s+AnswerRAG\b", content), (
        "AnswerRAG must be defined in answer.baml"
    )


def test_when_answer_baml_read_then_answer_cag_function_is_defined():
    """answer.baml must define the AnswerCAG function."""
    content = _read_baml("answer.baml")
    assert re.search(r"\bfunction\s+AnswerCAG\b", content), (
        "AnswerCAG must be defined in answer.baml"
    )


def test_when_answer_baml_read_then_both_functions_target_the_same_client():
    """
    AnswerRAG and AnswerCAG must both declare the same BAML client so that cost
    and accuracy gaps are attributable to retrieved-slice vs whole-doc alone,
    not to model differences.

    Verifies that every 'client <Name>' declaration in answer.baml names the
    same client (i.e. both functions use Gpt52).
    """
    content = _read_baml("answer.baml")

    # Extract all bare 'client <Name>' lines (not comments, not inside strings)
    client_refs = re.findall(r"^\s*client\s+(\w+)\s*$", content, re.MULTILINE)

    assert len(client_refs) >= 2, (
        f"Expected at least 2 'client <Name>' declarations in answer.baml "
        f"(one per function); found {len(client_refs)}: {client_refs}"
    )
    unique_clients = set(client_refs)
    assert len(unique_clients) == 1, (
        f"AnswerRAG and AnswerCAG must target the same client; "
        f"found multiple: {unique_clients}"
    )
    (client_name,) = unique_clients
    assert client_name == "Gpt52", (
        f"Both answer functions must target Gpt52; found '{client_name}'"
    )


# ── secrets hygiene ───────────────────────────────────────────────────────────


def test_when_env_example_checked_then_file_is_present():
    """A .env.example file must be shipped with the project."""
    env_example = _ROOT / ".env.example"
    assert env_example.exists(), ".env.example must exist at the project root"


def test_when_env_example_checked_then_openai_api_key_placeholder_is_present():
    """
    .env.example must contain OPENAI_API_KEY so users know which key to set.

    The value should be a placeholder (empty or example string), not a real key.
    """
    env_example = _ROOT / ".env.example"
    content = env_example.read_text()
    assert "OPENAI_API_KEY" in content, (
        ".env.example must document OPENAI_API_KEY as the required environment variable"
    )


def test_when_gitignore_checked_then_dot_env_is_excluded():
    """
    .gitignore must exclude .env (the file that holds the real key) so the key
    is never committed to the repository.
    """
    gitignore = _ROOT / ".gitignore"
    assert gitignore.exists(), ".gitignore must exist"
    lines = {line.strip() for line in gitignore.read_text().splitlines()}
    assert ".env" in lines or "*.env" in lines or ".env*" in lines, (
        ".gitignore must exclude .env so the real OPENAI_API_KEY is never committed"
    )
