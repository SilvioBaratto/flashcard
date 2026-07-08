"""
Source-blind tests for the "no secrets in source" global criterion — issue #3.

Criteria covered (UNIT-verifiable per oracle):
  - No secrets appear in source; the key is read only from OPENAI_API_KEY via
    env or the gitignored .env, and a .env.example is shipped.

Criteria NOT covered (oracle: NOT VERIFIABLE):
  - TUI legibility, SOLID code quality — no concrete runtime assertion.
  - Network retry behaviour — requires real network and live process state.
"""

import pathlib
import re

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
PACKAGE_DIR = PROJECT_ROOT / "flashcard"


# ---------------------------------------------------------------------------
# .env.example existence and content
# ---------------------------------------------------------------------------


def test_when_project_root_is_inspected_env_example_exists():
    """.env.example must be committed to the repo so contributors know what to set."""
    assert (PROJECT_ROOT / ".env.example").is_file(), (
        ".env.example not found at project root"
    )


def test_when_env_example_is_read_it_contains_openai_api_key():
    """.env.example must document the OPENAI_API_KEY variable (the only required key)."""
    content = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in content, (
        ".env.example must declare OPENAI_API_KEY so contributors know the required key"
    )


def test_when_env_example_is_read_it_does_not_contain_a_real_key_value():
    """.env.example must contain only a placeholder, not a real OpenAI key.

    Real OpenAI keys begin with 'sk-' followed by ≥10 alphanumeric/dash/underscore
    chars. Any such value in .env.example means a secret was accidentally committed.
    """
    content = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    real_key = re.compile(r"sk-[A-Za-z0-9_-]{10,}")
    match = real_key.search(content)
    assert match is None, (
        f".env.example contains what looks like a real API key: {match.group()!r}. "
        "Replace it with a placeholder such as sk-YOUR_KEY_HERE."
    )


# ---------------------------------------------------------------------------
# No hardcoded secrets in package source
# ---------------------------------------------------------------------------


def _iter_python_source():
    """Yield all .py files under the flashcard package (not tests)."""
    return PACKAGE_DIR.rglob("*.py")


def test_when_package_source_is_scanned_no_hardcoded_openai_key_is_found():
    """No Python source file inside the package must contain a literal OpenAI key.

    Criterion: 'the key is read only from OPENAI_API_KEY via env or the
    gitignored .env'.  A literal 'sk-...' string in source violates that.
    """
    real_key = re.compile(r"sk-[A-Za-z0-9_-]{10,}")
    offenders = []
    for path in _iter_python_source():
        text = path.read_text(encoding="utf-8", errors="replace")
        if real_key.search(text):
            offenders.append(str(path))
    assert not offenders, (
        "Hardcoded OpenAI key pattern (sk-...) found in: " + ", ".join(offenders)
    )


def test_when_package_source_is_scanned_openai_api_key_is_read_from_env():
    """At least one source file must reference OPENAI_API_KEY (env-var lookup),
    confirming the package knows the key name and reads it from the environment
    rather than from a hardcoded literal.
    """
    env_var_ref = re.compile(r"OPENAI_API_KEY")
    found = any(
        env_var_ref.search(path.read_text(encoding="utf-8", errors="replace"))
        for path in _iter_python_source()
    )
    assert found, (
        "No source file references 'OPENAI_API_KEY'. "
        "The package must read the key from os.environ or python-dotenv."
    )
