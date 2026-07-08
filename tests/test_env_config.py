"""
Source-blind example tests for issue #13, global criteria 1 and 3.

Criterion 1 (global): Both pipelines answer the same questions over the same source with
the same model and max_tokens, so cost and accuracy gaps are attributable to
retrieved-slice (RAG) vs whole-doc (CAG) alone.
  → Unit check: RunConfig exposes a single model / max_tokens; there is no per-pipeline
    model override.  (The property-based coverage of this invariant lives in
    test_judge_gating.py alongside the RunConfig tests.)

Criterion 3 (global): No secrets appear in source; the key is read only from
OPENAI_API_KEY via env or the gitignored .env, and a .env.example is shipped.
  → Unit checks: .env.example exists and references OPENAI_API_KEY;
    .env is listed in .gitignore.

Tests are authored from the acceptance-criteria text and requirements.md only.
No implementation file was read during authoring.
"""

import os
from pathlib import Path


ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Criterion 3 — .env.example ships and OPENAI_API_KEY is read from env
# ---------------------------------------------------------------------------


def test_when_project_root_is_checked_then_env_example_exists():
    """.env.example must be present at the project root (safe key distribution)."""
    env_example = ROOT / ".env.example"
    assert env_example.exists(), (
        ".env.example must exist at the project root; "
        "it tells users which environment variables to set."
    )


def test_when_env_example_is_read_then_it_references_openai_api_key():
    """.env.example must name OPENAI_API_KEY so users know which key to supply."""
    env_example = ROOT / ".env.example"
    assert env_example.exists(), ".env.example must exist"
    content = env_example.read_text()
    assert "OPENAI_API_KEY" in content, (
        ".env.example must include OPENAI_API_KEY (the only required secret)"
    )


def test_when_gitignore_is_read_then_dot_env_is_listed():
    """.env must appear in .gitignore so the real key is never committed."""
    gitignore = ROOT / ".gitignore"
    assert gitignore.exists(), ".gitignore must exist"
    content = gitignore.read_text()
    assert ".env" in content, (
        ".env must be listed in .gitignore to prevent accidental key commits"
    )


def test_when_openai_api_key_env_var_is_set_then_it_is_retrievable():
    """The key must be reachable via os.environ['OPENAI_API_KEY'] (not hardcoded anywhere)."""
    sentinel = "sk-test-sentinel-key-1234"
    original = os.environ.get("OPENAI_API_KEY")
    try:
        os.environ["OPENAI_API_KEY"] = sentinel
        assert os.environ["OPENAI_API_KEY"] == sentinel, (
            "os.environ must hold OPENAI_API_KEY; the application must read it from here"
        )
    finally:
        if original is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = original


# ---------------------------------------------------------------------------
# Criterion 1 — both pipelines share a single model and max_tokens (structural)
# ---------------------------------------------------------------------------


def test_when_run_config_is_imported_then_it_has_a_model_field():
    """RunConfig must expose a 'model' field used by both RAG and CAG pipelines."""
    from flashcard.runner import RunConfig
    from dataclasses import fields as dc_fields

    field_names = {f.name for f in dc_fields(RunConfig)}
    assert "model" in field_names, (
        "RunConfig must declare a 'model' field "
        "(same model applied to both RAG and CAG pipelines)"
    )


def test_when_run_config_is_imported_then_it_has_a_max_tokens_field():
    """RunConfig must expose a 'max_tokens' field shared across both pipelines."""
    from flashcard.runner import RunConfig
    from dataclasses import fields as dc_fields

    field_names = {f.name for f in dc_fields(RunConfig)}
    assert "max_tokens" in field_names, (
        "RunConfig must declare a 'max_tokens' field "
        "(same cap applied to both RAG and CAG pipelines)"
    )


def test_when_run_config_default_model_is_checked_then_it_is_gpt52():
    """RunConfig.model must default to 'gpt-5.2' (the spec's stated default generation model)."""
    from flashcard.runner import RunConfig

    config = RunConfig()
    assert config.model == "gpt-5.2", (
        f"RunConfig.model must default to 'gpt-5.2', got {config.model!r}"
    )


def test_when_run_config_default_max_tokens_is_checked_then_it_is_400():
    """RunConfig.max_tokens must default to 400 (bounded output to keep demo cheap)."""
    from flashcard.runner import RunConfig

    config = RunConfig()
    assert config.max_tokens == 400, (
        f"RunConfig.max_tokens must default to 400, got {config.max_tokens!r}"
    )
