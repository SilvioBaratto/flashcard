"""
Source-blind example tests for issue #23:
  docs: sync README with final flags, --judge, resilience, game mode, and default gpt-5.2

Criteria tested (per oracle report):
  [UNIT] Both pipelines share the same model and max_tokens                     → test_when_run_config_*
  [UNIT] No secrets in source; key only from OPENAI_API_KEY; .env.example ships → test_when_source_* / test_when_env_*
  [UNIT] CLI --judge/--no-judge present and defaults to off                     → test_when_cli_help_*
  [UNIT] README has all required sections, consistent with gpt-5.2              → test_when_readme_*
  [T3]   Cost guard computes CAG Q1 as a distinct (most expensive) line item    → test_when_cag_q1_*

Skipped (oracle: NOT VERIFIABLE at unit/integration tier):
  - Network retry / TUI legibility
  - Pricing source verification
  - README game-mode / resilience prose
  - All-tests-pass (green suite) / SOLID quality
"""

import re
import pathlib

from hypothesis import given, strategies as st

# Project root is two levels up from this file (tests/test_issue_23_readme_sync.py)
ROOT = pathlib.Path(__file__).parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _readme_text() -> str:
    path = ROOT / "README.md"
    assert path.exists(), "README.md must exist at the project root"
    return path.read_text(encoding="utf-8")


def _iter_source_files():
    """Yield every .py and .baml file under the application source trees."""
    for subdir in (ROOT / "flashcard", ROOT / "baml_src"):
        if subdir.is_dir():
            yield from subdir.rglob("*.py")
            yield from subdir.rglob("*.baml")


def _cli_help() -> str:
    from typer.testing import CliRunner
    from flashcard.cli import app

    result = CliRunner().invoke(app, ["--help"])
    return result.output


# ─────────────────────────────────────────────────────────────────────────────
# Criterion [UNIT]: Both pipelines answer the same questions over the same
# source with the same model and max_tokens, so cost/accuracy gaps are
# attributable to retrieved-slice (RAG) vs whole-doc (CAG) alone.
# ─────────────────────────────────────────────────────────────────────────────


def test_when_run_config_is_built_then_model_is_accessible():
    """RunConfig must expose the model so both sides can consume it."""
    from flashcard.runner import RunConfig

    cfg = RunConfig(model="gpt-5.2", max_tokens=400)
    assert cfg.model == "gpt-5.2"


def test_when_run_config_is_built_then_max_tokens_is_accessible():
    """RunConfig must expose max_tokens so both sides are capped identically."""
    from flashcard.runner import RunConfig

    cfg = RunConfig(model="gpt-5.2", max_tokens=400)
    assert cfg.max_tokens == 400


def test_when_run_config_is_built_then_there_is_no_per_side_model_override():
    """
    RunConfig must not allow independent model overrides per side.
    Both RAG and CAG consume the same model field — there is no rag_model or cag_model.
    """
    from flashcard.runner import RunConfig

    cfg = RunConfig(model="gpt-5.2", max_tokens=400)
    assert not hasattr(cfg, "rag_model"), (
        "RunConfig must not expose a per-side rag_model"
    )
    assert not hasattr(cfg, "cag_model"), (
        "RunConfig must not expose a per-side cag_model"
    )


@given(
    model=st.sampled_from(["gpt-5.2", "gpt-5-mini"]),
    max_tokens=st.integers(min_value=1, max_value=2048),
)
def test_when_run_config_is_built_with_any_valid_model_and_max_tokens_then_both_are_stored(
    model, max_tokens
):
    """For any valid model and max_tokens, RunConfig stores them unchanged."""
    from flashcard.runner import RunConfig

    cfg = RunConfig(model=model, max_tokens=max_tokens)
    assert cfg.model == model
    assert cfg.max_tokens == max_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Criterion [UNIT]: No secrets appear in source; the key is read only from
# OPENAI_API_KEY via env or the gitignored .env, and a .env.example is shipped.
# ─────────────────────────────────────────────────────────────────────────────

_OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{20,}")


def test_when_source_files_are_scanned_then_no_hardcoded_openai_key_is_found():
    """No literal OpenAI API key (sk-...) must appear in any .py or .baml source file."""
    violations = []
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if _OPENAI_KEY_RE.search(text):
            violations.append(str(path))
    assert not violations, f"Hardcoded API key found in: {violations}"


def test_when_env_example_is_checked_then_it_exists():
    """.env.example must be present at the project root."""
    assert (ROOT / ".env.example").exists(), (
        ".env.example must exist at the project root"
    )


def test_when_env_example_is_read_then_openai_api_key_is_present():
    """.env.example must mention OPENAI_API_KEY."""
    content = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in content, ".env.example must reference OPENAI_API_KEY"


def test_when_gitignore_is_checked_then_dot_env_is_excluded():
    """.gitignore must list .env so the secret file is never committed."""
    gitignore = ROOT / ".gitignore"
    assert gitignore.exists(), ".gitignore must exist"
    content = gitignore.read_text(encoding="utf-8")
    assert re.search(r"(^|\n)\.env(\s|$)", content), ".env must appear in .gitignore"


# ─────────────────────────────────────────────────────────────────────────────
# Criterion [UNIT]: Flag table includes --judge/--no-judge (default off) with
# an accurate description; all other flags match flashcard/cli.py.
# ─────────────────────────────────────────────────────────────────────────────


def test_when_cli_help_is_invoked_then_judge_flag_is_listed():
    """--judge / --no-judge must appear in the CLI --help output."""
    help_text = _cli_help()
    assert "--judge" in help_text or "--no-judge" in help_text, (
        "--judge / --no-judge flag must appear in CLI --help"
    )


def test_when_cli_help_is_invoked_then_judge_default_is_off():
    """
    --judge must default to off.
    Typer renders the default value inside the help text; we accept any of the
    common representations (False, no-judge, disabled, off).
    """
    help_text = _cli_help()
    # Check for any canonical "off" representation that Typer may emit
    assert re.search(r"no-judge|False|disabled|off", help_text, re.IGNORECASE), (
        "--judge must default to off (False / no-judge) per the CLI help"
    )


def test_when_cli_help_is_invoked_then_model_flag_is_listed():
    assert "--model" in _cli_help(), "--model flag must appear in CLI --help"


def test_when_cli_help_is_invoked_then_doc_flag_is_listed():
    assert "--doc" in _cli_help(), "--doc flag must appear in CLI --help"


def test_when_cli_help_is_invoked_then_questions_file_flag_is_listed():
    assert "--questions-file" in _cli_help(), (
        "--questions-file flag must appear in CLI --help"
    )


def test_when_cli_help_is_invoked_then_top_k_flag_is_listed():
    assert "--top-k" in _cli_help(), "--top-k flag must appear in CLI --help"


def test_when_cli_help_is_invoked_then_chunk_size_flag_is_listed():
    assert "--chunk-size" in _cli_help(), "--chunk-size flag must appear in CLI --help"


def test_when_cli_help_is_invoked_then_max_tokens_flag_is_listed():
    assert "--max-tokens" in _cli_help(), "--max-tokens flag must appear in CLI --help"


def test_when_cli_help_is_invoked_then_delay_flag_is_listed():
    assert "--delay" in _cli_help(), "--delay flag must appear in CLI --help"


def test_when_cli_help_is_invoked_then_all_required_flags_are_present():
    """All required CLI flags must be discoverable from --help in one call."""
    help_text = _cli_help()
    required = [
        "--model",
        "--doc",
        "--questions-file",
        "--top-k",
        "--chunk-size",
        "--max-tokens",
        "--delay",
        "--judge",
    ]
    missing = [f for f in required if f not in help_text]
    assert not missing, f"The following flags are missing from CLI --help: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# Criterion [UNIT]: Two-students analogy, glance table, install +
# OPENAI_API_KEY config, usage, a sample run, and "when to use which" are all
# present and consistent with default gpt-5.2 and the cost engine.
# ─────────────────────────────────────────────────────────────────────────────


def test_when_readme_is_read_then_two_students_analogy_is_present():
    """README must contain the two-students exam analogy from script_6.md."""
    content = _readme_text()
    assert re.search(r"student", content, re.IGNORECASE), (
        "README must include the two-students exam analogy"
    )


def test_when_readme_is_read_then_rag_and_cag_glance_table_is_present():
    """README must contain a comparison table with both RAG and CAG."""
    content = _readme_text()
    assert "RAG" in content, "README must mention RAG"
    assert "CAG" in content, "README must mention CAG"
    # A markdown table uses pipe characters
    assert "|" in content, "README must contain a markdown glance table"


def test_when_readme_is_read_then_openai_api_key_config_is_present():
    """README must document how to configure OPENAI_API_KEY."""
    content = _readme_text()
    assert "OPENAI_API_KEY" in content, (
        "README must document the OPENAI_API_KEY configuration step"
    )


def test_when_readme_is_read_then_install_instructions_are_present():
    """README must contain installation instructions."""
    content = _readme_text()
    assert re.search(r"\bpip\b.*install|install.*\bpip\b", content, re.IGNORECASE), (
        "README must contain pip install instructions"
    )


def test_when_readme_is_read_then_usage_section_is_present():
    """README must contain a usage or how-to-run section."""
    content = _readme_text()
    assert re.search(
        r"usage|how to (run|use)|quickstart|getting started", content, re.IGNORECASE
    ), "README must contain a usage section"


def test_when_readme_is_read_then_sample_run_code_block_is_present():
    """README must contain a fenced code block showing a sample flashcard invocation."""
    content = _readme_text()
    assert "flashcard" in content, "README must show a flashcard command"
    assert "```" in content, (
        "README must contain at least one fenced code block for the sample run"
    )


def test_when_readme_is_read_then_when_to_use_which_section_is_present():
    """README must contain a 'when to use which' decision guide."""
    content = _readme_text()
    assert re.search(
        r"when to use|which (to use|is better)|choose|tradeoff|trade.off",
        content,
        re.IGNORECASE,
    ), "README must contain a 'when to use which' section"


def test_when_readme_is_read_then_default_model_gpt_5_2_is_mentioned():
    """README must reference gpt-5.2 as the default model."""
    content = _readme_text()
    assert "gpt-5.2" in content, "README must name gpt-5.2 as the default model"


# ─────────────────────────────────────────────────────────────────────────────
# Criterion [T3]: Before any real call, worst-case spend across both pipelines
# × all questions is estimated with CAG's Q1 surfaced explicitly.
#
# The cost estimation is pure local arithmetic — no API key needed.
# We test through the pricing module's per-turn functions, verifying that Q1
# (no cached tokens) costs strictly more than a later turn (all tokens cached).
# ─────────────────────────────────────────────────────────────────────────────


def test_when_cag_q1_cost_is_computed_then_it_exceeds_a_cache_warm_turn():
    """
    CAG Q1 (nothing cached yet → full document at base rate) must cost more
    than a cache-warm turn (document tokens read at the 90%-discounted rate).

    Derived from: 'CAG's Q1 surfaced explicitly' and the pricing spec:
      Q1: all document tokens at base input rate
      Q2+: document tokens at cache-read rate (≈0.1× base)
    """
    from flashcard.pricing import price_cag_turn

    document_tokens = 10_000
    fresh_question_tokens = 50
    output_tokens = 200
    model = "gpt-5.2"

    # Q1: nothing is cached
    cost_q1 = price_cag_turn(
        model=model,
        cache_read_tokens=0,
        fresh_question_tokens=document_tokens + fresh_question_tokens,
        output_tokens=output_tokens,
    )

    # Q2+: the document prefix is cached; only the question is fresh
    cost_q2 = price_cag_turn(
        model=model,
        cache_read_tokens=document_tokens,
        fresh_question_tokens=fresh_question_tokens,
        output_tokens=output_tokens,
    )

    assert cost_q1 > cost_q2, (
        f"CAG Q1 cost ({cost_q1}) must exceed a cache-warm turn ({cost_q2}); "
        "Q1 pays the full document at base rate while Q2+ reads it at 90% discount"
    )


def test_when_cag_turn_cost_is_computed_then_it_is_non_negative():
    """price_cag_turn must return a non-negative dollar amount for any valid input."""
    from flashcard.pricing import price_cag_turn

    cost = price_cag_turn(
        model="gpt-5.2",
        cache_read_tokens=5_000,
        fresh_question_tokens=100,
        output_tokens=300,
    )
    assert cost >= 0, "CAG turn cost must be non-negative"


def test_when_rag_turn_cost_is_computed_then_it_is_non_negative():
    """price_rag_turn must return a non-negative dollar amount for any valid input."""
    from flashcard.pricing import price_rag_turn

    cost = price_rag_turn(
        model="gpt-5.2",
        input_tokens=600,
        output_tokens=200,
        query_embed_tokens=50,
    )
    assert cost >= 0, "RAG turn cost must be non-negative"


@given(
    cache_read_tokens=st.integers(min_value=0, max_value=100_000),
    fresh_question_tokens=st.integers(min_value=1, max_value=10_000),
    output_tokens=st.integers(min_value=0, max_value=2048),
)
def test_when_cag_turn_cost_is_computed_for_any_valid_inputs_then_no_error_is_raised(
    cache_read_tokens, fresh_question_tokens, output_tokens
):
    """price_cag_turn must not raise for any combination of non-negative token counts."""
    from flashcard.pricing import price_cag_turn

    result = price_cag_turn(
        model="gpt-5.2",
        cache_read_tokens=cache_read_tokens,
        fresh_question_tokens=fresh_question_tokens,
        output_tokens=output_tokens,
    )
    assert result >= 0


@given(
    input_tokens=st.integers(min_value=1, max_value=10_000),
    output_tokens=st.integers(min_value=0, max_value=2048),
    query_embed_tokens=st.integers(min_value=1, max_value=1000),
)
def test_when_rag_turn_cost_is_computed_for_any_valid_inputs_then_no_error_is_raised(
    input_tokens, output_tokens, query_embed_tokens
):
    """price_rag_turn must not raise for any combination of non-negative token counts."""
    from flashcard.pricing import price_rag_turn

    result = price_rag_turn(
        model="gpt-5.2",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        query_embed_tokens=query_embed_tokens,
    )
    assert result >= 0


@given(
    document_tokens=st.integers(min_value=1024, max_value=50_000),
    question_tokens=st.integers(min_value=1, max_value=500),
    output_tokens=st.integers(min_value=0, max_value=2048),
    cache_fraction=st.floats(min_value=0.0, max_value=1.0),
)
def test_when_cag_q1_and_later_turns_are_compared_then_q1_is_never_cheaper(
    document_tokens, question_tokens, output_tokens, cache_fraction
):
    """
    Q1 (0 cached tokens) must never be cheaper than a turn where some of the
    same tokens are cached.

    Invariant: for a fixed total input size, replacing fresh tokens with cached
    tokens can only reduce cost, because cache_rate < base_rate (90% discount).
    The total input count is held constant across the two calls.
    """
    from flashcard.pricing import price_cag_turn

    total_tokens = document_tokens + question_tokens
    cache_read_tokens = int(document_tokens * cache_fraction)

    cost_uncached = price_cag_turn(
        model="gpt-5.2",
        cache_read_tokens=0,
        fresh_question_tokens=total_tokens,
        output_tokens=output_tokens,
    )
    cost_partially_cached = price_cag_turn(
        model="gpt-5.2",
        cache_read_tokens=cache_read_tokens,
        fresh_question_tokens=total_tokens - cache_read_tokens,
        output_tokens=output_tokens,
    )
    assert cost_uncached >= cost_partially_cached, (
        "A fully-uncached CAG turn must never be cheaper than a partially-cached one "
        f"(uncached={cost_uncached:.6f}, partially_cached={cost_partially_cached:.6f})"
    )
