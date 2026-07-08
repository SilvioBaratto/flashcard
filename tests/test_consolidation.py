"""
Source-blind example tests for issue #14 (runner.py consolidation).

Covered criteria (UNIT-verifiable per oracle):
  Criterion 8:  runner.py exposes ONE dual-pipeline turn implementation via a ``Runner``
                class; redundant module-level stubs are removed; file < 600 lines.
  Criterion 9:  The optional judge (CompareAnswers, gated by RunConfig.judge) is reachable
                from Runner — the real-metrics path — not only from a zero-metrics stub.
  Criterion 10: flashcard/resilience.py exposes ``run_session``; it attaches
                ``similarity_scores`` and ``context_token_count`` to TurnResult.rag when
                the retriever provides them; it degrades gracefully when it does not.
  Criterion 11: CompareAnswers must NOT receive a ``client_registry`` override so BAML
                honours the static ``client Gpt5Mini`` declared in compare.baml.
  Criterion 12: The module-level ``run_turn(question, config, *, baml_client, retriever,
                document)`` signature from issue #13 is preserved or intentionally migrated.
  Criterion 13: A headless/no-API-key fallback (naive retriever, fallback document constant)
                exists and lets the pipeline complete without crashing.
  Criterion 14: runner.py is ≤ 600 lines after consolidation (SOLID).

Skipped criteria (per oracle):
  Global [T3]         — require live API calls.
  [NOT VERIFIABLE]    — TUI legibility, network retry, build-time price verification.
  Already covered     — RunConfig.model, .env.example, .gitignore (test_env_config.py).

Design choices recorded here (tests are in Red phase — they fail until implemented):
  - ``Runner`` is instantiated as
      ``Runner(config=config, baml_client=fake_baml, retriever=retriever)``
    with an async method ``run_turn(question: str, document: str, index: int = 0) → TurnResult``.
    If the actual signature differs, update the call sites below.
  - ``flashcard.resilience.run_session`` is an async generator:
      ``async for turn in run_session(config, baml_client, retriever, document, questions)``
    If it is a regular coroutine returning a list, replace the loop with ``await``.
  - ``TurnResult.rag.similarity_scores`` holds the retriever's ``last_similarity_scores``.
  - ``TurnResult.rag.context_token_count`` holds the retriever's ``last_context_token_count``.
  - BAML streams mocked via sync ``get_final_response``; ``_read_stream`` in runner.py
    handles both sync and async via ``inspect.iscoroutine`` (established in issue #13).
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, strategies as st

from flashcard.runner import RunConfig, TurnResult

ROOT = Path(__file__).parent.parent
RUNNER_PY = ROOT / "flashcard" / "runner.py"


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 8 — runner.py exposes ONE dual-pipeline implementation (Runner)
# ─────────────────────────────────────────────────────────────────────────────


def test_when_runner_class_is_imported_then_it_is_a_class():
    """Runner must be importable from flashcard.runner and must be a class, not a function."""
    from flashcard.runner import Runner

    assert isinstance(Runner, type), (
        "flashcard.runner.Runner must be a class — the single OO entry point for dual-pipeline turns"
    )


@pytest.mark.asyncio
async def test_when_runner_runs_a_turn_then_it_returns_turn_result():
    """Runner.run_turn must return a TurnResult containing the original question."""
    from flashcard.runner import Runner

    config = RunConfig()
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)

    runner = Runner(config=config, baml_client=fake_baml, retriever=_make_retriever())
    result = await runner.run_turn(question="What is the policy?", document="A short doc.")

    assert isinstance(result, TurnResult), "Runner.run_turn must return a TurnResult"
    assert result.question == "What is the policy?", (
        "TurnResult.question must match the question passed to run_turn"
    )


def test_when_runner_py_line_count_is_checked_then_it_is_under_600():
    """SOLID: runner.py must be ≤ 600 lines after deleting redundant stubs."""
    lines = RUNNER_PY.read_text().splitlines()
    assert len(lines) <= 600, (
        f"runner.py has {len(lines)} lines — consolidation must bring it to ≤ 600 "
        "(remove duplicate module-level _run_turn_raw / _run_rag / _run_cag / "
        "zero-metrics run_turn / _run_both_sides / _assemble_turn)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 9 — judge reachable from the real-metrics Runner path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_when_runner_judge_is_on_then_verdict_is_populated():
    """Runner with RunConfig.judge=True must populate TurnResult.verdict (real-metrics path)."""
    from flashcard.runner import Runner

    config = RunConfig(judge=True)
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)
    fake_baml.CompareAnswers.return_value = _make_verdict()

    runner = Runner(config=config, baml_client=fake_baml, retriever=_make_retriever())
    result = await runner.run_turn(question="What is covered?", document="A short doc.")

    assert result.verdict is not None, (
        "Runner with judge=True must attach a verdict — "
        "the judge must be reachable from the real-metrics path, "
        "not only from a zero-metrics stub"
    )


@pytest.mark.asyncio
async def test_when_runner_judge_is_off_then_verdict_is_none_and_compare_not_called():
    """Runner with RunConfig.judge=False must leave verdict=None and never call CompareAnswers."""
    from flashcard.runner import Runner

    config = RunConfig(judge=False)
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)

    runner = Runner(config=config, baml_client=fake_baml, retriever=_make_retriever())
    result = await runner.run_turn(question="What is covered?", document="A short doc.")

    assert result.verdict is None
    fake_baml.CompareAnswers.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 10 — resilience.py drives unified path; graceful degradation
# ─────────────────────────────────────────────────────────────────────────────


def test_when_resilience_module_is_imported_then_run_session_is_callable():
    """flashcard.resilience must expose a callable run_session."""
    import flashcard.resilience as res

    fn = getattr(res, "run_session", None)
    assert callable(fn), (
        "flashcard.resilience must expose run_session "
        "— the unified session driver that delegates to Runner"
    )


@pytest.mark.asyncio
async def test_when_retriever_lacks_last_attrs_then_run_session_does_not_crash():
    """Graceful degradation: run_session must not raise when retriever has no last_* attrs."""
    from flashcard.resilience import run_session

    config = RunConfig()
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)
    # Plain retriever with no last_similarity_scores / last_context_token_count attributes
    retriever = _make_retriever(last_scores=None, last_ctx=None, include_last=False)

    results = []
    async for turn in run_session(
        config,
        fake_baml,
        retriever,
        "A product manual.",
        ["What is the cancellation policy?"],
    ):
        results.append(turn)

    assert len(results) == 1, "run_session must yield one TurnResult per question"
    assert isinstance(results[0], TurnResult)


@pytest.mark.asyncio
async def test_when_retriever_has_last_similarity_scores_then_they_are_attached():
    """TurnResult.rag must expose similarity_scores when the retriever provides them."""
    from flashcard.resilience import run_session

    scores = [0.91, 0.84]
    config = RunConfig()
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)
    retriever = _make_retriever(last_scores=scores, include_last=True)

    results = []
    async for turn in run_session(
        config,
        fake_baml,
        retriever,
        "A product manual.",
        ["Which section covers billing?"],
    ):
        results.append(turn)

    rag = results[0].rag
    attached = getattr(rag, "similarity_scores", None)
    assert attached is not None, (
        "TurnResult.rag must expose similarity_scores when retriever.last_similarity_scores is set"
    )
    assert list(attached) == scores, (
        "TurnResult.rag.similarity_scores must match retriever.last_similarity_scores"
    )


@pytest.mark.asyncio
async def test_when_retriever_has_last_context_token_count_then_it_is_attached():
    """TurnResult.rag must expose context_token_count when the retriever provides it."""
    from flashcard.resilience import run_session

    config = RunConfig()
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)
    retriever = _make_retriever(last_ctx=120, include_last=True)

    results = []
    async for turn in run_session(
        config,
        fake_baml,
        retriever,
        "A product manual.",
        ["What does section 3 cover?"],
    ):
        results.append(turn)

    rag = results[0].rag
    attached = getattr(rag, "context_token_count", None)
    assert attached is not None, (
        "TurnResult.rag must expose context_token_count when retriever.last_context_token_count is set"
    )
    assert attached == 120, (
        "TurnResult.rag.context_token_count must match retriever.last_context_token_count"
    )


# Property: graceful degradation is an invariant over all retriever shapes
@given(
    has_scores=st.booleans(),
    has_ctx=st.booleans(),
)
@settings(max_examples=30)
def test_when_retriever_has_any_last_attr_combination_then_run_turn_never_raises(
    has_scores: bool,
    has_ctx: bool,
) -> None:
    """
    Invariant (criterion 10): for any combination of retriever last_* attributes,
    the unified path must not raise AttributeError.  Retriever may or may not
    expose last_similarity_scores / last_context_token_count.
    """
    from flashcard.runner import run_turn

    config = RunConfig()
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)
    retriever = _make_retriever(
        last_scores=[0.9] if has_scores else None,
        last_ctx=50 if has_ctx else None,
        include_last=has_scores or has_ctx,
    )

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(
            run_turn(
                "Does this fail?",
                config,
                baml_client=fake_baml,
                retriever=retriever,
                document="Fallback doc.",
            )
        )
        assert isinstance(result, TurnResult), (
            "run_turn must return TurnResult regardless of retriever last_* shape"
        )
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 11 — CompareAnswers must NOT receive client_registry
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_when_runner_calls_compare_answers_then_baml_options_has_no_client_registry():
    """Judge call via Runner must omit client_registry so BAML honours static Gpt5Mini."""
    from flashcard.runner import Runner

    config = RunConfig(judge=True)
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)
    fake_baml.CompareAnswers.return_value = _make_verdict()

    runner = Runner(config=config, baml_client=fake_baml, retriever=_make_retriever())
    await runner.run_turn(question="Question?", document="Doc.")

    call_args = fake_baml.CompareAnswers.call_args
    assert call_args is not None, "CompareAnswers must have been called when judge=True"
    baml_options = call_args.kwargs.get("baml_options", {})
    assert "client_registry" not in baml_options, (
        "CompareAnswers must NOT receive client_registry — "
        "passing FlashcardPrimary's registry would silently route the judge to gpt-5.2 "
        "instead of the static Gpt5Mini declared in compare.baml"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 12 — module-level run_turn signature from issue #13 preserved
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_when_module_level_run_turn_is_called_with_issue13_signature_then_it_works():
    """run_turn(question, config, *, baml_client, retriever, document) must still be callable."""
    from flashcard.runner import run_turn

    config = RunConfig(judge=False)
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)

    result = await run_turn(
        "What is the warranty period?",
        config,
        baml_client=fake_baml,
        retriever=_make_retriever(),
        document="A short product manual.",
    )

    assert isinstance(result, TurnResult), (
        "Module-level run_turn must return TurnResult with the issue #13 positional+keyword signature"
    )


def test_when_module_level_run_turn_is_inspected_then_it_is_a_coroutine_function():
    """run_turn must be an async function (coroutine function), not a sync wrapper."""
    from flashcard.runner import run_turn

    assert inspect.iscoroutinefunction(run_turn), (
        "flashcard.runner.run_turn must be declared with 'async def'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Criterion 13 — headless / no-API-key fallback
# ─────────────────────────────────────────────────────────────────────────────


def test_when_runner_module_is_imported_then_a_fallback_document_constant_exists():
    """A fallback document constant must be available so the demo runs without --doc."""
    import flashcard.runner as runner_mod

    candidates = ("_FALLBACK_DOC", "FALLBACK_DOC", "_DEFAULT_DOC", "DEFAULT_DOC")
    has_fallback = any(hasattr(runner_mod, name) for name in candidates)
    assert has_fallback, (
        "flashcard.runner must expose a fallback document constant "
        "(_FALLBACK_DOC or similar) so headless mode works without a user-supplied --doc"
    )


@pytest.mark.asyncio
async def test_when_run_turn_uses_naive_retriever_then_it_completes_without_error():
    """End-to-end headless: run_turn with a naive retriever and mock BAML must not raise."""
    from flashcard.runner import run_turn

    config = RunConfig()
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)
    # Naive retriever: term-overlap style, no embeddings needed
    retriever = _make_retriever()

    result = await run_turn(
        "What is covered by the warranty?",
        config,
        baml_client=fake_baml,
        retriever=retriever,
        document="A short fallback document.",
    )

    assert isinstance(result, TurnResult), (
        "run_turn with a naive retriever must complete and return TurnResult "
        "(headless/no-API-key fallback path)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


class _Retriever:
    """Minimal retriever conforming to the Retriever protocol (retrieve -> list[tuple[int,str]])."""

    def __init__(
        self,
        *,
        last_scores: list[float] | None = None,
        last_ctx: int | None = None,
        include_last: bool = False,
    ) -> None:
        if include_last:
            if last_scores is not None:
                self.last_similarity_scores = last_scores
            if last_ctx is not None:
                self.last_context_token_count = last_ctx

    def retrieve(self, query: str, top_k: int) -> list[tuple[int, str]]:
        return [(0, "A relevant chunk.")]


def _make_retriever(
    *,
    last_scores: list[float] | None = None,
    last_ctx: int | None = None,
    include_last: bool = False,
) -> _Retriever:
    return _Retriever(last_scores=last_scores, last_ctx=last_ctx, include_last=include_last)


def _make_verdict(
    *,
    agree: bool = True,
    more_complete: str = "tie",
    note: str = "Both answers agree.",
) -> MagicMock:
    v = MagicMock()
    v.agree = agree
    v.more_complete = more_complete
    v.note = note
    return v


def _make_stream(final_response: str = "Answer text.") -> AsyncMock:
    """Async-iterable mock whose get_final_response is sync (runner uses inspect.iscoroutine)."""
    stream = AsyncMock()
    stream.__aenter__ = AsyncMock(return_value=stream)
    stream.__aexit__ = AsyncMock(return_value=None)
    stream.get_final_response = MagicMock(return_value=final_response)
    return stream


def _patch_baml_streams(
    fake_baml: AsyncMock,
    rag_answer: str = "RAG answer.",
    cag_answer: str = "CAG answer.",
) -> None:
    fake_baml.stream.AnswerRAG.return_value = _make_stream(rag_answer)
    fake_baml.stream.AnswerCAG.return_value = _make_stream(cag_answer)
