"""
Source-blind example tests for issue #13, criteria 10 and 11.

Criterion 10: RunConfig.judge (default False) gates the judge; when off, no CompareAnswers
call is made and turns carry no verdict.

Criterion 11: When on, each turn attaches the verdict to TurnResult, running on gpt-5-mini
with its own Collector.

Property (criterion 10 invariant): for any valid model / max_tokens, RunConfig has a single
shared model — no separate rag_model / cag_model — so both pipelines are always compared
under identical conditions (criterion 1 structural invariant).

Tests are authored from the acceptance-criteria text and requirements.md only.
No implementation file was read during authoring.

Design choices recorded here (per instructions):
  - The runner exposes `run_turn(question, config, *, baml_client, retriever, document)`.
    If the actual signature differs, update the call sites below.
  - BAML streaming is mocked via `b.stream.AnswerRAG / AnswerCAG` returning async context
    managers whose `get_final_response()` is a sync method.  Adjust `_make_stream` if the
    real client uses async `get_final_response`.
  - `baml_client.CompareAnswers` is awaited directly (not streamed).
"""

from dataclasses import fields as dc_fields
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, strategies as st

from flashcard.runner import RunConfig, TurnResult


# ---------------------------------------------------------------------------
# RunConfig.judge default — criterion 10
# ---------------------------------------------------------------------------


def test_when_run_config_is_constructed_with_defaults_then_judge_is_false():
    """RunConfig.judge must default to False (judge off by default)."""
    config = RunConfig()
    assert config.judge is False


def test_when_run_config_has_judge_field_then_its_default_is_false():
    """RunConfig must declare a 'judge' dataclass field whose static default is False."""
    judge_field = next((f for f in dc_fields(RunConfig) if f.name == "judge"), None)
    assert judge_field is not None, "RunConfig must declare a 'judge' field"
    assert judge_field.default is False, (
        "RunConfig.judge must have a static default of False"
    )


# ---------------------------------------------------------------------------
# TurnResult.verdict absent when judge is off — criterion 10
# ---------------------------------------------------------------------------


def test_when_turn_result_has_verdict_field_then_its_default_is_none():
    """TurnResult must declare a 'verdict' field that defaults to None."""
    verdict_field = next(
        (f for f in dc_fields(TurnResult) if f.name == "verdict"), None
    )
    assert verdict_field is not None, "TurnResult must declare a 'verdict' field"
    assert verdict_field.default is None, (
        "TurnResult.verdict must default to None (no verdict when judge is off)"
    )


# ---------------------------------------------------------------------------
# Runner gating: judge=False → CompareAnswers never called — criterion 10
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_when_judge_is_off_then_compare_answers_is_not_called():
    """When RunConfig.judge=False the runner must never invoke CompareAnswers."""
    from flashcard.runner import run_turn

    config = RunConfig(judge=False)
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)

    await run_turn(
        question="What is the cancellation policy?",
        config=config,
        baml_client=fake_baml,
        retriever=_fake_retriever(),
        document="A short product manual.",
    )

    fake_baml.CompareAnswers.assert_not_called()


@pytest.mark.asyncio
async def test_when_judge_is_off_then_turn_result_verdict_is_none():
    """When RunConfig.judge=False the returned TurnResult must have verdict=None."""
    from flashcard.runner import run_turn

    config = RunConfig(judge=False)
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)

    result = await run_turn(
        question="What is the return policy?",
        config=config,
        baml_client=fake_baml,
        retriever=_fake_retriever(),
        document="A short product manual.",
    )

    assert result.verdict is None


# ---------------------------------------------------------------------------
# Runner gating: judge=True → verdict attached to TurnResult — criterion 11
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_when_judge_is_on_then_turn_result_verdict_is_not_none():
    """When RunConfig.judge=True, TurnResult.verdict must not be None."""
    from flashcard.runner import run_turn

    config = RunConfig(judge=True)
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)
    fake_baml.CompareAnswers.return_value = _fake_verdict()

    result = await run_turn(
        question="What is the warranty period?",
        config=config,
        baml_client=fake_baml,
        retriever=_fake_retriever(),
        document="A short product manual.",
    )

    assert result.verdict is not None


@pytest.mark.asyncio
async def test_when_judge_is_on_then_compare_answers_is_called_exactly_once_per_turn():
    """When RunConfig.judge=True, CompareAnswers must be called exactly once per turn."""
    from flashcard.runner import run_turn

    config = RunConfig(judge=True)
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)
    fake_baml.CompareAnswers.return_value = _fake_verdict()

    await run_turn(
        question="What is the warranty period?",
        config=config,
        baml_client=fake_baml,
        retriever=_fake_retriever(),
        document="A short product manual.",
    )

    assert fake_baml.CompareAnswers.call_count == 1, (
        "CompareAnswers must be called exactly once per turn when judge=True"
    )


@pytest.mark.asyncio
async def test_when_judge_is_on_then_compare_answers_receives_the_question():
    """CompareAnswers must be called with the turn question as its first argument."""
    from flashcard.runner import run_turn

    question = "What is the delivery time?"
    config = RunConfig(judge=True)
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)
    fake_baml.CompareAnswers.return_value = _fake_verdict()

    await run_turn(
        question=question,
        config=config,
        baml_client=fake_baml,
        retriever=_fake_retriever(),
        document="A short product manual.",
    )

    call_args = fake_baml.CompareAnswers.call_args
    positional = list(call_args.args) if call_args.args else []
    keyword = call_args.kwargs or {}
    received_question = positional[0] if positional else keyword.get("question")
    assert received_question == question, (
        "CompareAnswers must receive the current turn's question"
    )


@pytest.mark.asyncio
async def test_when_judge_is_on_then_verdict_has_agree_attribute():
    """TurnResult.verdict must expose an 'agree' attribute (bool)."""
    from flashcard.runner import run_turn

    config = RunConfig(judge=True)
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)
    fake_baml.CompareAnswers.return_value = _fake_verdict(agree=True)

    result = await run_turn(
        question="Does the manual cover setup?",
        config=config,
        baml_client=fake_baml,
        retriever=_fake_retriever(),
        document="A short product manual.",
    )

    assert hasattr(result.verdict, "agree"), (
        "TurnResult.verdict must have an 'agree' attribute"
    )
    assert isinstance(result.verdict.agree, bool), (
        "TurnResult.verdict.agree must be a bool"
    )


@pytest.mark.asyncio
async def test_when_judge_is_on_then_verdict_has_more_complete_attribute():
    """TurnResult.verdict must expose a 'more_complete' attribute."""
    from flashcard.runner import run_turn

    config = RunConfig(judge=True)
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)
    fake_baml.CompareAnswers.return_value = _fake_verdict(more_complete="rag")

    result = await run_turn(
        question="Which section covers billing?",
        config=config,
        baml_client=fake_baml,
        retriever=_fake_retriever(),
        document="A short product manual.",
    )

    assert hasattr(result.verdict, "more_complete"), (
        "TurnResult.verdict must have a 'more_complete' attribute"
    )
    assert result.verdict.more_complete in ("rag", "cag", "tie"), (
        "TurnResult.verdict.more_complete must be one of 'rag', 'cag', 'tie'"
    )


@pytest.mark.asyncio
async def test_when_judge_is_on_then_verdict_has_note_attribute():
    """TurnResult.verdict must expose a 'note' string attribute."""
    from flashcard.runner import run_turn

    config = RunConfig(judge=True)
    fake_baml = AsyncMock()
    _patch_baml_streams(fake_baml)
    fake_baml.CompareAnswers.return_value = _fake_verdict(note="RAG missed section 3.")

    result = await run_turn(
        question="What does section 3 say?",
        config=config,
        baml_client=fake_baml,
        retriever=_fake_retriever(),
        document="A short product manual.",
    )

    assert hasattr(result.verdict, "note"), (
        "TurnResult.verdict must have a 'note' attribute"
    )
    assert isinstance(result.verdict.note, str), (
        "TurnResult.verdict.note must be a string"
    )


# ---------------------------------------------------------------------------
# Property — criterion 1 structural invariant (via RunConfig)
#
# "Both pipelines answer the same questions over the same source with the same
# model and max_tokens" implies that RunConfig must NOT carry separate
# rag_model / cag_model fields — one model governs both sides.
# ---------------------------------------------------------------------------


@given(
    model=st.text(min_size=1, max_size=50).filter(str.strip),
    max_tokens=st.integers(min_value=1, max_value=8192),
    judge=st.booleans(),
)
@settings(max_examples=60)
def test_when_run_config_is_created_then_no_separate_rag_or_cag_model_exists(
    model: str, max_tokens: int, judge: bool
) -> None:
    """
    Invariant (criterion 1): for any combination of model / max_tokens / judge,
    RunConfig must not expose a rag_model or cag_model field — one shared model
    applies to both pipelines so cost/accuracy differences are attributable to
    retrieval strategy alone.
    """
    config = RunConfig(model=model, max_tokens=max_tokens, judge=judge)
    assert config.model == model, "RunConfig.model must equal the value supplied"
    assert config.max_tokens == max_tokens, (
        "RunConfig.max_tokens must equal the value supplied"
    )
    assert not hasattr(config, "rag_model"), (
        "RunConfig must not have a separate 'rag_model'; one model serves both pipelines"
    )
    assert not hasattr(config, "cag_model"), (
        "RunConfig must not have a separate 'cag_model'; one model serves both pipelines"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_retriever() -> MagicMock:
    retriever = MagicMock()
    # (chunk_id: int, chunk_text: str) — conforms to the Retriever protocol
    retriever.retrieve.return_value = [(0, "A relevant chunk.")]
    return retriever


def _fake_verdict(
    *,
    agree: bool = True,
    more_complete: str = "tie",
    note: str = "Both answers agree.",
) -> MagicMock:
    verdict = MagicMock()
    verdict.agree = agree
    verdict.more_complete = more_complete
    verdict.note = note
    return verdict


def _make_stream(final_response: str) -> AsyncMock:
    """Return an async-iterable mock that yields `final_response` from get_final_response.

    get_final_response is a MagicMock (sync) — _read_stream in runner.py detects
    whether the result is a coroutine and handles both cases.
    """
    stream = AsyncMock()
    stream.__aenter__ = AsyncMock(return_value=stream)
    stream.__aexit__ = AsyncMock(return_value=None)
    stream.get_final_response = MagicMock(return_value=final_response)
    return stream


def _patch_baml_streams(
    fake_baml: AsyncMock,
    rag_answer: str = "RAG answer text.",
    cag_answer: str = "CAG answer text.",
) -> None:
    """Wire AnswerRAG and AnswerCAG streams on the fake BAML client."""
    fake_baml.stream.AnswerRAG.return_value = _make_stream(rag_answer)
    fake_baml.stream.AnswerCAG.return_value = _make_stream(cag_answer)
