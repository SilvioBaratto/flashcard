"""
Source-blind tests for flashcard/cag.py (issue #10).

Every test is derived directly from the acceptance criteria — no implementation
source was read during authorship.  Fakes are built from the criteria text and
the requirements doc only.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

import pytest
from hypothesis import given, settings, strategies as st


# ---------------------------------------------------------------------------
# Fakes — constructed from acceptance-criteria text only
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_input_tokens = cached_input_tokens


class _FakeTiming:
    def __init__(self, duration_ms: float = 123.0) -> None:
        self.duration_ms = duration_ms


class _FakeLast:
    def __init__(self, duration_ms: float = 123.0) -> None:
        self.timing = _FakeTiming(duration_ms)


class FakeCollector:
    """
    Substitute for ``baml_py.Collector``.

    Criteria: the pipeline must create its own Collector and return a
    ``TurnMetrics`` snapshot parsed from it.  Pre-set usage values make
    ``TurnMetrics`` assertions deterministic without a live API call.
    """

    def __init__(
        self,
        *,
        input_tokens: int = 100,
        output_tokens: int = 20,
        cached_input_tokens: int = 80,
        duration_ms: float = 123.0,
    ) -> None:
        self.usage = _FakeUsage(input_tokens, output_tokens, cached_input_tokens)
        self.last = _FakeLast(duration_ms)


class _FakeStream:
    """Async-iterable BAML stream fake that yields partials then a final text."""

    def __init__(self, partials: List[str], final_text: str) -> None:
        self._partials = list(partials)
        self._final_text = final_text
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self._index >= len(self._partials):
            raise StopAsyncIteration
        value = self._partials[self._index]
        self._index += 1
        return value

    async def get_final_response(self) -> str:
        return self._final_text


class _FakeStreamNS:
    """Records ``AnswerCAG`` call arguments; mirrors the ``b.stream`` namespace."""

    def __init__(self, partials: List[str], final_text: str) -> None:
        self._partials = partials
        self._final_text = final_text
        self.received_document: Optional[str] = None
        self.received_question: Optional[str] = None
        self.received_baml_options: Optional[dict] = None

    def AnswerCAG(
        self,
        *,
        document: str,
        question: str,
        baml_options: Optional[dict] = None,
    ) -> _FakeStream:
        self.received_document = document
        self.received_question = question
        self.received_baml_options = dict(baml_options) if baml_options else {}
        return _FakeStream(self._partials, self._final_text)


class FakeBAML:
    """Fake BAML client injected in place of ``baml_client.async_client.b``."""

    def __init__(
        self,
        partials: List[str] = (),
        final_text: str = "answer",
    ) -> None:
        self.stream = _FakeStreamNS(list(partials), final_text)


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

_DOCUMENT = "Chapter 1: Intro. Chapter 2: Methods. Chapter 3: Results."
_QUESTION = "What does Chapter 2 describe?"
_PARTIALS = ["Chapter 2 describes", " the research methods."]
_FINAL_TEXT = "Chapter 2 describes the research methods."


# ---------------------------------------------------------------------------
# Fixture: resolve answer_cag from flashcard.cag
#
# The acceptance criteria allow either ``answer_cag`` (function) or
# ``CagPipeline`` (class); we prefer ``answer_cag`` and fail clearly when
# neither is present, which is the expected Red-phase state.
# ---------------------------------------------------------------------------


@pytest.fixture()
def answer_cag():
    """Return the ``answer_cag`` callable from ``flashcard.cag``."""
    from flashcard.cag import answer_cag as _fn  # fails until implemented (Red)

    return _fn


@pytest.fixture()
def fake_collector() -> FakeCollector:
    return FakeCollector(
        input_tokens=200,
        output_tokens=30,
        cached_input_tokens=160,
        duration_ms=250.0,
    )


@pytest.fixture()
def fake_b() -> FakeBAML:
    return FakeBAML(partials=_PARTIALS, final_text=_FINAL_TEXT)


# ---------------------------------------------------------------------------
# Criterion: pipeline returns (text, TurnMetrics) from its own Collector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_when_document_and_question_given_then_final_text_is_returned(
    answer_cag, fake_b: FakeBAML, fake_collector: FakeCollector
) -> None:
    """answer_cag returns the text produced by stream.get_final_response()."""
    text, _ = await answer_cag(
        _DOCUMENT, _QUESTION, b=fake_b, collector_factory=lambda: fake_collector
    )
    assert text == _FINAL_TEXT


@pytest.mark.asyncio
async def test_when_pipeline_runs_then_turn_metrics_instance_is_returned(
    answer_cag, fake_b: FakeBAML, fake_collector: FakeCollector
) -> None:
    """The second return value is an instance of TurnMetrics (criteria-named type)."""
    from flashcard.metrics import TurnMetrics  # referenced explicitly in criteria

    _, metrics = await answer_cag(
        _DOCUMENT, _QUESTION, b=fake_b, collector_factory=lambda: fake_collector
    )
    assert isinstance(metrics, TurnMetrics)


@pytest.mark.asyncio
async def test_when_collector_has_usage_values_then_metrics_reflect_them(
    answer_cag, fake_b: FakeBAML
) -> None:
    """TurnMetrics fields mirror the Collector's usage exactly."""
    collector = FakeCollector(
        input_tokens=300,
        output_tokens=45,
        cached_input_tokens=240,
        duration_ms=500.0,
    )
    _, metrics = await answer_cag(
        _DOCUMENT, _QUESTION, b=fake_b, collector_factory=lambda: collector
    )
    assert metrics.input_tokens == 300
    assert metrics.output_tokens == 45
    assert metrics.cached_input_tokens == 240


# ---------------------------------------------------------------------------
# Criterion: full document forwarded unchanged; no cache_control marker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_when_document_passed_then_baml_receives_it_unchanged(
    answer_cag, fake_collector: FakeCollector
) -> None:
    """The exact document string reaches b.stream.AnswerCAG without modification."""
    unique_doc = "unique\x00doc\nwith\ttabs and ñ special chars — verbatim"
    b = FakeBAML(final_text="ok")
    await answer_cag(
        unique_doc, _QUESTION, b=b, collector_factory=lambda: fake_collector
    )
    assert b.stream.received_document == unique_doc


@pytest.mark.asyncio
async def test_when_question_passed_then_baml_receives_it_unchanged(
    answer_cag, fake_collector: FakeCollector
) -> None:
    """The question string reaches b.stream.AnswerCAG without modification."""
    unique_q = "unique question: 漢字 emoji 🔥 — verbatim?"
    b = FakeBAML(final_text="ok")
    await answer_cag(_DOCUMENT, unique_q, b=b, collector_factory=lambda: fake_collector)
    assert b.stream.received_question == unique_q


@pytest.mark.asyncio
async def test_when_called_then_no_cache_control_key_in_baml_options(
    answer_cag, fake_b: FakeBAML, fake_collector: FakeCollector
) -> None:
    """
    No cache_control key appears in baml_options.

    Criteria: 'with NO cache_control marker'.  OpenAI caching is automatic;
    an explicit marker would be incorrect and potentially break the call.
    """
    await answer_cag(
        _DOCUMENT, _QUESTION, b=fake_b, collector_factory=lambda: fake_collector
    )
    opts = fake_b.stream.received_baml_options or {}
    assert "cache_control" not in opts


# ---------------------------------------------------------------------------
# Criterion: streamed text equals final response (not concatenation of partials)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_when_final_text_differs_from_partials_concat_then_final_text_is_returned(
    answer_cag, fake_collector: FakeCollector
) -> None:
    """
    Returned text is get_final_response(), not the join of stream partials.

    Criteria: 'the streamed text equals the final response'.  The pipeline
    must call get_final_response() — not accumulate partial strings.
    """
    b = FakeBAML(
        partials=["part one", " part two"],
        final_text="Canonical final answer — distinct from partials concat",
    )
    text, _ = await answer_cag(
        _DOCUMENT, _QUESTION, b=b, collector_factory=lambda: fake_collector
    )
    assert text == "Canonical final answer — distinct from partials concat"
    assert text != "part one part two"


# ---------------------------------------------------------------------------
# Criterion: fresh tokens = input − cached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_when_tokens_measured_then_fresh_equals_input_minus_cached(
    answer_cag, fake_b: FakeBAML
) -> None:
    """fresh_tokens = input_tokens − cached_input_tokens per the criteria formula."""
    # input=200, cached=160 → fresh must be 40
    collector = FakeCollector(
        input_tokens=200, output_tokens=30, cached_input_tokens=160
    )
    _, metrics = await answer_cag(
        _DOCUMENT, _QUESTION, b=fake_b, collector_factory=lambda: collector
    )
    fresh = metrics.input_tokens - metrics.cached_input_tokens
    assert fresh == 40


@pytest.mark.asyncio
async def test_when_cache_is_cold_then_fresh_equals_all_input_tokens(
    answer_cag, fake_b: FakeBAML
) -> None:
    """On Q1 (nothing cached), cached=0 so fresh = input_tokens (full base-rate charge)."""
    collector = FakeCollector(input_tokens=500, output_tokens=40, cached_input_tokens=0)
    _, metrics = await answer_cag(
        _DOCUMENT, _QUESTION, b=fake_b, collector_factory=lambda: collector
    )
    fresh = metrics.input_tokens - metrics.cached_input_tokens
    assert fresh == 500
    assert fresh == metrics.input_tokens


# ---------------------------------------------------------------------------
# Criterion: streaming partials surfaced via injected callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_when_on_token_callback_provided_then_each_partial_is_forwarded(
    answer_cag, fake_b: FakeBAML, fake_collector: FakeCollector
) -> None:
    """Each yielded partial is passed to on_token exactly once, in order."""
    received: List[str] = []
    await answer_cag(
        _DOCUMENT,
        _QUESTION,
        on_token=lambda p: received.append(p),
        b=fake_b,
        collector_factory=lambda: fake_collector,
    )
    assert received == _PARTIALS


@pytest.mark.asyncio
async def test_when_on_token_is_omitted_then_pipeline_completes_without_error(
    answer_cag, fake_b: FakeBAML, fake_collector: FakeCollector
) -> None:
    """on_token is optional; omitting it must not raise TypeError or AttributeError."""
    result = await answer_cag(
        _DOCUMENT, _QUESTION, b=fake_b, collector_factory=lambda: fake_collector
    )
    assert result is not None


@pytest.mark.asyncio
async def test_when_on_token_is_none_then_pipeline_completes_without_error(
    answer_cag, fake_b: FakeBAML, fake_collector: FakeCollector
) -> None:
    """Passing on_token=None explicitly must not raise."""
    result = await answer_cag(
        _DOCUMENT,
        _QUESTION,
        on_token=None,
        b=fake_b,
        collector_factory=lambda: fake_collector,
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Property-based tests derived from criteria invariants
#
# Two invariants extracted from the criteria:
#   1. Document identity — any document arrives at AnswerCAG unchanged.
#   2. Fresh-token non-negativity — fresh = input − cached ≥ 0 for valid usage.
# ---------------------------------------------------------------------------


@settings(max_examples=50)
@given(document=st.text(min_size=1))
def test_when_any_nonempty_document_given_then_it_arrives_unchanged_at_baml(
    document: str,
) -> None:
    """Invariant (identity): any non-empty document arrives at AnswerCAG unchanged."""
    from flashcard.cag import answer_cag

    b = FakeBAML(partials=[], final_text="ok")
    collector = FakeCollector()
    asyncio.run(answer_cag(document, "q?", b=b, collector_factory=lambda: collector))
    assert b.stream.received_document == document


@settings(max_examples=100)
@given(
    input_tokens=st.integers(min_value=0, max_value=50_000),
    cached_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_when_fresh_tokens_derived_then_result_is_non_negative(
    input_tokens: int, cached_fraction: float
) -> None:
    """
    Invariant (non-negativity): fresh = input − cached ≥ 0 for any valid usage.

    OpenAI guarantees cached_input_tokens ≤ input_tokens; we model this by
    deriving cached as a fraction of input (clamped to int) so the constraint
    is always satisfied in the fake.  The returned metrics must preserve it.
    """
    from flashcard.cag import answer_cag

    cached = min(int(input_tokens * cached_fraction), input_tokens)
    b = FakeBAML(partials=[], final_text="ok")
    collector = FakeCollector(
        input_tokens=input_tokens,
        output_tokens=0,
        cached_input_tokens=cached,
    )
    _, metrics = asyncio.run(
        answer_cag("doc", "q?", b=b, collector_factory=lambda: collector)
    )
    fresh = metrics.input_tokens - metrics.cached_input_tokens
    assert fresh >= 0
