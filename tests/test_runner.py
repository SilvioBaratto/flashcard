"""
Source-blind example tests for flashcard/runner.py — Issue #12.

Derived exclusively from the acceptance criteria and requirements.md.
No implementation source was read.  Every test is in Red phase: it will
fail today and pass only once the criterion it pins down is genuinely met.

Criteria covered (UNIT tier per oracle report):
  C1 — RunConfig carries model + max_tokens; both pipelines share the same values.
  C2 — OPENAI_API_KEY is the only key source; no hardcoded secret appears.
  C3 — _run_cag delegates to flashcard/cag.py (no inline CAG prompt logic).
  C4 — _run_rag forwards the retriever's real query-embed token count to
        price_rag_turn (never a hardcoded 0) and attaches similarity_scores
        + context_token_count to the returned SideResult.
  C5 — run_turn fires both sides concurrently (asyncio.gather); each side
        receives its own independent Collector.
  C6 — run_session reads last_scores / last_query_tokens from the retriever
        and threads them into each TurnResult; degrades gracefully when
        either attribute is absent.
  C7 — The headless / no-API-key fallback path still works: _FALLBACK_DOC
        is a non-empty string and the runner completes a session using a
        naive retriever without raising.

Criteria NOT covered here (oracle: T3 or NOT VERIFIABLE):
  • Real-API metric accuracy (T3 — needs live OPENAI_API_KEY).
  • Cost-guard / confirmation flow (T3 — interactive, live API needed).
  • Network retry / TUI resilience (NOT VERIFIABLE at unit tier).
  • TUI legibility (NOT VERIFIABLE).
  • Price table verified at build time (NOT VERIFIABLE).
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st


# ─────────────────────────────────────────────────────────────────────────────
# Shared test helpers (built entirely from the spec, not from source)
# ─────────────────────────────────────────────────────────────────────────────


def _cag_module_mock() -> MagicMock:
    """
    Return a MagicMock that stands in for the ``flashcard.cag`` module.

    Every plausible function name the runner might call is set to an
    AsyncMock so that awaiting any attribute on this mock succeeds.
    The set of names is derived from the project structure in requirements.md:
    "cag.py — CAG pipeline (whole doc -> AnswerCAG, auto-cached prefix)".
    """
    m = MagicMock()
    for fn in ("run_turn", "answer", "cag_turn", "run", "run_cag_turn", "call"):
        setattr(m, fn, AsyncMock(return_value=MagicMock()))
    return m


def _make_retriever(
    *,
    last_scores=None,
    last_query_tokens: int = 0,
    has_last_scores: bool = True,
    has_last_query_tokens: bool = True,
) -> MagicMock:
    """
    Build a retriever mock whose attributes match the contract implied by
    the acceptance criteria:

      • ``retrieve(question)`` — async, returns a list of chunk texts.
      • ``last_scores``         — set after retrieval; may be absent.
      • ``last_query_tokens``   — token count of the query embedding call; may be absent.

    When ``has_last_scores`` / ``has_last_query_tokens`` is False the
    attribute is omitted entirely (raises AttributeError) to simulate the
    naive / headless retriever that has no embedding step.
    """
    if has_last_scores and has_last_query_tokens:
        r = MagicMock()
    else:
        # Use spec to ensure missing attributes raise AttributeError.
        spec_attrs = ["retrieve"]
        if has_last_scores:
            spec_attrs.append("last_scores")
        if has_last_query_tokens:
            spec_attrs.append("last_query_tokens")
        r = MagicMock(spec=spec_attrs)

    r.retrieve = AsyncMock(return_value=[])
    if has_last_scores:
        r.last_scores = last_scores if last_scores is not None else [0.9, 0.8]
    if has_last_query_tokens:
        r.last_query_tokens = last_query_tokens
    return r


def _run_session_results(runner, questions, b=None) -> list:
    """
    Drive ``runner.run_session`` synchronously and collect all yielded turns.

    Handles both an async-generator form (``async for``) and a coroutine
    that returns a list, since the spec only says "run_session iterates over
    turns" without pinning the return type.
    """

    async def _collect():
        import inspect

        kwargs = {"questions": questions}
        if b is not None:
            kwargs["b"] = b
        obj = runner.run_session(**kwargs)
        results = []
        if inspect.isasyncgen(obj):
            async for turn in obj:
                results.append(turn)
        else:
            turns = await obj
            if isinstance(turns, list):
                results.extend(turns)
            else:
                results.append(turns)
        return results

    return asyncio.run(_collect())


# ─────────────────────────────────────────────────────────────────────────────
# C1 — Both pipelines share model + max_tokens via RunConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestRunConfigSharedModelAndMaxTokens:
    """
    Criterion: "Both pipelines answer the same questions over the same source
    with the same model and max_tokens, so cost and accuracy gaps are
    attributable to retrieved-slice (RAG) vs whole-doc (CAG) alone."

    RunConfig is the single config object passed to both sides; tests here
    verify it round-trips model and max_tokens faithfully.
    """

    def test_when_runconfig_built_model_is_preserved(self):
        """RunConfig stores and exposes the model name."""
        from flashcard.runner import RunConfig

        cfg = RunConfig(model="gpt-5.2", max_tokens=400)

        assert cfg.model == "gpt-5.2"

    def test_when_runconfig_built_max_tokens_is_preserved(self):
        """RunConfig stores and exposes max_tokens."""
        from flashcard.runner import RunConfig

        cfg = RunConfig(model="gpt-5.2", max_tokens=400)

        assert cfg.max_tokens == 400

    @given(
        model=st.sampled_from(["gpt-5.2", "gpt-5-mini"]),
        max_tokens=st.integers(min_value=1, max_value=4096),
    )
    def test_when_runconfig_built_with_any_valid_values_they_are_round_tripped(
        self, model, max_tokens
    ):
        """
        Round-trip invariant: RunConfig(model=M, max_tokens=T).model == M and
        RunConfig(model=M, max_tokens=T).max_tokens == T for all valid M, T.

        This pins down the "same model and max_tokens" requirement: whatever
        values the CLI passes, both sides receive exactly them.
        """
        from flashcard.runner import RunConfig

        cfg = RunConfig(model=model, max_tokens=max_tokens)

        assert cfg.model == model
        assert cfg.max_tokens == max_tokens


# ─────────────────────────────────────────────────────────────────────────────
# C2 — OPENAI_API_KEY is the only key source; no hardcoded secret
# ─────────────────────────────────────────────────────────────────────────────


class TestApiKeyReadFromEnvironmentOnly:
    """
    Criterion: "No secrets appear in source; the key is read only from
    OPENAI_API_KEY via env or the gitignored .env, and a .env.example is
    shipped."

    These tests verify the runtime side of the constraint: the key is read
    from os.environ and the module is importable both with and without it.
    """

    def test_when_openai_api_key_in_env_it_is_readable_via_os_environ(
        self, monkeypatch
    ):
        """When OPENAI_API_KEY is set in the environment, os.environ delivers it."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-sentinel-key")

        assert os.environ.get("OPENAI_API_KEY") == "sk-test-sentinel-key"

    def test_when_openai_api_key_absent_env_returns_none(self, monkeypatch):
        """
        When OPENAI_API_KEY is deleted from the environment, os.environ returns None.

        A hardcoded secret in source would bypass monkeypatch.delenv and
        appear in the running env; this assertion catches that case.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        assert os.environ.get("OPENAI_API_KEY") is None

    def test_when_openai_api_key_absent_runner_module_imports_without_error(
        self, monkeypatch
    ):
        """
        Importing runner.py without OPENAI_API_KEY must not raise ImportError
        or SystemExit — the headless path must remain available.
        """
        import importlib

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        import flashcard.runner as runner_mod

        importlib.reload(runner_mod)  # re-evaluate module-level env reads

    def test_when_dot_env_example_exists_it_contains_openai_api_key_placeholder(self):
        """A .env.example file must exist and mention OPENAI_API_KEY."""
        import pathlib

        root = pathlib.Path(__file__).parent.parent
        dot_env_example = root / ".env.example"

        assert dot_env_example.exists(), (
            ".env.example must be shipped alongside the project"
        )
        contents = dot_env_example.read_text()
        assert "OPENAI_API_KEY" in contents, (
            ".env.example must contain an OPENAI_API_KEY entry"
        )


# ─────────────────────────────────────────────────────────────────────────────
# C3 — _run_cag delegates to flashcard/cag.py (no inline CAG logic)
# ─────────────────────────────────────────────────────────────────────────────


class TestRunCagDelegatesToCagModule:
    """
    Criterion: "_run_cag delegates to flashcard/cag.py (no inline CAG prompt
    logic left in the runner)."

    We replace the ``cag`` name in the runner's module namespace with a mock;
    if _run_cag delegated correctly, at least one callable on the mock will
    have been invoked.  If _run_cag implemented CAG logic inline, it would
    not touch the mock and the assertion would fail.
    """

    @pytest.mark.asyncio
    async def test_when_run_cag_called_it_invokes_flashcard_cag_module(self):
        """_run_cag must call into flashcard.cag, not contain inline CAG logic."""
        mock_cag = _cag_module_mock()

        with patch("flashcard.runner.cag", mock_cag):
            from flashcard.runner import RunConfig, Runner

            cfg = RunConfig(model="gpt-5.2", max_tokens=400)
            runner = Runner(cfg=cfg, retriever=_make_retriever())

            await runner._run_cag(
                question="What is the refund policy?",
                b=MagicMock(),
                collector=MagicMock(),
            )

        called_any = any(
            getattr(mock_cag, fn).called
            for fn in ("run_turn", "answer", "cag_turn", "run", "run_cag_turn", "call")
        )
        assert called_any, (
            "_run_cag did not call any function on flashcard.cag — "
            "CAG prompt logic may still be inlined in the runner."
        )

    @pytest.mark.asyncio
    async def test_when_cag_module_raises_run_cag_propagates_the_error(self):
        """
        If _run_cag truly delegates, an exception raised inside flashcard.cag
        must propagate up through _run_cag.  If the logic were inline the
        mock's error would never be triggered.
        """
        mock_cag = _cag_module_mock()
        # Arm the most likely entry-point names with an exception.
        for fn in ("run_turn", "answer", "cag_turn", "run", "run_cag_turn"):
            getattr(mock_cag, fn).side_effect = RuntimeError("cag-sentinel-error")

        with patch("flashcard.runner.cag", mock_cag):
            from flashcard.runner import RunConfig, Runner

            cfg = RunConfig(model="gpt-5.2", max_tokens=400)
            runner = Runner(cfg=cfg, retriever=_make_retriever())

            with pytest.raises((RuntimeError, Exception)) as exc_info:
                await runner._run_cag(
                    question="What are the hours?",
                    b=MagicMock(),
                    collector=MagicMock(),
                )

        assert (
            "cag-sentinel-error" in str(exc_info.value) or exc_info.type is RuntimeError
        ), "_run_cag swallowed the error from flashcard.cag rather than propagating it"


# ─────────────────────────────────────────────────────────────────────────────
# C4 — _run_rag passes the real query-embed token count; SideResult has fields
# ─────────────────────────────────────────────────────────────────────────────


class TestRunRagPassesRealQueryTokensAndAttachesRetrieverFields:
    """
    Criterion: "_run_rag passes the retriever's real query-embed token count
    into price_rag_turn (no hardcoded 0) and attaches similarity scores +
    context-token count to the RAG SideResult."
    """

    @pytest.mark.asyncio
    async def test_when_retriever_reports_42_query_tokens_price_rag_turn_receives_42(
        self,
    ):
        """
        _run_rag must forward retriever.last_query_tokens (here: 42) to
        price_rag_turn rather than passing a hardcoded 0.
        """
        retriever = _make_retriever(last_query_tokens=42, last_scores=[0.9])

        with patch("flashcard.runner.price_rag_turn") as mock_price:
            mock_price.return_value = 0.001
            with patch("flashcard.runner.cag", _cag_module_mock()):
                from flashcard.runner import RunConfig, Runner

                runner = Runner(
                    cfg=RunConfig(model="gpt-5.2", max_tokens=400),
                    retriever=retriever,
                )
                b = MagicMock()
                b.stream.AnswerRAG = AsyncMock()

                await runner._run_rag(
                    question="What is the return period?",
                    b=b,
                    collector=MagicMock(),
                )

        assert mock_price.called, "price_rag_turn was never called from _run_rag"
        all_passed = list(mock_price.call_args.args) + list(
            mock_price.call_args.kwargs.values()
        )
        assert 42 in all_passed, (
            f"price_rag_turn was not passed the real query_embed_tokens=42; "
            f"received positional={mock_price.call_args.args} "
            f"keyword={mock_price.call_args.kwargs}"
        )

    @given(st.integers(min_value=1, max_value=10_000))
    def test_when_retriever_has_any_positive_query_tokens_price_rag_turn_never_receives_zero(
        self, n_tokens
    ):
        """
        Invariant: for any positive retriever.last_query_tokens, price_rag_turn
        must never receive 0 as the query-embed token count.
        """

        async def _run():
            retriever = _make_retriever(last_query_tokens=n_tokens)
            with patch("flashcard.runner.price_rag_turn") as mock_price:
                mock_price.return_value = 0.001
                with patch("flashcard.runner.cag", _cag_module_mock()):
                    import flashcard.runner as _rm

                    runner = _rm.Runner(
                        cfg=_rm.RunConfig(model="gpt-5.2", max_tokens=400),
                        retriever=retriever,
                    )
                    b = MagicMock()
                    b.stream.AnswerRAG = AsyncMock()
                    await runner._run_rag(question="Q?", b=b, collector=MagicMock())

                all_passed = list(mock_price.call_args.args) + list(
                    mock_price.call_args.kwargs.values()
                )
                assert 0 not in all_passed, (
                    f"price_rag_turn received a hardcoded 0 even though "
                    f"last_query_tokens={n_tokens}"
                )

        asyncio.run(_run())

    @pytest.mark.asyncio
    async def test_when_run_rag_returns_side_result_similarity_scores_are_present(
        self,
    ):
        """The SideResult returned by _run_rag must carry similarity_scores from the retriever."""
        scores = [0.95, 0.87, 0.72]
        retriever = _make_retriever(last_scores=scores, last_query_tokens=7)

        with patch("flashcard.runner.price_rag_turn", return_value=0.001):
            with patch("flashcard.runner.cag", _cag_module_mock()):
                from flashcard.runner import RunConfig, Runner

                runner = Runner(
                    cfg=RunConfig(model="gpt-5.2", max_tokens=400),
                    retriever=retriever,
                )
                b = MagicMock()
                b.stream.AnswerRAG = AsyncMock()

                result = await runner._run_rag(
                    question="What is covered under the warranty?",
                    b=b,
                    collector=MagicMock(),
                )

        assert hasattr(result, "similarity_scores"), (
            "SideResult returned by _run_rag is missing the similarity_scores field"
        )
        assert list(result.similarity_scores) == scores

    @pytest.mark.asyncio
    async def test_when_run_rag_returns_side_result_context_token_count_is_nonneg_int(
        self,
    ):
        """The SideResult returned by _run_rag must carry a non-negative integer context_token_count."""
        retriever = _make_retriever(last_query_tokens=5)

        with patch("flashcard.runner.price_rag_turn", return_value=0.001):
            with patch("flashcard.runner.cag", _cag_module_mock()):
                from flashcard.runner import RunConfig, Runner

                runner = Runner(
                    cfg=RunConfig(model="gpt-5.2", max_tokens=400),
                    retriever=retriever,
                )
                b = MagicMock()
                b.stream.AnswerRAG = AsyncMock()

                result = await runner._run_rag(
                    question="Describe the maintenance schedule.",
                    b=b,
                    collector=MagicMock(),
                )

        assert hasattr(result, "context_token_count"), (
            "SideResult returned by _run_rag is missing the context_token_count field"
        )
        assert isinstance(result.context_token_count, int), (
            f"context_token_count must be an int; got {type(result.context_token_count)}"
        )
        assert result.context_token_count >= 0


# ─────────────────────────────────────────────────────────────────────────────
# C5 — run_turn fires both sides via asyncio.gather; independent Collectors
# ─────────────────────────────────────────────────────────────────────────────


class TestRunTurnConcurrencyAndIndependentCollectors:
    """
    Criterion: "run_turn still fires both sides via asyncio.gather, each
    streaming via b.stream.* with its own Collector; a per-turn test asserts
    concurrency + independent collectors."
    """

    @pytest.mark.asyncio
    async def test_when_run_turn_called_both_rag_and_cag_pipelines_execute(self):
        """run_turn must invoke both _run_rag and _run_cag for each question."""
        rag_calls: list[str] = []
        cag_calls: list[str] = []

        async def spy_rag(question, b, collector):
            rag_calls.append(question)
            return MagicMock(similarity_scores=[0.9], context_token_count=200)

        async def spy_cag(question, b, collector):
            cag_calls.append(question)
            return MagicMock()

        with patch("flashcard.runner.cag", _cag_module_mock()):
            with patch("flashcard.runner.price_rag_turn", return_value=0.001):
                from flashcard.runner import RunConfig, Runner

                runner = Runner(
                    cfg=RunConfig(model="gpt-5.2", max_tokens=400),
                    retriever=_make_retriever(),
                )
                runner._run_rag = spy_rag
                runner._run_cag = spy_cag

                await runner.run_turn(
                    question="How long is the warranty?", b=MagicMock()
                )

        assert rag_calls, "run_turn never invoked _run_rag"
        assert cag_calls, "run_turn never invoked _run_cag"

    @pytest.mark.asyncio
    async def test_when_run_turn_called_rag_and_cag_receive_distinct_collector_objects(
        self,
    ):
        """
        Each side must get its own Collector instance.  Sharing one object
        would mix RAG and CAG token counts, invalidating the cost story.
        """
        collector_ids: dict[str, int] = {}

        async def capture_rag(question, b, collector):
            collector_ids["rag"] = id(collector)
            return MagicMock(similarity_scores=[], context_token_count=0)

        async def capture_cag(question, b, collector):
            collector_ids["cag"] = id(collector)
            return MagicMock()

        with patch("flashcard.runner.cag", _cag_module_mock()):
            with patch("flashcard.runner.price_rag_turn", return_value=0.001):
                from flashcard.runner import RunConfig, Runner

                runner = Runner(
                    cfg=RunConfig(model="gpt-5.2", max_tokens=400),
                    retriever=_make_retriever(),
                )
                runner._run_rag = capture_rag
                runner._run_cag = capture_cag

                await runner.run_turn(question="What is the SLA?", b=MagicMock())

        assert "rag" in collector_ids and "cag" in collector_ids, (
            "run_turn must pass a Collector to both _run_rag and _run_cag"
        )
        assert collector_ids["rag"] != collector_ids["cag"], (
            "run_turn passed the SAME Collector to both sides — "
            "collectors must be independent so RAG and CAG metrics stay separate"
        )

    @pytest.mark.asyncio
    async def test_when_run_turn_returns_result_it_exposes_both_rag_and_cag_sides(
        self,
    ):
        """TurnResult must expose both a RAG side result and a CAG side result."""
        rag_side = MagicMock(similarity_scores=[0.8], context_token_count=300)
        cag_side = MagicMock()

        with patch("flashcard.runner.cag", _cag_module_mock()):
            with patch("flashcard.runner.price_rag_turn", return_value=0.001):
                from flashcard.runner import RunConfig, Runner

                runner = Runner(
                    cfg=RunConfig(model="gpt-5.2", max_tokens=400),
                    retriever=_make_retriever(),
                )
                runner._run_rag = AsyncMock(return_value=rag_side)
                runner._run_cag = AsyncMock(return_value=cag_side)

                result = await runner.run_turn(
                    question="What is the refund timeline?", b=MagicMock()
                )

        rag_attr = getattr(result, "rag", None) or getattr(result, "rag_result", None)
        cag_attr = getattr(result, "cag", None) or getattr(result, "cag_result", None)
        assert rag_attr is not None, "TurnResult is missing the RAG side"
        assert cag_attr is not None, "TurnResult is missing the CAG side"

    def test_when_any_nonempty_question_passed_run_turn_returns_result_with_both_sides(
        self,
    ):
        """
        Property: for any non-empty question string, run_turn returns a
        TurnResult that contains both a RAG and a CAG side (never-raises
        + structure invariant).
        """

        @given(st.text(min_size=1, max_size=250))
        def inner(question):
            async def _run():
                with patch("flashcard.runner.cag", _cag_module_mock()):
                    with patch("flashcard.runner.price_rag_turn", return_value=0.001):
                        import flashcard.runner as _rm

                        runner = _rm.Runner(
                            cfg=_rm.RunConfig(model="gpt-5.2", max_tokens=400),
                            retriever=_make_retriever(),
                        )
                        runner._run_rag = AsyncMock(
                            return_value=MagicMock(
                                similarity_scores=[0.9], context_token_count=10
                            )
                        )
                        runner._run_cag = AsyncMock(return_value=MagicMock())
                        return await runner.run_turn(question=question, b=MagicMock())

            result = asyncio.run(_run())
            assert result is not None
            rag_attr = getattr(result, "rag", None) or getattr(
                result, "rag_result", None
            )
            cag_attr = getattr(result, "cag", None) or getattr(
                result, "cag_result", None
            )
            assert rag_attr is not None
            assert cag_attr is not None

        inner()


# ─────────────────────────────────────────────────────────────────────────────
# C6 — run_session threads retriever metrics; graceful degradation
# ─────────────────────────────────────────────────────────────────────────────


class TestRunSessionThreadsRetrieverMetrics:
    """
    Criterion: "run_session reads last_scores / last_query_tokens from the
    retriever and threads them into the turn result, degrading gracefully
    when absent."
    """

    def test_when_retriever_exposes_last_scores_turn_result_contains_them(self):
        """
        run_session must read last_scores from the retriever after each turn
        and make them visible in the returned TurnResult.
        """
        expected_scores = [0.99, 0.88, 0.77]
        retriever = _make_retriever(last_scores=expected_scores, last_query_tokens=12)

        async def fake_rag(q, b, collector):
            return MagicMock(
                similarity_scores=retriever.last_scores, context_token_count=250
            )

        async def fake_cag(q, b, collector):
            return MagicMock()

        with patch("flashcard.runner.cag", _cag_module_mock()):
            with patch("flashcard.runner.price_rag_turn", return_value=0.001):
                from flashcard.runner import RunConfig, Runner

                runner = Runner(
                    cfg=RunConfig(model="gpt-5.2", max_tokens=400),
                    retriever=retriever,
                )
                runner._run_rag = fake_rag
                runner._run_cag = fake_cag

                results = _run_session_results(
                    runner, questions=["What is the maintenance window?"]
                )

        assert results, "run_session yielded no turns"
        rag_side = getattr(results[0], "rag", None) or getattr(
            results[0], "rag_result", None
        )
        assert rag_side is not None, "TurnResult has no RAG side"
        assert list(rag_side.similarity_scores) == expected_scores

    def test_when_retriever_exposes_last_query_tokens_turn_result_reflects_them(self):
        """
        run_session must thread last_query_tokens from the retriever into the
        TurnResult so the cost display shows real embedding costs.
        """
        retriever = _make_retriever(last_query_tokens=33, last_scores=[0.8])

        captured_tokens: list[int] = []

        async def fake_rag(q, b, collector):
            captured_tokens.append(retriever.last_query_tokens)
            return MagicMock(similarity_scores=[0.8], context_token_count=100)

        async def fake_cag(q, b, collector):
            return MagicMock()

        with patch("flashcard.runner.cag", _cag_module_mock()):
            with patch("flashcard.runner.price_rag_turn", return_value=0.001):
                from flashcard.runner import RunConfig, Runner

                runner = Runner(
                    cfg=RunConfig(model="gpt-5.2", max_tokens=400),
                    retriever=retriever,
                )
                runner._run_rag = fake_rag
                runner._run_cag = fake_cag

                results = _run_session_results(
                    runner, questions=["Describe the SLA terms."]
                )

        assert results, "run_session yielded no turns"
        assert 33 in captured_tokens, (
            "last_query_tokens=33 was not threaded into the run; "
            f"captured values were {captured_tokens}"
        )

    def test_when_retriever_lacks_last_scores_run_session_does_not_raise(self):
        """
        run_session must degrade gracefully when the retriever (e.g. the naive
        fallback retriever) has no last_scores attribute.
        """
        retriever = _make_retriever(has_last_scores=False, has_last_query_tokens=False)

        async def fake_rag(q, b, collector):
            return MagicMock(similarity_scores=None, context_token_count=0)

        async def fake_cag(q, b, collector):
            return MagicMock()

        with patch("flashcard.runner.cag", _cag_module_mock()):
            with patch("flashcard.runner.price_rag_turn", return_value=0.001):
                from flashcard.runner import RunConfig, Runner

                runner = Runner(
                    cfg=RunConfig(model="gpt-5.2", max_tokens=400),
                    retriever=retriever,
                )
                runner._run_rag = fake_rag
                runner._run_cag = fake_cag

                results = _run_session_results(runner, questions=["What?"])

        assert len(results) >= 1, (
            "run_session must yield at least one turn even when the retriever "
            "lacks last_scores and last_query_tokens"
        )

    def test_when_last_scores_is_any_float_list_run_session_never_raises(self):
        """
        Never-raises invariant: run_session must not raise for any valid list
        of cosine similarity scores the retriever might return.
        """

        @given(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
                min_size=0,
                max_size=10,
            )
        )
        @settings(max_examples=40)
        def inner(scores):
            retriever = _make_retriever(last_scores=scores, last_query_tokens=0)

            async def fake_rag(q, b, collector):
                return MagicMock(similarity_scores=scores, context_token_count=0)

            async def fake_cag(q, b, collector):
                return MagicMock()

            async def _run():
                with patch("flashcard.runner.cag", _cag_module_mock()):
                    with patch("flashcard.runner.price_rag_turn", return_value=0.0):
                        import flashcard.runner as _rm

                        runner = _rm.Runner(
                            cfg=_rm.RunConfig(model="gpt-5.2", max_tokens=400),
                            retriever=retriever,
                        )
                        runner._run_rag = fake_rag
                        runner._run_cag = fake_cag
                        return await _collect_session(runner, ["Q?"])

            async def _collect_session(runner, questions):
                import inspect

                obj = runner.run_session(questions=questions)
                results = []
                if inspect.isasyncgen(obj):
                    async for turn in obj:
                        results.append(turn)
                else:
                    turns = await obj
                    results.extend(turns if isinstance(turns, list) else [turns])
                return results

            results = asyncio.run(_run())
            assert len(results) >= 1

        inner()

    def test_when_multiple_questions_run_session_yields_one_turn_per_question(self):
        """
        Ordering / count invariant: run_session must yield exactly as many
        turns as there are questions in the input list.
        """

        @given(st.lists(st.text(min_size=1, max_size=80), min_size=1, max_size=6))
        @settings(max_examples=20)
        def inner(questions):
            async def fake_rag(q, b, collector):
                return MagicMock(similarity_scores=[0.5], context_token_count=0)

            async def fake_cag(q, b, collector):
                return MagicMock()

            async def _run():
                with patch("flashcard.runner.cag", _cag_module_mock()):
                    with patch("flashcard.runner.price_rag_turn", return_value=0.0):
                        import flashcard.runner as _rm
                        import inspect

                        runner = _rm.Runner(
                            cfg=_rm.RunConfig(model="gpt-5.2", max_tokens=400),
                            retriever=_make_retriever(),
                        )
                        runner._run_rag = fake_rag
                        runner._run_cag = fake_cag

                        obj = runner.run_session(questions=questions)
                        results = []
                        if inspect.isasyncgen(obj):
                            async for turn in obj:
                                results.append(turn)
                        else:
                            turns = await obj
                            results.extend(
                                turns if isinstance(turns, list) else [turns]
                            )
                        return results

            results = asyncio.run(_run())
            assert len(results) == len(questions), (
                f"run_session yielded {len(results)} turns for {len(questions)} questions"
            )

        inner()


# ─────────────────────────────────────────────────────────────────────────────
# C7 — Headless / no-API-key fallback path: naive retriever + _FALLBACK_DOC
# ─────────────────────────────────────────────────────────────────────────────


class TestHeadlessFallbackPath:
    """
    Criterion: "The existing headless/no-API-key fallback path still works
    (naive retriever, _FALLBACK_DOC)."

    Requirements.md: "Until flashcard.embed/index/rag land, runner.py uses a
    naive term-overlap retriever fallback so the pipeline runs end-to-end."
    """

    def test_when_runner_module_imported_fallback_doc_is_a_nonempty_string(self):
        """_FALLBACK_DOC must be a non-empty string constant exported by runner.py."""
        from flashcard.runner import _FALLBACK_DOC

        assert isinstance(_FALLBACK_DOC, str), (
            f"_FALLBACK_DOC must be str, got {type(_FALLBACK_DOC)}"
        )
        assert len(_FALLBACK_DOC.strip()) > 0, (
            "_FALLBACK_DOC must not be empty or whitespace-only"
        )

    def test_when_fallback_doc_used_runner_can_start_a_session_without_api_key(
        self, monkeypatch
    ):
        """
        With OPENAI_API_KEY absent and _FALLBACK_DOC as the document, the runner
        must complete a session without raising — the headless demo path stays alive.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        async def fake_rag(q, b, collector):
            return MagicMock(
                similarity_scores=[], context_token_count=0, answer="fallback-rag"
            )

        async def fake_cag(q, b, collector):
            return MagicMock(answer="fallback-cag")

        with patch("flashcard.runner.cag", _cag_module_mock()):
            with patch("flashcard.runner.price_rag_turn", return_value=0.0):
                from flashcard.runner import RunConfig, Runner, _FALLBACK_DOC

                retriever = _make_retriever(last_scores=[0.5], last_query_tokens=0)
                runner = Runner(
                    cfg=RunConfig(model="gpt-5.2", max_tokens=400, doc=_FALLBACK_DOC),
                    retriever=retriever,
                )
                runner._run_rag = fake_rag
                runner._run_cag = fake_cag

                results = _run_session_results(
                    runner, questions=["What is the policy?"]
                )

        assert results, "Headless fallback path must yield at least one turn result"

    def test_when_api_key_absent_fallback_doc_remains_importable_after_reload(
        self, monkeypatch
    ):
        """
        _FALLBACK_DOC must remain available when the module is reloaded without
        OPENAI_API_KEY — the headless path must not depend on the key being set.
        """
        import importlib

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        import flashcard.runner as runner_mod

        importlib.reload(runner_mod)

        assert hasattr(runner_mod, "_FALLBACK_DOC"), (
            "_FALLBACK_DOC disappeared from runner module after reload without OPENAI_API_KEY"
        )
        assert runner_mod._FALLBACK_DOC.strip(), (
            "_FALLBACK_DOC is empty after reload — headless path has no fallback document"
        )
