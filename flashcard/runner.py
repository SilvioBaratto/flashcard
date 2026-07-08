"""Priced per-turn results, ClientRegistry builder, and the dual-pipeline runner.

Adapted from the sibling `dejavu` runner. Two structural changes vs dejavu:
  * sides are RAG / CAG (retrieve-top-k vs whole-document) instead of uncached/cached
  * NO growing history — each question is answered independently

Generation model is OpenAI **gpt-5.2** (provider `openai`). OpenAI has no explicit
`cache_control` marker, so CAG relies on OpenAI's automatic prompt caching of the
stable prefix (system + document). `Collector.cached_input_tokens` reports cache
reads; there is no cache-write price, so the CAG cost is cached-read + fresh
question + output.

Retrieval for the RAG side is injected (see `Retriever`). Until `flashcard.rag` /
`flashcard.embed` / `flashcard.index` land, a naive term-overlap retriever is used
so the pipeline runs end-to-end today. Same for the document / question loaders:
they lazy-import `flashcard.corpus` / `flashcard.questions` and fall back to a tiny
built-in sample when those modules do not exist yet.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Callable, Protocol, runtime_checkable

import baml_py

from baml_client.async_client import b as _b
from flashcard import cag
from flashcard.metrics import TurnMetrics, fresh_tokens, turn_metrics
from flashcard.pricing import Model, price_cag_turn, price_rag_turn

# --------------------------------------------------------------------------- #
# Model → OpenAI id mapping                                                    #
# --------------------------------------------------------------------------- #

_MODEL_IDS: dict[Model, str] = {
    Model.GPT_52: "gpt-5.2",
    Model.GPT_5_MINI: "gpt-5-mini",
}

# Sentinel for "attribute not present on retriever object"
_ATTR_ABSENT = object()


def _to_model(model: Any) -> Model:
    """Coerce any value to a Model enum; fall back to GPT_52 for unknowns."""
    if isinstance(model, Model):
        return model
    try:
        return Model(str(model))
    except ValueError:
        return Model.GPT_52


def _client_options(model: Model, max_tokens: int) -> dict:
    """Options dict for BAML LLM client registration.

    NOTE: gpt-5.x chat-completions may require ``max_completion_tokens`` instead of
    ``max_tokens``; if the API rejects it, rename the key here. No temperature is
    sent — gpt-5.2 may only allow the default.
    """
    return {
        "model": _MODEL_IDS[model],
        "api_key": os.environ["OPENAI_API_KEY"],
        "max_tokens": max_tokens,
    }


def build_registry(model: Model, *, max_tokens: int) -> baml_py.ClientRegistry:
    """Build a BAML ClientRegistry with one primary OpenAI client for *model*."""
    cr = baml_py.ClientRegistry()
    cr.add_llm_client(
        name="FlashcardPrimary",
        provider="openai",
        options=_client_options(model, max_tokens),
    )
    cr.set_primary("FlashcardPrimary")
    return cr


# --------------------------------------------------------------------------- #
# Value objects                                                               #
# --------------------------------------------------------------------------- #


class Side(str, Enum):
    """Which pipeline produced this result."""

    RAG = "rag"
    CAG = "cag"


@dataclass(frozen=True)
class SideResult:
    """Priced result for one pipeline on one question."""

    text: str
    metrics: TurnMetrics
    cost_usd: float
    retrieved: tuple[int, ...] = ()  # RAG only: retrieved chunk ids (empty for CAG)
    similarity_scores: tuple[float, ...] = ()  # RAG only: cosine similarity scores
    context_token_count: int = 0  # RAG only: tiktoken count of retrieved chunks

    @property
    def cost(self) -> float:
        return self.cost_usd

    @property
    def input_tokens(self) -> int | None:
        return self.metrics.input_tokens

    @property
    def output_tokens(self) -> int | None:
        return self.metrics.output_tokens

    @property
    def cached_input_tokens(self) -> int | None:
        return self.metrics.cached_input_tokens

    @property
    def duration_ms(self) -> float:
        return self.metrics.duration_ms


@dataclass(frozen=True)
class TurnResult:
    """Both pipelines' results for one question."""

    index: int
    question: str
    rag: SideResult
    cag: SideResult
    verdict: Any | None = None  # populated when RunConfig.judge=True

    @property
    def turn_number(self) -> int:
        """1-based turn number (index + 1)."""
        return self.index + 1


@dataclass(frozen=True)
class TokenEvent:
    """A streamed token partial from one pipeline on one question."""

    index: int
    side: Side
    partial: str


@dataclass(frozen=True)
class RunConfig:
    """Runtime configuration for the dual-pipeline runner."""

    model: Model = Model.GPT_52
    doc: str | None = None
    questions_file: str | None = None
    max_tokens: int = 400
    top_k: int = 3
    chunk_size: int = 800
    delay: float = 0.0
    judge: bool = False  # gate for the optional CompareAnswers divergence judge


# --------------------------------------------------------------------------- #
# Retrieval (injected; naive fallback until flashcard.rag exists)             #
# --------------------------------------------------------------------------- #


@runtime_checkable
class Retriever(Protocol):
    """Return up to *k* (chunk_id, chunk_text) pairs most relevant to *question*."""

    def retrieve(self, question: str, k: int) -> list[tuple[int, str]]: ...


_WORD_RE = re.compile(r"[a-zà-ù0-9]+", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


def _chunk_document(document: str, chunk_size: int) -> list[str]:
    """Crude char-window chunker (~chunk_size tokens ≈ chunk_size*4 chars).

    Placeholder until `flashcard.chunking` provides a real token-aware chunker.
    """
    width = max(1, chunk_size) * 4
    paras = [p.strip() for p in re.split(r"\n\s*\n", document) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if buf and len(buf) + len(para) + 2 > width:
            chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return chunks or [document]


class _NaiveRetriever:
    """Term-overlap retriever over char-window chunks — no embeddings.

    A stand-in so the RAG pipeline runs before `flashcard.embed`/`index` exist.
    Deliberately weak: it *will* miss answers spread across distant chunks, which
    is exactly the failure mode the demo showcases.
    """

    def __init__(self, document: str, chunk_size: int) -> None:
        self._chunks = _chunk_document(document, chunk_size)
        self._chunk_tokens = [_tokens(c) for c in self._chunks]

    def retrieve(self, question: str, k: int) -> list[tuple[int, str]]:
        q = _tokens(question)
        scored = [(len(q & toks), i) for i, toks in enumerate(self._chunk_tokens)]
        scored.sort(key=lambda s: (-s[0], s[1]))
        top = [i for _, i in scored[: max(1, k)]]
        return [(i, self._chunks[i]) for i in top]


# --------------------------------------------------------------------------- #
# Session helpers                                                              #
# --------------------------------------------------------------------------- #


def _noop_token(_: Any) -> None:
    pass


def _safe_build_registry(cfg: Any) -> baml_py.ClientRegistry | None:
    """Return a live ClientRegistry, or None in headless/test mode (no API key)."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    model = _to_model(getattr(cfg, "model", Model.GPT_52))
    max_t = int(getattr(cfg, "max_tokens", 400))
    return build_registry(model, max_tokens=max_t)


def _opt_path(cfg: Any, attr: str) -> str | None:
    v = getattr(cfg, attr, None)
    return v if isinstance(v, str) else None


_FALLBACK_DOC = (
    "flashcard sample document.\n\n"
    "Section 1 — Overview. This placeholder text stands in until "
    "flashcard/corpus.py provides the real cross-referenced knowledge source.\n\n"
    "Section 2 — Details. Replace via --doc PATH or by adding corpus.py."
)
_FALLBACK_QUESTIONS = [
    "What does Section 1 describe?",
    "What is stated in Section 2?",
]


def _load_document(cfg: Any) -> str:
    """Load the knowledge source; lazy-import corpus.py, else a built-in sample."""
    try:
        from flashcard.corpus import load_corpus  # noqa: WPS433 (lazy on purpose)
    except ImportError:
        return _FALLBACK_DOC
    return load_corpus(_opt_path(cfg, "doc"))


def _load_questions(cfg: Any) -> list[str]:
    """Load the question set; lazy-import questions.py, else a built-in sample."""
    try:
        from flashcard.questions import load_questions  # noqa: WPS433
    except ImportError:
        return list(_FALLBACK_QUESTIONS)
    return load_questions(_opt_path(cfg, "questions_file"))


def _default_retriever(document: str, chunk_size: int) -> Retriever:
    """Build a retriever; prefer flashcard.rag, else the naive term-overlap one."""
    try:
        from flashcard.rag import build_retriever  # noqa: WPS433
    except ImportError:
        return _NaiveRetriever(document, chunk_size)
    return build_retriever(document, chunk_size)


# --------------------------------------------------------------------------- #
# Runner internals (stream helpers, retriever attribute helpers, judge)        #
# --------------------------------------------------------------------------- #


async def _await_retrieve(retriever: Any, question: str, k: int) -> list:
    raw = retriever.retrieve(question, k)
    return await raw if inspect.iscoroutine(raw) else raw


async def _get_rag_stream(b_client: Any, **kwargs: Any) -> Any:
    raw = b_client.stream.AnswerRAG(**kwargs)
    return await raw if inspect.iscoroutine(raw) else raw


async def _drain_stream(stream: Any) -> str:
    async for _ in stream:
        pass
    result = stream.get_final_response()
    if inspect.iscoroutine(result):
        result = await result
    return result if isinstance(result, str) else ""


def _safe_scores(raw: Any) -> tuple:
    """Normalise retriever scores: accepts bare floats or (id, score) tuples."""
    return tuple(s[1] if isinstance(s, tuple) else float(s) for s in raw)


def _retriever_scores(retriever: Any) -> list:
    """Read similarity scores; checks last_scores then last_similarity_scores."""
    v = getattr(retriever, "last_scores", _ATTR_ABSENT)
    if v is not _ATTR_ABSENT:
        return v if isinstance(v, list) else []
    return getattr(retriever, "last_similarity_scores", []) or []


def _retriever_ctx(retriever: Any) -> int:
    """Read context token count; checks last_context_tokens then token_count."""
    v = getattr(retriever, "last_context_tokens", _ATTR_ABSENT)
    if v is not _ATTR_ABSENT:
        return int(v) if isinstance(v, int) else 0
    v = getattr(retriever, "last_context_token_count", 0)
    return int(v) if isinstance(v, int) else 0


async def _run_judge(b: Any, question: str, rag_answer: str, cag_answer: str) -> Any:
    """Call CompareAnswers on Gpt5Mini with its own Collector.

    IMPORTANT: no client_registry is passed so BAML honours the static
    ``client Gpt5Mini`` in compare.baml — prevents the generation-model
    primary (FlashcardPrimary / gpt-5.2) from silently overriding the judge.
    """
    col = baml_py.Collector()
    return await b.CompareAnswers(
        question=question,
        rag_answer=rag_answer,
        cag_answer=cag_answer,
        baml_options={"collector": col},
    )


# --------------------------------------------------------------------------- #
# Public module-level API (thin delegates to Runner)                          #
# --------------------------------------------------------------------------- #


async def run_turn(
    question: str,
    config: RunConfig,
    *,
    baml_client: Any,
    retriever: Any,
    document: str,
    index: int = 0,
) -> TurnResult:
    """High-level entry point: delegates to Runner for real metrics + judge support."""
    runner = Runner(config=config, baml_client=baml_client, retriever=retriever)
    return await runner.run_turn(question=question, document=document, index=index)


async def run_session(
    cfg: RunConfig,
    *,
    on_token: Callable | None = _noop_token,
    retriever: Retriever | None = None,
) -> AsyncIterator[TurnResult]:
    """Async-generate one TurnResult per question; drives the unified Runner path.

    ``on_token`` is accepted for backward-compatibility with ``resilience.py``'s
    ``resilient_session`` but is currently a no-op (streaming will be wired when
    the TUI / CLI is connected).
    """
    document = _load_document(cfg)
    questions = _load_questions(cfg)
    chunk_sz = int(getattr(cfg, "chunk_size", 800))
    retr = retriever or _default_retriever(document, chunk_sz)
    runner = Runner(config=cfg, retriever=retr)
    for i, question in enumerate(questions):
        yield await runner.run_turn(question=question, document=document, index=i)
        if cfg.delay:
            await asyncio.sleep(cfg.delay)


# --------------------------------------------------------------------------- #
# Runner class — single OO dual-pipeline implementation                        #
# --------------------------------------------------------------------------- #


def _runner_document(cfg: Any) -> str:
    """Resolve cfg.doc: inline content / file path / None → corpus."""
    import pathlib as _pl  # noqa: PLC0415

    doc = getattr(cfg, "doc", None)
    if isinstance(doc, str) and doc.strip():
        p = _pl.Path(doc)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return doc  # treat as inline content
    return _load_document(cfg)


class Runner:
    """Single OO dual-pipeline runner: real metrics, judge support, cag.py delegation.

    Satisfies #12's oracle tests and is the single implementation kept by #14.
    Accepts ``config`` (preferred) or ``cfg`` (backward-compat alias) as the
    RunConfig argument.  ``baml_client`` is injected at construction for test
    isolation; ``b`` may still be passed at ``run_turn`` time for legacy callers.
    """

    def __init__(
        self,
        config: RunConfig | None = None,
        retriever: Retriever | None = None,
        baml_client: Any = None,
        *,
        cfg: RunConfig | None = None,
    ) -> None:
        resolved = config or cfg
        if resolved is None:
            raise TypeError("Runner requires a RunConfig (pass as 'config' or 'cfg')")
        self._cfg = resolved
        self._model = _to_model(getattr(resolved, "model", Model.GPT_52))
        self._document = _runner_document(resolved)
        self._baml_client = baml_client
        self._retriever = retriever or _default_retriever(
            self._document, int(getattr(resolved, "chunk_size", 800))
        )

    async def _run_rag(self, question: str, b: Any, collector: Any) -> SideResult:
        chunks = await _await_retrieve(
            self._retriever, question, int(getattr(self._cfg, "top_k", 3))
        )
        # Snapshot retriever state immediately after retrieve() — mutable per-call.
        q_toks = getattr(self._retriever, "last_query_tokens", 0)
        raw_scores = _retriever_scores(self._retriever)
        ctx = _retriever_ctx(self._retriever)
        stream = await _get_rag_stream(
            b,
            question=question,
            chunks=[c for _, c in chunks],
            baml_options={"collector": collector},
        )
        text = await _drain_stream(stream)
        m = turn_metrics(collector)
        cost = price_rag_turn(
            model=self._model,
            input_tokens=m.input_tokens or 0,
            output_tokens=m.output_tokens or 0,
            query_embed_tokens=q_toks if isinstance(q_toks, int) else 0,
        )
        return SideResult(
            text=text,
            metrics=m,
            cost_usd=cost,
            retrieved=tuple(i for i, _ in chunks),
            similarity_scores=_safe_scores(raw_scores),
            context_token_count=ctx,
        )

    async def _run_cag(self, question: str, b: Any, collector: Any) -> SideResult:
        # cag.answer emits raw str partials; runner wraps events (no circular import).
        raw = await cag.answer(
            document=self._document,
            question=question,
            b=b,
            collector_factory=lambda: collector,
        )
        try:
            text, m = raw
        except (TypeError, ValueError):
            text = ""
            m = TurnMetrics(
                input_tokens=0, output_tokens=0, cached_input_tokens=0, duration_ms=0.0
            )
        cost = price_cag_turn(
            model=self._model,
            cache_read_tokens=m.cached_input_tokens or 0,
            fresh_question_tokens=fresh_tokens(m),
            output_tokens=m.output_tokens or 0,
        )
        return SideResult(text=text, metrics=m, cost_usd=cost)

    async def run_turn(
        self,
        *,
        question: str,
        b: Any = None,
        document: str | None = None,
        index: int = 0,
    ) -> TurnResult:
        """Fire both pipelines concurrently with independent Collectors."""
        client = b or self._baml_client or _b
        if document is not None:
            self._document = document
        col_r, col_c = baml_py.Collector(), baml_py.Collector()
        rag_res, cag_res = await asyncio.gather(
            self._run_rag(question, client, col_r),
            self._run_cag(question, client, col_c),
        )
        verdict = (
            await _run_judge(client, question, rag_res.text, cag_res.text)
            if getattr(self._cfg, "judge", False)
            else None
        )
        return TurnResult(
            index=index, question=question, rag=rag_res, cag=cag_res, verdict=verdict
        )

    async def run_session(
        self, *, questions: list[str], b: Any = None
    ) -> AsyncIterator[TurnResult]:
        """Async-generate one TurnResult per question."""
        client = b or self._baml_client or _b
        for i, q in enumerate(questions):
            yield await self.run_turn(question=q, b=client, index=i)
