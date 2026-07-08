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

from baml_client.async_client import b
from baml_client.runtime import BamlCallOptions
from flashcard.metrics import TurnMetrics, turn_metrics
from flashcard.pricing import Model, price_cag_turn, price_rag_turn

# --------------------------------------------------------------------------- #
# Model → OpenAI id mapping                                                    #
# --------------------------------------------------------------------------- #

_MODEL_IDS: dict[Model, str] = {
    Model.GPT_52: "gpt-5.2",
    Model.GPT_5_MINI: "gpt-5-mini",
}


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
# Streaming a single side                                                     #
# --------------------------------------------------------------------------- #


async def _emit(on_token: Callable, event: TokenEvent) -> None:
    """Call on_token(event); await the result if it is a coroutine."""
    result = on_token(event)
    if inspect.isawaitable(result):
        await result


def _fresh_question_tokens(m: TurnMetrics) -> int:
    """Fresh (non-cached) prompt tokens for a CAG turn.

    OpenAI reports `prompt_tokens` (=> input_tokens) INCLUSIVE of cached tokens,
    so fresh = input_tokens − cached_input_tokens.
    """
    return max((m.input_tokens or 0) - (m.cached_input_tokens or 0), 0)


async def _run_rag(
    *,
    index: int,
    question: str,
    chunks: list[tuple[int, str]],
    registry: Any,
    model: Model,
    on_token: Callable,
) -> SideResult:
    """Stream AnswerRAG over the retrieved chunks; return a priced SideResult."""
    collector = baml_py.Collector()
    opts: BamlCallOptions = {"collector": collector, "client_registry": registry}
    stream = b.stream.AnswerRAG(
        question=question, chunks=[c for _, c in chunks], baml_options=opts
    )
    async for partial in stream:
        await _emit(on_token, TokenEvent(index=index, side=Side.RAG, partial=partial))
    text = await stream.get_final_response()
    m = turn_metrics(collector)
    cost = price_rag_turn(
        model=model,
        input_tokens=m.input_tokens or 0,
        output_tokens=m.output_tokens or 0,
        # query embed tokens are tiny; wire real counts once flashcard.embed lands.
        query_embed_tokens=0,
    )
    return SideResult(
        text=text, metrics=m, cost_usd=cost, retrieved=tuple(i for i, _ in chunks)
    )


async def _run_cag(
    *,
    index: int,
    document: str,
    question: str,
    registry: Any,
    model: Model,
    on_token: Callable,
) -> SideResult:
    """Stream AnswerCAG over the whole document; return a priced SideResult."""
    collector = baml_py.Collector()
    opts: BamlCallOptions = {"collector": collector, "client_registry": registry}
    stream = b.stream.AnswerCAG(document=document, question=question, baml_options=opts)
    async for partial in stream:
        await _emit(on_token, TokenEvent(index=index, side=Side.CAG, partial=partial))
    text = await stream.get_final_response()
    m = turn_metrics(collector)
    cost = price_cag_turn(
        model=model,
        cache_read_tokens=m.cached_input_tokens or 0,
        fresh_question_tokens=_fresh_question_tokens(m),
        output_tokens=m.output_tokens or 0,
    )
    return SideResult(text=text, metrics=m, cost_usd=cost)


# --------------------------------------------------------------------------- #
# Turn + session orchestration                                                #
# --------------------------------------------------------------------------- #


def _noop_token(_: TokenEvent) -> None:
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


async def run_turn(
    index: int,
    question: str,
    document: str,
    chunks: list[tuple[int, str]],
    registry: Any,
    model: Model,
    on_token: Callable,
) -> TurnResult:
    """Fire the RAG and CAG pipelines concurrently and return a paired TurnResult."""
    rag, cag = await asyncio.gather(
        _run_rag(
            index=index,
            question=question,
            chunks=chunks,
            registry=registry,
            model=model,
            on_token=on_token,
        ),
        _run_cag(
            index=index,
            document=document,
            question=question,
            registry=registry,
            model=model,
            on_token=on_token,
        ),
    )
    return TurnResult(index=index, question=question, rag=rag, cag=cag)


async def run_session(
    cfg: RunConfig,
    *,
    on_token: Callable | None = _noop_token,
    retriever: Retriever | None = None,
) -> AsyncIterator[TurnResult]:
    """Async-generate one TurnResult per question, streaming tokens to on_token."""
    sink = on_token if on_token is not None else _noop_token
    document = _load_document(cfg)
    questions = _load_questions(cfg)
    model = _to_model(getattr(cfg, "model", Model.GPT_52))
    registry = _safe_build_registry(cfg)
    retr = retriever or _default_retriever(
        document, int(getattr(cfg, "chunk_size", 800))
    )
    top_k = int(getattr(cfg, "top_k", 3))
    for i, question in enumerate(questions):
        chunks = retr.retrieve(question, top_k)
        result = await run_turn(i, question, document, chunks, registry, model, sink)
        yield result
        if cfg.delay:
            await asyncio.sleep(cfg.delay)


# --------------------------------------------------------------------------- #
# Resilience — re-exported for backward-compatible imports                    #
# --------------------------------------------------------------------------- #
from flashcard.resilience import (  # noqa: E402,F401
    _backoff_delay,
    _is_transient,
    resilient_session,
)
