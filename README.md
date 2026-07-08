# flashcard — RAG vs CAG, side by side

> Same questions, same model. One looks things up. The other already knows. Watch them disagree.

## The closed book and the crammed mind

Imagine a hard exam and two students.

The **first student** keeps the textbook closed. For every question they send a runner to the library, who searches the shelves, grabs the three pages that *seem* most relevant, and sprints back. The student answers using only those three pages. Fast, cheap per question — but they never see the whole book. If the runner grabs the wrong pages, the answer is confidently wrong, and the student never even knows the right page existed.

The **second student** crammed. Before the exam they read the entire textbook into memory — once, slowly, expensively. Now every question is answered from the whole book at once. Nothing is missed, every connection is visible. The catch: they can only hold what fits in their head, and that first read cost real effort.

A large language model can play either student.

- The first student is **RAG** — *Retrieval-Augmented Generation*. Your corpus lives in an external store. At query time you embed the question, retrieve the top-k chunks, and paste only those into the prompt. The model sees a keyhole view, chosen by the retriever.
- The second student is **CAG** — *Cache-Augmented Generation*. You load the *entire* document into the context window as a stable prefix; OpenAI's automatic prompt caching holds it, so every later question reuses it cheaply. Every question runs against the full text. No retriever, no keyhole — but bounded by the context window.

**flashcard** runs both students against the same questions and shows you, live, where they agree, where they diverge, and what each one costs.

## What flashcard does

**flashcard** takes one knowledge source and one set of questions, then answers every question **twice, concurrently** — once through a RAG pipeline, once through a CAG pipeline — and renders both in a Rich live terminal UI.

```
┌──────────────────────────────┬──────────────────────────────┐
│  RAG  (retrieve top-k)       │  CAG  (whole doc, cached)    │
│  Running cost: $0.0091       │  Running cost: $0.0136       │
│  Q3  Retrieved: chunks 4,7,9 │  Q3  Cache read: 42 118 tok  │
│  ctx tokens: 1 240           │  ctx tokens: 42 118          │
│  answer: "Clause 12 says..." │  answer: "Clause 12 says..." │
│  ⚠ missed clause 12(b)       │  ✓ full recall               │
└──────────────────────────────┴──────────────────────────────┘
      RAG cheaper per call · CAG never misses · you pick
```

- **Left panel (RAG):** the question is embedded, the top-k most similar chunks are pulled from the vector index, and only those chunks enter the prompt. Small context, low per-question cost — but the answer is only as good as what the retriever fetched.
- **Right panel (CAG):** the full document sits in context as a stable prefix. The first question pays the full prompt; OpenAI caches the prefix automatically, so every question after reads it cheaply. Large context, perfect recall inside the window.

**Teaching beat:** on a question whose answer is spread across distant parts of the document, RAG's keyhole misses a piece and CAG answers in full — the two panels visibly disagree. On a narrow lookup over a giant corpus, RAG stays cheap while CAG pays to hold the whole book. Neither wins everywhere; flashcard shows you *where* the line is.

## RAG vs CAG at a glance

| | **RAG** | **CAG** |
|---|---|---|
| Knowledge lives in | External vector store | The context window (KV cache) |
| Per-question context | Top-k retrieved chunks (small) | The whole document (large) |
| Recall | Only what the retriever fetched | Everything in the window |
| Failure mode | Retriever misses the right chunk | Document exceeds the context window |
| Corpus size | Effectively unbounded | Bounded by the context window |
| First-call cost | Low | High (full prompt, uncached) |
| Repeat-call cost | Low | Very low (cache read) |
| Infra | Embedder + vector index | Prompt caching only |

## Install

```bash
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and set your keys:

```bash
cp .env.example .env
# Edit .env and set:
# OPENAI_API_KEY=sk-...        # generation + RAG embeddings + optional judge
```

Keys are read exclusively from the environment / `.env` file. They are never committed to source.

## Usage

```bash
flashcard [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `gpt-5.2` | Generation model (OpenAI) |
| `--doc` | *(built-in sample corpus)* | Path to the knowledge source |
| `--questions-file` | *(built-in questions)* | Path to a custom questions JSON file |
| `--top-k` | `3` | RAG: chunks retrieved per question |
| `--chunk-size` | `800` | RAG: chunk length in tokens |
| `--max-tokens` | `400` | Max output tokens per answer |
| `--delay` | `0.0` | Pacing delay between questions (seconds, useful for recording) |

Before firing any real API calls, flashcard prints a **worst-case cost estimate** and asks for confirmation.

## Sample run

```bash
# Default: built-in corpus + questions, top-3 retrieval
flashcard

# Bigger keyhole for RAG
flashcard --top-k 6 --max-tokens 400

# Your own document and questions
flashcard --doc ./handbook.txt --questions-file ./questions.json
```

The terminal shows both pipelines updating in real time. The footer tracks cumulative cost per side, retrieval hit/miss count for RAG, and cache-read totals for CAG.

## When to use which

- **Reach for RAG** when the corpus is far larger than any context window, most questions touch a small slice, and freshness matters (re-index without re-prompting).
- **Reach for CAG** when the whole source fits in context, questions span the document, retrieval misses are costly, and the same source is reused across many questions (caching amortizes the load).
- **Often the answer is both** — retrieve to narrow a huge corpus to a chapter, then cache that chapter for a burst of questions. flashcard makes the tradeoff legible so you can decide.

## Security

- API keys live only in environment variables or `.env` (gitignored).
- `.env.example` ships without any real key.
- No secrets ever appear in source.
