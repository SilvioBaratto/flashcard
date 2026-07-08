// Answer functions for the flashcard RAG-vs-CAG demo (OpenAI gpt-5.2).
//
// The runner overrides the client at runtime via a ClientRegistry primary. Each
// question is answered INDEPENDENTLY (no growing conversation history) — the only
// difference between the two panels is what context the model sees:
//
//   AnswerRAG — only the top-k retrieved chunks (small keyhole view).
//   AnswerCAG — the ENTIRE document, placed as a stable prefix BEFORE the question
//               so OpenAI's automatic prompt caching keys on it across questions.
//               Q1 pays the full prompt; Q2+ read the cached (system + document)
//               prefix at the discounted rate. No cache marker exists on OpenAI.

// ── RAG: answer from retrieved chunks only ───────────────────────────────────

function AnswerRAG(question: string, chunks: string[]) -> string {
  client Gpt52
  prompt #"
    {{ _.role("system") }}
    You answer questions using ONLY the provided context snippets.
    If the snippets do not contain the answer, say you do not have enough
    information — do not use outside knowledge. Be concise and accurate.

    {{ _.role("user") }}
    Context snippets:
    {% for chunk in chunks %}
    [{{ loop.index }}] {{ chunk }}
    {% endfor %}

    Question: {{ question }}
  "#
}

// ── CAG: answer from the whole document (auto-cached prefix) ──────────────────

function AnswerCAG(document: string, question: string) -> string {
  client Gpt52
  prompt #"
    {{ _.role("system") }}
    You answer questions using ONLY the provided document.
    If the document does not contain the answer, say you do not have enough
    information — do not use outside knowledge. Be concise and accurate.

    {{ _.role("user") }}
    Document:
    {{ document }}

    {{ _.role("user") }}
    Question: {{ question }}
  "#
}
