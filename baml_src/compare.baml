// Optional divergence / completeness judge for the flashcard RAG-vs-CAG demo.
//
// Runs on Gpt5Mini (gpt-5-mini) — NOT on the generation-model primary — to keep
// judge cost separate and cheap.  Do NOT invoke this function with a ClientRegistry
// that overrides the primary; let BAML honour the static `client Gpt5Mini` here.

class Verdict {
  agree bool @description("True if RAG and CAG answers are substantively equivalent.")
  more_complete "rag" | "cag" | "tie" @description("Which answer covers more of the topic, or 'tie'.")
  note string @description("One-sentence explanation of the verdict.")
}

function CompareAnswers(question: string, rag_answer: string, cag_answer: string) -> Verdict {
  client Gpt5Mini
  prompt #"
    {{ _.role("system") }}
    You are a neutral judge comparing two answers to the same question.
    RAG retrieved only a small slice of the source document; CAG used the whole
    document. Your job is to determine whether they agree and which is more complete.
    Be concise and objective. Do not favour either approach by default.

    {{ _.role("user") }}
    Question: {{ question }}

    RAG answer:
    {{ rag_answer }}

    CAG answer:
    {{ cag_answer }}

    Return a JSON object with:
      agree         — true if both answers convey the same information
      more_complete — "rag" | "cag" | "tie"
      note          — one sentence explaining the verdict
  "#
}
