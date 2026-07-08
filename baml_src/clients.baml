// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview
//
// flashcard runs on OpenAI only. OpenAI prompt caching is automatic (no explicit
// cache_control marker needed — caching is automatic and has no off-switch).

// Default generation model for both the RAG and CAG panels.
client<llm> Gpt52 {
  provider openai
  retry_policy Exponential
  options {
    model "gpt-5.2"
    api_key env.OPENAI_API_KEY
    // NOTE: if gpt-5.2 chat-completions rejects `max_tokens`, switch the runtime
    // ClientRegistry option to `max_completion_tokens` (see flashcard/runner.py).
  }
}

// Cheaper model for the optional CompareAnswers judge.
client<llm> Gpt5Mini {
  provider openai
  retry_policy Exponential
  options {
    model "gpt-5-mini"
    api_key env.OPENAI_API_KEY
  }
}

// https://docs.boundaryml.com/docs/snippets/clients/retry
retry_policy Exponential {
  max_retries 2
  strategy {
    type exponential_backoff
    delay_ms 300
    multiplier 1.5
    max_delay_ms 10000
  }
}
