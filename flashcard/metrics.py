"""Per-turn metrics parsed from a BAML Collector (OpenAI usage).

BAML's typed usage gives input/output tokens, cached input tokens (cache reads),
and call timing. OpenAI reports `prompt_tokens` INCLUSIVE of cached tokens, so the
runner derives fresh tokens as `input_tokens − cached_input_tokens`. OpenAI has no
separate cache-write count, so none is tracked here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TurnMetrics:
    """Immutable snapshot of one turn's token and latency usage."""

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    duration_ms: float


def turn_metrics(collector: Any) -> TurnMetrics:
    """Parse a BAML Collector into a frozen TurnMetrics record."""
    usage = collector.usage
    log = collector.last
    return TurnMetrics(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        duration_ms=_duration_ms(log),
    )


def _duration_ms(log: Any | None) -> float:
    if log is None:
        return 0.0
    return log.timing.duration_ms
