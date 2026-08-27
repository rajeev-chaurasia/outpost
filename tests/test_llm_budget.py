"""Budget done-tests: a completion slower than the tenant's latency
ceiling is rejected rather than accepted, one that uses more tokens
than the ceiling is rejected too, and wrapping a budgeted provider in
FallbackProvider turns a budget breach into an actual fallback, the
same as any other provider failure.
"""

import time
from dataclasses import dataclass

import pytest

from outpost.llm.base import Completion, Message, ToolSpec, Usage
from outpost.llm.budget import BudgetedProvider
from outpost.llm.errors import LatencyBudgetExceededError, TokenBudgetExceededError
from outpost.llm.fallback import FallbackProvider


@dataclass
class _SlowProvider:
    delay_seconds: float
    completion: Completion

    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion:
        time.sleep(self.delay_seconds)
        return self.completion


@dataclass
class _FixedProvider:
    completion: Completion

    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion:
        return self.completion


def _completion(prompt_tokens: int = 10, completion_tokens: int = 10) -> Completion:
    return Completion(
        content="ok",
        tool_calls=[],
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        model="m",
    )


def test_completion_within_budget_passes_through() -> None:
    provider = BudgetedProvider(
        inner=_FixedProvider(_completion()),
        model="m",
        latency_p99_ms=8000,
        max_tokens_per_request=4000,
    )
    completion = provider.complete([Message(role="user", content="hi")])
    assert completion.content == "ok"


def test_completion_over_latency_budget_raises() -> None:
    provider = BudgetedProvider(
        inner=_SlowProvider(delay_seconds=0.05, completion=_completion()),
        model="m",
        latency_p99_ms=10,
        max_tokens_per_request=4000,
    )
    with pytest.raises(LatencyBudgetExceededError) as exc_info:
        provider.complete([Message(role="user", content="hi")])
    assert exc_info.value.budget_ms == 10


def test_completion_over_token_budget_raises() -> None:
    provider = BudgetedProvider(
        inner=_FixedProvider(_completion(prompt_tokens=3000, completion_tokens=3000)),
        model="m",
        latency_p99_ms=8000,
        max_tokens_per_request=4000,
    )
    with pytest.raises(TokenBudgetExceededError) as exc_info:
        provider.complete([Message(role="user", content="hi")])
    assert exc_info.value.total_tokens == 6000


def test_latency_budget_breach_triggers_fallback() -> None:
    primary = BudgetedProvider(
        inner=_SlowProvider(delay_seconds=0.05, completion=_completion()),
        model="primary",
        latency_p99_ms=10,
        max_tokens_per_request=4000,
    )
    secondary = _FixedProvider(_completion())
    fallback = FallbackProvider(primary=primary, secondary=secondary)

    completion = fallback.complete([Message(role="user", content="hi")])

    assert fallback.fell_back is True
    assert completion.content == "ok"
