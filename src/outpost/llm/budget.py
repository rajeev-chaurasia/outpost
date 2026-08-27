"""Enforces a tenant's latency and token budget on a provider.

A completion that takes longer than the tenant's declared latency
ceiling is rejected rather than accepted slow: it raises
LatencyBudgetExceededError, a ProviderError, so a FallbackProvider
wrapping this switches to the secondary provider the same way it does
for any other provider failure, without needing to know budgets exist.
"""

import time
from dataclasses import dataclass

from outpost.llm.base import Completion, Message, Provider, ToolSpec
from outpost.llm.errors import LatencyBudgetExceededError, TokenBudgetExceededError


@dataclass
class BudgetedProvider:
    inner: Provider
    model: str
    latency_p99_ms: int
    max_tokens_per_request: int

    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion:
        start = time.monotonic()
        completion = self.inner.complete(messages, tools=tools)
        elapsed_ms = (time.monotonic() - start) * 1000

        if elapsed_ms > self.latency_p99_ms:
            raise LatencyBudgetExceededError(
                model=self.model, elapsed_ms=elapsed_ms, budget_ms=self.latency_p99_ms
            )

        total_tokens = completion.usage.prompt_tokens + completion.usage.completion_tokens
        if total_tokens > self.max_tokens_per_request:
            raise TokenBudgetExceededError(
                model=self.model,
                total_tokens=total_tokens,
                budget_tokens=self.max_tokens_per_request,
            )

        return completion
