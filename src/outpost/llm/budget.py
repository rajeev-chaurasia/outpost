"""Enforces a tenant's latency and token budget on a provider.

The latency ceiling is enforced two ways, and only the first one bounds
what a user actually waits:

1. As a deadline on the underlying transport. build_budgeted_provider
   builds the client with its socket timeout set to the budget, so a
   slow call is aborted at the budget rather than run to completion.
2. As a post-hoc check here, which catches time spent outside the
   transport's own timeout (local processing, a provider that streams
   headers early and then stalls).

The second alone cannot bound anything: by the time it fires, the user
has already waited the full duration, and the fallback it triggers makes
them wait again. It stays as a backstop, not as the mechanism.

Both raise ProviderError subclasses, so a FallbackProvider wrapping this
switches to the secondary provider the same way it does for any other
provider failure, without needing to know budgets exist.
"""

import time
from dataclasses import dataclass

from outpost.llm.base import Completion, Message, Provider, ToolSpec
from outpost.llm.errors import LatencyBudgetExceededError, TokenBudgetExceededError
from outpost.llm.openai_compatible import OpenAICompatibleProvider


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


def build_budgeted_provider(
    model: str, *, latency_p99_ms: int, max_tokens_per_request: int, base_url: str | None = None
) -> BudgetedProvider:
    """Builds a provider whose transport deadline is the latency budget.

    This is the constructor callers should use, because setting the
    timeout is what actually cuts off a slow call; constructing
    BudgetedProvider directly around a client with a longer timeout
    leaves the tail uncapped.
    """
    client_kwargs = {"model": model, "timeout_seconds": latency_p99_ms / 1000}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    return BudgetedProvider(
        inner=OpenAICompatibleProvider(**client_kwargs),  # type: ignore[arg-type]
        model=model,
        latency_p99_ms=latency_p99_ms,
        max_tokens_per_request=max_tokens_per_request,
    )
