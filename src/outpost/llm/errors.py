"""Typed errors for the llm layer."""


class LLMError(Exception):
    """Base class for all llm provider errors."""


class ProviderError(LLMError):
    """A provider request failed, or returned something outpost cannot
    parse (an unexpected response shape, malformed tool call arguments).
    """

    def __init__(self, *, model: str, detail: str) -> None:
        self.model = model
        self.detail = detail
        super().__init__(f"provider error from {model!r}: {detail}")


class LatencyBudgetExceededError(ProviderError):
    """A completion came back, but too slowly to accept.

    Subclasses ProviderError so a FallbackProvider wrapping a budgeted
    provider switches to the secondary the same way it does for any
    other provider failure, without needing to know budgets exist.
    """

    def __init__(self, *, model: str, elapsed_ms: float, budget_ms: int) -> None:
        self.elapsed_ms = elapsed_ms
        self.budget_ms = budget_ms
        super().__init__(
            model=model,
            detail=f"took {elapsed_ms:.0f}ms, over the {budget_ms}ms latency budget",
        )


class TokenBudgetExceededError(ProviderError):
    """A completion used more tokens than the tenant's request ceiling."""

    def __init__(self, *, model: str, total_tokens: int, budget_tokens: int) -> None:
        self.total_tokens = total_tokens
        self.budget_tokens = budget_tokens
        super().__init__(
            model=model,
            detail=f"used {total_tokens} tokens, over the {budget_tokens} token budget",
        )
