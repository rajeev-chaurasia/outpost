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
