"""LLM provider protocol and implementations."""

from outpost.llm.base import Completion, Message, Provider, ToolCall, ToolSpec, Usage
from outpost.llm.budget import BudgetedProvider
from outpost.llm.errors import (
    LatencyBudgetExceededError,
    LLMError,
    ProviderError,
    TokenBudgetExceededError,
)
from outpost.llm.fallback import FallbackProvider
from outpost.llm.openai_compatible import DEFAULT_BASE_URL, OpenAICompatibleProvider
from outpost.llm.recorded import RecordedProvider, request_key

__all__ = [
    "DEFAULT_BASE_URL",
    "BudgetedProvider",
    "Completion",
    "FallbackProvider",
    "LLMError",
    "LatencyBudgetExceededError",
    "Message",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderError",
    "RecordedProvider",
    "TokenBudgetExceededError",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "request_key",
]
