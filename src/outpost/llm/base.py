"""Provider protocol and the completion types every provider returns."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """One function call a model asked to make."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    """A tool definition to offer a provider, in a vocabulary-agnostic
    shape; each provider translates this to its own wire format.
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Message:
    role: str
    content: str | None
    tool_call_id: str | None = None
    name: str | None = None
    # Only set on an assistant message that requested tool calls, so a
    # provider can reconstruct the exact turn it needs to pair the
    # following tool result messages against.
    tool_calls: list[ToolCall] | None = None


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class Completion:
    """One provider response: either text, or a request to call tools."""

    content: str | None
    tool_calls: list[ToolCall]
    usage: Usage
    model: str


class Provider(Protocol):
    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion: ...
