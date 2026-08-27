"""Falls back to a secondary provider when the primary one errors.

Phase 6 extends the trigger to exceeding the latency budget too; for
now, only a provider error causes the switch. Construct a fresh instance
per request: fell_back is per-instance state, and reusing one across
requests would leak a fallback from an earlier request into a later
one's rung.
"""

from dataclasses import dataclass, field

from outpost.llm.base import Completion, Message, Provider, ToolSpec
from outpost.llm.errors import ProviderError


@dataclass
class FallbackProvider:
    primary: Provider
    secondary: Provider
    fell_back: bool = field(default=False, init=False)

    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion:
        try:
            return self.primary.complete(messages, tools=tools)
        except ProviderError:
            self.fell_back = True
            return self.secondary.complete(messages, tools=tools)
