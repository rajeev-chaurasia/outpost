"""Tool protocol, and the policy gate every write tool goes through.

A tool is something the agent can call by name with json-shaped
arguments, returning a json-shaped result. Read-only tools are used
directly; write tools are wrapped in PolicyGatedTool so a call outside
the tenant's declared policy is refused before it ever reaches the
underlying tool, not after.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from outpost.llm.base import ToolSpec


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    def invoke(self, arguments: dict[str, Any]) -> Any: ...


@dataclass
class PolicyGatedTool:
    """Wraps a write tool with a tenant's allowed-actions list.

    invoke() checks the policy before ever calling through, so a
    declined action never executes and the caller can tell a refusal
    apart from a real result by the executed field.
    """

    tool: Tool
    allowed_actions: frozenset[str]

    @property
    def spec(self) -> ToolSpec:
        return self.tool.spec

    def invoke(self, arguments: dict[str, Any]) -> Any:
        name = self.spec.name
        if name not in self.allowed_actions:
            return {"executed": False, "reason": f"{name} is not permitted by tenant policy"}
        return self.tool.invoke(arguments)
