"""Write tool: records a discrepancy against an entity for human
follow-up. Wrap in PolicyGatedTool to enforce a tenant's action policy;
this tool itself only knows how to do the mechanical part.
"""

from dataclasses import dataclass
from typing import Any

from outpost.llm.base import ToolSpec


@dataclass
class FlagDiscrepancyTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="flag_discrepancy",
            description="Flag a discrepancy against an entity for human follow-up.",
            parameters={
                "type": "object",
                "properties": {
                    "entity_key": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["entity_key", "reason"],
            },
        )

    def invoke(self, arguments: dict[str, Any]) -> Any:
        return {
            "executed": True,
            "entity_key": arguments["entity_key"],
            "reason": arguments["reason"],
        }
