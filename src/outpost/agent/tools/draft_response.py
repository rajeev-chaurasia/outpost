"""Write tool: drafts a response for a human to review. Wrap in
PolicyGatedTool to enforce a tenant's action policy; this tool itself
only knows how to do the mechanical part.
"""

from dataclasses import dataclass
from typing import Any

from outpost.llm.base import ToolSpec


@dataclass
class DraftResponseTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="draft_response",
            description="Draft a response for a human to review before it is sent.",
            parameters={
                "type": "object",
                "properties": {"draft": {"type": "string"}},
                "required": ["draft"],
            },
        )

    def invoke(self, arguments: dict[str, Any]) -> Any:
        return {"executed": True, "draft": arguments["draft"]}
