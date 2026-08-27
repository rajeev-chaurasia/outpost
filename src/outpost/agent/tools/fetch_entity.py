"""Read-only fetch tool: looks up one structured entity record by its
ontology key, e.g. a specific record from a mapped csv source.
"""

from dataclasses import dataclass
from typing import Any

from outpost.llm.base import ToolSpec


@dataclass
class FetchEntityTool:
    entity_name: str
    key_field: str
    records_by_key: dict[str, dict[str, Any]]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="fetch_entity",
            description=f"Fetch one {self.entity_name} record by its {self.key_field}.",
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        )

    def invoke(self, arguments: dict[str, Any]) -> Any:
        key = arguments["key"]
        record = self.records_by_key.get(key)
        if record is None:
            return {"error": f"no {self.entity_name} found for key {key!r}"}
        return record
