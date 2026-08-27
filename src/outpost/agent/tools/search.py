"""Read-only search tool: runs a tenant-scoped hybrid search and returns
each result as its exact source span, so grounding can bind an answer's
claims back to real text without a second lookup.
"""

from dataclasses import dataclass
from typing import Any

from outpost.llm.base import ToolSpec
from outpost.retrieval.dense import DenseStore
from outpost.retrieval.isolation import search
from outpost.retrieval.lexical import BM25Index


@dataclass
class SearchTool:
    lexical_index: BM25Index
    dense_store: DenseStore
    tenant_id: str
    top_k: int = 5

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search",
            description="Search this tenant's indexed documents for text relevant to a query.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

    def invoke(self, arguments: dict[str, Any]) -> Any:
        chunk_ids = search(
            self.lexical_index,
            self.dense_store,
            tenant_id=self.tenant_id,
            query=arguments["query"],
            top_k=self.top_k,
        )
        return [self.lexical_index.chunks[chunk_id].span.model_dump() for chunk_id in chunk_ids]
