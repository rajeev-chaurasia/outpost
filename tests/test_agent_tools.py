"""Tool done-tests: read-only tools return usable results, and a write
tool wrapped in ActionGatedTool refuses without executing when the
action is outside the tenant's allowed actions, but runs normally when
it is inside them.
"""

import numpy as np

from outpost.agent.tools import (
    ActionGatedTool,
    FetchEntityTool,
    FlagDiscrepancyTool,
    SearchTool,
)
from outpost.retrieval.dense import DenseStore, EmbeddingCache
from outpost.retrieval.document import Chunk, Span
from outpost.retrieval.lexical import BM25Index


def _index_with_one_chunk() -> tuple[BM25Index, DenseStore]:
    lexical_index = BM25Index()
    chunk_text = "account balance short"
    chunk = Chunk(
        chunk_id="c1",
        tenant_id="t1",
        span=Span(source_id="s", document_id="d", start=0, end=len(chunk_text), text=chunk_text),
    )
    lexical_index.add(chunk)

    cache = EmbeddingCache()
    cache.put(chunk_text, "passage", np.array([1.0, 0.0], dtype=np.float32))
    cache.put("account balance", "query", np.array([1.0, 0.0], dtype=np.float32))
    dense_store = DenseStore(cache=cache)
    dense_store.index_chunk(chunk)

    return lexical_index, dense_store


def test_search_tool_returns_span_shaped_results() -> None:
    lexical_index, dense_store = _index_with_one_chunk()
    tool = SearchTool(lexical_index=lexical_index, dense_store=dense_store, tenant_id="t1")

    results = tool.invoke({"query": "account balance"})

    assert results
    assert results[0]["text"] == "account balance short"
    assert results[0]["source_id"] == "s"


def test_fetch_entity_tool_returns_matching_record() -> None:
    tool = FetchEntityTool(
        entity_name="widget",
        key_field="widget_id",
        records_by_key={"W-1": {"widget_id": "W-1", "status": "open"}},
    )
    assert tool.invoke({"key": "W-1"}) == {"widget_id": "W-1", "status": "open"}


def test_fetch_entity_tool_reports_a_missing_key_without_raising() -> None:
    tool = FetchEntityTool(entity_name="widget", key_field="widget_id", records_by_key={})
    result = tool.invoke({"key": "missing"})
    assert "error" in result


def test_action_gated_tool_declines_action_outside_allowed_actions() -> None:
    gated = ActionGatedTool(tool=FlagDiscrepancyTool(), allowed_actions=frozenset())

    result = gated.invoke({"entity_key": "W-1", "reason": "short paid"})

    assert result == {
        "executed": False,
        "reason": "flag_discrepancy is not in the tenant's allowed actions",
    }


def test_action_gated_tool_executes_action_inside_allowed_actions() -> None:
    gated = ActionGatedTool(
        tool=FlagDiscrepancyTool(), allowed_actions=frozenset({"flag_discrepancy"})
    )

    result = gated.invoke({"entity_key": "W-1", "reason": "short paid"})

    assert result == {"executed": True, "entity_key": "W-1", "reason": "short paid"}
