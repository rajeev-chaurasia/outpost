"""Orchestration done-tests: a search result becomes real citation
evidence in the audit record, and a write action outside the tenant's
allowed actions is refused and that refusal is visible in the audit log.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from outpost.agent.audit import AuditLog
from outpost.agent.handle import handle_request
from outpost.agent.tools import ActionGatedTool, FlagDiscrepancyTool, SearchTool
from outpost.llm.base import Completion, Message, ToolCall, ToolSpec, Usage
from outpost.retrieval.dense import DenseStore, EmbeddingCache
from outpost.retrieval.document import Chunk, Span
from outpost.retrieval.lexical import BM25Index


@dataclass
class ScriptedProvider:
    completions: list[Completion]
    calls: int = 0

    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion:
        completion = self.completions[self.calls]
        self.calls += 1
        return completion


def _usage() -> Usage:
    return Usage(prompt_tokens=1, completion_tokens=1)


def _search_setup() -> tuple[BM25Index, DenseStore]:
    lexical_index = BM25Index()
    text = "invoice INV-1001 was paid in full on 2026-03-15 via ach"
    chunk = Chunk(
        chunk_id="c1",
        tenant_id="dealer_ar",
        span=Span(source_id="statements", document_id="d1", start=0, end=len(text), text=text),
    )
    lexical_index.add(chunk)

    cache = EmbeddingCache()
    cache.put(text, "passage", np.array([1.0, 0.0], dtype=np.float32))
    cache.put("INV-1001", "query", np.array([1.0, 0.0], dtype=np.float32))
    dense_store = DenseStore(cache=cache)
    dense_store.index_chunk(chunk)

    return lexical_index, dense_store


def test_search_result_becomes_a_real_citation_in_the_audit_record(tmp_path: Path) -> None:
    lexical_index, dense_store = _search_setup()
    search_tool = SearchTool(
        lexical_index=lexical_index, dense_store=dense_store, tenant_id="dealer_ar"
    )

    provider = ScriptedProvider(
        completions=[
            Completion(
                content=None,
                tool_calls=[ToolCall(id="c1", name="search", arguments={"query": "INV-1001"})],
                usage=_usage(),
                model="m",
            ),
            Completion(
                content="Invoice INV-1001 was paid in full on 2026-03-15 via ACH.",
                tool_calls=[],
                usage=_usage(),
                model="m",
            ),
        ]
    )
    audit_log = AuditLog(tmp_path / "audit.sqlite")

    result = handle_request(
        provider,
        {"search": search_tool},
        audit_log,
        tenant_id="dealer_ar",
        system_prompt="s",
        user_request="has invoice INV-1001 been paid?",
    )

    assert len(result.grounding.citations) == 1
    citation = result.grounding.citations[0]
    assert citation.span.source_id == "statements"
    assert "INV-1001" in citation.span.text

    stored = audit_log.get(result.request_id)
    assert stored is not None
    assert stored.citations == result.grounding.citations
    assert stored.final_content == "Invoice INV-1001 was paid in full on 2026-03-15 via ACH."


def test_write_action_outside_allowed_actions_is_refused_and_audited(tmp_path: Path) -> None:
    gated_tool = ActionGatedTool(tool=FlagDiscrepancyTool(), allowed_actions=frozenset())

    provider = ScriptedProvider(
        completions=[
            Completion(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="flag_discrepancy",
                        arguments={"entity_key": "INV-1005", "reason": "short paid"},
                    )
                ],
                usage=_usage(),
                model="m",
            ),
            Completion(content="I could not flag this.", tool_calls=[], usage=_usage(), model="m"),
        ]
    )
    audit_log = AuditLog(tmp_path / "audit.sqlite")

    result = handle_request(
        provider,
        {"flag_discrepancy": gated_tool},
        audit_log,
        tenant_id="dealer_ar",
        system_prompt="s",
        user_request="flag invoice INV-1005 as short paid",
    )

    assert result.plan.steps[0].result == {
        "executed": False,
        "reason": "flag_discrepancy is not in the tenant's allowed actions",
        "draft": {"entity_key": "INV-1005", "reason": "short paid"},
    }

    stored = audit_log.get(result.request_id)
    assert stored is not None
    assert stored.steps[0].result["executed"] is False
    assert "not in the tenant's allowed actions" in stored.steps[0].result["reason"]
