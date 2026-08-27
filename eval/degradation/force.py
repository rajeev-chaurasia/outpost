"""Forces each rung of the failure ladder and reports whether the
correct one fired. A rung that cannot be forced this way is not real.

Every scenario uses a scripted provider rather than a live model: the
point here is to prove degrade.py's own decision logic against known
inputs, not to re-prove that a real model can be made to call a tool
(phase 4's integration tests already cover that).
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from outpost.agent.audit import AuditLog
from outpost.agent.degrade import Rung
from outpost.agent.handle import RequestResult, handle_request
from outpost.agent.tools import ActionGatedTool, FlagDiscrepancyTool, SearchTool
from outpost.llm.base import Completion, Message, ToolCall, ToolSpec, Usage
from outpost.llm.errors import ProviderError
from outpost.llm.fallback import FallbackProvider
from outpost.retrieval.dense import DenseStore, EmbeddingCache
from outpost.retrieval.document import Chunk, Span
from outpost.retrieval.lexical import BM25Index

ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2] / "eval" / "artifacts" / "degradation_results.json"
)

_EVIDENCE_TEXT = "invoice INV-1001 was paid in full on 2026-03-15 via ach"
_QUERY = "INV-1001"


@dataclass
class _ScriptedProvider:
    completions: list[Completion]
    calls: int = 0

    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion:
        completion = self.completions[self.calls]
        self.calls += 1
        return completion


@dataclass
class _FailingProvider:
    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion:
        raise ProviderError(model="primary", detail="simulated outage")


def _usage() -> Usage:
    return Usage(prompt_tokens=1, completion_tokens=1)


def _search_index() -> tuple[BM25Index, DenseStore]:
    lexical_index = BM25Index()
    chunk = Chunk(
        chunk_id="c1",
        tenant_id="dealer_ar",
        span=Span(
            source_id="statements",
            document_id="d1",
            start=0,
            end=len(_EVIDENCE_TEXT),
            text=_EVIDENCE_TEXT,
        ),
    )
    lexical_index.add(chunk)

    cache = EmbeddingCache()
    cache.put(_EVIDENCE_TEXT, "passage", np.array([1.0, 0.0], dtype=np.float32))
    cache.put(_QUERY, "query", np.array([1.0, 0.0], dtype=np.float32))
    dense_store = DenseStore(cache=cache)
    dense_store.index_chunk(chunk)

    return lexical_index, dense_store


def _search_tool_call() -> ToolCall:
    return ToolCall(id="c1", name="search", arguments={"query": _QUERY})


def force_full(audit_log: AuditLog) -> RequestResult:
    lexical_index, dense_store = _search_index()
    search_tool = SearchTool(
        lexical_index=lexical_index, dense_store=dense_store, tenant_id="dealer_ar"
    )
    provider = _ScriptedProvider(
        completions=[
            Completion(content=None, tool_calls=[_search_tool_call()], usage=_usage(), model="m"),
            Completion(
                content="Invoice INV-1001 was paid in full on 2026-03-15 via ACH.",
                tool_calls=[],
                usage=_usage(),
                model="m",
            ),
        ]
    )
    return handle_request(
        provider,
        {"search": search_tool},
        audit_log,
        tenant_id="dealer_ar",
        system_prompt="s",
        user_request="was INV-1001 paid?",
    )


def force_partial(audit_log: AuditLog) -> RequestResult:
    lexical_index, dense_store = _search_index()
    search_tool = SearchTool(
        lexical_index=lexical_index, dense_store=dense_store, tenant_id="dealer_ar"
    )
    provider = _ScriptedProvider(
        completions=[
            Completion(content=None, tool_calls=[_search_tool_call()], usage=_usage(), model="m"),
            Completion(
                content=(
                    "Invoice INV-1001 was paid in full on 2026-03-15 via ACH. "
                    "The warranty on this vehicle expires next year."
                ),
                tool_calls=[],
                usage=_usage(),
                model="m",
            ),
        ]
    )
    return handle_request(
        provider,
        {"search": search_tool},
        audit_log,
        tenant_id="dealer_ar",
        system_prompt="s",
        user_request="was INV-1001 paid, and what about the warranty?",
    )


def force_action_declined(audit_log: AuditLog) -> RequestResult:
    gated_tool = ActionGatedTool(tool=FlagDiscrepancyTool(), allowed_actions=frozenset())
    provider = _ScriptedProvider(
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
    return handle_request(
        provider,
        {"flag_discrepancy": gated_tool},
        audit_log,
        tenant_id="dealer_ar",
        system_prompt="s",
        user_request="flag INV-1005 as short paid",
    )


def force_provider_fallback(audit_log: AuditLog) -> RequestResult:
    secondary = _ScriptedProvider(
        completions=[
            Completion(
                content="Handled by the fallback provider.",
                tool_calls=[],
                usage=_usage(),
                model="m",
            )
        ]
    )
    provider = FallbackProvider(primary=_FailingProvider(), secondary=secondary)
    return handle_request(
        provider,
        {},
        audit_log,
        tenant_id="dealer_ar",
        system_prompt="s",
        user_request="anything",
    )


def force_refused(audit_log: AuditLog) -> RequestResult:
    lexical_index, dense_store = _search_index()
    search_tool = SearchTool(
        lexical_index=lexical_index, dense_store=dense_store, tenant_id="dealer_ar"
    )
    provider = _ScriptedProvider(
        completions=[
            Completion(content=None, tool_calls=[_search_tool_call()], usage=_usage(), model="m"),
            Completion(
                content="I have no information about spaceship warranties.",
                tool_calls=[],
                usage=_usage(),
                model="m",
            ),
        ]
    )
    return handle_request(
        provider,
        {"search": search_tool},
        audit_log,
        tenant_id="dealer_ar",
        system_prompt="s",
        user_request="what is the spaceship warranty policy?",
    )


SCENARIOS: dict[str, tuple[Any, Rung]] = {
    "full": (force_full, Rung.FULL),
    "partial": (force_partial, Rung.PARTIAL),
    "action_declined": (force_action_declined, Rung.ACTION_DECLINED),
    "provider_fallback": (force_provider_fallback, Rung.PROVIDER_FALLBACK),
    "refused": (force_refused, Rung.REFUSED),
}


def run_all(db_path: Path) -> dict[str, dict[str, Any]]:
    audit_log = AuditLog(db_path)
    results: dict[str, dict[str, Any]] = {}
    for name, (force_fn, expected) in SCENARIOS.items():
        result = force_fn(audit_log)
        results[name] = {
            "expected_rung": expected.name,
            "actual_rung": result.rung.name,
            "correct": result.rung == expected,
            "answer": result.answer,
        }
    return results


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_all(Path(tmp_dir) / "audit.sqlite")

    correct_count = sum(1 for r in results.values() if r["correct"])
    summary = {"correct_rung_rate": correct_count / len(results), "scenarios": results}

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
