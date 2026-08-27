"""Orchestrates one request end to end: runs the tool loop, grounds the
answer against whatever evidence search returned, and appends an audit
record with the full decision path.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from outpost.agent.audit import AuditLog, AuditRecord
from outpost.agent.ground import GroundingResult, ground_answer
from outpost.agent.plan import PlanResult, Step
from outpost.agent.plan import run as run_plan
from outpost.agent.tools.base import Tool
from outpost.llm.base import Provider
from outpost.retrieval.document import Span


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    plan: PlanResult
    grounding: GroundingResult


def handle_request(
    provider: Provider,
    tools: dict[str, Tool],
    audit_log: AuditLog,
    *,
    tenant_id: str,
    system_prompt: str,
    user_request: str,
) -> RequestResult:
    plan_result = run_plan(provider, tools, system_prompt=system_prompt, user_request=user_request)

    evidence_spans = _collect_evidence_spans(plan_result.steps)
    grounding = (
        ground_answer(plan_result.final_content, evidence_spans)
        if plan_result.final_content
        else GroundingResult(citations=[], unsupported_assertions=[])
    )

    request_id = str(uuid.uuid4())
    audit_log.append(
        AuditRecord(
            request_id=request_id,
            tenant_id=tenant_id,
            request_text=user_request,
            steps=plan_result.steps,
            final_content=plan_result.final_content,
            citations=grounding.citations,
            unsupported_assertions=grounding.unsupported_assertions,
            rung=None,
            created_at=datetime.now(UTC).isoformat(),
        )
    )

    return RequestResult(request_id=request_id, plan=plan_result, grounding=grounding)


_SPAN_FIELDS = {"source_id", "document_id", "start", "end", "text"}


def _collect_evidence_spans(steps: list[Step]) -> list[Span]:
    spans: list[Span] = []
    for step in steps:
        if step.tool_name != "search" or not isinstance(step.result, list):
            continue
        for item in step.result:
            if isinstance(item, dict) and item.keys() >= _SPAN_FIELDS:
                spans.append(Span(**item))
    return spans
