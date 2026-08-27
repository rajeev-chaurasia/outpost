"""Orchestrates one request end to end: runs the tool loop, grounds the
answer against whatever evidence search returned, decides which rung of
the failure ladder it landed on, and appends an audit record with the
full decision path.
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from outpost.agent.audit import AuditLog, AuditRecord
from outpost.agent.degrade import Rung, describe_refusal, determine_rung
from outpost.agent.ground import GroundingResult, ground_answer
from outpost.agent.plan import PlanResult, Step
from outpost.agent.plan import run as run_plan
from outpost.agent.tools.base import Tool
from outpost.llm.base import Provider
from outpost.llm.fallback import FallbackProvider
from outpost.retrieval.document import Span


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    plan: PlanResult
    grounding: GroundingResult
    rung: Rung
    answer: str | None


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

    declined_actions = _collect_declined_actions(plan_result.steps)
    rung = determine_rung(
        final_content=plan_result.final_content,
        citation_count=len(grounding.citations),
        unsupported_count=len(grounding.unsupported_assertions),
        declined_actions=declined_actions,
        provider_fell_back=isinstance(provider, FallbackProvider) and provider.fell_back,
    )

    # A refusal replaces whatever the model said, grounded or not: the
    # whole point of rung 5 is that nothing it said cleared the bar, so
    # the user should see a controlled refusal, not an ungrounded guess.
    answer = (
        describe_refusal(plan_result.steps) if rung is Rung.REFUSED else plan_result.final_content
    )

    request_id = str(uuid.uuid4())
    audit_log.append(
        AuditRecord(
            request_id=request_id,
            tenant_id=tenant_id,
            request_text=user_request,
            steps=plan_result.steps,
            final_content=answer,
            citations=grounding.citations,
            unsupported_assertions=grounding.unsupported_assertions,
            rung=rung.value,
            created_at=datetime.now(UTC).isoformat(),
        )
    )

    return RequestResult(
        request_id=request_id, plan=plan_result, grounding=grounding, rung=rung, answer=answer
    )


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


def _collect_declined_actions(steps: list[Step]) -> list[str]:
    return [
        step.tool_name
        for step in steps
        if isinstance(step.result, dict) and step.result.get("executed") is False
    ]
