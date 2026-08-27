"""Runs one request through the agent for a given tenant."""

from fastapi import APIRouter, HTTPException, Request

from outpost.agent.handle import handle_request
from outpost.serve.schemas import CitationResponse, QueryRequest, QueryResponse
from outpost.serve.state import AppState

router = APIRouter()


@router.post("/tenants/{tenant_id}/query", response_model=QueryResponse)
def query_tenant(tenant_id: str, body: QueryRequest, request: Request) -> QueryResponse:
    state: AppState = request.app.state.outpost
    runtime = state.tenants.get(tenant_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail=f"unknown tenant {tenant_id!r}")

    result = handle_request(
        state.provider(runtime.config.budget),
        runtime.tools,
        state.audit_log,
        tenant_id=tenant_id,
        system_prompt=runtime.system_prompt,
        user_request=body.user_request,
    )

    return QueryResponse(
        request_id=result.request_id,
        tenant_id=tenant_id,
        answer=result.answer,
        rung=result.rung.value,
        rung_name=result.rung.name,
        citations=[
            CitationResponse(
                assertion=citation.assertion,
                source_id=citation.span.source_id,
                document_id=citation.span.document_id,
                text=citation.span.text,
            )
            for citation in result.grounding.citations
        ],
        unsupported_assertions=result.grounding.unsupported_assertions,
    )
