"""Lists a tenant's recent audit trail."""

from fastapi import APIRouter, HTTPException, Request

from outpost.agent.degrade import Rung
from outpost.serve.schemas import AuditEntryResponse
from outpost.serve.state import AppState

router = APIRouter()


@router.get("/tenants/{tenant_id}/audit", response_model=list[AuditEntryResponse])
def list_audit(tenant_id: str, request: Request, limit: int = 50) -> list[AuditEntryResponse]:
    state: AppState = request.app.state.outpost
    if tenant_id not in state.tenants:
        raise HTTPException(status_code=404, detail=f"unknown tenant {tenant_id!r}")

    records = state.audit_log.list_by_tenant(tenant_id, limit=limit)
    return [
        AuditEntryResponse(
            request_id=record.request_id,
            tenant_id=record.tenant_id,
            request_text=record.request_text,
            answer=record.final_content,
            rung=record.rung,
            rung_name=Rung(record.rung).name if record.rung is not None else None,
            created_at=record.created_at,
        )
        for record in records
    ]
