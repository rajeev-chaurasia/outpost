"""Lists the tenants this instance is serving."""

from fastapi import APIRouter, Request

from outpost.serve.schemas import TenantSummary
from outpost.serve.state import AppState

router = APIRouter()


@router.get("/tenants", response_model=list[TenantSummary])
def list_tenants(request: Request) -> list[TenantSummary]:
    state: AppState = request.app.state.outpost
    return [
        TenantSummary(tenant_id=tenant_id, display_name=runtime.config.display_name)
        for tenant_id, runtime in state.tenants.items()
    ]
