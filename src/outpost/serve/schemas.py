"""Request and response models for the served API."""

from pydantic import BaseModel, ConfigDict


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_request: str


class CitationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assertion: str
    source_id: str
    document_id: str
    text: str


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    tenant_id: str
    answer: str | None
    rung: int
    rung_name: str
    citations: list[CitationResponse]
    unsupported_assertions: list[str]


class TenantSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    display_name: str


class AuditEntryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    tenant_id: str
    request_text: str
    answer: str | None
    rung: int | None
    rung_name: str | None
    created_at: str
