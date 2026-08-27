"""Shared retrieval types."""

from pydantic import BaseModel, ConfigDict


class Span(BaseModel):
    """A byte range into one source document, and the text at that range.

    Every citation the agent produces resolves to a Span; grounding checks
    that span.text still matches the source at [start, end).
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str
    document_id: str
    start: int
    end: int
    text: str


class Document(BaseModel):
    """One source document, before chunking."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_id: str
    tenant_id: str
    text: str


class Chunk(BaseModel):
    """One retrievable slice of a document, still traceable back to the
    exact source text it came from via span.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    tenant_id: str
    span: Span
