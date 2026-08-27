"""Shared retrieval types.

Document and Chunk are added in phase 3, once chunking and indexing exist
to produce them.
"""

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
