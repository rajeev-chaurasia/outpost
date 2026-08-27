"""Splits a document into chunks, each carrying the exact span it came
from so a citation can always resolve back to source text.
"""

from outpost.retrieval.document import Chunk, Document, Span


def chunk_document(document: Document, *, max_chars: int = 400) -> list[Chunk]:
    text = document.text
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Prefer breaking on whitespace over cutting mid-word; fall
            # back to a hard cut only if no whitespace exists in range.
            boundary = end
            while boundary > start and not text[boundary].isspace():
                boundary -= 1
            if boundary > start:
                end = boundary
        chunk_text = text[start:end]
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}:{index}",
                tenant_id=document.tenant_id,
                span=Span(
                    source_id=document.source_id,
                    document_id=document.document_id,
                    start=start,
                    end=end,
                    text=chunk_text,
                ),
            )
        )
        index += 1
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return chunks
