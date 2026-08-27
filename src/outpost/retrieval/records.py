"""Renders mapped structured records into citable documents.

Retrieval and grounding both work on text spans, so a record has to
become text before the agent can cite it. Rendering each record as one
canonical line, then indexing that line the same way a document is
indexed, means a record answer gets a citation with real offsets through
exactly the same path a document answer does, with no second grounding
mechanism to keep in sync.

Only fields that mapped cleanly are rendered. A value the mapping layer
flagged NEEDS_REVIEW or could not map is left out rather than presented
as fact, so the agent cannot cite a value a human has not confirmed.
"""

from typing import Any

from outpost.retrieval.document import Document


def render_record(fields: dict[str, Any]) -> str:
    return " | ".join(f"{name}: {value}" for name, value in fields.items() if value is not None)


def record_documents(
    rows: list[dict[str, Any]], *, tenant_id: str, source_id: str, key_field: str
) -> list[Document]:
    documents = []
    # A tenant's data can repeat a key: the fixtures include a duplicate
    # row differing in one field, and mapping deliberately keeps both.
    # Repeated keys are suffixed so the second row gets its own document
    # instead of overwriting the first, since silently indexing one of
    # two conflicting records is exactly the failure the mapping report
    # exists to prevent.
    seen_keys: dict[str, int] = {}
    for index, fields in enumerate(rows):
        text = render_record(fields)
        if not text:
            continue

        key = str(fields.get(key_field) or f"row{index + 1}")
        seen_keys[key] = seen_keys.get(key, 0) + 1
        occurrence = seen_keys[key]
        document_key = key if occurrence == 1 else f"{key}#{occurrence}"

        documents.append(
            Document(
                document_id=f"{tenant_id}:{source_id}:{document_key}",
                source_id=source_id,
                tenant_id=tenant_id,
                text=text,
            )
        )
    return documents
