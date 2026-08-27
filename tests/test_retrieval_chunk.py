"""Chunking done-test: every chunk resolves back to exact source text at
its recorded offsets, including on a document long enough to need more
than one chunk.
"""

from outpost.retrieval.chunk import chunk_document
from outpost.retrieval.document import Document


def test_short_document_produces_one_chunk() -> None:
    document = Document(document_id="d1", source_id="s1", tenant_id="t1", text="hello world")
    chunks = chunk_document(document)
    assert len(chunks) == 1
    assert chunks[0].span.text == "hello world"


def test_every_chunk_resolves_to_exact_source_text() -> None:
    words = " ".join(f"word{i}" for i in range(200))
    document = Document(document_id="d1", source_id="s1", tenant_id="t1", text=words)
    chunks = chunk_document(document, max_chars=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert document.text[chunk.span.start : chunk.span.end] == chunk.span.text
        assert chunk.span.document_id == document.document_id
        assert chunk.tenant_id == document.tenant_id


def test_chunk_ids_are_unique_and_ordered() -> None:
    words = " ".join(f"word{i}" for i in range(200))
    document = Document(document_id="d1", source_id="s1", tenant_id="t1", text=words)
    chunks = chunk_document(document, max_chars=50)
    ids = [chunk.chunk_id for chunk in chunks]
    assert ids == sorted(set(ids), key=ids.index)
    assert len(set(ids)) == len(ids)
