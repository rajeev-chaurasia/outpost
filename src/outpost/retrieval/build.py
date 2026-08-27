"""Builds the shared, multi-tenant retrieval index from each tenant's
unstructured (pdf_text) sources.

One index spans every tenant on purpose: this is the shared-index
architecture isolation.py's traversal-time filtering is designed for,
not something reserved for evaluation. The served API and the isolation
eval suite both build their index this way.
"""

from pathlib import Path

from outpost.connectors.pdf_text import PdfTextConnector
from outpost.ontology import load_tenant_config
from outpost.retrieval.chunk import chunk_document
from outpost.retrieval.dense import DenseStore, EmbeddingSource
from outpost.retrieval.document import Document
from outpost.retrieval.lexical import BM25Index


def build_multi_tenant_index(
    tenant_ids: list[str], tenants_dir: Path, embedding_cache: EmbeddingSource
) -> tuple[BM25Index, DenseStore]:
    lexical_index = BM25Index()
    dense_store = DenseStore(cache=embedding_cache)

    for tenant_id in tenant_ids:
        config = load_tenant_config(tenants_dir / tenant_id / "config.yaml")
        for source in config.sources:
            if source.connector != "pdf_text":
                continue
            records = PdfTextConnector(
                source_id=source.id, path=tenants_dir / tenant_id / source.path
            ).read()
            for record in records:
                document = Document(
                    document_id=f"{tenant_id}:{record.fields['document_id']}",
                    source_id=source.id,
                    tenant_id=tenant_id,
                    text=record.fields["text"],
                )
                for chunk in chunk_document(document):
                    lexical_index.add(chunk)
                    dense_store.index_chunk(chunk)

    return lexical_index, dense_store
