"""Builds the shared, multi-tenant retrieval index.

Both kinds of tenant source end up in the same index: unstructured
documents through the pdf_text connector, and structured records
rendered to text through the mapping layer. Indexing records here rather
than giving them a separate lookup path means a record answer is
grounded, cited, and tenant-isolated by exactly the same code that
handles a document answer.

One index spans every tenant on purpose: this is the shared-index
architecture isolation.py's traversal-time filtering is designed for,
not something reserved for evaluation. The served API, the onboarding
cli, and the isolation eval suite all build their index this way.
"""

from pathlib import Path

from outpost.connectors.csv_export import CsvExportConnector
from outpost.connectors.pdf_text import PdfTextConnector
from outpost.mapping import resolve_records
from outpost.ontology import load_tenant_config
from outpost.retrieval.chunk import chunk_document
from outpost.retrieval.dense import DenseStore, EmbeddingSource
from outpost.retrieval.document import Document
from outpost.retrieval.lexical import BM25Index
from outpost.retrieval.records import record_documents


def build_multi_tenant_index(
    tenant_ids: list[str], tenants_dir: Path, embedding_cache: EmbeddingSource
) -> tuple[BM25Index, DenseStore]:
    lexical_index = BM25Index()
    dense_store = DenseStore(cache=embedding_cache)

    for tenant_id in tenant_ids:
        for document in _tenant_documents(tenant_id, tenants_dir):
            for chunk in chunk_document(document):
                lexical_index.add(chunk)
                dense_store.index_chunk(chunk)

    return lexical_index, dense_store


def _tenant_documents(tenant_id: str, tenants_dir: Path) -> list[Document]:
    config = load_tenant_config(tenants_dir / tenant_id / "config.yaml")
    entities_by_name = {entity.name: entity for entity in config.ontology.entities}
    documents: list[Document] = []

    for source in config.sources:
        path = tenants_dir / tenant_id / source.path

        if source.connector == "pdf_text":
            documents.extend(
                Document(
                    document_id=f"{tenant_id}:{record.fields['document_id']}",
                    source_id=source.id,
                    tenant_id=tenant_id,
                    text=record.fields["text"],
                )
                for record in PdfTextConnector(source_id=source.id, path=path).read()
            )

        elif source.connector == "csv_export" and source.entity is not None:
            entity = entities_by_name[source.entity]
            rows, _ = resolve_records(
                CsvExportConnector(source_id=source.id, path=path).read(),
                entity_fields=entity.fields,
                field_map=source.field_map,
            )
            documents.extend(
                record_documents(
                    rows, tenant_id=tenant_id, source_id=source.id, key_field=entity.key
                )
            )

    return documents
