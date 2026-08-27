"""Ingests one tenant's structured and unstructured sources, producing a
report of what mapped, what needs review, and how much got indexed.

This is what the onboarding CLI runs, and what the onboarding
measurement reads to score how much of a tenant's data made it in
without a human touching anything beyond the tenant's own config file.
"""

from dataclasses import dataclass
from pathlib import Path

from outpost.connectors.csv_export import CsvExportConnector
from outpost.connectors.pdf_text import PdfTextConnector
from outpost.mapping import MappingReport, resolve_records
from outpost.ontology import discover_tenant_ids, load_tenant_config
from outpost.retrieval.build import build_multi_tenant_index
from outpost.retrieval.dense import (
    DenseStore,
    EmbeddingCache,
    LiveFallbackEmbeddingCache,
    NvidiaEmbeddingClient,
)
from outpost.retrieval.lexical import BM25Index


@dataclass(frozen=True)
class OnboardingReport:
    tenant_id: str
    display_name: str
    source_record_counts: dict[str, int]
    mapping: MappingReport
    indexed_chunk_count: int
    indexed_document_count: int

    @property
    def mapped_count(self) -> int:
        return len(self.mapping.mapped())

    @property
    def needs_review_count(self) -> int:
        return len(self.mapping.needs_review())

    @property
    def unmapped_count(self) -> int:
        return len(self.mapping.unmapped())

    @property
    def auto_mapped_percentage(self) -> float:
        total = self.mapped_count + self.needs_review_count + self.unmapped_count
        return 100.0 * self.mapped_count / total if total else 0.0


def ingest_tenant(
    tenant_id: str, tenants_dir: Path, embedding_cache_path: Path
) -> tuple[OnboardingReport, BM25Index, DenseStore]:
    config = load_tenant_config(tenants_dir / tenant_id / "config.yaml")
    entities_by_name = {entity.name: entity for entity in config.ontology.entities}

    mapping = MappingReport()
    source_record_counts: dict[str, int] = {}
    document_count = 0

    for source in config.sources:
        path = tenants_dir / tenant_id / source.path
        if source.connector == "csv_export":
            entity = entities_by_name[source.entity] if source.entity else None
            records = CsvExportConnector(source_id=source.id, path=path).read()
            source_record_counts[source.id] = len(records)
            if entity is not None:
                _, source_mapping = resolve_records(
                    records, entity_fields=entity.fields, field_map=source.field_map
                )
                mapping.entries.extend(source_mapping.entries)
        elif source.connector == "pdf_text":
            records = PdfTextConnector(source_id=source.id, path=path).read()
            source_record_counts[source.id] = len(records)
            document_count += len(records)
        else:
            source_record_counts[source.id] = 0

    embedding_cache = LiveFallbackEmbeddingCache(
        cache=EmbeddingCache.load(embedding_cache_path),
        client=NvidiaEmbeddingClient(),
        save_path=embedding_cache_path,
    )
    all_tenant_ids = discover_tenant_ids(tenants_dir)
    lexical_index, dense_store = build_multi_tenant_index(
        all_tenant_ids, tenants_dir, embedding_cache
    )

    report = OnboardingReport(
        tenant_id=tenant_id,
        display_name=config.display_name,
        source_record_counts=source_record_counts,
        mapping=mapping,
        indexed_chunk_count=len(
            [chunk for chunk in lexical_index.chunks.values() if chunk.tenant_id == tenant_id]
        ),
        indexed_document_count=document_count,
    )
    return report, lexical_index, dense_store
