"""Builds the shared multi-tenant retrieval index from the pdf_text
fixtures, and runs the adversarial probes in cases.yaml against it, both
with traversal-time filtering and the post-filter negative control.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from outpost.connectors.pdf_text import PdfTextConnector
from outpost.ontology import load_tenant_config
from outpost.retrieval.chunk import chunk_document
from outpost.retrieval.dense import DenseStore, EmbeddingCache
from outpost.retrieval.document import Document
from outpost.retrieval.isolation import search, search_post_filtered
from outpost.retrieval.lexical import BM25Index

REPO_ROOT = Path(__file__).resolve().parents[2]
TENANTS_DIR = REPO_ROOT / "tenants"
CASES_PATH = Path(__file__).resolve().parent / "cases.yaml"
EMBEDDING_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "embeddings" / "retrieval.npz"
TENANT_IDS = ("dealer_ar", "claims_intake")


@dataclass(frozen=True)
class IsolationCase:
    tenant_id: str
    query: str


def load_cases() -> list[IsolationCase]:
    raw = yaml.safe_load(CASES_PATH.read_text())
    return [
        IsolationCase(tenant_id=case["tenant_id"], query=case["query"]) for case in raw["cases"]
    ]


def build_multi_tenant_index(
    cache_path: Path = EMBEDDING_CACHE_PATH,
) -> tuple[BM25Index, DenseStore]:
    lexical_index = BM25Index()
    dense_store = DenseStore(cache=EmbeddingCache.load(cache_path))

    for tenant_id in TENANT_IDS:
        config = load_tenant_config(TENANTS_DIR / tenant_id / "config.yaml")
        for source in config.sources:
            if source.connector != "pdf_text":
                continue
            records = PdfTextConnector(
                source_id=source.id, path=TENANTS_DIR / tenant_id / source.path
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


@dataclass(frozen=True)
class CaseResult:
    tenant_id: str
    query: str
    traversal_result_ids: list[str]
    traversal_leaks: int
    post_filter_result_ids: list[str]


def run_isolation_suite(
    lexical_index: BM25Index,
    dense_store: DenseStore,
    cases: list[IsolationCase],
    *,
    top_k: int = 3,
) -> list[CaseResult]:
    results = []
    for case in cases:
        traversal_ids = search(
            lexical_index, dense_store, tenant_id=case.tenant_id, query=case.query, top_k=top_k
        )
        leaks = sum(
            1
            for chunk_id in traversal_ids
            if lexical_index.chunks[chunk_id].tenant_id != case.tenant_id
        )
        post_filter_ids = search_post_filtered(
            lexical_index, dense_store, tenant_id=case.tenant_id, query=case.query, top_k=top_k
        )
        results.append(
            CaseResult(
                tenant_id=case.tenant_id,
                query=case.query,
                traversal_result_ids=traversal_ids,
                traversal_leaks=leaks,
                post_filter_result_ids=post_filter_ids,
            )
        )
    return results
