"""Structured records are retrievable and citable through the same path
documents are.

This closes the gap where a question answerable only from a tenant's csv
data refused instead of answering, because grounding could only cite the
unstructured index.
"""

from pathlib import Path
from typing import Any

from outpost.ontology import discover_tenant_ids
from outpost.retrieval.build import build_multi_tenant_index
from outpost.retrieval.dense import EmbeddingCache
from outpost.retrieval.records import record_documents, render_record

REPO_ROOT = Path(__file__).resolve().parents[1]
TENANTS_DIR = REPO_ROOT / "tenants"
EMBEDDING_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "embeddings" / "retrieval.npz"


def _rows() -> list[dict[str, Any]]:
    return [
        {"widget_id": "W-1", "status": "open", "amount": None},
        {"widget_id": "W-2", "status": "closed", "amount": 12},
    ]


def test_render_record_omits_fields_that_did_not_map() -> None:
    text = render_record(_rows()[0])
    assert "widget_id: W-1" in text
    assert "status: open" in text
    # amount did not map cleanly, so it must not appear as if it were fact
    assert "amount" not in text


def test_record_documents_carry_tenant_and_source() -> None:
    documents = record_documents(
        _rows(), tenant_id="t1", source_id="widgets", key_field="widget_id"
    )
    assert [d.document_id for d in documents] == ["t1:widgets:W-1", "t1:widgets:W-2"]
    assert all(d.tenant_id == "t1" and d.source_id == "widgets" for d in documents)


def test_repeated_keys_get_distinct_documents() -> None:
    rows = [
        {"widget_id": "W-1", "status": "open"},
        {"widget_id": "W-1", "status": "closed"},
    ]
    documents = record_documents(rows, tenant_id="t1", source_id="widgets", key_field="widget_id")

    assert [d.document_id for d in documents] == ["t1:widgets:W-1", "t1:widgets:W-1#2"]
    assert documents[0].text != documents[1].text


def test_rows_with_nothing_mapped_are_skipped() -> None:
    documents = record_documents(
        [{"widget_id": None}], tenant_id="t1", source_id="widgets", key_field="widget_id"
    )
    assert documents == []


def test_every_tenants_structured_records_reach_the_shared_index() -> None:
    lexical_index, _ = build_multi_tenant_index(
        discover_tenant_ids(TENANTS_DIR), TENANTS_DIR, EmbeddingCache.load(EMBEDDING_CACHE_PATH)
    )

    for tenant_id in discover_tenant_ids(TENANTS_DIR):
        record_chunks = [
            chunk
            for chunk in lexical_index.chunks.values()
            if chunk.tenant_id == tenant_id and " | " in chunk.span.text
        ]
        assert record_chunks, f"{tenant_id} has no indexed structured records"


def test_a_fact_held_only_in_csv_is_retrievable() -> None:
    """work_order status lives only in utility_ops' csv, never in its
    documents. Before structured records were indexed, a question about
    it had nothing to ground against and correctly refused.
    """
    lexical_index, _ = build_multi_tenant_index(
        discover_tenant_ids(TENANTS_DIR), TENANTS_DIR, EmbeddingCache.load(EMBEDDING_CACHE_PATH)
    )

    open_work_orders = [
        chunk
        for chunk in lexical_index.chunks.values()
        if chunk.tenant_id == "utility_ops"
        and "work_orders" in chunk.span.document_id
        and "status: open" in chunk.span.text
    ]

    assert open_work_orders
    assert any("WO-3003" in chunk.span.document_id for chunk in open_work_orders)
