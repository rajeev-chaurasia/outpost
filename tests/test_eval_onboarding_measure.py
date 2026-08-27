"""Unit test for the isolation check in the onboarding measurement:
this is a structural check on chunk_id/tenant_id tagging, not an
adversarial probe, so it needs no live provider and no embeddings.
"""

from eval.onboarding.measure import _check_isolation
from outpost.retrieval.document import Chunk, Span
from outpost.retrieval.lexical import BM25Index


def _chunk(chunk_id: str, tenant_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        tenant_id=tenant_id,
        span=Span(source_id="s", document_id="d", start=0, end=4, text="text"),
    )


def test_check_isolation_reports_disjoint_chunk_sets() -> None:
    index = BM25Index()
    index.add(_chunk("a", "tenant_c"))
    index.add(_chunk("b", "tenant_c"))
    index.add(_chunk("c", "dealer_ar"))

    result = _check_isolation(index, "tenant_c")

    assert result["this_tenant_chunk_count"] == 2
    assert result["other_tenant_chunk_count"] == 1
    assert result["shares_no_chunk_ids"] is True


def test_check_isolation_with_no_other_tenants_present() -> None:
    index = BM25Index()
    index.add(_chunk("a", "tenant_c"))

    result = _check_isolation(index, "tenant_c")

    assert result["this_tenant_chunk_count"] == 1
    assert result["other_tenant_chunk_count"] == 0
    assert result["shares_no_chunk_ids"] is True
