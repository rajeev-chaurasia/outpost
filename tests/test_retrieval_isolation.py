"""Phase 3 done-tests: the adversarial suite finds zero cross-tenant
leaks under traversal-time filtering, the post-filter negative control
measurably returns fewer authorized results on the same queries, and
every chunk resolves back to exact source text at its recorded offsets.
"""

from pathlib import Path

from eval.isolation.adversarial import (
    TENANTS_DIR,
    build_multi_tenant_index,
    load_cases,
    run_isolation_suite,
)
from outpost.ontology import discover_tenant_ids
from outpost.retrieval.document import Chunk


def test_traversal_filtering_has_zero_cross_tenant_leaks() -> None:
    lexical_index, dense_store = build_multi_tenant_index()
    results = run_isolation_suite(lexical_index, dense_store, load_cases())

    assert results
    assert all(result.traversal_leaks == 0 for result in results)


def test_post_filter_control_returns_fewer_authorized_results() -> None:
    lexical_index, dense_store = build_multi_tenant_index()
    results = run_isolation_suite(lexical_index, dense_store, load_cases())

    total_traversal = sum(len(result.traversal_result_ids) for result in results)
    total_post_filter = sum(len(result.post_filter_result_ids) for result in results)

    # Real numbers from the fixture corpus, not asserted in principle:
    # traversal-time filtering returns 20 authorized results across the
    # 8 adversarial cases, the post-filter control returns only 6, because
    # unauthorized chunks from the other tenant crowd the shared top-k
    # window before authorized results ever get a chance to appear in it.
    assert total_post_filter < total_traversal


def test_every_case_result_belongs_to_the_querying_tenant() -> None:
    lexical_index, dense_store = build_multi_tenant_index()
    results = run_isolation_suite(lexical_index, dense_store, load_cases())

    for result in results:
        for chunk_id in result.traversal_result_ids:
            assert lexical_index.chunks[chunk_id].tenant_id == result.tenant_id
        for chunk_id in result.post_filter_result_ids:
            assert lexical_index.chunks[chunk_id].tenant_id == result.tenant_id


def test_indexed_chunks_resolve_back_to_exact_fixture_text() -> None:
    lexical_index, _ = build_multi_tenant_index()

    checked = 0
    for chunk in lexical_index.chunks.values():
        tenant_dir = TENANTS_DIR / chunk.tenant_id
        source_path = _document_path(tenant_dir, chunk)
        original_text = source_path.read_text(encoding="utf-8")
        assert original_text[chunk.span.start : chunk.span.end] == chunk.span.text
        checked += 1

    assert checked == len(lexical_index.chunks)
    assert {chunk.tenant_id for chunk in lexical_index.chunks.values()} == set(
        discover_tenant_ids(TENANTS_DIR)
    )


def _document_path(tenant_dir: Path, chunk: Chunk) -> Path:
    _, stem = chunk.span.document_id.split(":", 1)
    matches = list(tenant_dir.glob(f"fixtures/**/{stem}.txt"))
    assert len(matches) == 1, f"expected exactly one fixture file named {stem}.txt"
    return matches[0]
