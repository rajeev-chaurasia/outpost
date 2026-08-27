"""Builds the shared multi-tenant retrieval index from the pdf_text
fixtures, and runs the adversarial probes in cases.yaml against it, both
with traversal-time filtering and the post-filter negative control.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from outpost.ontology import discover_tenant_ids
from outpost.retrieval.build import build_multi_tenant_index as _build_multi_tenant_index
from outpost.retrieval.dense import DenseStore, EmbeddingCache
from outpost.retrieval.isolation import search, search_post_filtered
from outpost.retrieval.lexical import BM25Index

REPO_ROOT = Path(__file__).resolve().parents[2]
TENANTS_DIR = REPO_ROOT / "tenants"
CASES_PATH = Path(__file__).resolve().parent / "cases.yaml"
EMBEDDING_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "embeddings" / "retrieval.npz"
ARTIFACT_PATH = REPO_ROOT / "eval" / "artifacts" / "isolation_results.json"


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
    return _build_multi_tenant_index(
        discover_tenant_ids(TENANTS_DIR), TENANTS_DIR, EmbeddingCache.load(cache_path)
    )


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


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    """Rolls case results into the committed artifact shape.

    traversal_authorized_results and post_filter_authorized_results are
    the measured argument for filtering during traversal rather than
    after it: both numbers count only results the querying tenant is
    allowed to see, so the gap between them is authorized recall the
    post-filter approach loses.
    """
    per_tenant: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = per_tenant.setdefault(
            result.tenant_id,
            {"cases": 0, "leaks": 0, "traversal_results": 0, "post_filter_results": 0},
        )
        bucket["cases"] += 1
        bucket["leaks"] += result.traversal_leaks
        bucket["traversal_results"] += len(result.traversal_result_ids)
        bucket["post_filter_results"] += len(result.post_filter_result_ids)

    return {
        "case_count": len(results),
        "total_leaks": sum(r.traversal_leaks for r in results),
        "zero_leak_invariant_met": all(r.traversal_leaks == 0 for r in results),
        "traversal_authorized_results": sum(len(r.traversal_result_ids) for r in results),
        "post_filter_authorized_results": sum(len(r.post_filter_result_ids) for r in results),
        "per_tenant": per_tenant,
    }


def main() -> None:
    lexical_index, dense_store = build_multi_tenant_index()
    results = run_isolation_suite(lexical_index, dense_store, load_cases())
    summary = summarize(results)

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
