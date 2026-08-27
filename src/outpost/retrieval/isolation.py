"""Tenant isolation for retrieval.

The guarantee is structural, not a filter bolted on afterward: a search
restricted to one tenant only ever traverses that tenant's postings and
embedding rows, so a chunk belonging to another tenant is never scored,
let alone returned.
"""

from outpost.retrieval.dense import DenseStore
from outpost.retrieval.hybrid import reciprocal_rank_fusion
from outpost.retrieval.lexical import BM25Index


def search(
    lexical_index: BM25Index,
    dense_store: DenseStore,
    *,
    tenant_id: str,
    query: str,
    top_k: int = 5,
) -> list[str]:
    """Traversal-time isolation: the candidate set is restricted to the
    tenant before either index is scored.
    """
    candidate_ids = lexical_index.chunk_ids_for_tenant(tenant_id)
    lexical_ranked = [
        chunk_id for chunk_id, _ in lexical_index.score(query, candidate_ids=candidate_ids)
    ]
    query_vector = dense_store.embed_query(query)
    dense_ranked = [
        chunk_id for chunk_id, _ in dense_store.score(query_vector, candidate_ids=candidate_ids)
    ]
    return reciprocal_rank_fusion([lexical_ranked, dense_ranked])[:top_k]


def search_post_filtered(
    lexical_index: BM25Index,
    dense_store: DenseStore,
    *,
    tenant_id: str,
    query: str,
    top_k: int = 5,
) -> list[str]:
    """The negative control this design argues against: scores across
    every tenant's chunks, then filters to the caller's tenant afterward.
    Exists only to measure how much authorized recall that loses
    compared to traversal-time filtering.
    """
    lexical_ranked = [chunk_id for chunk_id, _ in lexical_index.score(query)]
    query_vector = dense_store.embed_query(query)
    dense_ranked = [chunk_id for chunk_id, _ in dense_store.score(query_vector)]
    fused = reciprocal_rank_fusion([lexical_ranked, dense_ranked])[:top_k]
    authorized = lexical_index.chunk_ids_for_tenant(tenant_id)
    return [chunk_id for chunk_id in fused if chunk_id in authorized]
