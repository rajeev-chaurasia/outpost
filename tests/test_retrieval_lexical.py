"""Unit tests for the BM25 index: relevant chunks outscore irrelevant
ones, and restricting candidate_ids changes what gets scored, not just
what gets returned afterward.
"""

from outpost.retrieval.document import Chunk, Span
from outpost.retrieval.lexical import BM25Index


def _chunk(chunk_id: str, tenant_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        tenant_id=tenant_id,
        span=Span(source_id="s", document_id="d", start=0, end=len(text), text=text),
    )


def test_relevant_chunk_scores_higher_than_irrelevant() -> None:
    index = BM25Index()
    index.add(_chunk("a", "t1", "the account balance is short by fifty dollars"))
    index.add(_chunk("b", "t1", "completely unrelated text about gardening"))

    ranked = index.score("account balance short")
    assert ranked[0][0] == "a"


def test_candidate_ids_restricts_before_scoring() -> None:
    index = BM25Index()
    index.add(_chunk("a", "t1", "account balance short"))
    index.add(_chunk("b", "t2", "account balance short"))

    restricted = index.score("account balance short", candidate_ids={"a"})
    assert [chunk_id for chunk_id, _ in restricted] == ["a"]


def test_empty_index_returns_no_results() -> None:
    assert BM25Index().score("anything") == []
