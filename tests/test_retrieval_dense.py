"""Unit tests for the embedding cache and dense store: a cache miss
raises instead of silently falling back, and candidate_ids restricts the
similarity matrix before scoring, not just the returned results.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from outpost.retrieval.dense import DenseStore, EmbeddingCache, LiveFallbackEmbeddingCache
from outpost.retrieval.document import Chunk, Span
from outpost.retrieval.errors import EmbeddingCacheMissError


def _chunk(chunk_id: str, tenant_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        tenant_id=tenant_id,
        span=Span(source_id="s", document_id="d", start=0, end=len(text), text=text),
    )


def test_cache_load_on_missing_path_returns_empty() -> None:
    cache = EmbeddingCache.load(Path("/nonexistent/path/does-not-exist.npz"))
    assert cache.vectors == {}


def test_cache_put_and_get_roundtrip(tmp_path: Path) -> None:
    cache = EmbeddingCache()
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    cache.put("hello", "passage", vector)

    retrieved = cache.get("hello", "passage")
    assert retrieved is not None
    np.testing.assert_allclose(retrieved, vector, rtol=1e-2)
    assert cache.get("hello", "query") is None

    saved_path = tmp_path / "cache.npz"
    cache.save(saved_path)
    reloaded = EmbeddingCache.load(saved_path)
    reloaded_vector = reloaded.get("hello", "passage")
    assert reloaded_vector is not None
    np.testing.assert_allclose(reloaded_vector, vector, rtol=1e-2)


def test_index_chunk_raises_on_cache_miss() -> None:
    store = DenseStore(cache=EmbeddingCache())
    with pytest.raises(EmbeddingCacheMissError) as exc_info:
        store.index_chunk(_chunk("a", "t1", "never embedded"))
    assert exc_info.value.input_type == "passage"


def test_embed_query_raises_on_cache_miss() -> None:
    store = DenseStore(cache=EmbeddingCache())
    with pytest.raises(EmbeddingCacheMissError) as exc_info:
        store.embed_query("never embedded")
    assert exc_info.value.input_type == "query"


def test_score_restricts_candidates_before_building_similarity_matrix() -> None:
    cache = EmbeddingCache()
    cache.put("a text", "passage", np.array([1.0, 0.0], dtype=np.float32))
    cache.put("b text", "passage", np.array([0.0, 1.0], dtype=np.float32))
    store = DenseStore(cache=cache)
    store.index_chunk(_chunk("a", "t1", "a text"))
    store.index_chunk(_chunk("b", "t2", "b text"))

    query_vector = np.array([1.0, 0.0], dtype=np.float32)
    restricted = store.score(query_vector, candidate_ids={"a"})
    assert [chunk_id for chunk_id, _ in restricted] == ["a"]


def test_score_on_empty_store_returns_no_results() -> None:
    store = DenseStore(cache=EmbeddingCache())
    assert store.score(np.array([1.0, 0.0], dtype=np.float32)) == []


@dataclass
class _FakeEmbeddingClient:
    calls: list[tuple[str, str]]

    def embed(self, texts: list[str], input_type: str) -> list[NDArray[np.float32]]:
        self.calls.append((texts[0], input_type))
        return [np.array([9.0, 9.0], dtype=np.float32)]


def test_live_fallback_cache_computes_and_persists_a_miss(tmp_path: Path) -> None:
    save_path = tmp_path / "cache.npz"
    fake_client = _FakeEmbeddingClient(calls=[])
    live_cache = LiveFallbackEmbeddingCache(
        cache=EmbeddingCache(),
        client=fake_client,  # type: ignore[arg-type]
        save_path=save_path,
    )

    vector = live_cache.get("a brand new question", "query")

    assert vector is not None
    assert list(vector) == [9.0, 9.0]
    assert fake_client.calls == [("a brand new question", "query")]
    assert save_path.exists()

    reloaded = EmbeddingCache.load(save_path)
    assert reloaded.get("a brand new question", "query") is not None


def test_live_fallback_cache_reuses_an_existing_hit_without_calling_the_client() -> None:
    cache = EmbeddingCache()
    cache.put("known text", "passage", np.array([1.0, 2.0], dtype=np.float32))
    fake_client = _FakeEmbeddingClient(calls=[])
    live_cache = LiveFallbackEmbeddingCache(cache=cache, client=fake_client)  # type: ignore[arg-type]

    vector = live_cache.get("known text", "passage")

    assert vector is not None
    np.testing.assert_allclose(vector, [1.0, 2.0], rtol=1e-2)
    assert fake_client.calls == []
