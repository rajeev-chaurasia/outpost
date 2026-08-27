"""Dense retrieval backed by NVIDIA's hosted embedding model.

CI never calls the network: every text this project needs embedded has
its vector pre-computed and committed to tests/fixtures/embeddings/,
keyed by a hash of (input_type, text). A cache miss raises rather than
silently falling back to a live call or a zero vector, so a missing
fixture fails loudly instead of quietly depending on network access.
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import httpx
import numpy as np
from numpy.typing import NDArray

from outpost.retrieval.document import Chunk
from outpost.retrieval.errors import EmbeddingCacheMissError

EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"
_API_URL = "https://integrate.api.nvidia.com/v1/embeddings"

InputType = Literal["query", "passage"]


def cache_key(text: str, input_type: InputType) -> str:
    return hashlib.sha256(f"{input_type}\x00{text}".encode()).hexdigest()


class EmbeddingSource(Protocol):
    """Anything DenseStore can ask for an embedding: the static,
    committed EmbeddingCache tests and CI use, or the served app's
    LiveFallbackEmbeddingCache.
    """

    def get(self, text: str, input_type: InputType) -> NDArray[np.float32] | None: ...

    def put(self, text: str, input_type: InputType, vector: NDArray[np.float32]) -> None: ...


@dataclass
class EmbeddingCache:
    """cache_key -> vector, persisted as a single float16 .npz file."""

    vectors: dict[str, NDArray[np.float16]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "EmbeddingCache":
        if not path.exists():
            return cls()
        data = np.load(path)
        return cls(vectors={key: data[key] for key in data.files})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # numpy's savez_compressed stub does not accept a **dict[str,
        # NDArray] unpack directly; the cast just satisfies that, the
        # runtime call is unaffected.
        np.savez_compressed(path, **cast("dict[str, Any]", self.vectors))

    def get(self, text: str, input_type: InputType) -> NDArray[np.float32] | None:
        vector = self.vectors.get(cache_key(text, input_type))
        return vector.astype(np.float32) if vector is not None else None

    def put(self, text: str, input_type: InputType, vector: NDArray[np.float32]) -> None:
        self.vectors[cache_key(text, input_type)] = vector.astype(np.float16)


class NvidiaEmbeddingClient:
    """Calls the live NVIDIA embedding endpoint.

    The api key is only read from the environment when embed() actually
    runs, not at construction: LiveFallbackEmbeddingCache always holds
    one of these ready in case of a cache miss, and constructing it must
    not require a key that a fully cache-hit run never ends up needing.
    """

    def __init__(self, api_key: str | None = None, model: str = EMBEDDING_MODEL) -> None:
        self._api_key = api_key
        self._model = model

    def embed(self, texts: list[str], input_type: InputType) -> list[NDArray[np.float32]]:
        api_key = self._api_key or os.environ["LLM_API_KEY"]
        response = httpx.post(
            _API_URL,
            json={
                "model": self._model,
                "input": texts,
                "input_type": input_type,
                "encoding_format": "float",
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        return [np.array(item["embedding"], dtype=np.float32) for item in payload["data"]]


@dataclass
class LiveFallbackEmbeddingCache:
    """Wraps an EmbeddingCache; a miss is computed live through the
    client, stored back into the wrapped cache, and persisted to disk
    if a save path is given.

    Used only by the served app, so a user's actual question can be
    embedded even though it was never in the pre-committed fixture set.
    Tests and CI use EmbeddingCache directly, never this: their whole
    point is staying deterministic and keyless.
    """

    cache: EmbeddingCache
    client: NvidiaEmbeddingClient
    save_path: Path | None = None

    def get(self, text: str, input_type: InputType) -> NDArray[np.float32] | None:
        vector = self.cache.get(text, input_type)
        if vector is not None:
            return vector
        computed = self.client.embed([text], input_type)[0]
        self.put(text, input_type, computed)
        return computed

    def put(self, text: str, input_type: InputType, vector: NDArray[np.float32]) -> None:
        self.cache.put(text, input_type, vector)
        if self.save_path is not None:
            self.cache.save(self.save_path)


@dataclass
class DenseStore:
    """Chunk vectors plus cosine similarity scoring, restricted to a
    candidate set before the similarity matrix is ever built.
    """

    cache: EmbeddingSource
    vectors: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    chunk_tenant: dict[str, str] = field(default_factory=dict)

    def index_chunk(self, chunk: Chunk) -> None:
        vector = self.cache.get(chunk.span.text, "passage")
        if vector is None:
            raise EmbeddingCacheMissError(text=chunk.span.text, input_type="passage")
        self.vectors[chunk.chunk_id] = vector
        self.chunk_tenant[chunk.chunk_id] = chunk.tenant_id

    def embed_query(self, query: str) -> NDArray[np.float32]:
        vector = self.cache.get(query, "query")
        if vector is None:
            raise EmbeddingCacheMissError(text=query, input_type="query")
        return vector

    def chunk_ids_for_tenant(self, tenant_id: str) -> set[str]:
        return {chunk_id for chunk_id, tid in self.chunk_tenant.items() if tid == tenant_id}

    def score(
        self, query_vector: NDArray[np.float32], *, candidate_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        # Row selection happens here, before any similarity is computed:
        # restricting candidate_ids changes what enters the matrix, not
        # just what gets returned.
        ids = [
            chunk_id
            for chunk_id in self.vectors
            if candidate_ids is None or chunk_id in candidate_ids
        ]
        if not ids:
            return []
        matrix = np.stack([self.vectors[chunk_id] for chunk_id in ids])
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-9)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        similarities = matrix_norm @ query_norm
        return sorted(
            zip(ids, similarities.tolist(), strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
