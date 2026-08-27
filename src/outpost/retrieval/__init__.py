"""Retrieval indexes and tenant isolation."""

from outpost.retrieval.chunk import chunk_document
from outpost.retrieval.dense import DenseStore, EmbeddingCache, NvidiaEmbeddingClient
from outpost.retrieval.document import Chunk, Document, Span
from outpost.retrieval.errors import EmbeddingCacheMissError, RetrievalError
from outpost.retrieval.hybrid import reciprocal_rank_fusion
from outpost.retrieval.isolation import search, search_post_filtered
from outpost.retrieval.lexical import BM25Index

__all__ = [
    "BM25Index",
    "Chunk",
    "DenseStore",
    "Document",
    "EmbeddingCache",
    "EmbeddingCacheMissError",
    "NvidiaEmbeddingClient",
    "RetrievalError",
    "Span",
    "chunk_document",
    "reciprocal_rank_fusion",
    "search",
    "search_post_filtered",
]
