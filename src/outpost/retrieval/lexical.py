"""Hand-written BM25 lexical index.

One index spans every tenant's chunks, since a real deployment shares
infrastructure across customers: the isolation guarantee has to come
from how a query is scored, not from running a separate index per
tenant.
"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from outpost.retrieval.document import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    chunks: dict[str, Chunk] = field(default_factory=dict)
    postings: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(dict))
    doc_lengths: dict[str, int] = field(default_factory=dict)

    def add(self, chunk: Chunk) -> None:
        tokens = tokenize(chunk.span.text)
        self.chunks[chunk.chunk_id] = chunk
        self.doc_lengths[chunk.chunk_id] = len(tokens)
        for token, count in Counter(tokens).items():
            self.postings[token][chunk.chunk_id] = count

    def chunk_ids_for_tenant(self, tenant_id: str) -> set[str]:
        return {chunk_id for chunk_id, chunk in self.chunks.items() if chunk.tenant_id == tenant_id}

    def score(
        self, query: str, *, candidate_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """Scores chunks against query.

        If candidate_ids is given, the posting lists are intersected with
        it before any score is computed: restricting the candidate set
        changes what gets scored, not just what gets returned afterward.
        """
        if not self.chunks:
            return []
        avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        n_docs = len(self.chunks)
        scores: dict[str, float] = defaultdict(float)

        for token in tokenize(query):
            postings = self.postings.get(token, {})
            doc_freq = len(postings)
            if doc_freq == 0:
                continue
            idf = math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            for chunk_id, term_freq in postings.items():
                if candidate_ids is not None and chunk_id not in candidate_ids:
                    continue
                doc_length = self.doc_lengths[chunk_id]
                denom = term_freq + self.k1 * (1 - self.b + self.b * doc_length / avg_doc_length)
                scores[chunk_id] += idf * (term_freq * (self.k1 + 1)) / denom

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)
