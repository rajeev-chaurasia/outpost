"""One-off dev tool: computes real NVIDIA embeddings for every chunk and
adversarial query text this project's tests need, and writes them to the
committed cache at tests/fixtures/embeddings/retrieval.npz.

Run locally with LLM_API_KEY set. Never invoked by tests or CI, both of
which only ever read the cache this script produces.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from eval.isolation.adversarial import (  # noqa: E402
    EMBEDDING_CACHE_PATH,
    TENANT_IDS,
    TENANTS_DIR,
    load_cases,
)
from outpost.connectors.pdf_text import PdfTextConnector  # noqa: E402
from outpost.ontology import load_tenant_config  # noqa: E402
from outpost.retrieval.chunk import chunk_document  # noqa: E402
from outpost.retrieval.dense import EmbeddingCache, NvidiaEmbeddingClient  # noqa: E402
from outpost.retrieval.document import Document  # noqa: E402


def main() -> None:
    passages: list[str] = []
    for tenant_id in TENANT_IDS:
        config = load_tenant_config(TENANTS_DIR / tenant_id / "config.yaml")
        for source in config.sources:
            if source.connector != "pdf_text":
                continue
            records = PdfTextConnector(
                source_id=source.id, path=TENANTS_DIR / tenant_id / source.path
            ).read()
            for record in records:
                document = Document(
                    document_id=f"{tenant_id}:{record.fields['document_id']}",
                    source_id=source.id,
                    tenant_id=tenant_id,
                    text=record.fields["text"],
                )
                passages.extend(chunk.span.text for chunk in chunk_document(document))

    queries = [case.query for case in load_cases()]

    client = NvidiaEmbeddingClient()
    cache = EmbeddingCache.load(EMBEDDING_CACHE_PATH)

    print(f"embedding {len(passages)} passages and {len(queries)} queries")
    for text, vector in zip(passages, client.embed(passages, "passage"), strict=True):
        cache.put(text, "passage", vector)
    for text, vector in zip(queries, client.embed(queries, "query"), strict=True):
        cache.put(text, "query", vector)

    cache.save(EMBEDDING_CACHE_PATH)
    print(f"wrote {len(cache.vectors)} vectors to {EMBEDDING_CACHE_PATH}")


if __name__ == "__main__":
    main()
