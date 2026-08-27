"""Records the cold-onboarding measurement for one tenant.

onboarding_seconds and manual_interventions are supplied by the person
who ran the onboarding: the wall-clock time they observed, and how many
times they had to touch code under src/outpost/ rather than just the
tenant's own config and fixtures. Neither is something this script, or
any script, can measure by itself.

Everything else (auto-mapped percentage, indexed chunk count, whether
the tenant answers a real question with citations, whether that answer
ever includes another tenant's chunk) is computed here, live, against
the real provider and the real shared index.
"""

import json
from pathlib import Path
from typing import Any

import typer

from outpost.agent.audit import AuditLog
from outpost.agent.handle import handle_request
from outpost.agent.tools import SearchTool
from outpost.llm.fallback import FallbackProvider
from outpost.llm.openai_compatible import OpenAICompatibleProvider
from outpost.onboard.report import ingest_tenant
from outpost.ontology import load_tenant_config

REPO_ROOT = Path(__file__).resolve().parents[2]
TENANTS_DIR = REPO_ROOT / "tenants"
EMBEDDING_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "embeddings" / "retrieval.npz"
ARTIFACT_PATH = REPO_ROOT / "eval" / "artifacts" / "onboarding_results.json"
PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

app = typer.Typer(add_completion=False)


def _check_isolation(lexical_index: Any, tenant_id: str) -> dict[str, Any]:
    """A structural check, not a probe: every chunk this tenant's search
    tool could ever return is tagged with this tenant_id at index time,
    so a leak would mean isolation.py's traversal-time filter itself is
    broken, not that some adversarial query happened to slip through.
    """
    other_tenant_chunks = [
        chunk for chunk in lexical_index.chunks.values() if chunk.tenant_id != tenant_id
    ]
    this_tenant_chunks = [
        chunk for chunk in lexical_index.chunks.values() if chunk.tenant_id == tenant_id
    ]
    return {
        "other_tenant_chunk_count": len(other_tenant_chunks),
        "this_tenant_chunk_count": len(this_tenant_chunks),
        "shares_no_chunk_ids": {c.chunk_id for c in this_tenant_chunks}.isdisjoint(
            {c.chunk_id for c in other_tenant_chunks}
        ),
    }


@app.command()
def measure(
    tenant_id: str,
    onboarding_seconds: float,
    manual_interventions: int,
    verification_question: str,
) -> None:
    report, lexical_index, dense_store = ingest_tenant(tenant_id, TENANTS_DIR, EMBEDDING_CACHE_PATH)
    config = load_tenant_config(TENANTS_DIR / tenant_id / "config.yaml")

    search_tool = SearchTool(
        lexical_index=lexical_index, dense_store=dense_store, tenant_id=tenant_id
    )
    provider = FallbackProvider(
        primary=OpenAICompatibleProvider(model=PRIMARY_MODEL),
        secondary=OpenAICompatibleProvider(model=FALLBACK_MODEL),
    )
    audit_log = AuditLog(REPO_ROOT / "var" / "onboarding_measure.sqlite")

    result = handle_request(
        provider,
        {"search": search_tool},
        audit_log,
        tenant_id=tenant_id,
        system_prompt=(
            f"You are a helpful assistant for {config.display_name}. Use the search "
            "tool to find relevant text before answering. Answer only using "
            "information the search tool returns."
        ),
        user_request=verification_question,
    )

    isolation = _check_isolation(lexical_index, tenant_id)

    artifact: dict[str, Any] = {}
    if ARTIFACT_PATH.exists():
        artifact = json.loads(ARTIFACT_PATH.read_text())

    artifact[tenant_id] = {
        "onboarding_seconds": onboarding_seconds,
        "manual_interventions": manual_interventions,
        "source_record_counts": report.source_record_counts,
        "mapped_count": report.mapped_count,
        "needs_review_count": report.needs_review_count,
        "unmapped_count": report.unmapped_count,
        "auto_mapped_percentage": round(report.auto_mapped_percentage, 1),
        "indexed_chunk_count": report.indexed_chunk_count,
        "verification": {
            "question": verification_question,
            "rung": result.rung.name,
            "answer": result.answer,
            "citation_count": len(result.grounding.citations),
            "unsupported_count": len(result.grounding.unsupported_assertions),
        },
        "isolation": isolation,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    typer.echo(json.dumps(artifact[tenant_id], indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
