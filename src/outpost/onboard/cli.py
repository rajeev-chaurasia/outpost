"""The onboarding command.

Everything a new tenant needs is a config file and its data under
tenants/<tenant_id>/. This CLI is the only tool an operator should need
to bring one online: index it, then ask it a question to check the
result.
"""

from pathlib import Path

import typer

from outpost.agent.audit import AuditLog
from outpost.agent.handle import handle_request
from outpost.agent.tools import ActionGatedTool, DraftResponseTool, FlagDiscrepancyTool, SearchTool
from outpost.agent.tools.base import Tool
from outpost.llm.fallback import FallbackProvider
from outpost.llm.openai_compatible import OpenAICompatibleProvider
from outpost.onboard.report import ingest_tenant
from outpost.ontology import load_tenant_config

REPO_ROOT = Path(__file__).resolve().parents[3]
TENANTS_DIR = REPO_ROOT / "tenants"
EMBEDDING_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "embeddings" / "retrieval.npz"
AUDIT_DB_PATH = REPO_ROOT / "var" / "audit.sqlite"
PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

app = typer.Typer(add_completion=False)


@app.command()
def index(tenant_id: str) -> None:
    """Ingests a tenant's structured and unstructured sources and
    reports what mapped, what needs review, and how much got indexed.
    """
    report, _, _ = ingest_tenant(tenant_id, TENANTS_DIR, EMBEDDING_CACHE_PATH)

    typer.echo(f"tenant: {report.tenant_id} ({report.display_name})")
    typer.echo(f"sources: {report.source_record_counts}")
    typer.echo(
        f"fields mapped: {report.mapped_count}, "
        f"needs review: {report.needs_review_count}, "
        f"unmapped: {report.unmapped_count} "
        f"({report.auto_mapped_percentage:.1f}% auto-mapped)"
    )
    typer.echo(
        f"indexed {report.indexed_document_count} document(s) into "
        f"{report.indexed_chunk_count} chunk(s)"
    )
    for entry in report.mapping.unmapped():
        typer.echo(f"  unmapped: {entry.source_id} row {entry.row} field {entry.field!r}")
    for entry in report.mapping.needs_review():
        typer.echo(
            f"  needs review: {entry.source_id} row {entry.row} field {entry.field!r} "
            f"({entry.reason})"
        )


@app.command()
def ask(tenant_id: str, question: str) -> None:
    """Asks a tenant a question through the real agent, for a quick
    smoke check right after indexing.
    """
    config = load_tenant_config(TENANTS_DIR / tenant_id / "config.yaml")
    _, lexical_index, dense_store = ingest_tenant(tenant_id, TENANTS_DIR, EMBEDDING_CACHE_PATH)

    allowed = frozenset(config.actions.allowed)
    tools: dict[str, Tool] = {
        "search": SearchTool(
            lexical_index=lexical_index, dense_store=dense_store, tenant_id=tenant_id
        ),
        "flag_discrepancy": ActionGatedTool(tool=FlagDiscrepancyTool(), allowed_actions=allowed),
        "draft_response": ActionGatedTool(tool=DraftResponseTool(), allowed_actions=allowed),
    }
    provider = FallbackProvider(
        primary=OpenAICompatibleProvider(model=PRIMARY_MODEL),
        secondary=OpenAICompatibleProvider(model=FALLBACK_MODEL),
    )
    AUDIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_log = AuditLog(AUDIT_DB_PATH)

    result = handle_request(
        provider,
        tools,
        audit_log,
        tenant_id=tenant_id,
        system_prompt=(
            f"You are a helpful assistant for {config.display_name}. Use the search "
            "tool to find relevant text before answering. Answer only using "
            "information the search tool returns."
        ),
        user_request=question,
    )

    typer.echo(f"rung: {result.rung.name}")
    typer.echo(f"answer: {result.answer}")
    for citation in result.grounding.citations:
        typer.echo(f"  citation: {citation.span.source_id}/{citation.span.document_id}")
    for assertion in result.grounding.unsupported_assertions:
        typer.echo(f"  unsupported: {assertion}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
