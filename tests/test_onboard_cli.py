"""Onboarding done-tests: ingesting an existing tenant through the same
path the CLI uses produces a real mapping report and indexes real
chunks, without touching anything the tenant's own config and fixtures
don't already declare.
"""

from pathlib import Path

from outpost.mapping import MappingOutcome
from outpost.onboard.report import ingest_tenant

REPO_ROOT = Path(__file__).resolve().parents[1]
TENANTS_DIR = REPO_ROOT / "tenants"
EMBEDDING_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "embeddings" / "retrieval.npz"


def test_ingest_dealer_ar_produces_a_real_mapping_report() -> None:
    report, lexical_index, _ = ingest_tenant("dealer_ar", TENANTS_DIR, EMBEDDING_CACHE_PATH)

    assert report.tenant_id == "dealer_ar"
    assert report.source_record_counts["invoices"] == 7
    assert report.mapped_count > 0
    assert report.unmapped_count > 0  # the memo/note columns have no ontology field
    assert 0.0 < report.auto_mapped_percentage < 100.0

    dealer_chunks = [c for c in lexical_index.chunks.values() if c.tenant_id == "dealer_ar"]
    assert len(dealer_chunks) == report.indexed_chunk_count
    assert report.indexed_chunk_count > 0


def test_ingest_reports_unmapped_columns_by_name() -> None:
    report, _, _ = ingest_tenant("dealer_ar", TENANTS_DIR, EMBEDDING_CACHE_PATH)

    unmapped_fields = {entry.field for entry in report.mapping.unmapped()}
    assert "memo" in unmapped_fields
    assert "note" in unmapped_fields


def test_ingest_reports_ambiguous_dates_as_needing_review() -> None:
    report, _, _ = ingest_tenant("dealer_ar", TENANTS_DIR, EMBEDDING_CACHE_PATH)

    review_reasons = {entry.reason for entry in report.mapping.needs_review()}
    assert "ambiguous date format" in review_reasons


def test_ingesting_two_tenants_keeps_their_chunks_isolated() -> None:
    dealer_report, dealer_index, _ = ingest_tenant("dealer_ar", TENANTS_DIR, EMBEDDING_CACHE_PATH)
    claims_report, claims_index, _ = ingest_tenant(
        "claims_intake", TENANTS_DIR, EMBEDDING_CACHE_PATH
    )

    assert dealer_report.tenant_id != claims_report.tenant_id
    dealer_chunks = {c.chunk_id for c in dealer_index.chunks.values() if c.tenant_id == "dealer_ar"}
    claims_chunks = {
        c.chunk_id for c in claims_index.chunks.values() if c.tenant_id == "claims_intake"
    }
    assert dealer_chunks
    assert claims_chunks
    assert dealer_chunks.isdisjoint(claims_chunks)


def test_mapping_outcome_helpers_are_consistent_with_the_report() -> None:
    report, _, _ = ingest_tenant("dealer_ar", TENANTS_DIR, EMBEDDING_CACHE_PATH)

    total_entries = len(report.mapping.entries)
    outcome_counts = sum(
        1
        for entry in report.mapping.entries
        if entry.outcome
        in {MappingOutcome.MAPPED, MappingOutcome.NEEDS_REVIEW, MappingOutcome.UNMAPPED}
    )
    assert total_entries == outcome_counts
    assert total_entries == report.mapped_count + report.needs_review_count + report.unmapped_count


def test_cli_index_command_reports_the_same_numbers_as_ingest_tenant() -> None:
    from typer.testing import CliRunner

    from outpost.onboard.cli import app

    expected, _, _ = ingest_tenant("dealer_ar", TENANTS_DIR, EMBEDDING_CACHE_PATH)

    result = CliRunner().invoke(app, ["index", "dealer_ar"])

    assert result.exit_code == 0
    assert "dealer_ar" in result.stdout
    assert f"fields mapped: {expected.mapped_count}" in result.stdout
    assert "memo" in result.stdout
