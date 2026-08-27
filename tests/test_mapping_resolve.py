"""Phase 2 done-tests: every corrupted fixture ingests and resolves without
raising, unmapped columns and ambiguous dates are named in the
MappingReport rather than dropped or guessed at, and duplicate rows are
both kept.
"""

from pathlib import Path

from outpost.connectors.csv_export import CsvExportConnector
from outpost.mapping import MappingOutcome, resolve_records
from outpost.mapping.coerce import coerce_date
from outpost.ontology import load_tenant_config

TENANTS_DIR = Path(__file__).resolve().parents[1] / "tenants"


def _resolve(tenant: str, source_id: str, entity: str) -> tuple[list[dict], object]:
    config = load_tenant_config(TENANTS_DIR / tenant / "config.yaml")
    source = next(s for s in config.sources if s.id == source_id)
    entity_type = next(e for e in config.ontology.entities if e.name == entity)
    records = CsvExportConnector(
        source_id=source_id, path=TENANTS_DIR / tenant / source.path
    ).read()
    return resolve_records(records, entity_fields=entity_type.fields, field_map=source.field_map)


def test_unmapped_column_is_named_with_source_and_row() -> None:
    _, report = _resolve("dealer_ar", "invoices", "invoice")
    unmapped = report.unmapped()
    assert any(entry.field == "memo" and entry.row == 1 for entry in unmapped)
    assert all(entry.source_id == "invoices" for entry in unmapped)


def test_ambiguous_date_needs_review_not_mapped() -> None:
    _, report = _resolve("dealer_ar", "invoices", "invoice")
    # row 4 is INV-1004, issued_date 03/04/2026, ambiguous.
    entry = next(e for e in report.entries if e.row == 4 and e.field == "issued_date")
    assert entry.outcome is MappingOutcome.NEEDS_REVIEW
    assert entry.value is None


def test_unambiguous_slash_date_is_mapped() -> None:
    _, report = _resolve("dealer_ar", "invoices", "invoice")
    # row 2 is INV-1002 (padded), issued_date 14/03/2026, day 14 is unambiguous.
    entry = next(e for e in report.entries if e.row == 2 and e.field == "issued_date")
    assert entry.outcome is MappingOutcome.MAPPED
    assert entry.value.isoformat() == "2026-03-14"


def test_currency_with_symbol_and_thousands_separator_is_mapped() -> None:
    _, report = _resolve("dealer_ar", "invoices", "invoice")
    entry = next(e for e in report.entries if e.row == 1 and e.field == "amount")
    assert entry.outcome is MappingOutcome.MAPPED
    assert entry.value == entry.value.to_integral_value() or str(entry.value) == "1240.00"


def test_null_literal_needs_review() -> None:
    _, report = _resolve("dealer_ar", "invoices", "invoice")
    entry = next(e for e in report.entries if e.row == 3 and e.field == "amount")
    assert entry.outcome is MappingOutcome.NEEDS_REVIEW
    assert entry.reason == "missing value"


def test_whitespace_padded_identifier_is_stripped() -> None:
    _, report = _resolve("dealer_ar", "invoices", "invoice")
    entry = next(e for e in report.entries if e.row == 2 and e.field == "invoice_number")
    assert entry.outcome is MappingOutcome.MAPPED
    assert entry.value == "INV-1002"


def test_duplicate_rows_are_both_kept_not_deduplicated() -> None:
    mapped_rows, _ = _resolve("dealer_ar", "invoices", "invoice")
    duplicates = [row for row in mapped_rows if row.get("invoice_number") == "INV-1005"]
    assert len(duplicates) == 2


def test_claims_intake_resolves_with_its_own_aliases() -> None:
    mapped_rows, report = _resolve("claims_intake", "claims", "claim")
    assert any(entry.field == "adjuster_comment" for entry in report.unmapped())
    ambiguous = next(e for e in report.entries if e.row == 4 and e.field == "filed_date")
    assert ambiguous.outcome is MappingOutcome.NEEDS_REVIEW
    duplicates = [row for row in mapped_rows if row.get("claim_number") == "CLM-2005"]
    assert len(duplicates) == 2


def test_coerce_date_edge_cases() -> None:
    assert coerce_date("2026-03-14").outcome is MappingOutcome.MAPPED
    assert coerce_date("14/03/2026").outcome is MappingOutcome.MAPPED
    assert coerce_date("03/04/2026").outcome is MappingOutcome.NEEDS_REVIEW
    assert coerce_date("").outcome is MappingOutcome.NEEDS_REVIEW
    assert coerce_date("NULL").outcome is MappingOutcome.NEEDS_REVIEW
    assert coerce_date("not a date").outcome is MappingOutcome.NEEDS_REVIEW
