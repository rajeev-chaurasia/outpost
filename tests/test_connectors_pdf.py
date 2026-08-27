"""PDF text connector done-tests: every corrupted statement and policy
fixture (broken line wrapping, OCR digit/letter confusion, a repeated
page header) reads without raising, and no document's text is dropped.
"""

from pathlib import Path

from outpost.connectors.pdf_text import PdfTextConnector

TENANTS_DIR = Path(__file__).resolve().parents[1] / "tenants"


def test_reads_every_dealer_ar_statement_without_raising() -> None:
    records = PdfTextConnector(
        source_id="statements", path=TENANTS_DIR / "dealer_ar" / "fixtures" / "statements"
    ).read()
    document_ids = {record.fields["document_id"] for record in records}
    assert document_ids == {"statement_clean", "statement_repeated_header", "statement_scan"}
    assert all(record.fields["text"].strip() for record in records)


def test_reads_every_claims_intake_policy_without_raising() -> None:
    records = PdfTextConnector(
        source_id="policies", path=TENANTS_DIR / "claims_intake" / "fixtures" / "policies"
    ).read()
    document_ids = {record.fields["document_id"] for record in records}
    assert document_ids == {"policy_clean", "policy_scan"}
    assert all(record.fields["text"].strip() for record in records)
