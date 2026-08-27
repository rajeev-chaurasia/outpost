"""CSV connector done-tests: every corrupted dealer_ar fixture ingests
without raising, a byte-order mark is stripped transparently, and raw
values are handed back untouched (stripping and coercion are mapping's
job, not the connector's).
"""

from pathlib import Path

from outpost.connectors.csv_export import CsvExportConnector

FIXTURES = Path(__file__).resolve().parents[1] / "tenants" / "dealer_ar" / "fixtures"


def test_reads_every_row_without_raising() -> None:
    records = CsvExportConnector(source_id="invoices", path=FIXTURES / "invoices.csv").read()
    assert len(records) == 7
    assert records[0].source_id == "invoices"
    assert records[0].row == 1


def test_preserves_raw_values_including_whitespace() -> None:
    records = CsvExportConnector(source_id="invoices", path=FIXTURES / "invoices.csv").read()
    padded = records[1]
    assert padded.fields["Invoice #"] == " INV-1002 "


def test_strips_byte_order_mark_from_first_header() -> None:
    records = CsvExportConnector(source_id="payments", path=FIXTURES / "payments.csv").read()
    assert len(records) == 5
    assert "pay_id" in records[0].fields
    assert "﻿pay_id" not in records[0].fields
