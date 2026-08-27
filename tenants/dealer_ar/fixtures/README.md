# dealer_ar fixtures

Deliberately messy, so the connectors and mapping resolver have something
real to survive. Nothing here should crash the pipeline, and nothing
should be silently dropped: every corruption below should show up either
as a correctly coerced value or as an entry in the MappingReport.

## invoices.csv

- Header row uses the alias forms declared in config.yaml ("Invoice #",
  "Amount Due", "Invoice Date") instead of the canonical field names, to
  exercise alias resolution.
- `INV-1001`: amount is `$1,240.00`, symbol and thousands separator.
- `INV-1002`: invoice number has leading and trailing whitespace
  (" INV-1002 "), and the date is `14/03/2026` (day 14, unambiguous
  since no month goes past 12).
- `INV-1003`: amount is the literal string `NULL`.
- `INV-1004`: date is `03/04/2026`, ambiguous (day and month both valid
  either way), amount is a bare number with no symbol.
- `INV-1005` appears twice with different amounts (one empty, one
  `650.00`): an empty value and a duplicate row differing in one field.
- `memo` is an extra column with no ontology field, on every row.

## payments.csv

- File is saved with a byte-order mark.
- `PAY-3`: `received_date` is `NULL`.
- `PAY-4`: amount is `$650.00`.
- `PAY-5`: invoice number has leading and trailing whitespace.
- `note` is an extra column, populated only on the last row.

## statements/

- `statement_clean.txt`: a normal, clean extraction.
- `statement_scan.txt`: broken line wrapping and the classic OCR
  confusion between the digit `1` and the letter `l` (`lnvoice`,
  `lNV-l002`, `Ba1ance`).
- `statement_repeated_header.txt`: the same header line repeats at the
  top of every simulated page, as a multi-page PDF's letterhead would.
