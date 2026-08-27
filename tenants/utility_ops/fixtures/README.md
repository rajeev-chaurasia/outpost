# utility_ops fixtures

Onboarded cold as the third tenant (phase 8's headline measurement), a
vertical not touched anywhere else in this project. Same idea as the
other two tenants' fixtures: genuinely messy, so nothing here should
crash the pipeline and nothing should be silently dropped.

## work_orders.csv

- Header row uses the alias forms from config.yaml ("WO #",
  "date_scheduled", "date_completed").
- `WO-3002`: work order id has leading and trailing whitespace,
  completed date is `21/04/2026` (day 21, unambiguous).
- `WO-3003`: completed date is the literal string `NULL`.
- `WO-3004`: completed date is `05/04/2026`, ambiguous.
- `WO-3005` appears twice with different completed dates and statuses
  (one still open, one closed).
- `crew_notes` is an extra column with no ontology field.

## technician_notes.csv

- File is saved with a byte-order mark.
- `NOTE-2`: note date is `22/04/2026` (day 22, unambiguous).
- `NOTE-3`: note date is `NULL`, work order id has leading and
  trailing whitespace.
- `NOTE-4`: note date is `04/05/2026`, ambiguous.
- `internal_flag` is an extra column with no ontology field.

## service_agreements/

- `agreement_clean.txt`: a normal, clean extraction.
- `agreement_scan.txt`: broken line wrapping and the same `1`/`l` OCR
  confusion as the other two tenants' scanned documents.
