# claims_intake fixtures

Same idea as dealer_ar's fixtures: genuinely messy, so nothing here
should crash the pipeline and nothing should be silently dropped.

## claims.csv

- Header row uses the alias forms from config.yaml ("Claim #", "Filed
  Date", "Amount Requested").
- `CLM-2001`: amount is `$4,500.00`, symbol and thousands separator.
- `CLM-2002`: claim number has leading and trailing whitespace, filed
  date is `21/02/2026` (day 21, unambiguous).
- `CLM-2003`: filed date is the literal string `NULL`.
- `CLM-2004`: filed date is `05/03/2026`, ambiguous, amount is a bare
  number.
- `CLM-2005` appears twice with different amounts (one empty, one
  `950.00`).
- `adjuster_comment` is an extra column with no ontology field.

## adjuster_notes.csv

- File is saved with a byte-order mark.
- `NOTE-2`: note date is `22/02/2026` (day 22, unambiguous).
- `NOTE-3`: note date is `NULL`, claim number has leading and trailing
  whitespace.
- `NOTE-4`: note date is `03/03/2026`, ambiguous.
- `internal_flag` is an extra column with no ontology field.

## policies/

- `policy_clean.txt`: a normal, clean extraction.
- `policy_scan.txt`: broken line wrapping and the same `1`/`l` OCR
  confusion as dealer_ar's scanned statement.
