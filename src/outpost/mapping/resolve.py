"""Resolves a source's raw column headers to ontology fields using a
tenant's declared aliases, then coerces each resolved value.

Nothing here guesses: a header that matches no known field or alias, and
a value that cannot be coerced with confidence, both go into the
MappingReport instead of being dropped or faked.
"""

from typing import Any

from outpost.connectors.base import RawRecord
from outpost.mapping.coerce import classify_field_kind, coerce_value
from outpost.mapping.report import MappingEntry, MappingOutcome, MappingReport


def _resolve_field_name(
    header: str, fields: list[str], field_map: dict[str, list[str]]
) -> str | None:
    normalized = header.strip().lower()
    for candidate in fields:
        if candidate.lower() == normalized:
            return candidate
    for field_name, aliases in field_map.items():
        if any(alias.strip().lower() == normalized for alias in aliases):
            return field_name
    return None


def resolve_records(
    records: list[RawRecord],
    *,
    entity_fields: list[str],
    field_map: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], MappingReport]:
    report = MappingReport()
    mapped_rows: list[dict[str, Any]] = []

    for record in records:
        mapped_row: dict[str, Any] = {}
        for header, raw_value in record.fields.items():
            resolved_field = _resolve_field_name(header, entity_fields, field_map)
            if resolved_field is None:
                report.add(
                    MappingEntry(
                        source_id=record.source_id,
                        row=record.row,
                        field=header,
                        outcome=MappingOutcome.UNMAPPED,
                        raw_value=raw_value,
                        reason="no matching ontology field or alias",
                    )
                )
                continue

            kind = classify_field_kind(resolved_field)
            result = coerce_value(kind, raw_value)
            report.add(
                MappingEntry(
                    source_id=record.source_id,
                    row=record.row,
                    field=resolved_field,
                    outcome=result.outcome,
                    raw_value=raw_value,
                    value=result.value,
                    reason=result.reason,
                )
            )
            if result.outcome is MappingOutcome.MAPPED:
                mapped_row[resolved_field] = result.value

        mapped_rows.append(mapped_row)

    return mapped_rows, report
