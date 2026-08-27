"""Field mapping between raw tenant data and the ontology."""

from outpost.mapping.coerce import CoerceResult, classify_field_kind, coerce_value
from outpost.mapping.report import MappingEntry, MappingOutcome, MappingReport
from outpost.mapping.resolve import resolve_records

__all__ = [
    "CoerceResult",
    "MappingEntry",
    "MappingOutcome",
    "MappingReport",
    "classify_field_kind",
    "coerce_value",
    "resolve_records",
]
