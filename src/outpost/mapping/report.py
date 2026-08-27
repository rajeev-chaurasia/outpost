"""Mapping outcomes and the report that collects them.

MappingReport is what a tenant admin reads to see exactly what did and
did not make it from raw source data into the ontology, and why.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MappingOutcome(StrEnum):
    """What happened when a tenant's raw field was resolved to an ontology field."""

    MAPPED = "mapped"
    NEEDS_REVIEW = "needs_review"
    UNMAPPED = "unmapped"


@dataclass(frozen=True)
class MappingEntry:
    """The outcome for one field in one row from one source."""

    source_id: str
    row: int | str
    field: str
    outcome: MappingOutcome
    raw_value: str | None
    value: Any = None
    reason: str | None = None


@dataclass
class MappingReport:
    """Every field outcome for one ingest run, across every source."""

    entries: list[MappingEntry] = field(default_factory=list)

    def add(self, entry: MappingEntry) -> None:
        self.entries.append(entry)

    def mapped(self) -> list[MappingEntry]:
        return [entry for entry in self.entries if entry.outcome is MappingOutcome.MAPPED]

    def needs_review(self) -> list[MappingEntry]:
        return [entry for entry in self.entries if entry.outcome is MappingOutcome.NEEDS_REVIEW]

    def unmapped(self) -> list[MappingEntry]:
        return [entry for entry in self.entries if entry.outcome is MappingOutcome.UNMAPPED]
