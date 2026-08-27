"""Connector protocol and the raw record every connector produces."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RawRecord:
    """One row of raw, unmapped data from a source, with its position for
    error reporting.

    row is an int for file-backed sources (a csv row number, a file index)
    and a str for id-addressed sources (a rest record id), so a mapping
    report entry can always point back at exactly where the value came
    from.
    """

    source_id: str
    row: int | str
    fields: dict[str, str]


class Connector(Protocol):
    """Something that can read a tenant data source into RawRecords.

    A connector never resolves column headers to ontology fields and never
    coerces values, that is mapping's job. It only has to survive whatever
    is actually in the source without crashing or dropping a record.
    """

    def read(self) -> list[RawRecord]: ...
