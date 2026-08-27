"""CSV connector.

Reads a csv file into RawRecords as-is: whatever headers and values the
file actually has, byte-order mark and all. Resolving a header to an
ontology field, and deciding whether a value is usable, are mapping's job,
not this connector's.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from outpost.connectors.base import RawRecord


@dataclass(frozen=True)
class CsvExportConnector:
    source_id: str
    path: Path

    def read(self) -> list[RawRecord]:
        # utf-8-sig strips a leading byte-order mark if present and behaves
        # like plain utf-8 otherwise, so one connector handles both.
        with self.path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return [
                RawRecord(source_id=self.source_id, row=index, fields=dict(row))
                for index, row in enumerate(reader, start=1)
            ]
