"""PDF text connector.

Real text extraction happens upstream of this connector; it just reads
whatever text came out. The fixtures are plain .txt files standing in for
already-extracted pdf text, one clean and one simulating a noisy scan, so
this connector's only real job is to not choke on either.
"""

from dataclasses import dataclass
from pathlib import Path

from outpost.connectors.base import RawRecord


@dataclass(frozen=True)
class PdfTextConnector:
    source_id: str
    path: Path

    def read(self) -> list[RawRecord]:
        records = []
        for index, file_path in enumerate(sorted(self.path.glob("*.txt")), start=1):
            text = file_path.read_text(encoding="utf-8")
            records.append(
                RawRecord(
                    source_id=self.source_id,
                    row=index,
                    fields={"document_id": file_path.stem, "text": text},
                )
            )
        return records
