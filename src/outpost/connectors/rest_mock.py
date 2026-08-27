"""A simulated REST connector.

Exercises the failure modes a real API integration has to survive: rate
limiting, a server error, a timeout, and a response carrying a field the
schema does not know about. The first three raise typed errors; the last
one is not a failure at all, it has to come back successfully so the
extra field can flow into the mapping report as unmapped instead of being
dropped before mapping ever sees it.
"""

from dataclasses import dataclass
from typing import Any

from outpost.connectors.base import RawRecord
from outpost.connectors.errors import RateLimitedError, RecordFetchError, RequestTimedOutError


@dataclass(frozen=True)
class MockResponse:
    """One canned response the mock connector returns for a record id."""

    status: int
    body: dict[str, Any] | None = None
    retry_after_seconds: int | None = None
    timed_out: bool = False


@dataclass
class RestMockConnector:
    """Reads a fixed set of canned responses instead of calling a real API."""

    source_id: str
    responses: dict[str, MockResponse]
    request_timeout_seconds: float = 5.0

    def fetch(self, record_id: str) -> RawRecord:
        response = self.responses.get(record_id)
        if response is None:
            raise RecordFetchError(record_id=record_id, status=404, detail="no such record")

        if response.timed_out:
            raise RequestTimedOutError(
                record_id=record_id, timeout_seconds=self.request_timeout_seconds
            )

        if response.status == 429:
            raise RateLimitedError(
                record_id=record_id, retry_after_seconds=response.retry_after_seconds
            )

        if response.status >= 500:
            raise RecordFetchError(
                record_id=record_id, status=response.status, detail="server error"
            )

        body = response.body or {}
        return RawRecord(
            source_id=self.source_id,
            row=record_id,
            fields={key: str(value) for key, value in body.items()},
        )
