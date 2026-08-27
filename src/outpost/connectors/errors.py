"""Typed errors for connector failure modes.

Each carries enough detail (record id, status, timeout) for a caller to
log or retry without parsing a message string.
"""


class ConnectorError(Exception):
    """Base class for all connector errors."""


class RecordFetchError(ConnectorError):
    """A record could not be fetched for a reason that is not rate
    limiting or a timeout, e.g. a 404 or a 500."""

    def __init__(self, *, record_id: str, status: int, detail: str) -> None:
        self.record_id = record_id
        self.status = status
        self.detail = detail
        super().__init__(f"failed to fetch {record_id!r}: {status} {detail}")


class RateLimitedError(ConnectorError):
    """The source rejected a request with a rate limit response."""

    def __init__(self, *, record_id: str, retry_after_seconds: int | None) -> None:
        self.record_id = record_id
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"rate limited fetching {record_id!r}, retry after {retry_after_seconds}s")


class RequestTimedOutError(ConnectorError):
    """A request exceeded the connector's configured deadline."""

    def __init__(self, *, record_id: str, timeout_seconds: float) -> None:
        self.record_id = record_id
        self.timeout_seconds = timeout_seconds
        super().__init__(f"timed out fetching {record_id!r} after {timeout_seconds}s")
