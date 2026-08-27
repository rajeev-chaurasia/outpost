"""Connectors that read tenant data sources into RawRecords."""

from outpost.connectors.base import Connector, RawRecord
from outpost.connectors.errors import (
    ConnectorError,
    RateLimitedError,
    RecordFetchError,
    RequestTimedOutError,
)

__all__ = [
    "Connector",
    "ConnectorError",
    "RateLimitedError",
    "RawRecord",
    "RecordFetchError",
    "RequestTimedOutError",
]
