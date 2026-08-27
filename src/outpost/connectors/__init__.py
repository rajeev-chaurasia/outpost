"""Connectors that read tenant data sources into RawRecords."""

from outpost.connectors.base import Connector, RawRecord
from outpost.connectors.csv_export import CsvExportConnector
from outpost.connectors.errors import (
    ConnectorError,
    RateLimitedError,
    RecordFetchError,
    RequestTimedOutError,
)
from outpost.connectors.pdf_text import PdfTextConnector
from outpost.connectors.rest_mock import MockResponse, RestMockConnector

__all__ = [
    "Connector",
    "ConnectorError",
    "CsvExportConnector",
    "MockResponse",
    "PdfTextConnector",
    "RateLimitedError",
    "RawRecord",
    "RecordFetchError",
    "RequestTimedOutError",
    "RestMockConnector",
]
