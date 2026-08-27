"""REST mock connector done-tests: each of the three real failure modes
(rate limiting, a server error, a timeout) raises a distinct typed error
with the detail a caller needs to retry, and the fourth case, a response
carrying a field the schema does not know about, is not an error at all,
it fetches successfully so the unknown field can reach the mapping report
instead of being dropped before mapping ever sees it.
"""

import pytest

from outpost.connectors.errors import RateLimitedError, RecordFetchError, RequestTimedOutError
from outpost.connectors.rest_mock import MockResponse, RestMockConnector
from outpost.mapping import MappingOutcome, resolve_records


def _connector() -> RestMockConnector:
    return RestMockConnector(
        source_id="claims_api",
        responses={
            "ok-1": MockResponse(status=200, body={"claim_number": "CLM-9001", "status": "open"}),
            "rate-limited": MockResponse(status=429, retry_after_seconds=30),
            "server-error": MockResponse(status=500),
            "slow": MockResponse(status=200, timed_out=True),
            "surprise-field": MockResponse(
                status=200, body={"claim_number": "CLM-9002", "escalation_tier": "gold"}
            ),
        },
        request_timeout_seconds=2.0,
    )


def test_rate_limited_raises_with_retry_after() -> None:
    with pytest.raises(RateLimitedError) as exc_info:
        _connector().fetch("rate-limited")
    assert exc_info.value.retry_after_seconds == 30


def test_server_error_raises_record_fetch_error() -> None:
    with pytest.raises(RecordFetchError) as exc_info:
        _connector().fetch("server-error")
    assert exc_info.value.status == 500


def test_timeout_raises_with_configured_deadline() -> None:
    with pytest.raises(RequestTimedOutError) as exc_info:
        _connector().fetch("slow")
    assert exc_info.value.timeout_seconds == 2.0


def test_missing_record_raises_not_found() -> None:
    with pytest.raises(RecordFetchError) as exc_info:
        _connector().fetch("does-not-exist")
    assert exc_info.value.status == 404


def test_unknown_field_does_not_raise_and_reaches_the_mapping_report() -> None:
    record = _connector().fetch("surprise-field")
    assert record.fields["escalation_tier"] == "gold"

    _, report = resolve_records([record], entity_fields=["claim_number", "status"], field_map={})
    assert any(
        entry.field == "escalation_tier" and entry.outcome is MappingOutcome.UNMAPPED
        for entry in report.entries
    )
