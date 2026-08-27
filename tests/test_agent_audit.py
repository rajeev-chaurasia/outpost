"""Audit log done-tests: a single request's full decision path (steps,
final answer, citations, unsupported assertions) round-trips through the
append-only log with no gaps, and there is no update or delete path to
lose along the way.
"""

from dataclasses import replace
from pathlib import Path

from outpost.agent.audit import AuditLog, AuditRecord
from outpost.agent.ground import Citation
from outpost.agent.plan import Step
from outpost.retrieval.document import Span


def _record(request_id: str = "req-1") -> AuditRecord:
    return AuditRecord(
        request_id=request_id,
        tenant_id="dealer_ar",
        request_text="has invoice INV-1001 been paid?",
        steps=[
            Step(
                tool_name="search",
                arguments={"query": "INV-1001"},
                result=[{"source_id": "statements", "document_id": "d1", "text": "paid"}],
            )
        ],
        final_content="Invoice INV-1001 was paid in full.",
        citations=[
            Citation(
                assertion="Invoice INV-1001 was paid in full.",
                span=Span(source_id="statements", document_id="d1", start=0, end=4, text="paid"),
            )
        ],
        unsupported_assertions=["some unrelated aside"],
        rung=None,
        created_at="2026-08-27T00:00:00+00:00",
    )


def test_appended_record_reconstructs_with_no_gaps(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.sqlite")
    original = _record()
    log.append(original)

    reloaded = log.get("req-1")

    assert reloaded is not None
    assert reloaded.request_id == original.request_id
    assert reloaded.tenant_id == original.tenant_id
    assert reloaded.request_text == original.request_text
    assert reloaded.steps == original.steps
    assert reloaded.final_content == original.final_content
    assert reloaded.citations == original.citations
    assert reloaded.unsupported_assertions == original.unsupported_assertions
    assert reloaded.rung == original.rung
    assert reloaded.created_at == original.created_at


def test_missing_request_id_returns_none(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.sqlite")
    assert log.get("does-not-exist") is None


def test_multiple_requests_are_all_retrievable_independently(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.sqlite")
    log.append(_record("req-1"))
    log.append(_record("req-2"))

    assert log.get("req-1") is not None
    assert log.get("req-2") is not None
    assert log.get("req-1").request_id != log.get("req-2").request_id  # type: ignore[union-attr]


def test_audit_log_class_exposes_no_update_or_delete_method() -> None:
    public_methods = {name for name in dir(AuditLog) if not name.startswith("_")}
    assert public_methods == {"append", "get", "list_by_tenant"}


def test_list_by_tenant_returns_only_that_tenants_records_newest_first(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.sqlite")
    dealer_record = _record("req-1")
    other_record = replace(dealer_record, request_id="req-2", tenant_id="claims_intake")
    later_dealer_record = replace(
        dealer_record, request_id="req-3", created_at="2026-08-27T01:00:00+00:00"
    )
    log.append(dealer_record)
    log.append(other_record)
    log.append(later_dealer_record)

    results = log.list_by_tenant("dealer_ar")

    assert [record.request_id for record in results] == ["req-3", "req-1"]
    assert all(record.tenant_id == "dealer_ar" for record in results)


def test_list_by_tenant_respects_limit(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.sqlite")
    for i in range(5):
        log.append(_record(f"req-{i}"))

    assert len(log.list_by_tenant("dealer_ar", limit=2)) == 2
