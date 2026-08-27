"""Append-only audit log.

Every request's plan, tool calls, citations, and rung is written here.
There is no update or delete anywhere in this class: the only way to
change what happened on a past request is to make a new request.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from outpost.agent.ground import Citation
from outpost.agent.plan import Step


@dataclass(frozen=True)
class AuditRecord:
    request_id: str
    tenant_id: str
    request_text: str
    steps: list[Step]
    final_content: str | None
    citations: list[Citation]
    unsupported_assertions: list[str]
    rung: int | None
    created_at: str


class AuditLog:
    def __init__(self, db_path: Path) -> None:
        self._connection = sqlite3.connect(db_path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                request_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                request_text TEXT NOT NULL,
                steps TEXT NOT NULL,
                final_content TEXT,
                citations TEXT NOT NULL,
                unsupported_assertions TEXT NOT NULL,
                rung INTEGER,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def append(self, record: AuditRecord) -> None:
        self._connection.execute(
            "INSERT INTO audit_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.request_id,
                record.tenant_id,
                record.request_text,
                json.dumps([_step_to_dict(step) for step in record.steps]),
                record.final_content,
                json.dumps([citation.model_dump() for citation in record.citations]),
                json.dumps(record.unsupported_assertions),
                record.rung,
                record.created_at,
            ),
        )
        self._connection.commit()

    def get(self, request_id: str) -> AuditRecord | None:
        row = self._connection.execute(
            "SELECT * FROM audit_log WHERE request_id = ?", (request_id,)
        ).fetchone()
        return None if row is None else _row_to_record(row)


def _step_to_dict(step: Step) -> dict[str, Any]:
    return {"tool_name": step.tool_name, "arguments": step.arguments, "result": step.result}


def _row_to_record(row: tuple[Any, ...]) -> AuditRecord:
    (
        request_id,
        tenant_id,
        request_text,
        steps_json,
        final_content,
        citations_json,
        unsupported_json,
        rung,
        created_at,
    ) = row
    steps = [
        Step(tool_name=raw["tool_name"], arguments=raw["arguments"], result=raw["result"])
        for raw in json.loads(steps_json)
    ]
    citations = [Citation.model_validate(raw) for raw in json.loads(citations_json)]
    return AuditRecord(
        request_id=request_id,
        tenant_id=tenant_id,
        request_text=request_text,
        steps=steps,
        final_content=final_content,
        citations=citations,
        unsupported_assertions=json.loads(unsupported_json),
        rung=rung,
        created_at=created_at,
    )
