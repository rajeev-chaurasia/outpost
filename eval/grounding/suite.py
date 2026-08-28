"""Runs the full grounding question set through the real agent and scores
it per tenant.

Two rates, because either alone is gameable. A grounder scores zero
unsupported by answering only what it is sure of, and scores a perfect
refusal rate by refusing everything, so both are reported together:

unsupported_rate
    over sentences of answered questions, how many no retrieved span
    supports.
correct_refusal_rate
    over questions whose answer is not in the corpus, how many the agent
    actually refused instead of producing something.

Replays committed fixtures, so this needs no api key and no network.
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from eval.grounding.scenarios import SCENARIOS
from outpost.agent.audit import AuditLog
from outpost.agent.degrade import Rung
from outpost.agent.handle import handle_request
from outpost.agent.tools import SearchTool
from outpost.llm.recorded import RecordedProvider
from outpost.ontology import discover_tenant_ids
from outpost.retrieval.build import build_multi_tenant_index
from outpost.retrieval.dense import EmbeddingCache

REPO_ROOT = Path(__file__).resolve().parents[2]
TENANTS_DIR = REPO_ROOT / "tenants"
CASES_PATH = Path(__file__).resolve().parent / "cases.yaml"
LLM_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "llm"
EMBEDDING_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "embeddings" / "retrieval.npz"
ARTIFACT_PATH = REPO_ROOT / "eval" / "artifacts" / "grounding_results.json"
MODEL = "openai/gpt-oss-120b"

# The per-tenant prompts already used to record fixtures. Reused rather
# than redefined so a prompt change cannot silently invalidate the
# recorded set.
_PROMPTS = {scenario.tenant_id: scenario.system_prompt for scenario in SCENARIOS}


@dataclass(frozen=True)
class GroundingCase:
    tenant_id: str
    kind: str
    question: str


def load_cases() -> list[GroundingCase]:
    raw = yaml.safe_load(CASES_PATH.read_text())
    return [GroundingCase(**case) for case in raw["cases"]]


def system_prompt_for(tenant_id: str) -> str:
    return _PROMPTS[tenant_id]


def run(db_path: Path, cases: list[GroundingCase]) -> dict[str, Any]:
    lexical_index, dense_store = build_multi_tenant_index(
        discover_tenant_ids(TENANTS_DIR), TENANTS_DIR, EmbeddingCache.load(EMBEDDING_CACHE_PATH)
    )
    provider = RecordedProvider(fixtures_dir=LLM_FIXTURES_DIR, model=MODEL)
    audit_log = AuditLog(db_path)

    per_tenant: dict[str, dict[str, Any]] = {}
    for case in cases:
        bucket = per_tenant.setdefault(
            case.tenant_id,
            {
                "answerable": 0,
                "refusable": 0,
                "citations": 0,
                "unsupported": 0,
                "correct_refusals": 0,
                "answered_when_it_should_have_refused": [],
                "refused_when_it_should_have_answered": [],
            },
        )
        search_tool = SearchTool(
            lexical_index=lexical_index, dense_store=dense_store, tenant_id=case.tenant_id
        )
        result = handle_request(
            provider,
            {"search": search_tool},
            audit_log,
            tenant_id=case.tenant_id,
            system_prompt=system_prompt_for(case.tenant_id),
            user_request=case.question,
        )
        refused = result.rung is Rung.REFUSED

        if case.kind == "refusable":
            bucket["refusable"] += 1
            if refused:
                bucket["correct_refusals"] += 1
            else:
                bucket["answered_when_it_should_have_refused"].append(case.question)
            continue

        bucket["answerable"] += 1
        if refused:
            bucket["refused_when_it_should_have_answered"].append(case.question)
            continue
        bucket["citations"] += len(result.grounding.citations)
        bucket["unsupported"] += len(result.grounding.unsupported_assertions)

    summary: dict[str, Any] = {}
    for tenant_id, b in sorted(per_tenant.items()):
        assertions = b["citations"] + b["unsupported"]
        summary[tenant_id] = {
            "answerable_questions": b["answerable"],
            "refusable_questions": b["refusable"],
            "assertions_scored": assertions,
            "citations": b["citations"],
            "unsupported": b["unsupported"],
            "unsupported_rate": round(b["unsupported"] / assertions, 3) if assertions else 0.0,
            "correct_refusals": b["correct_refusals"],
            "correct_refusal_rate": (
                round(b["correct_refusals"] / b["refusable"], 3) if b["refusable"] else 0.0
            ),
            "answered_when_it_should_have_refused": b["answered_when_it_should_have_refused"],
            "refused_when_it_should_have_answered": b["refused_when_it_should_have_answered"],
        }

    totals = {
        "questions": len(cases),
        "assertions_scored": sum(t["assertions_scored"] for t in summary.values()),
        "citations": sum(t["citations"] for t in summary.values()),
        "unsupported": sum(t["unsupported"] for t in summary.values()),
        "refusable_questions": sum(t["refusable_questions"] for t in summary.values()),
        "correct_refusals": sum(t["correct_refusals"] for t in summary.values()),
    }
    totals["unsupported_rate"] = (
        round(totals["unsupported"] / totals["assertions_scored"], 3)
        if totals["assertions_scored"]
        else 0.0
    )
    totals["correct_refusal_rate"] = (
        round(totals["correct_refusals"] / totals["refusable_questions"], 3)
        if totals["refusable_questions"]
        else 0.0
    )

    return {"totals": totals, "per_tenant": summary}


def score(db_path: Path) -> dict[str, Any]:
    return run(db_path, load_cases())


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = score(Path(tmp_dir) / "audit.sqlite")
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
