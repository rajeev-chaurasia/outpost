"""Computes the unsupported-assertion rate per tenant from recorded
agent scenarios, and writes the result to a committed artifact.

Report per tenant, never pooled: a weak tenant should never hide behind
a strong one.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

from eval.grounding.scenarios import SCENARIOS
from eval.isolation.adversarial import EMBEDDING_CACHE_PATH, build_multi_tenant_index
from outpost.agent.audit import AuditLog
from outpost.agent.handle import handle_request
from outpost.agent.tools import SearchTool
from outpost.llm.recorded import RecordedProvider

REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "llm"
ARTIFACT_PATH = REPO_ROOT / "eval" / "artifacts" / "grounding_results.json"
MODEL = "openai/gpt-oss-120b"


def score(db_path: Path) -> dict[str, dict[str, Any]]:
    lexical_index, dense_store = build_multi_tenant_index(EMBEDDING_CACHE_PATH)
    provider = RecordedProvider(fixtures_dir=LLM_FIXTURES_DIR, model=MODEL)
    audit_log = AuditLog(db_path)

    results: dict[str, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        search_tool = SearchTool(
            lexical_index=lexical_index, dense_store=dense_store, tenant_id=scenario.tenant_id
        )
        result = handle_request(
            provider,
            {"search": search_tool},
            audit_log,
            tenant_id=scenario.tenant_id,
            system_prompt=scenario.system_prompt,
            user_request=scenario.user_request,
        )
        results[scenario.tenant_id] = {
            "request": scenario.user_request,
            "answer": result.plan.final_content,
            "citation_count": len(result.grounding.citations),
            "unsupported_count": len(result.grounding.unsupported_assertions),
            "unsupported_assertions": result.grounding.unsupported_assertions,
            "unsupported_rate": result.grounding.unsupported_rate,
        }
    return results


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = score(Path(tmp_dir) / "audit.sqlite")

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
