"""Done-test: swapping the model is a config change, not a code change.

Runs the exact same dealer_ar grounding request through two different
recorded models (gpt-oss-120b and gpt-oss-20b) via the same
OpenAICompatibleProvider class, changing only the model string, and
asserts both produce a structurally equivalent, fully grounded answer.
"""

from pathlib import Path

from eval.grounding.scenarios import SCENARIOS
from outpost.agent.audit import AuditLog
from outpost.agent.handle import handle_request
from outpost.agent.tools import SearchTool
from outpost.llm.recorded import RecordedProvider
from outpost.retrieval.dense import DenseStore
from outpost.retrieval.lexical import BM25Index

REPO_ROOT = Path(__file__).resolve().parents[1]
LLM_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "llm"
EMBEDDING_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "embeddings" / "retrieval.npz"

DEALER_AR_SCENARIO = next(s for s in SCENARIOS if s.tenant_id == "dealer_ar")
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


def _dealer_ar_index() -> tuple[BM25Index, DenseStore]:
    from eval.isolation.adversarial import build_multi_tenant_index

    return build_multi_tenant_index(EMBEDDING_CACHE_PATH)


def test_swapping_the_model_string_still_produces_a_grounded_answer(tmp_path: Path) -> None:
    for model in MODELS:
        lexical_index, dense_store = _dealer_ar_index()
        search_tool = SearchTool(
            lexical_index=lexical_index, dense_store=dense_store, tenant_id="dealer_ar"
        )
        provider = RecordedProvider(fixtures_dir=LLM_FIXTURES_DIR, model=model)
        audit_log = AuditLog(tmp_path / f"audit-{model.replace('/', '_')}.sqlite")

        result = handle_request(
            provider,
            {"search": search_tool},
            audit_log,
            tenant_id="dealer_ar",
            system_prompt=DEALER_AR_SCENARIO.system_prompt,
            user_request=DEALER_AR_SCENARIO.user_request,
        )

        assert result.plan.steps[0].tool_name == "search", model
        assert result.answer is not None, model
        assert result.grounding.citations, model
        assert not result.grounding.unsupported_assertions, model
        for citation in result.grounding.citations:
            assert citation.span.source_id == "statements", model
