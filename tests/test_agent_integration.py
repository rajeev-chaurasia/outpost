"""End-to-end done-tests against real recorded model behavior (not a
hand-scripted fake): a real gpt-oss-120b turn that calls search produces
an answer whose citations resolve to real spans, and a real turn that
calls a declined write action shows the refusal in the audit log. Both
replay committed fixtures; neither touches the network.
"""

from pathlib import Path

from outpost.agent.audit import AuditLog
from outpost.agent.handle import handle_request
from outpost.agent.tools import ActionGatedTool, FlagDiscrepancyTool, SearchTool
from outpost.llm.recorded import RecordedProvider
from outpost.retrieval.dense import DenseStore
from outpost.retrieval.lexical import BM25Index

REPO_ROOT = Path(__file__).resolve().parents[1]
TENANTS_DIR = REPO_ROOT / "tenants"
LLM_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "llm"
EMBEDDING_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "embeddings" / "retrieval.npz"

SYSTEM_PROMPT = (
    "You are a helpful assistant for a dealership accounts receivable team. "
    "Use the search tool to find relevant statement text before answering. "
    "Answer only using information the search tool returns, in one or two "
    "short sentences."
)
USER_REQUEST = "According to the account statements, was invoice INV-1001 paid, and how?"


def _dealer_ar_index() -> tuple[BM25Index, DenseStore]:
    from eval.isolation.adversarial import build_multi_tenant_index

    return build_multi_tenant_index(EMBEDDING_CACHE_PATH)


def test_real_model_search_answer_grounds_with_real_citations(tmp_path: Path) -> None:
    lexical_index, dense_store = _dealer_ar_index()
    search_tool = SearchTool(
        lexical_index=lexical_index, dense_store=dense_store, tenant_id="dealer_ar"
    )
    provider = RecordedProvider(fixtures_dir=LLM_FIXTURES_DIR, model="openai/gpt-oss-120b")
    audit_log = AuditLog(tmp_path / "audit.sqlite")

    result = handle_request(
        provider,
        {"search": search_tool},
        audit_log,
        tenant_id="dealer_ar",
        system_prompt=SYSTEM_PROMPT,
        user_request=USER_REQUEST,
    )

    assert result.plan.steps[0].tool_name == "search"
    assert result.plan.final_content is not None
    assert result.grounding.citations
    for citation in result.grounding.citations:
        assert citation.span.text  # a real, non-empty source span backs every citation
        assert citation.span.source_id == "statements"

    stored = audit_log.get(result.request_id)
    assert stored is not None
    assert stored.citations == result.grounding.citations


def test_real_model_declined_write_action_is_refused_and_audited(tmp_path: Path) -> None:
    gated_tool = ActionGatedTool(tool=FlagDiscrepancyTool(), allowed_actions=frozenset())
    provider = RecordedProvider(fixtures_dir=LLM_FIXTURES_DIR, model="openai/gpt-oss-120b")
    audit_log = AuditLog(tmp_path / "audit.sqlite")

    result = handle_request(
        provider,
        {"flag_discrepancy": gated_tool},
        audit_log,
        tenant_id="dealer_ar",
        system_prompt=(
            "You are a helpful assistant for a dealership accounts receivable team. "
            "If asked to flag a discrepancy, call the flag_discrepancy tool with the "
            "entity_key and reason."
        ),
        user_request="Invoice INV-1005 was short paid by $650. Please flag this discrepancy.",
    )

    assert result.plan.steps[0].tool_name == "flag_discrepancy"
    assert result.plan.steps[0].result["executed"] is False

    stored = audit_log.get(result.request_id)
    assert stored is not None
    assert stored.steps[0].result["executed"] is False
    assert "not in the tenant's allowed actions" in stored.steps[0].result["reason"]
