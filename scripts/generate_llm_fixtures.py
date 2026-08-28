"""One-off dev tool: runs real requests through the live nvidia model
against a real tenant index, and records each turn's request/response
pair to tests/fixtures/llm/, keyed the same way RecordedProvider looks
them up, so tests and CI replay real model behavior deterministically.

Run locally with LLM_API_KEY set. Never invoked by tests or CI.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from eval.grounding.scenarios import SCENARIOS, GroundingScenario  # noqa: E402
from eval.grounding.suite import load_cases as load_grounding_cases  # noqa: E402
from eval.grounding.suite import system_prompt_for  # noqa: E402
from outpost.agent.plan import PlanResult  # noqa: E402
from outpost.agent.plan import run as run_plan  # noqa: E402
from outpost.agent.tools import ActionGatedTool, FlagDiscrepancyTool, SearchTool  # noqa: E402
from outpost.llm.base import Completion, Message, ToolSpec  # noqa: E402
from outpost.llm.errors import ProviderError  # noqa: E402
from outpost.llm.openai_compatible import OpenAICompatibleProvider  # noqa: E402
from outpost.llm.recorded import request_key  # noqa: E402
from outpost.ontology import discover_tenant_ids  # noqa: E402
from outpost.retrieval.build import build_multi_tenant_index  # noqa: E402
from outpost.retrieval.dense import DenseStore, EmbeddingCache, NvidiaEmbeddingClient  # noqa: E402
from outpost.retrieval.errors import EmbeddingCacheMissError  # noqa: E402
from outpost.retrieval.lexical import BM25Index  # noqa: E402

TENANTS_DIR = REPO_ROOT / "tenants"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "llm"
EMBEDDING_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "embeddings" / "retrieval.npz"
MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"


def _build_index() -> tuple[BM25Index, DenseStore]:
    return build_multi_tenant_index(
        discover_tenant_ids(TENANTS_DIR), TENANTS_DIR, EmbeddingCache.load(EMBEDDING_CACHE_PATH)
    )


def _run_filling_embedding_gaps(
    run_once: "callable[[], PlanResult]",
    dense_store: DenseStore,
    embedding_client: NvidiaEmbeddingClient,
    max_attempts: int = 20,
) -> PlanResult:
    """The agent's own model decides what to search for, so the exact
    query text isn't known ahead of time. On a cache miss this embeds
    the missing text live, persists it, and retries, rather than trying
    to pre-guess every phrasing the model might choose.

    The retry budget is generous because a question the corpus cannot
    answer is the expensive case: the model reformulates its search
    several times before giving up, and each new phrasing is another
    cache miss.
    """
    for _ in range(max_attempts):
        try:
            return run_once()
        except EmbeddingCacheMissError as exc:
            vector = embedding_client.embed([exc.text], exc.input_type)[0]  # type: ignore[arg-type]
            dense_store.cache.put(exc.text, exc.input_type, vector)  # type: ignore[arg-type]
            dense_store.cache.save(EMBEDDING_CACHE_PATH)
    raise RuntimeError("exceeded retries filling embedding cache gaps")


@dataclass
class RecordingProvider:
    """Wraps a real provider; every call is written to a fixture file
    keyed the same way RecordedProvider looks them up.
    """

    inner: OpenAICompatibleProvider
    fixtures_dir: Path

    def complete(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None
    ) -> Completion:
        completion = self.inner.complete(messages, tools=tools)
        key = request_key(self.inner.model, messages, tools)
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "response": {
                "content": completion.content,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in completion.tool_calls
                ],
                "usage": {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                },
            }
        }
        (self.fixtures_dir / f"{key}.json").write_text(json.dumps(payload, indent=2))
        return completion


def _record_one_scenario(
    scenario: GroundingScenario,
    lexical_index: BM25Index,
    dense_store: DenseStore,
    embedding_client: NvidiaEmbeddingClient,
    model: str = MODEL,
) -> None:
    search_tool = SearchTool(
        lexical_index=lexical_index, dense_store=dense_store, tenant_id=scenario.tenant_id
    )
    provider = RecordingProvider(
        inner=OpenAICompatibleProvider(model=model, timeout_seconds=90.0),
        fixtures_dir=FIXTURES_DIR,
    )

    def run_once() -> PlanResult:
        return run_plan(
            provider,
            {"search": search_tool},
            system_prompt=scenario.system_prompt,
            user_request=scenario.user_request,
        )

    result = _run_filling_embedding_gaps(run_once, dense_store, embedding_client)
    print(f"[{scenario.tenant_id}] steps:", result.steps)
    print(f"[{scenario.tenant_id}] final_content:", result.final_content)


def record_grounding_scenarios() -> None:
    """Records one search-then-answer turn per tenant from
    eval.grounding.scenarios, the same source eval/grounding/suite.py
    reads, so a fixture is always generated for the exact wording that
    will later be replayed.
    """
    lexical_index, dense_store = _build_index()
    embedding_client = NvidiaEmbeddingClient()

    for scenario in SCENARIOS:
        _record_one_scenario(scenario, lexical_index, dense_store, embedding_client)


def record_dealer_ar_with_fallback_model() -> None:
    """Records the dealer_ar grounding scenario again with the fallback
    model, proving a model swap is a config change: the same request
    goes through OpenAICompatibleProvider with nothing touched but the
    model string, and gpt-oss-20b answers it the same way gpt-oss-120b
    does, structurally.
    """
    lexical_index, dense_store = _build_index()
    embedding_client = NvidiaEmbeddingClient()
    dealer_ar_scenario = next(s for s in SCENARIOS if s.tenant_id == "dealer_ar")
    _record_one_scenario(
        dealer_ar_scenario, lexical_index, dense_store, embedding_client, model=FALLBACK_MODEL
    )


def record_declined_write_action() -> None:
    tool = ActionGatedTool(tool=FlagDiscrepancyTool(), allowed_actions=frozenset())
    provider = RecordingProvider(
        inner=OpenAICompatibleProvider(model=MODEL, timeout_seconds=90.0), fixtures_dir=FIXTURES_DIR
    )

    result = run_plan(
        provider,
        {"flag_discrepancy": tool},
        system_prompt=(
            "You are a helpful assistant for a dealership accounts receivable team. "
            "If asked to flag a discrepancy, call the flag_discrepancy tool with the "
            "entity_key and reason."
        ),
        user_request="Invoice INV-1005 was short paid by $650. Please flag this discrepancy.",
    )
    print("declined_write_action steps:", result.steps)
    print("declined_write_action final_content:", result.final_content)


def record_grounding_suite() -> None:
    """Records every question in eval/grounding/cases.yaml, including the
    ones the corpus cannot answer, since a refusal has to be recorded to
    be replayed.
    """
    lexical_index, dense_store = _build_index()
    embedding_client = NvidiaEmbeddingClient()

    for case in load_grounding_cases():
        scenario = GroundingScenario(
            tenant_id=case.tenant_id,
            system_prompt=system_prompt_for(case.tenant_id),
            user_request=case.question,
        )
        # One case failing must not abandon the rest, so the run can be
        # repeated to fill whatever is still missing.
        try:
            _record_one_scenario(scenario, lexical_index, dense_store, embedding_client)
        except (RuntimeError, ProviderError) as exc:
            print(f"[{case.tenant_id}] SKIPPED {case.question!r}: {exc}")


if __name__ == "__main__":
    record_grounding_scenarios()
    record_dealer_ar_with_fallback_model()
    record_declined_write_action()
