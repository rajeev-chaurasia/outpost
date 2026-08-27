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

from eval.isolation.adversarial import EMBEDDING_CACHE_PATH, build_multi_tenant_index  # noqa: E402
from outpost.agent.plan import PlanResult  # noqa: E402
from outpost.agent.plan import run as run_plan  # noqa: E402
from outpost.agent.tools import ActionGatedTool, FlagDiscrepancyTool, SearchTool  # noqa: E402
from outpost.llm.base import Completion, Message, ToolSpec  # noqa: E402
from outpost.llm.openai_compatible import OpenAICompatibleProvider  # noqa: E402
from outpost.llm.recorded import request_key  # noqa: E402
from outpost.retrieval.dense import DenseStore, NvidiaEmbeddingClient  # noqa: E402
from outpost.retrieval.errors import EmbeddingCacheMissError  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "llm"
MODEL = "openai/gpt-oss-120b"


def _run_filling_embedding_gaps(
    run_once: "callable[[], PlanResult]",
    dense_store: DenseStore,
    embedding_client: NvidiaEmbeddingClient,
    max_attempts: int = 5,
) -> PlanResult:
    """The agent's own model decides what to search for, so the exact
    query text isn't known ahead of time. On a cache miss this embeds
    the missing text live, persists it, and retries, rather than trying
    to pre-guess every phrasing the model might choose.
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


def record_search_then_answer() -> None:
    lexical_index, dense_store = build_multi_tenant_index()
    search_tool = SearchTool(
        lexical_index=lexical_index, dense_store=dense_store, tenant_id="dealer_ar"
    )
    provider = RecordingProvider(
        inner=OpenAICompatibleProvider(model=MODEL, timeout_seconds=90.0), fixtures_dir=FIXTURES_DIR
    )
    embedding_client = NvidiaEmbeddingClient()

    result = _run_filling_embedding_gaps(
        lambda: run_plan(
            provider,
            {"search": search_tool},
            system_prompt=(
                "You are a helpful assistant for a dealership accounts receivable team. "
                "Use the search tool to find relevant statement text before answering. "
                "Answer only using information the search tool returns, in one or two "
                "short sentences."
            ),
            user_request="According to the account statements, was invoice INV-1001 paid, and how?",
        ),
        dense_store,
        embedding_client,
    )
    print("search_then_answer steps:", result.steps)
    print("search_then_answer final_content:", result.final_content)


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


if __name__ == "__main__":
    record_search_then_answer()
    record_declined_write_action()
