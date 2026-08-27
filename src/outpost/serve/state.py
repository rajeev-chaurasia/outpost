"""Builds and holds the runtime state the API needs: the shared
multi-tenant retrieval index, each tenant's tools, the provider, and
the audit log, built once at startup rather than per request.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from outpost.agent.audit import AuditLog
from outpost.agent.tools import ActionGatedTool, DraftResponseTool, FlagDiscrepancyTool, SearchTool
from outpost.agent.tools.base import Tool
from outpost.llm.base import Provider
from outpost.llm.fallback import FallbackProvider
from outpost.llm.openai_compatible import OpenAICompatibleProvider
from outpost.ontology import TenantConfig, discover_tenant_ids, load_tenant_config
from outpost.retrieval.build import build_multi_tenant_index
from outpost.retrieval.dense import (
    DenseStore,
    EmbeddingCache,
    LiveFallbackEmbeddingCache,
    NvidiaEmbeddingClient,
)
from outpost.retrieval.lexical import BM25Index

REPO_ROOT = Path(__file__).resolve().parents[3]
TENANTS_DIR = REPO_ROOT / "tenants"
EMBEDDING_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "embeddings" / "retrieval.npz"
AUDIT_DB_PATH = REPO_ROOT / "var" / "audit.sqlite"
PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant for {display_name}. Use the search tool to find "
    "relevant text before answering. Answer only using information the search "
    "tool returns."
)


@dataclass(frozen=True)
class TenantRuntime:
    config: TenantConfig
    tools: dict[str, Tool]
    system_prompt: str


def _default_provider_factory() -> Provider:
    # A fresh FallbackProvider per call: fell_back is per-instance state,
    # and reusing one across requests would leak a fallback from an
    # earlier request into a later one's rung.
    return FallbackProvider(
        primary=OpenAICompatibleProvider(model=PRIMARY_MODEL),
        secondary=OpenAICompatibleProvider(model=FALLBACK_MODEL),
    )


@dataclass
class AppState:
    tenants: dict[str, TenantRuntime]
    audit_log: AuditLog
    # Injectable so tests can substitute a RecordedProvider instead of
    # ever constructing a live OpenAICompatibleProvider.
    provider_factory: Callable[[], Provider] = field(default=_default_provider_factory)

    def provider(self) -> Provider:
        return self.provider_factory()


def _build_tenant_runtime(
    tenant_id: str, lexical_index: BM25Index, dense_store: DenseStore
) -> TenantRuntime:
    config = load_tenant_config(TENANTS_DIR / tenant_id / "config.yaml")
    allowed = frozenset(config.actions.allowed)
    tools: dict[str, Tool] = {
        "search": SearchTool(
            lexical_index=lexical_index, dense_store=dense_store, tenant_id=tenant_id
        ),
        "flag_discrepancy": ActionGatedTool(tool=FlagDiscrepancyTool(), allowed_actions=allowed),
        "draft_response": ActionGatedTool(tool=DraftResponseTool(), allowed_actions=allowed),
    }
    return TenantRuntime(
        config=config,
        tools=tools,
        system_prompt=SYSTEM_PROMPT_TEMPLATE.format(display_name=config.display_name),
    )


def build_app_state() -> AppState:
    AUDIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # A user's actual question is never fully covered by the committed
    # fixture cache, so the served app wraps it with a live fallback:
    # a miss is embedded on demand and persisted back to the same file,
    # so the cache grows in place as the app is used.
    embedding_cache = LiveFallbackEmbeddingCache(
        cache=EmbeddingCache.load(EMBEDDING_CACHE_PATH),
        client=NvidiaEmbeddingClient(),
        save_path=EMBEDDING_CACHE_PATH,
    )
    tenant_ids = discover_tenant_ids(TENANTS_DIR)
    lexical_index, dense_store = build_multi_tenant_index(tenant_ids, TENANTS_DIR, embedding_cache)
    tenants = {
        tenant_id: _build_tenant_runtime(tenant_id, lexical_index, dense_store)
        for tenant_id in tenant_ids
    }
    return AppState(tenants=tenants, audit_log=AuditLog(AUDIT_DB_PATH))
