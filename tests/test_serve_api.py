"""API done-tests: querying a tenant returns a grounded response through
the real routes, listing tenants and the audit trail works, and an
unknown tenant returns 404. Uses a recorded provider throughout, never
a live one.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from eval.grounding.scenarios import SCENARIOS
from outpost.agent.audit import AuditLog
from outpost.agent.tools import SearchTool
from outpost.llm.recorded import RecordedProvider
from outpost.ontology import load_tenant_config
from outpost.retrieval.build import build_multi_tenant_index
from outpost.retrieval.dense import EmbeddingCache
from outpost.serve.routes import audit, query, tenants
from outpost.serve.state import AppState, TenantRuntime

REPO_ROOT = Path(__file__).resolve().parents[1]
TENANTS_DIR = REPO_ROOT / "tenants"
LLM_FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "llm"
EMBEDDING_CACHE_PATH = REPO_ROOT / "tests" / "fixtures" / "embeddings" / "retrieval.npz"
TENANT_IDS = ["dealer_ar", "claims_intake"]
DEALER_AR_SCENARIO = next(s for s in SCENARIOS if s.tenant_id == "dealer_ar")


def _test_app(tmp_path: Path) -> FastAPI:
    lexical_index, dense_store = build_multi_tenant_index(
        TENANT_IDS, TENANTS_DIR, EmbeddingCache.load(EMBEDDING_CACHE_PATH)
    )
    tenants_runtime = {}
    for tenant_id in TENANT_IDS:
        config = load_tenant_config(TENANTS_DIR / tenant_id / "config.yaml")
        search_tool = SearchTool(
            lexical_index=lexical_index, dense_store=dense_store, tenant_id=tenant_id
        )
        system_prompt = (
            DEALER_AR_SCENARIO.system_prompt
            if tenant_id == "dealer_ar"
            else "unused in these tests"
        )
        tenants_runtime[tenant_id] = TenantRuntime(
            config=config, tools={"search": search_tool}, system_prompt=system_prompt
        )

    state = AppState(
        tenants=tenants_runtime,
        audit_log=AuditLog(tmp_path / "audit.sqlite"),
        provider_factory=lambda: RecordedProvider(
            fixtures_dir=LLM_FIXTURES_DIR, model="openai/gpt-oss-120b"
        ),
    )

    app = FastAPI()
    app.state.outpost = state
    app.include_router(tenants.router)
    app.include_router(query.router)
    app.include_router(audit.router)
    return app


def test_list_tenants(tmp_path: Path) -> None:
    client = TestClient(_test_app(tmp_path))
    response = client.get("/tenants")
    assert response.status_code == 200
    tenant_ids = {entry["tenant_id"] for entry in response.json()}
    assert tenant_ids == set(TENANT_IDS)


def test_query_tenant_returns_grounded_answer(tmp_path: Path) -> None:
    client = TestClient(_test_app(tmp_path))
    response = client.post(
        "/tenants/dealer_ar/query", json={"user_request": DEALER_AR_SCENARIO.user_request}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["tenant_id"] == "dealer_ar"
    assert body["rung_name"] == "FULL"
    assert body["citations"]
    assert body["unsupported_assertions"] == []


def test_query_unknown_tenant_returns_404(tmp_path: Path) -> None:
    client = TestClient(_test_app(tmp_path))
    response = client.post("/tenants/does-not-exist/query", json={"user_request": "hi"})
    assert response.status_code == 404


def test_audit_lists_the_query_that_just_ran(tmp_path: Path) -> None:
    client = TestClient(_test_app(tmp_path))
    client.post("/tenants/dealer_ar/query", json={"user_request": DEALER_AR_SCENARIO.user_request})

    response = client.get("/tenants/dealer_ar/audit")

    assert response.status_code == 200
    entries = response.json()
    assert len(entries) == 1
    assert entries[0]["tenant_id"] == "dealer_ar"
    assert entries[0]["rung_name"] == "FULL"


def test_audit_unknown_tenant_returns_404(tmp_path: Path) -> None:
    client = TestClient(_test_app(tmp_path))
    response = client.get("/tenants/does-not-exist/audit")
    assert response.status_code == 404
