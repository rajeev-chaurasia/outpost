"""FastAPI application: ask a tenant-scoped question, list tenants, and
read the audit trail.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from outpost.serve.routes import audit, query, tenants
from outpost.serve.state import build_app_state

DASHBOARD_ORIGIN = "http://localhost:5173"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.outpost = build_app_state()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="outpost", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[DASHBOARD_ORIGIN],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(tenants.router)
    app.include_router(query.router)
    app.include_router(audit.router)
    return app


app = create_app()
