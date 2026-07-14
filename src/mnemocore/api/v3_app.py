"""Standalone composition root for the isolated v3 memory API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from mnemocore.api.routes.v3_memories import router as v3_memories_router
from mnemocore.api.scope_authorization import ScopeAuthorizer


def create_v3_app(
    sqlite_path: str | Path,
    *,
    scope_authorizer: ScopeAuthorizer | None = None,
) -> FastAPI:
    """Build the v3-only app without starting legacy HAIM infrastructure.

    A deployment must inject an authorizer that binds authenticated credentials
    to the exact requested ``MemoryScope``.  Without one the v3 routes remain
    deliberately unavailable.
    """

    app = FastAPI(title="MnemoCore v3 API")
    app.state.v3_sqlite_path = Path(sqlite_path)
    app.state.v3_scope_authorizer = scope_authorizer
    app.include_router(v3_memories_router)
    return app
