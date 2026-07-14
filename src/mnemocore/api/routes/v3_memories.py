"""Scoped v3 memory API backed only by request-owned SQLite runtimes."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mnemocore.agent_memory import MemoryKind, MemoryNotFoundError, MemoryScope
from mnemocore.api.scope_authorization import ScopeAuthorizer
from mnemocore.hybrid import RuntimeFactory

router = APIRouter(prefix="/v3/memories", tags=["v3 memories"])

_METADATA_MAX_DEPTH = 3
_METADATA_MAX_ITEMS = 50
_METADATA_MAX_KEY_LENGTH = 64
_METADATA_MAX_STRING_LENGTH = 1000
_METADATA_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")


def _validate_metadata_value(value: Any, *, path: str, depth: int) -> None:
    """Validate bounded JSON metadata accepted from the public v3 API."""
    if depth > _METADATA_MAX_DEPTH:
        raise ValueError(f"Metadata at '{path}' exceeds maximum nesting depth")
    if isinstance(value, str):
        if len(value) > _METADATA_MAX_STRING_LENGTH:
            raise ValueError(f"Metadata value at '{path}' is too long")
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Metadata value at '{path}' must be finite")
        return
    if isinstance(value, list):
        if len(value) > _METADATA_MAX_ITEMS:
            raise ValueError(f"Metadata list at '{path}' is too large")
        for index, item in enumerate(value):
            _validate_metadata_value(item, path=f"{path}[{index}]", depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > _METADATA_MAX_ITEMS:
            raise ValueError(f"Metadata object at '{path}' is too large")
        for key, item in value.items():
            _validate_metadata_key(key, path=path)
            _validate_metadata_value(item, path=f"{path}.{key}", depth=depth + 1)
        return
    raise ValueError(f"Metadata value at '{path}' must be JSON-compatible")


def _validate_metadata_key(key: Any, *, path: str) -> None:
    if not isinstance(key, str):
        raise ValueError(f"Metadata key at '{path}' must be a string")
    if len(key) > _METADATA_MAX_KEY_LENGTH:
        raise ValueError(f"Metadata key '{key[:20]}...' is too long")
    if not _METADATA_KEY_PATTERN.match(key):
        raise ValueError(f"Metadata key '{key}' contains invalid characters")
    if key.startswith("_") or key.startswith("internal_"):
        raise ValueError(f"Metadata key '{key}' is reserved for internal use")


class V3ScopeRequest(BaseModel):
    """Every v3 operation must provide an exact scope; no defaults exist."""

    model_config = ConfigDict(extra="forbid")
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    project_id: str | None = Field(default=None, min_length=1)
    session_id: str | None = Field(default=None, min_length=1)

    def scope(self) -> MemoryScope:
        return MemoryScope(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            agent_id=self.agent_id,
            project_id=self.project_id,
            session_id=self.session_id,
        )


class V3RememberRequest(V3ScopeRequest):
    content: str = Field(min_length=1)
    kind: MemoryKind = MemoryKind.OBSERVATION
    metadata: dict[str, Any] | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(
        cls, metadata: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if metadata is None:
            return metadata
        if len(metadata) > _METADATA_MAX_ITEMS:
            raise ValueError("Metadata object is too large")
        for key, value in metadata.items():
            _validate_metadata_key(key, path="metadata")
            _validate_metadata_value(value, path=key, depth=1)
        return metadata


class V3RecallRequest(V3ScopeRequest):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)


class V3ForgetRequest(V3ScopeRequest):
    reason: str | None = Field(default=None, min_length=1)


def _sqlite_path(request: Request) -> Path:
    configured = getattr(request.app.state, "v3_sqlite_path", None)
    if configured is None:
        raise RuntimeError("v3 SQLite path is not configured")
    return Path(configured)


async def _authorize(request: Request, scope: MemoryScope) -> None:
    authorizer: ScopeAuthorizer | None = getattr(
        request.app.state, "v3_scope_authorizer", None
    )
    if authorizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="v3 scope authorization is not configured",
        )
    if not await authorizer.authorize(request, scope):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="scope is not authorized"
        )


def _memory(record: Any) -> dict[str, Any]:
    return {
        "id": record.id,
        "kind": record.kind.value,
        "content": record.content,
        "metadata": dict(record.metadata),
        "status": record.status.value,
        "confidence": record.confidence,
        "observed_at": record.observed_at.isoformat(),
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }


async def _open(request: Request, scope: MemoryScope):
    factory = RuntimeFactory(_sqlite_path(request))
    return factory, await factory.open(scope=scope)


@router.post("", status_code=status.HTTP_201_CREATED)
async def remember(request: Request, body: V3RememberRequest):
    scope = body.scope()
    await _authorize(request, scope)
    factory, runtime = await _open(request, scope)
    try:
        record = await runtime.remember(
            body.content,
            kind=body.kind,
            metadata=body.metadata,
            confidence=body.confidence,
        )
        return {"scope_key": scope.scope_key, "memory": _memory(record)}
    finally:
        await factory.close()


@router.post("/recall")
async def recall(request: Request, body: V3RecallRequest):
    scope = body.scope()
    await _authorize(request, scope)
    factory, runtime = await _open(request, scope)
    try:
        results = await runtime.recall(scope, body.query, limit=body.limit)
        return {
            "scope_key": scope.scope_key,
            "results": [
                {
                    "memory": _memory(result.memory),
                    "score": result.score,
                    "scoring_version": result.scoring_version,
                    "score_components": dict(result.score_components),
                }
                for result in results
            ],
        }
    finally:
        await factory.close()


@router.get("/{memory_id}")
async def get(memory_id: str, request: Request, scope: V3ScopeRequest = Depends()):
    exact_scope = scope.scope()
    await _authorize(request, exact_scope)
    factory, runtime = await _open(request, exact_scope)
    try:
        record = await runtime.get(memory_id)
        return {"scope_key": exact_scope.scope_key, "memory": _memory(record)}
    except MemoryNotFoundError as error:
        raise HTTPException(status_code=404, detail="memory not found") from error
    finally:
        await factory.close()


@router.delete("/{memory_id}")
async def forget(memory_id: str, request: Request, scope: V3ForgetRequest = Depends()):
    exact_scope = scope.scope()
    await _authorize(request, exact_scope)
    factory, runtime = await _open(request, exact_scope)
    try:
        record = await runtime.forget(memory_id, reason=scope.reason)
        return {"scope_key": exact_scope.scope_key, "memory": _memory(record)}
    except MemoryNotFoundError as error:
        raise HTTPException(status_code=404, detail="memory not found") from error
    finally:
        await factory.close()
