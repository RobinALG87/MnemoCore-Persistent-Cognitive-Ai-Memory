"""Composition boundary for SQLite-backed hybrid runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mnemocore.agent_memory import MemoryScope

from .runtime import HybridMemoryRuntime


@dataclass(frozen=True, slots=True)
class RuntimeMetadata:
    """Content-free details suitable for runtime observability."""

    scope_key: str
    storage_backend: Literal["sqlite"]
    runtime_kind: Literal["hybrid-memory"]


class RuntimeFactory:
    """Open and own one exact-scope SQLite-backed hybrid runtime at a time."""

    def __init__(self, sqlite_path: str | Path) -> None:
        self._sqlite_path = Path(sqlite_path)
        self._runtime: HybridMemoryRuntime | None = None
        self._metadata: RuntimeMetadata | None = None

    @property
    def metadata(self) -> RuntimeMetadata | None:
        """Return content-free metadata for the runtime currently or last owned."""
        return self._metadata

    async def open(self, *, scope: MemoryScope) -> HybridMemoryRuntime:
        """Open an owned runtime bound to the explicitly supplied exact scope."""
        if not isinstance(scope, MemoryScope):
            raise TypeError("scope must be a MemoryScope")
        if self._runtime is not None:
            raise RuntimeError("factory already owns an open runtime")

        runtime = await HybridMemoryRuntime.open(self._sqlite_path, scope=scope)
        self._runtime = runtime
        self._metadata = RuntimeMetadata(
            scope_key=scope.scope_key,
            storage_backend="sqlite",
            runtime_kind="hybrid-memory",
        )
        return runtime

    async def close(self) -> None:
        """Close the owned runtime, if one is open."""
        if self._runtime is None:
            return
        runtime, self._runtime = self._runtime, None
        await runtime.close()

    async def __aenter__(self) -> "RuntimeFactory":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
