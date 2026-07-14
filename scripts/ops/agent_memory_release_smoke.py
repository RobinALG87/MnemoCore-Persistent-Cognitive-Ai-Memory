"""Local-only release smoke for MnemoCore's stable AgentMemory API."""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile

from mnemocore._version import __version__
from mnemocore.agent_memory import AgentMemory, MemoryKind, MemoryScope

EXPECTED_VERSION = "3.0.0"


async def main() -> None:
    if __version__ != EXPECTED_VERSION:
        raise RuntimeError(
            f"Expected MnemoCore {EXPECTED_VERSION}, found {__version__}"
        )

    scope = MemoryScope(
        tenant_id="release-smoke",
        user_id="release",
        agent_id="agent-memory",
        project_id="v3",
    )
    with tempfile.TemporaryDirectory(prefix="mnemocore-release-") as directory:
        database = Path(directory) / "agent-memory.sqlite3"
        async with await AgentMemory.open(database, scope=scope) as memory:
            original = await memory.remember(
                "Release smoke remembers a local fact.",
                kind=MemoryKind.FACT,
                idempotency_key="release-smoke-v3",
            )
            recalled = await memory.recall("local fact", limit=5)
            if not any(result.memory.id == original.id for result in recalled):
                raise RuntimeError("Remembered memory was not recalled")

            context = await memory.compile_context("local fact", token_budget=120)
            if original.id not in {item.receipt.memory_id for item in context.semantic}:
                raise RuntimeError("Remembered memory was not included in context")

            replacement = await memory.supersede(
                original.id,
                "Release smoke confirms the revised local fact.",
                effective_at="2026-07-14T00:00:00Z",
                reason="release smoke",
                idempotency_key="release-smoke-v3-revision",
            )
            forgotten = await memory.forget(
                replacement.id, reason="release smoke cleanup"
            )
            if forgotten.id != replacement.id:
                raise RuntimeError("Forget returned an unexpected memory")
            await memory.rebuild()

        async with await AgentMemory.open(database, scope=scope) as reopened:
            if await reopened.list():
                raise RuntimeError("Forgotten release-smoke memory remained active")

    print(f"AgentMemory release smoke passed for {EXPECTED_VERSION}")


if __name__ == "__main__":
    asyncio.run(main())
