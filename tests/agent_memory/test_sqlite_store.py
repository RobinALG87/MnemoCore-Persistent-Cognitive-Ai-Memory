import sqlite3

import pytest

from mnemocore.agent_memory import MemoryScope
from mnemocore.agent_memory.sqlite_store import SQLiteMemoryStore


@pytest.fixture
def scope():
    return MemoryScope(user_id="robin", agent_id="codex", project_id="mnemocore")


@pytest.mark.asyncio
async def test_open_creates_versioned_schema(tmp_path):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    await store.close()

    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {
        "memory_events",
        "memories",
        "memory_evidence",
        "memory_relations",
        "memory_history",
        "memory_fts",
    } <= tables


@pytest.mark.asyncio
async def test_open_releases_schema_connection(tmp_path):
    path = tmp_path / "memory.db"
    store = await SQLiteMemoryStore.open(path)
    await store.close()

    path.unlink()

    assert not path.exists()
