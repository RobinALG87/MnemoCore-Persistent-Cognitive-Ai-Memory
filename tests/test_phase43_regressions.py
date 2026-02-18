import asyncio
import os
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from src.core.binary_hdv import BinaryHDV
from src.core.config import reset_config
from src.core.node import MemoryNode

try:
    from src.core.engine import HAIMEngine
    _ENGINE_IMPORT_ERROR = None
except (ModuleNotFoundError, ImportError) as exc:
    HAIMEngine = None
    _ENGINE_IMPORT_ERROR = exc
    pytestmark = pytest.mark.skip(
        reason=f"HAIMEngine unavailable in current branch state: {exc}"
    )


@pytest_asyncio.fixture
async def isolated_engine():
    root = Path(".tmp_phase43_tests") / str(uuid.uuid4())
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HAIM_DATA_DIR"] = str(data_dir)
    os.environ["HAIM_MEMORY_FILE"] = str(data_dir / "memory.jsonl")
    os.environ["HAIM_CODEBOOK_FILE"] = str(data_dir / "codebook.json")
    os.environ["HAIM_SYNAPSES_FILE"] = str(data_dir / "synapses.json")
    os.environ["HAIM_WARM_MMAP_DIR"] = str(data_dir / "warm")
    os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(data_dir / "cold")
    os.environ["HAIM_DIMENSIONALITY"] = "1024"

    reset_config()
    engine = HAIMEngine()
    yield engine

    for key in [
        "HAIM_DATA_DIR",
        "HAIM_MEMORY_FILE",
        "HAIM_CODEBOOK_FILE",
        "HAIM_SYNAPSES_FILE",
        "HAIM_WARM_MMAP_DIR",
        "HAIM_COLD_ARCHIVE_DIR",
        "HAIM_DIMENSIONALITY",
    ]:
        os.environ.pop(key, None)
    reset_config()


@pytest.mark.asyncio
async def test_query_chrono_uses_batch_lookup(isolated_engine):
    engine = isolated_engine
    now = datetime.now(timezone.utc)

    node1 = MemoryNode(id="n1", hdv=BinaryHDV.random(engine.dimension), content="c1", created_at=now)
    node2 = MemoryNode(id="n2", hdv=BinaryHDV.random(engine.dimension), content="c2", created_at=now)

    engine.tier_manager.search = AsyncMock(return_value=[("n1", 0.9), ("n2", 0.8)])
    engine.tier_manager.get_memories_batch = AsyncMock(return_value=[node1, node2])
    engine.tier_manager.get_memory = AsyncMock(
        side_effect=AssertionError("Per-node get_memory() should not be used in chrono loop")
    )
    engine.tier_manager.get_hot_recent = AsyncMock(return_value=[])

    results = await engine.query(
        "chrono",
        top_k=2,
        associative_jump=False,
        track_gaps=False,
        chrono_weight=True,
        include_neighbors=False,
    )

    assert len(results) <= 2
    engine.tier_manager.get_memories_batch.assert_awaited_once()
    engine.tier_manager.get_memory.assert_not_awaited()


@pytest.mark.asyncio
async def test_query_include_neighbors_preserves_top_k_contract(isolated_engine):
    engine = isolated_engine
    now = datetime.now(timezone.utc)

    n1 = MemoryNode(
        id="n1",
        hdv=BinaryHDV.random(engine.dimension),
        content="n1",
        created_at=now,
        previous_id="p1",
    )
    n2 = MemoryNode(
        id="n2",
        hdv=BinaryHDV.random(engine.dimension),
        content="n2",
        created_at=now,
        previous_id="p2",
    )
    p1 = MemoryNode(id="p1", hdv=BinaryHDV.random(engine.dimension), content="p1", created_at=now)
    p2 = MemoryNode(id="p2", hdv=BinaryHDV.random(engine.dimension), content="p2", created_at=now)

    by_id: Dict[str, Optional[MemoryNode]] = {"n1": n1, "n2": n2, "p1": p1, "p2": p2}

    async def _get_memory(node_id: str):
        return by_id.get(node_id)

    engine.tier_manager.search = AsyncMock(return_value=[("n1", 0.9), ("n2", 0.8), ("n3", 0.7)])
    engine.tier_manager.get_hot_recent = AsyncMock(return_value=[])
    engine.tier_manager.get_memory = AsyncMock(side_effect=_get_memory)
    engine.tier_manager.use_qdrant = False

    results = await engine.query(
        "neighbors",
        top_k=2,
        associative_jump=False,
        track_gaps=False,
        chrono_weight=False,
        include_neighbors=True,
    )

    assert len(results) == 2


@pytest.mark.asyncio
async def test_background_dream_uses_semaphore_locked_api(isolated_engine):
    engine = isolated_engine
    engine.subconscious_queue.append("x")
    await engine._dream_sem.acquire()
    try:
        engine.tier_manager.get_memory = AsyncMock(
            side_effect=AssertionError("Should return early while semaphore is locked")
        )
        await engine._background_dream()
        engine.tier_manager.get_memory.assert_not_awaited()
    finally:
        engine._dream_sem.release()


def _assert_linear_chain(nodes):
    ids = [n.id for n in nodes]
    prev = {n.id: n.previous_id for n in nodes}

    roots = [nid for nid, p in prev.items() if p is None]
    assert len(roots) == 1

    prev_non_none = [p for p in prev.values() if p is not None]
    assert len(prev_non_none) == len(nodes) - 1
    assert len(set(prev_non_none)) == len(prev_non_none)
    assert all(p in ids for p in prev_non_none)

    child_by_prev = {p: nid for nid, p in prev.items() if p is not None}
    current = roots[0]
    visited = {current}
    for _ in range(len(nodes) - 1):
        current = child_by_prev[current]
        assert current not in visited
        visited.add(current)
    assert len(visited) == len(nodes)


@pytest.mark.asyncio
async def test_persist_memory_concurrent_stores_keep_linear_previous_chain(isolated_engine):
    engine = isolated_engine
    engine.tier_manager.add_memory = AsyncMock(return_value=None)
    engine._append_persisted = AsyncMock(return_value=None)

    vec_a = BinaryHDV.random(engine.dimension)
    vec_b = BinaryHDV.random(engine.dimension)
    vec_c = BinaryHDV.random(engine.dimension)

    a, b, c = await asyncio.gather(
        engine._persist_memory("a", vec_a, {"eig": 0.1}),
        engine._persist_memory("b", vec_b, {"eig": 0.2}),
        engine._persist_memory("c", vec_c, {"eig": 0.3}),
    )

    _assert_linear_chain([a, b, c])


@pytest.mark.asyncio
async def test_get_stats_reports_engine_version_43(isolated_engine):
    engine = isolated_engine
    engine.tier_manager.get_stats = AsyncMock(return_value={"hot_count": 0, "warm_count": 0})
    stats = await engine.get_stats()
    assert stats["engine_version"] == "4.3.0"


@pytest.mark.asyncio
async def test_tier_manager_search_applies_hot_time_range_filter(isolated_engine):
    engine = isolated_engine
    tm = engine.tier_manager
    tm.use_qdrant = False

    now = datetime.now(timezone.utc)
    old_node = MemoryNode(
        id="old",
        hdv=BinaryHDV.random(engine.dimension),
        content="old",
        created_at=now - timedelta(days=2),
    )
    new_node = MemoryNode(
        id="new",
        hdv=BinaryHDV.random(engine.dimension),
        content="new",
        created_at=now,
    )

    tm.hot = {"old": old_node, "new": new_node}
    tm.search_hot = lambda query_vec, top_k=5: [("old", 0.95), ("new", 0.90)]

    query_vec = BinaryHDV.random(engine.dimension)
    results = await tm.search(
        query_vec,
        top_k=5,
        time_range=(now - timedelta(hours=1), now + timedelta(hours=1)),
    )

    assert [nid for nid, _ in results] == ["new"]


@pytest.mark.asyncio
async def test_orchestrate_orch_or_is_async_and_lock_guarded(isolated_engine):
    engine = isolated_engine
    node = MemoryNode(
        id="orch",
        hdv=BinaryHDV.random(engine.dimension),
        content="orch content",
        created_at=datetime.now(timezone.utc),
    )
    node.ltp_strength = 0.8
    node.epistemic_value = 0.4
    node.access_count = 5
    engine.tier_manager.hot[node.id] = node

    await engine.tier_manager.lock.acquire()
    task = asyncio.create_task(engine.orchestrate_orch_or(max_collapse=1))
    await asyncio.sleep(0.05)
    assert not task.done()
    engine.tier_manager.lock.release()

    collapsed = await task
    assert len(collapsed) == 1
    assert collapsed[0].id == "orch"
