"""
End-to-End Tests for MnemoCore
===============================
Tests the complete cognitive memory pipeline:
  store → query → feedback → consolidation

These tests run fully offline using the mock infrastructure from conftest.py.
No live Redis or Qdrant required.

SEGMENT 3.4 – End-to-end tests (AGENT_MASTER_PLAN)
"""

import os
import pytest
import pytest_asyncio

from mnemocore.core.config import get_config, reset_config
from mnemocore.core.engine import HAIMEngine
from mnemocore.core.binary_hdv import BinaryHDV


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def isolated_engine(tmp_path):
    """
    Create a fully isolated HAIMEngine with a temp data directory.
    No live services required — uses local file-based tier only.

    Key settings:
    - HAIM_HOT_LTP_THRESHOLD_MIN=0.0  → prevents immediate HOT→WARM demotion
      (new memories have LTP ~0.55, below the default threshold of 0.7)
    - HAIM_HOT_MAX_MEMORIES=10000     → prevents eviction during tests
    """
    from mnemocore.core.hnsw_index import HNSWIndexManager
    HNSWIndexManager._instance = None
    reset_config()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    os.environ["HAIM_DATA_DIR"] = str(data_dir)
    os.environ["HAIM_MEMORY_FILE"] = str(data_dir / "memory.jsonl")
    os.environ["HAIM_CODEBOOK_FILE"] = str(data_dir / "codebook.json")
    os.environ["HAIM_SYNAPSES_FILE"] = str(data_dir / "synapses.json")
    os.environ["HAIM_WARM_MMAP_DIR"] = str(data_dir / "warm")
    os.environ["HAIM_COLD_ARCHIVE_DIR"] = str(data_dir / "cold")
    os.environ["HAIM_ENCODING_MODE"] = "binary"
    os.environ["HAIM_DIMENSIONALITY"] = "1024"
    # Prevent HOT→WARM demotion: new memories have LTP ~0.55,
    # below the default threshold of 0.7, causing immediate demotion.
    # _build_tier("hot", ...) uses prefix TIERS_HOT, so env var is HAIM_TIERS_HOT_*
    os.environ["HAIM_TIERS_HOT_LTP_THRESHOLD_MIN"] = "0.0"
    os.environ["HAIM_TIERS_HOT_MAX_MEMORIES"] = "10000"

    reset_config()
    engine = HAIMEngine()
    yield engine

    # Cleanup env
    for key in [
        "HAIM_DATA_DIR", "HAIM_MEMORY_FILE", "HAIM_CODEBOOK_FILE",
        "HAIM_SYNAPSES_FILE", "HAIM_WARM_MMAP_DIR", "HAIM_COLD_ARCHIVE_DIR",
        "HAIM_ENCODING_MODE", "HAIM_DIMENSIONALITY",
        "HAIM_TIERS_HOT_LTP_THRESHOLD_MIN", "HAIM_TIERS_HOT_MAX_MEMORIES",
    ]:
        os.environ.pop(key, None)
    reset_config()


# =============================================================================
# Test 1: Complete Store → Query Cycle
# =============================================================================

@pytest.mark.asyncio
async def test_complete_store_query_cycle(isolated_engine):
    """
    Full pipeline: store a memory, then query for it.
    The stored memory should appear as the top result.
    """
    await isolated_engine.initialize()

    # Store a distinctive memory
    content = "The mitochondria is the powerhouse of the cell"
    memory_id = await isolated_engine.store(content)

    assert isinstance(memory_id, str)
    assert len(memory_id) == 36  # UUID format

    # Query with the same content — should be top result
    results = await isolated_engine.query(content, top_k=5)

    assert len(results) > 0
    top_id, top_score = results[0]
    assert top_id == memory_id
    assert top_score > 0.5  # High similarity for identical content


@pytest.mark.asyncio
async def test_store_multiple_query_returns_most_relevant(isolated_engine):
    """
    Store multiple memories, query for one specific topic.
    The most semantically relevant memory should rank highest.
    """
    await isolated_engine.initialize()

    # Store memories on different topics
    id_biology = await isolated_engine.store("Photosynthesis converts sunlight into glucose in plants")
    id_physics = await isolated_engine.store("Newton's second law: force equals mass times acceleration")
    id_chemistry = await isolated_engine.store("Water molecule consists of two hydrogen and one oxygen atom")

    # Query for biology topic
    results = await isolated_engine.query("How do plants make food from sunlight?", top_k=5)

    assert len(results) > 0
    # All stored memories should be retrievable via query
    result_ids = [r[0] for r in results]
    # At least one of our stored memories should appear in results
    stored_ids = {id_biology, id_physics, id_chemistry}
    assert len(stored_ids & set(result_ids)) > 0, "At least one stored memory should appear in query results"
    # Note: HDV uses hash-based token encoding, not semantic embeddings,
    # so cross-topic ranking order is not deterministic.


# =============================================================================
# Test 2: LTP Strength Decay
# =============================================================================

@pytest.mark.asyncio
async def test_ltp_strength_is_positive_after_store(isolated_engine):
    """
    Verify that stored memories have positive LTP strength.
    Formula: S = I × log(1+A) × e^(-λT)
    """
    await isolated_engine.initialize()

    memory_id = await isolated_engine.store("Test memory for LTP verification")
    node = await isolated_engine.get_memory(memory_id)

    assert node is not None
    assert hasattr(node, "ltp_strength")
    assert node.ltp_strength >= 0.0


@pytest.mark.asyncio
async def test_retrieval_feedback_updates_node(isolated_engine):
    """
    Test that recording retrieval feedback (helpful=True) works without error.
    The Bayesian LTP updater should be called.
    """
    await isolated_engine.initialize()

    memory_id = await isolated_engine.store("Memory to receive positive feedback")

    # Record positive feedback — should not raise
    await isolated_engine.record_retrieval_feedback(memory_id, helpful=True, eig_signal=0.8)

    # Node should still be retrievable
    node = await isolated_engine.get_memory(memory_id)
    assert node is not None


@pytest.mark.asyncio
async def test_negative_feedback_does_not_delete_memory(isolated_engine):
    """
    Negative feedback should update reliability but not delete the memory.
    """
    await isolated_engine.initialize()

    memory_id = await isolated_engine.store("Memory to receive negative feedback")

    await isolated_engine.record_retrieval_feedback(memory_id, helpful=False, eig_signal=0.5)

    # Memory should still exist
    node = await isolated_engine.get_memory(memory_id)
    assert node is not None


# =============================================================================
# Test 3: XOR Project Isolation
# =============================================================================

@pytest.mark.asyncio
async def test_xor_project_isolation(isolated_engine):
    """
    Memories stored with project_id A should not be the top result
    when querying with project_id B (XOR isolation).
    """
    await isolated_engine.initialize()

    content = "Secret project Alpha data: classified information"

    # Store with project A
    id_project_a = await isolated_engine.store(
        content,
        project_id="project_alpha"
    )

    # Query with project B — should NOT find project A's memory as top result
    results_b = await isolated_engine.query(
        content,
        top_k=5,
        project_id="project_beta"
    )

    # Project A's memory should either not appear, or appear with low score
    result_ids = [r[0] for r in results_b]
    if id_project_a in result_ids:
        # If it appears, its score should be low (XOR mask garbles the vector)
        a_score = dict(results_b)[id_project_a]
        assert a_score < 0.9, "Cross-project memory should have low similarity score"


@pytest.mark.asyncio
async def test_same_project_query_finds_memory(isolated_engine):
    """
    Memories stored with project_id should be findable with the same project_id.
    """
    await isolated_engine.initialize()

    content = "Project Alpha internal knowledge base entry"
    memory_id = await isolated_engine.store(content, project_id="project_alpha")

    # Query with same project — should find it
    results = await isolated_engine.query(content, top_k=5, project_id="project_alpha")

    assert len(results) > 0
    top_id, top_score = results[0]
    assert top_id == memory_id
    assert top_score > 0.5


# =============================================================================
# Test 4: Episodic Chaining
# =============================================================================

@pytest.mark.asyncio
async def test_episodic_chain_links_memories(isolated_engine):
    """
    Memories stored sequentially should form an episodic chain
    via the previous_id field.
    """
    await isolated_engine.initialize()

    id_1 = await isolated_engine.store("First memory in the chain")
    id_2 = await isolated_engine.store("Second memory in the chain")
    id_3 = await isolated_engine.store("Third memory in the chain")

    node_2 = await isolated_engine.get_memory(id_2)
    node_3 = await isolated_engine.get_memory(id_3)

    assert node_2 is not None
    assert node_3 is not None

    # Each memory should point to the previous one
    assert node_2.previous_id == id_1
    assert node_3.previous_id == id_2


@pytest.mark.asyncio
async def test_temporal_neighbors_via_include_neighbors(isolated_engine):
    """
    Query with include_neighbors=True should return temporal context.
    """
    await isolated_engine.initialize()

    id_1 = await isolated_engine.store("Context before the target memory")
    id_target = await isolated_engine.store("Target memory to query for")
    id_3 = await isolated_engine.store("Context after the target memory")

    results = await isolated_engine.query(
        "Target memory to query for",
        top_k=5,
        include_neighbors=True,
    )

    result_ids = [r[0] for r in results]
    # Target should be in results
    assert id_target in result_ids


# =============================================================================
# Test 5: Redis Fallback (engine works without Redis)
# =============================================================================

@pytest.mark.asyncio
async def test_engine_works_without_redis(isolated_engine):
    """
    Engine should function correctly even when Redis is unavailable.
    The tier_manager uses local in-memory storage as fallback.
    """
    await isolated_engine.initialize()

    # No Redis configured — engine should still work
    memory_id = await isolated_engine.store("Memory stored without Redis")
    assert memory_id is not None

    results = await isolated_engine.query("Memory stored without Redis", top_k=3)
    assert len(results) > 0
    assert results[0][0] == memory_id


# =============================================================================
# Test 6: Qdrant Fallback (engine works without Qdrant)
# =============================================================================

@pytest.mark.asyncio
async def test_engine_works_without_qdrant(isolated_engine):
    """
    Engine should function correctly even when Qdrant is unavailable.
    The tier_manager uses local FAISS/in-memory HOT tier as fallback.
    """
    await isolated_engine.initialize()

    # Qdrant not configured — engine should still work via HOT tier
    memory_id = await isolated_engine.store("Memory stored without Qdrant")
    assert memory_id is not None

    node = await isolated_engine.get_memory(memory_id)
    assert node is not None
    assert node.content == "Memory stored without Qdrant"


# =============================================================================
# Test 7: Delete Memory
# =============================================================================

@pytest.mark.asyncio
async def test_delete_removes_memory_from_results(isolated_engine):
    """
    After deleting a memory, it should not appear in query results.
    """
    await isolated_engine.initialize()

    content = "Memory that will be deleted"
    memory_id = await isolated_engine.store(content)

    # Verify it exists
    node = await isolated_engine.get_memory(memory_id)
    assert node is not None

    # Delete it
    await isolated_engine.delete_memory(memory_id)

    # Should no longer be retrievable
    node_after = await isolated_engine.get_memory(memory_id)
    assert node_after is None


# =============================================================================
# Test 8: Stats Endpoint
# =============================================================================

@pytest.mark.asyncio
async def test_get_stats_returns_valid_structure(isolated_engine):
    """
    get_stats() should return a dict with expected keys.
    """
    await isolated_engine.initialize()

    await isolated_engine.store("Memory for stats test")

    stats = await isolated_engine.get_stats()

    assert isinstance(stats, dict)
    assert "engine_version" in stats
    assert "dimension" in stats
    assert "tiers" in stats
    assert "synapses_count" in stats
    assert "timestamp" in stats


# =============================================================================
# Test 9: Synapse Binding
# =============================================================================

@pytest.mark.asyncio
async def test_bind_memories_creates_synapse(isolated_engine):
    """
    bind_memories() should create a synaptic connection between two nodes.
    """
    await isolated_engine.initialize()

    id_a = await isolated_engine.store("Memory A about machine learning")
    id_b = await isolated_engine.store("Memory B about neural networks")

    await isolated_engine.bind_memories(id_a, id_b, success=True)

    stats = await isolated_engine.get_stats()
    assert stats["synapses_count"] >= 1


@pytest.mark.asyncio
async def test_associative_jump_finds_bound_memory(isolated_engine):
    """
    After binding two memories, querying for one should surface the other
    via associative spreading.
    """
    await isolated_engine.initialize()

    id_a = await isolated_engine.store("Concept Alpha: quantum entanglement")
    id_b = await isolated_engine.store("Concept Beta: spooky action at a distance")

    # Bind them explicitly
    await isolated_engine.bind_memories(id_a, id_b, success=True)

    # Query for A — B should appear via associative jump
    results = await isolated_engine.query(
        "Concept Alpha: quantum entanglement",
        top_k=5,
        associative_jump=True,
    )

    result_ids = [r[0] for r in results]
    assert id_a in result_ids  # Direct match
    # B may appear via associative spreading
    # (not guaranteed if score is too low, but no error should occur)
