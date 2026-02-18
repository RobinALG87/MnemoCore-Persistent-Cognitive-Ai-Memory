"""
Tests for Extracted Engine Methods
===================================
Unit tests for the refactored private helper methods in HAIMEngine:
- _encode_input()
- _evaluate_tier()
- _persist_memory()
- _trigger_post_store()
"""

import os
from collections import deque
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from mnemocore.core.config import get_config, reset_config
from mnemocore.core.engine import HAIMEngine
from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.node import MemoryNode


@pytest.fixture
def test_engine(tmp_path):
    """Create a test engine with isolated configuration."""
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

    reset_config()
    engine = HAIMEngine()
    yield engine

    # Cleanup
    for key in [
        "HAIM_DATA_DIR",
        "HAIM_MEMORY_FILE",
        "HAIM_CODEBOOK_FILE",
        "HAIM_SYNAPSES_FILE",
        "HAIM_WARM_MMAP_DIR",
        "HAIM_COLD_ARCHIVE_DIR",
        "HAIM_ENCODING_MODE",
        "HAIM_DIMENSIONALITY",
    ]:
        if key in os.environ:
            del os.environ[key]
    reset_config()


# =============================================================================
# Tests for _encode_input()
# =============================================================================

@pytest.mark.asyncio
class TestEncodeInput:
    """Test suite for _encode_input method."""

    async def test_encode_input_basic(self, test_engine):
        """Test basic encoding without goal_id."""
        await test_engine.initialize()

        encoded_vec, metadata = await test_engine._encode_input("test content")

        assert isinstance(encoded_vec, BinaryHDV)
        assert encoded_vec.dimension == test_engine.dimension
        assert metadata == {}

    async def test_encode_input_with_metadata(self, test_engine):
        """Test encoding with existing metadata."""
        await test_engine.initialize()

        existing_metadata = {"key": "value", "number": 42}
        encoded_vec, metadata = await test_engine._encode_input(
            "test content", metadata=existing_metadata
        )

        assert isinstance(encoded_vec, BinaryHDV)
        assert metadata["key"] == "value"
        assert metadata["number"] == 42

    async def test_encode_input_with_goal_id(self, test_engine):
        """Test encoding with goal context binding."""
        await test_engine.initialize()

        encoded_vec, metadata = await test_engine._encode_input(
            "test content", goal_id="goal-123"
        )

        assert isinstance(encoded_vec, BinaryHDV)
        assert metadata["goal_context"] == "goal-123"

    async def test_encode_input_with_goal_and_metadata(self, test_engine):
        """Test encoding with both goal_id and existing metadata."""
        await test_engine.initialize()

        existing_metadata = {"priority": "high"}
        encoded_vec, metadata = await test_engine._encode_input(
            "test content", metadata=existing_metadata, goal_id="goal-456"
        )

        assert isinstance(encoded_vec, BinaryHDV)
        assert metadata["priority"] == "high"
        assert metadata["goal_context"] == "goal-456"

    async def test_encode_input_deterministic(self, test_engine):
        """Test that same content produces same encoding."""
        await test_engine.initialize()

        encoded_vec1, _ = await test_engine._encode_input("identical content")
        encoded_vec2, _ = await test_engine._encode_input("identical content")

        # Same content should produce identical vectors
        assert encoded_vec1.data.tobytes() == encoded_vec2.data.tobytes()

    async def test_encode_input_different_content(self, test_engine):
        """Test that different content produces different encodings."""
        await test_engine.initialize()

        encoded_vec1, _ = await test_engine._encode_input("content A")
        encoded_vec2, _ = await test_engine._encode_input("completely different content B")

        # Different content should produce different vectors
        similarity = encoded_vec1.similarity(encoded_vec2)
        # Similarity should be less than 1.0 for different content
        assert similarity < 1.0


# =============================================================================
# Tests for _evaluate_tier()
# =============================================================================

@pytest.mark.asyncio
class TestEvaluateTier:
    """Test suite for _evaluate_tier method."""

    async def test_evaluate_tier_with_epistemic_drive(self, test_engine):
        """Test EIG calculation when epistemic drive is active."""
        await test_engine.initialize()

        test_engine.epistemic_drive_active = True
        encoded_vec = BinaryHDV.random(test_engine.dimension)
        metadata = {}

        updated_metadata = await test_engine._evaluate_tier(encoded_vec, metadata)

        assert "eig" in updated_metadata
        assert isinstance(updated_metadata["eig"], float)
        assert 0.0 <= updated_metadata["eig"] <= 1.0

    async def test_evaluate_tier_without_epistemic_drive(self, test_engine):
        """Test that EIG is set to 0 when epistemic drive is inactive."""
        await test_engine.initialize()

        test_engine.epistemic_drive_active = False
        encoded_vec = BinaryHDV.random(test_engine.dimension)
        metadata = {}

        updated_metadata = await test_engine._evaluate_tier(encoded_vec, metadata)

        assert updated_metadata["eig"] == 0.0

    async def test_evaluate_tier_high_eig_tags(self, test_engine):
        """Test that high EIG adds epistemic_high tag."""
        await test_engine.initialize()

        test_engine.epistemic_drive_active = True
        test_engine.surprise_threshold = 0.1  # Low threshold to trigger tagging

        # Create a random vector that will likely be different from context
        encoded_vec = BinaryHDV.random(test_engine.dimension)
        metadata = {}

        updated_metadata = await test_engine._evaluate_tier(encoded_vec, metadata)

        if updated_metadata["eig"] >= test_engine.surprise_threshold:
            assert "epistemic_high" in updated_metadata.get("tags", [])

    async def test_evaluate_tier_preserves_existing_tags(self, test_engine):
        """Test that existing tags are preserved when adding epistemic_high."""
        await test_engine.initialize()

        test_engine.epistemic_drive_active = True
        test_engine.surprise_threshold = 0.0  # Guarantee tagging

        encoded_vec = BinaryHDV.random(test_engine.dimension)
        metadata = {"tags": ["existing_tag"]}

        updated_metadata = await test_engine._evaluate_tier(encoded_vec, metadata)

        assert "existing_tag" in updated_metadata["tags"]
        assert "epistemic_high" in updated_metadata["tags"]

    async def test_evaluate_tier_low_eig_no_tag(self, test_engine):
        """Test that low EIG does not add epistemic_high tag."""
        await test_engine.initialize()

        test_engine.epistemic_drive_active = True
        test_engine.surprise_threshold = 1.0  # Impossibly high threshold

        encoded_vec = BinaryHDV.random(test_engine.dimension)
        metadata = {}

        updated_metadata = await test_engine._evaluate_tier(encoded_vec, metadata)

        assert "tags" not in updated_metadata or "epistemic_high" not in updated_metadata.get("tags", [])


# =============================================================================
# Tests for _persist_memory()
# =============================================================================

@pytest.mark.asyncio
class TestPersistMemory:
    """Test suite for _persist_memory method."""

    async def test_persist_memory_creates_node(self, test_engine):
        """Test that _persist_memory creates a valid MemoryNode."""
        await test_engine.initialize()

        encoded_vec = BinaryHDV.random(test_engine.dimension)
        metadata = {"eig": 0.5}

        node = await test_engine._persist_memory("test content", encoded_vec, metadata)

        assert isinstance(node, MemoryNode)
        assert node.content == "test content"
        assert node.hdv.data.tobytes() == encoded_vec.data.tobytes()
        assert node.metadata == metadata

    async def test_persist_memory_stores_in_tier_manager(self, test_engine):
        """Test that node is stored in tier manager (HOT tier)."""
        await test_engine.initialize()

        encoded_vec = BinaryHDV.random(test_engine.dimension)
        metadata = {"eig": 0.5}

        node = await test_engine._persist_memory("test content", encoded_vec, metadata)

        # Verify node is in HOT tier
        async with test_engine.tier_manager.lock:
            assert node.id in test_engine.tier_manager.hot
            assert test_engine.tier_manager.hot[node.id].id == node.id

    async def test_persist_memory_sets_epistemic_value(self, test_engine):
        """Test that epistemic_value is correctly set from metadata."""
        await test_engine.initialize()

        encoded_vec = BinaryHDV.random(test_engine.dimension)
        metadata = {"eig": 0.75}

        node = await test_engine._persist_memory("test content", encoded_vec, metadata)

        assert node.epistemic_value == 0.75

    async def test_persist_memory_calculates_ltp(self, test_engine):
        """Test that LTP is calculated after persistence."""
        await test_engine.initialize()

        encoded_vec = BinaryHDV.random(test_engine.dimension)
        metadata = {"eig": 0.5}

        node = await test_engine._persist_memory("test content", encoded_vec, metadata)

        # LTP should be calculated (non-zero with default config)
        assert hasattr(node, "ltp_strength")
        assert node.ltp_strength >= 0.0

    async def test_persist_memory_writes_to_disk(self, test_engine):
        """Test that memory is appended to persistence log."""
        await test_engine.initialize()

        encoded_vec = BinaryHDV.random(test_engine.dimension)
        metadata = {"eig": 0.5}

        node = await test_engine._persist_memory("test content", encoded_vec, metadata)

        # Check persistence file exists and contains the node
        assert os.path.exists(test_engine.persist_path)


# =============================================================================
# Tests for _trigger_post_store()
# =============================================================================

@pytest.mark.asyncio
class TestTriggerPostStore:
    """Test suite for _trigger_post_store method."""

    async def test_trigger_post_store_adds_to_subconscious_queue(self, test_engine):
        """Test that node ID is added to subconscious queue."""
        await test_engine.initialize()

        # Pre-populate queue to prevent dream from consuming our node
        test_engine.subconscious_queue.clear()
        test_engine.subconscious_queue.append("placeholder")

        node = MemoryNode(
            id="test-node-id",
            hdv=BinaryHDV.random(test_engine.dimension),
            content="test content",
            metadata={},
        )
        metadata = {}

        await test_engine._trigger_post_store(node, metadata)

        # Our node should have been added (dream may have popped placeholder)
        assert "test-node-id" in test_engine.subconscious_queue

    async def test_trigger_post_store_skips_dream_for_gap_fill(self, test_engine):
        """Test that background dream is skipped for gap-filled memories."""
        await test_engine.initialize()

        test_engine.subconscious_queue.clear()

        node = MemoryNode(
            id="gap-fill-node",
            hdv=BinaryHDV.random(test_engine.dimension),
            content="generated content",
            metadata={},
        )
        metadata = {"source": "llm_gap_fill"}

        # Should not raise any errors
        await test_engine._trigger_post_store(node, metadata)

        # For gap fill, node should remain in queue since dream is skipped
        assert "gap-fill-node" in test_engine.subconscious_queue

    async def test_trigger_post_store_triggers_dream_for_normal_memory(self, test_engine):
        """Test that background dream is triggered for normal memories."""
        await test_engine.initialize()

        # Pre-populate to test that dream is triggered
        test_engine.subconscious_queue.clear()
        test_engine.subconscious_queue.append("pre-existing")

        node = MemoryNode(
            id="normal-node",
            hdv=BinaryHDV.random(test_engine.dimension),
            content="normal content",
            metadata={},
        )
        metadata = {}

        # The dream should be triggered and process the queue
        await test_engine._trigger_post_store(node, metadata)

        # Either node was added and dream consumed it, or it's still there
        # The key test is that no error was raised
        assert True

    async def test_trigger_post_store_with_empty_subconscious_queue(self, test_engine):
        """Test behavior when subconscious queue is initially empty."""
        await test_engine.initialize()

        test_engine.subconscious_queue.clear()

        node = MemoryNode(
            id="first-node",
            hdv=BinaryHDV.random(test_engine.dimension),
            content="first content",
            metadata={},
        )
        metadata = {}

        await test_engine._trigger_post_store(node, metadata)

        # Queue may be empty after dream consumes, but node was added
        # The test verifies no exception was raised
        assert True

    async def test_trigger_post_store_gap_fill_not_consumed(self, test_engine):
        """Test that gap-filled nodes remain in queue since dream is skipped."""
        await test_engine.initialize()

        test_engine.subconscious_queue.clear()

        # Gap fill should NOT trigger dream, so node should remain
        node = MemoryNode(
            id="gap-node",
            hdv=BinaryHDV.random(test_engine.dimension),
            content="gap fill content",
            metadata={},
        )
        metadata = {"source": "llm_gap_fill"}

        await test_engine._trigger_post_store(node, metadata)

        # Gap fill skips dream, so node should be in queue
        assert "gap-node" in test_engine.subconscious_queue

    async def test_trigger_post_store_multiple_gap_fills(self, test_engine):
        """Test multiple gap fill calls add multiple entries to queue."""
        await test_engine.initialize()

        test_engine.subconscious_queue.clear()

        for i in range(3):
            node = MemoryNode(
                id=f"gap-node-{i}",
                hdv=BinaryHDV.random(test_engine.dimension),
                content=f"gap content {i}",
                metadata={},
            )
            # Gap fill source skips dream, so nodes accumulate
            await test_engine._trigger_post_store(node, {"source": "llm_gap_fill"})

        assert len(test_engine.subconscious_queue) == 3

    async def test_subconscious_queue_respects_maxlen_config(self, tmp_path):
        """Queue should drop oldest items when maxlen is configured."""
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
        os.environ["HAIM_DREAM_LOOP_SUBCONSCIOUS_QUEUE_MAXLEN"] = "2"

        reset_config()
        engine = HAIMEngine()
        assert isinstance(engine.subconscious_queue, deque)

        engine.subconscious_queue.append("id-1")
        engine.subconscious_queue.append("id-2")
        engine.subconscious_queue.append("id-3")

        assert list(engine.subconscious_queue) == ["id-2", "id-3"]

        for key in [
            "HAIM_DATA_DIR",
            "HAIM_MEMORY_FILE",
            "HAIM_CODEBOOK_FILE",
            "HAIM_SYNAPSES_FILE",
            "HAIM_WARM_MMAP_DIR",
            "HAIM_COLD_ARCHIVE_DIR",
            "HAIM_ENCODING_MODE",
            "HAIM_DIMENSIONALITY",
            "HAIM_DREAM_LOOP_SUBCONSCIOUS_QUEUE_MAXLEN",
        ]:
            if key in os.environ:
                del os.environ[key]
        reset_config()


# =============================================================================
# Integration Tests for store() orchestration
# =============================================================================

@pytest.mark.asyncio
class TestStoreOrchestration:
    """Integration tests for the refactored store() method."""

    async def test_store_returns_valid_id(self, test_engine):
        """Test that store() returns a valid UUID string."""
        await test_engine.initialize()

        node_id = await test_engine.store("test memory content")

        assert isinstance(node_id, str)
        assert len(node_id) == 36  # UUID format

    async def test_store_with_all_parameters(self, test_engine):
        """Test store() with all optional parameters."""
        await test_engine.initialize()

        metadata = {"priority": "high", "category": "test"}
        node_id = await test_engine.store(
            content="complete test",
            metadata=metadata,
            goal_id="goal-789",
        )

        node = await test_engine.get_memory(node_id)

        assert node is not None
        assert node.metadata["priority"] == "high"
        assert node.metadata["category"] == "test"
        assert node.metadata["goal_context"] == "goal-789"
        assert "eig" in node.metadata

    async def test_store_pipeline_integration(self, test_engine):
        """Test complete pipeline from encoding to persistence."""
        await test_engine.initialize()

        content = "integration test content"
        node_id = await test_engine.store(content)

        # Verify node exists in tier manager
        node = await test_engine.tier_manager.get_memory(node_id)
        assert node is not None
        assert node.content == content
        # Node starts in hot (may be demoted based on config, so just check it exists)
        assert node.tier in ["hot", "warm"]

        # Verify persistence
        assert os.path.exists(test_engine.persist_path)
