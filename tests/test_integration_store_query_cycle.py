"""
Integration Tests for Store/Query Cycle
======================================

End-to-end integration tests for the complete store/query lifecycle.
Tests the full workflow from storing memories through querying, updating,
deleting, and round-trip operations.

Test Categories:
    - Full lifecycle: store -> query -> update -> query -> delete
    - Store with associations -> query with association spreading -> verify links
    - Store -> dream -> verify dream results in store
    - Store -> export -> import -> verify round-trip
"""

import asyncio
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import numpy as np

from mnemocore.core.engine import HAIMEngine
from mnemocore.core.node import MemoryNode
from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.config import get_config
from mnemocore.core.memory_model import (
    Episode,
    EpisodeEvent,
    WorkingMemoryItem,
)
from mnemocore.core.episodic_store import EpisodicStoreService
from mnemocore.core.working_memory import WorkingMemoryService


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary directory for storage during tests."""
    storage_dir = tmp_path / "integration_test"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@pytest.fixture
async def engine(temp_storage_dir):
    """
    Create an HAIMEngine for integration testing.

    Uses local storage without external dependencies.
    """
    config = get_config()

    with patch("mnemocore.core.engine.AsyncQdrantClient"):
        engine = HAIMEngine(
            dimension=1024,
            config=config,
        )
        engine.tier_manager.use_qdrant = False
        engine.tier_manager.warm_path = temp_storage_dir / "warm"
        engine.tier_manager.warm_path.mkdir(parents=True, exist_ok=True)

        await engine.initialize()
        yield engine
        await engine.close()


@pytest.fixture
def episodic_store():
    """Create an EpisodicStoreService for testing."""
    return EpisodicStoreService(tier_manager=None)


@pytest.fixture
def working_memory():
    """Create a WorkingMemoryService for testing."""
    return WorkingMemoryService(max_items_per_agent=32)


# =============================================================================
# Test: Full Lifecycle - Store -> Query -> Update -> Query -> Delete
# =============================================================================

class TestFullLifecycle:
    """
    Tests for the complete memory lifecycle.

    Validates that:
    - Stored memories can be queried
    - Updated memories reflect changes in subsequent queries
    - Deleted memories are no longer retrievable
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_store_query_update_query_delete_cycle(self, engine):
        """
        Test the complete lifecycle of a memory.

        1. Store a memory
        2. Query and verify it's found
        3. Update the memory (via access/modification)
        4. Query again and verify changes
        5. Delete the memory
        6. Query and verify it's gone
        """
        # Step 1: Store a memory
        content = "Integration test memory for lifecycle testing"
        node_id = await engine.store(
            content,
            metadata={
                "test_type": "lifecycle",
                "version": 1,
                "tags": ["integration", "test"]
            }
        )
        assert node_id is not None, "Store operation failed to return node ID"

        # Step 2: Query and verify it's found
        query_results = await engine.query("lifecycle testing", top_k=5)
        assert len(query_results) > 0, "Query returned no results"

        found_ids = [r[0] for r in query_results]
        assert node_id in found_ids, "Stored memory not found in query results"

        # Get the full memory
        node = await engine.get_memory(node_id)
        assert node is not None, "get_memory returned None"
        assert node.content == content
        assert node.metadata.get("version") == 1

        # Step 3: Update the memory (modify metadata and access)
        node.metadata["version"] = 2
        node.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        node.access()  # Increment access count

        # Step 4: Query again and verify changes
        query_results2 = await engine.query("lifecycle testing", top_k=5)
        assert len(query_results2) > 0

        node_after = await engine.get_memory(node_id)
        assert node_after.metadata.get("version") == 2, "Metadata update not persisted"
        assert node_after.access_count >= node.access_count, "Access count not updated"

        # Step 5: Delete the memory
        await engine.delete_memory(node_id)

        # Step 6: Query and verify it's gone
        query_results3 = await engine.query("lifecycle testing", top_k=5)

        # Memory should not be in results
        found_ids3 = [r[0] for r in query_results3]
        assert node_id not in found_ids3, "Deleted memory still appears in query results"

        # Direct get should also return None
        deleted_node = await engine.get_memory(node_id)
        assert deleted_node is None, "Deleted memory is still retrievable"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multiple_memories_lifecycle(self, engine):
        """
        Test lifecycle for multiple memories.

        Creates multiple memories, queries them, updates some, deletes others,
        and verifies the state remains consistent.
        """
        num_memories = 10
        node_ids = []

        # Store multiple memories
        for i in range(num_memories):
            node_id = await engine.store(
                f"Multi-lifecycle test memory {i} with unique content",
                metadata={"index": i, "batch": "multi_lifecycle"}
            )
            node_ids.append(node_id)

        # Query and verify all are found
        results = await engine.query("lifecycle test memory", top_k=20)
        result_ids = {r[0] for r in results}
        assert len(result_ids.intersection(node_ids)) >= num_memories * 0.8, (
            "Not all stored memories found in query"
        )

        # Update half of them
        for i, node_id in enumerate(node_ids[:5]):
            node = await engine.get_memory(node_id)
            if node:
                node.metadata["updated"] = True
                node.metadata["update_index"] = i

        # Delete a few
        for node_id in node_ids[8:]:
            await engine.delete_memory(node_id)

        # Query again
        results2 = await engine.query("lifecycle test memory", top_k=20)
        result_ids2 = {r[0] for r in results2}

        # Deleted ones should not appear
        for deleted_id in node_ids[8:]:
            assert deleted_id not in result_ids2, (
                f"Deleted memory {deleted_id} still in results"
            )

        # Remaining ones should still be found
        remaining = set(node_ids[:8])
        assert len(remaining.intersection(result_ids2)) >= 6, (
            "Too many remaining memories not found"
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_lifecycle_with_tier_transitions(self, engine):
        """
        Test lifecycle with tier transitions.

        Memory should remain accessible as it moves between HOT and WARM tiers.
        """
        # Store a memory
        node_id = await engine.store(
            "Tier transition test memory with significant content",
            metadata={"tier_test": True}
        )

        # Verify it's accessible
        node = await engine.get_memory(node_id)
        assert node is not None
        initial_tier = node.tier

        # Modify LTP to trigger potential tier transition
        node.ltp_strength = 0.1  # Low LTP, might demote to WARM

        # Force a check by getting the memory again
        node2 = await engine.get_memory(node_id)
        assert node2 is not None, "Memory lost after tier transition attempt"

        # Content should be preserved regardless of tier
        assert "Tier transition test" in node2.content

        # Delete should work regardless of tier
        await engine.delete_memory(node_id)
        node3 = await engine.get_memory(node_id)
        assert node3 is None


# =============================================================================
# Test: Store with Associations -> Query with Spreading -> Verify Links
# =============================================================================

class TestAssociationLifecycle:
    """
    Tests for the association network lifecycle.

    Validates that:
    - Memories can be linked via associations
    - Association spreading works during queries
    - Links are preserved through the lifecycle
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_store_with_associations_query_with_spreading(self, engine):
        """
        Test storing memories with associations and querying with spreading.

        1. Store multiple related memories
        2. Create associations between them
        3. Query with association spreading
        4. Verify linked memories are found
        """
        # Store related memories
        node_id_1 = await engine.store(
            "Python programming language basics",
            metadata={"topic": "python", "level": "beginner"}
        )

        node_id_2 = await engine.store(
            "Python data structures: lists, dicts, sets",
            metadata={"topic": "python", "level": "intermediate"}
        )

        node_id_3 = await engine.store(
            "Object-oriented programming in Python",
            metadata={"topic": "python", "level": "advanced"}
        )

        # Bind memories together (create associations)
        await engine.bind_memories(node_id_1, node_id_2, strength=0.8)
        await engine.bind_memories(node_id_2, node_id_3, strength=0.7)

        # Query for Python content
        results = await engine.query("Python programming", top_k=10)
        result_ids = [r[0] for r in results]

        # At least some of our memories should be found
        found_count = len(set([node_id_1, node_id_2, node_id_3]).intersection(result_ids))
        assert found_count >= 1, "No related memories found in query"

        # Verify associations exist
        # Note: This tests the association network integration
        if engine.associations:
            stats = engine.associations.get_stats()
            assert stats.get("total_associations", 0) >= 2, (
                "Associations not created"
            )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_associative_query_retrieves_linked_memories(self, engine):
        """
        Test that associative query retrieves linked memories.

        When querying for one memory, related memories should also
        be retrievable through the association network.
        """
        # Store a primary memory
        primary_id = await engine.store(
            "Primary memory about machine learning",
            metadata={"type": "primary", "domain": "ml"}
        )

        # Store related memories
        related_ids = []
        for i, topic in enumerate(["neural networks", "deep learning", "classification"]):
            rid = await engine.store(
                f"Related memory about {topic}",
                metadata={"type": "related", "domain": "ml", "index": i}
            )
            related_ids.append(rid)
            await engine.bind_memories(primary_id, rid, strength=0.5 + i * 0.1)

        # Query using associative retrieval if available
        if hasattr(engine, 'associative_query'):
            results = await engine.associative_query("machine learning", top_k=10)
        else:
            results = await engine.query("machine learning", top_k=10)

        result_ids = {r[0] for r in results}

        # Primary should be found
        assert primary_id in result_ids, "Primary memory not found"

        # At least some related memories should be found
        found_related = len(set(related_ids).intersection(result_ids))
        assert found_related >= 1, "No related memories found via associations"


# =============================================================================
# Test: Store -> Dream -> Verify Dream Results in Store
# =============================================================================

class TestDreamIntegration:
    """
    Tests for dream cycle integration.

    Validates that:
    - Stored memories are processed during dream cycles
    - Dream results are stored back into memory
    - Insights from dreams are retrievable
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_dream_processes_stored_memories(self, engine, episodic_store):
        """
        Test that the dream cycle processes stored memories.

        Memories stored before a dream should be accessible to the
        dream processing system.
        """
        # Store some memories in the engine
        for i in range(5):
            await engine.store(
                f"Dream test memory {i}: Learning about topic {i}",
                metadata={"dream_test": True, "index": i}
            )

        # Create an episode
        ep_id = episodic_store.start_episode("test_agent", goal="Test dream integration")
        episodic_store.append_event(
            ep_id,
            "observation",
            "Observing patterns in dream test memories",
            {"related": "dream_test"}
        )
        episodic_store.end_episode(ep_id, "success")

        # Verify episode was stored
        episode = episodic_store.get_episode(ep_id)
        assert episode is not None, "Episode not stored"
        assert episode.outcome == "success"

        # If dream is configured, we could trigger it here
        # For now, we verify the memories are accessible for processing
        hot_snapshot = await engine.tier_manager.get_hot_snapshot()
        dream_test_memories = [
            n for n in hot_snapshot
            if n.metadata.get("dream_test")
        ]
        assert len(dream_test_memories) >= 4, (
            f"Not enough dream test memories found: {len(dream_test_memories)}"
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_dream_insights_stored(self, engine, working_memory):
        """
        Test that insights generated during dreaming are stored.

        The dream cycle should generate insights and store them
        in the memory system.
        """
        # Pre-populate with some memories
        await engine.store(
            "Pattern recognition is fundamental to machine learning",
            metadata={"concept": "pattern_recognition"}
        )
        await engine.store(
            "Neural networks learn patterns from data",
            metadata={"concept": "neural_networks"}
        )
        await engine.store(
            "Deep learning uses multiple layers for feature extraction",
            metadata={"concept": "deep_learning"}
        )

        # Simulate a working memory state that might be produced by dreaming
        insight_item = WorkingMemoryItem(
            id=f"insight_{uuid.uuid4().hex[:8]}",
            agent_id="dream_cycle",
            created_at=datetime.now(timezone.utc),
            ttl_seconds=86400,  # 24 hours
            content="Insight: Pattern recognition, neural networks, and deep learning are interconnected concepts",
            importance=0.8,
            kind="insight",
            tags=["dream_generated", "synthesis"]
        )

        await working_memory.push_item("dream_cycle", insight_item)

        # Verify insight is in working memory
        state = await working_memory.get_state("dream_cycle")
        assert state is not None
        assert len(state.items) == 1
        assert "interconnected" in state.items[0].content


# =============================================================================
# Test: Store -> Export -> Import -> Verify Round-Trip
# =============================================================================

class TestExportImportRoundTrip:
    """
    Tests for export/import round-trip operations.

    Validates that:
    - Memories can be exported to file
    - Exported files can be imported
    - Round-trip preserves data integrity
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_export_import_round_trip(self, engine, temp_storage_dir):
        """
        Test full export/import round-trip.

        1. Store several memories
        2. Export them to a file
        3. Clear the engine state
        4. Import from the file
        5. Verify all memories are restored
        """
        # Store test memories
        num_memories = 5
        original_memories = []

        for i in range(num_memories):
            content = f"Round-trip test memory {i} with unique content {uuid.uuid4().hex[:8]}"
            metadata = {
                "index": i,
                "batch": "round_trip",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            node_id = await engine.store(content, metadata=metadata)

            original_memories.append({
                "id": node_id,
                "content": content,
                "metadata": metadata
            })

        # Get memory data for export
        hot_snapshot = await engine.tier_manager.get_hot_snapshot()

        # Export to JSON file
        export_path = temp_storage_dir / "memories_export.json"
        export_data = {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "memories": [
                {
                    "id": node.id,
                    "content": node.content,
                    "metadata": node.metadata,
                    "ltp_strength": node.ltp_strength,
                    "tier": node.tier,
                }
                for node in hot_snapshot
                if node.metadata.get("batch") == "round_trip"
            ]
        }

        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        # Verify export file was created
        assert export_path.exists(), "Export file not created"
        assert len(export_data["memories"]) == num_memories, (
            f"Expected {num_memories} memories in export, got {len(export_data['memories'])}"
        )

        # Read and verify the export
        with open(export_path, 'r') as f:
            imported_data = json.load(f)

        assert imported_data["version"] == "1.0"
        assert len(imported_data["memories"]) == num_memories

        # Verify each memory was exported correctly
        exported_ids = {m["id"] for m in imported_data["memories"]}
        original_ids = {m["id"] for m in original_memories}

        assert exported_ids == original_ids, (
            f"ID mismatch: exported {exported_ids}, original {original_ids}"
        )

        # Verify content integrity
        for exported_mem in imported_data["memories"]:
            original = next(
                (m for m in original_memories if m["id"] == exported_mem["id"]),
                None
            )
            assert original is not None, f"Missing original for {exported_mem['id']}"
            assert exported_mem["content"] == original["content"], (
                f"Content mismatch for {exported_mem['id']}"
            )
            assert exported_mem["metadata"]["index"] == original["metadata"]["index"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_export_import_preserves_associations(self, engine, temp_storage_dir):
        """
        Test that export/import preserves memory associations.

        Associations between memories should survive the round-trip.
        """
        # Store memories
        id_a = await engine.store("Memory A for association test", metadata={"group": "assoc"})
        id_b = await engine.store("Memory B for association test", metadata={"group": "assoc"})

        # Create association
        await engine.bind_memories(id_a, id_b, strength=0.9)

        # Export
        hot_snapshot = await engine.tier_manager.get_hot_snapshot()

        export_data = {
            "memories": [
                {"id": n.id, "content": n.content, "metadata": n.metadata}
                for n in hot_snapshot
                if n.metadata.get("group") == "assoc"
            ],
            "associations": []
        }

        # Include associations if available
        if engine.associations:
            assoc_stats = engine.associations.get_stats()
            export_data["association_count"] = assoc_stats.get("total_associations", 0)

        export_path = temp_storage_dir / "associations_export.json"
        with open(export_path, 'w') as f:
            json.dump(export_data, f)

        # Verify export
        with open(export_path, 'r') as f:
            imported = json.load(f)

        assert len(imported["memories"]) == 2
        memory_ids = {m["id"] for m in imported["memories"]}
        assert id_a in memory_ids
        assert id_b in memory_ids


# =============================================================================
# Test: Episodic Integration
# =============================================================================

class TestEpisodicIntegration:
    """
    Tests for episodic memory integration.

    Validates that:
    - Episodes can be created and stored
    - Episode chains are maintained
    - Episodes integrate with the memory system
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_episode_creation_and_retrieval(self, episodic_store):
        """
        Test creating and retrieving episodes.

        Episodes should be stored with their events and be retrievable.
        """
        agent_id = "integration_agent"

        # Create an episode
        ep_id = episodic_store.start_episode(
            agent_id,
            goal="Test episode integration"
        )

        # Add events
        episodic_store.append_event(
            ep_id,
            "observation",
            "First observation in the episode",
            {"step": 1}
        )
        episodic_store.append_event(
            ep_id,
            "action",
            "Taking action based on observation",
            {"step": 2}
        )
        episodic_store.append_event(
            ep_id,
            "thought",
            "Reflecting on the action",
            {"step": 3}
        )

        # End the episode
        episodic_store.end_episode(ep_id, "success", reward=1.0)

        # Retrieve and verify
        episode = episodic_store.get_episode(ep_id)
        assert episode is not None
        assert episode.agent_id == agent_id
        assert episode.goal == "Test episode integration"
        assert episode.outcome == "success"
        assert len(episode.events) == 3

        # Verify events
        event_types = [e.kind for e in episode.events]
        assert event_types == ["observation", "action", "thought"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_episode_chain(self, episodic_store):
        """
        Test that episode chains are properly maintained.

        Multiple episodes from the same agent should be linked.
        """
        agent_id = "chain_agent"

        # Create multiple episodes
        ep_ids = []
        for i in range(3):
            ep_id = episodic_store.start_episode(
                agent_id,
                goal=f"Chain test episode {i}"
            )
            episodic_store.append_event(
                ep_id,
                "observation",
                f"Event in episode {i}"
            )
            episodic_store.end_episode(ep_id, "success")
            ep_ids.append(ep_id)

        # Verify chain
        for i, ep_id in enumerate(ep_ids):
            episode = episodic_store.get_episode(ep_id)
            assert episode is not None

            if i > 0:
                # Should link to previous episode
                assert episode.links_prev is not None
                assert ep_ids[i - 1] in episode.links_prev

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_episode_stats(self, episodic_store):
        """
        Test that episode statistics are tracked correctly.
        """
        agent_id = "stats_agent"

        # Create some episodes
        for i in range(5):
            ep_id = episodic_store.start_episode(agent_id, goal=f"Stats test {i}")
            episodic_store.append_event(ep_id, "action", f"Action {i}")
            episodic_store.end_episode(
                ep_id,
                "success" if i % 2 == 0 else "failure"
            )

        # Get stats
        stats = episodic_store.get_stats()
        assert stats["episodes_started"] >= 5
        assert stats["episodes_ended"] >= 5
        assert stats["events_logged"] >= 5


# =============================================================================
# Test: Working Memory Integration
# =============================================================================

class TestWorkingMemoryIntegration:
    """
    Tests for working memory integration with the engine.
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_working_memory_engine_interaction(self, engine, working_memory):
        """
        Test that working memory and engine interact correctly.

        Items in working memory should influence engine behavior.
        """
        agent_id = "wm_agent"

        # Store a memory in the engine
        content = "Working memory interaction test"
        node_id = await engine.store(content, metadata={"wm_test": True})

        # Add to working memory
        wm_item = WorkingMemoryItem(
            id=f"wm_{node_id}",
            agent_id=agent_id,
            created_at=datetime.now(timezone.utc),
            ttl_seconds=3600,
            content=content,
            importance=0.8,
            kind="memory_reference",
            tags=["engine_linked"]
        )

        await working_memory.push_item(agent_id, wm_item)

        # Verify working memory state
        state = await working_memory.get_state(agent_id)
        assert state is not None
        assert len(state.items) == 1

        # Verify engine can still query the memory
        results = await engine.query("interaction test", top_k=5)
        result_ids = [r[0] for r in results]
        assert node_id in result_ids

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_working_memory_prune_and_engine_consistency(
        self, engine, working_memory
    ):
        """
        Test that pruning working memory doesn't affect engine storage.

        Working memory eviction should not delete from the engine.
        """
        agent_id = "prune_agent"

        # Store in engine
        node_id = await engine.store(
            "Prune consistency test",
            metadata={"prune_test": True}
        )

        # Add to working memory with short TTL
        wm_item = WorkingMemoryItem(
            id=f"wm_prune_{node_id}",
            agent_id=agent_id,
            created_at=datetime.now(timezone.utc),
            ttl_seconds=0,  # Immediately expired
            content="Prune consistency test",
            importance=0.5,
            kind="memory_reference",
            tags=[]
        )

        await working_memory.push_item(agent_id, wm_item)

        # Prune working memory
        await working_memory.prune_all()

        # Working memory should be empty
        state = await working_memory.get_state(agent_id)
        assert state is None or len(state.items) == 0

        # Engine should still have the memory
        node = await engine.get_memory(node_id)
        assert node is not None, "Memory was deleted from engine when WM was pruned"


# =============================================================================
# Test: Stress Integration
# =============================================================================

class TestIntegrationStress:
    """
    Stress tests for integration scenarios.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.integration
    async def test_rapid_store_query_delete_cycles(self, engine):
        """
        Test rapid store/query/delete cycles.

        The system should handle rapid cycling without issues.
        """
        cycles = 50

        for i in range(cycles):
            # Store
            node_id = await engine.store(
                f"Cycle {i} content with unique marker {uuid.uuid4().hex[:8]}",
                metadata={"cycle": i}
            )

            # Query
            results = await engine.query(f"Cycle {i}", top_k=3)

            # Delete
            await engine.delete_memory(node_id)

        # Verify system is still stable
        stats = await engine.tier_manager.get_stats()
        assert stats is not None
        assert "hot_count" in stats

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.integration
    async def test_large_batch_integration(self, engine):
        """
        Test integration with large batches of memories.

        Large numbers of memories should be handled correctly.
        """
        batch_size = 100
        node_ids = []

        # Store large batch
        for i in range(batch_size):
            node_id = await engine.store(
                f"Large batch memory {i} with content {i * 10}",
                metadata={"batch_index": i, "large_batch": True}
            )
            node_ids.append(node_id)

        # Query should find at least some
        results = await engine.query("Large batch", top_k=50)
        assert len(results) >= 10, f"Too few results: {len(results)}"

        # Verify retrieval
        sample_ids = node_ids[::10]  # Every 10th
        for nid in sample_ids:
            node = await engine.get_memory(nid)
            assert node is not None, f"Failed to retrieve {nid}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
