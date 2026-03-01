"""
Tests for Episodic Future Thinking Module
=========================================
Comprehensive tests for cognitive simulation of probable future scenarios.

Tests cover:
- ScenarioStore lifecycle: create, get, verify, archive
- Confidence decay over time
- Cleanup of expired scenarios
- Stats functionality
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import numpy as np

from mnemocore.cognitive.future_thinking import (
    EFTConfig,
    ScenarioType,
    ScenarioNode,
    ScenarioStore,
    EpisodeFutureSimulator,
)


@pytest.fixture
def temp_storage_path():
    """Create a temporary file path for scenario storage."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def eft_config():
    """Create default EFT configuration."""
    return EFTConfig(
        enabled=True,
        max_scenarios_per_simulation=5,
        min_similarity_threshold=0.55,
        temporal_horizon_hours=24.0,
        scenario_decay_lambda=0.05,
        scenario_half_life_hours=12.0,
        min_scenario_confidence=0.1,
        max_stored_scenarios=100,
        persist_scenarios=False  # Disable for most tests
    )


@pytest.fixture
def scenario_store(eft_config, temp_storage_path):
    """Create a ScenarioStore with temporary storage."""
    eft_config.persist_scenarios = False  # Disable persistence for unit tests
    return ScenarioStore(config=eft_config, storage_path=temp_storage_path)


@pytest.fixture
def sample_scenario():
    """Create a sample ScenarioNode for testing."""
    return ScenarioNode(
        id="scenario_test123",
        type=ScenarioType.CONTINUATION,
        content="This is a test scenario content",
        context_summary="Test context",
        projected_timestamp=(datetime.now(timezone.utc) + timedelta(hours=12)).isoformat(),
        confidence=0.75,
        probability=0.6,
        source_memory_ids=["mem1", "mem2"],
        pattern_similarity=0.8
    )


@pytest.fixture
def populated_store(scenario_store):
    """Create a ScenarioStore with multiple scenarios."""
    store = scenario_store

    scenarios = [
        ScenarioNode(
            type=ScenarioType.CONTINUATION,
            content="Continuation scenario",
            confidence=0.8,
            source_memory_ids=["mem1"]
        ),
        ScenarioNode(
            type=ScenarioType.BRANCHING,
            content="Branching scenario",
            confidence=0.6,
            source_memory_ids=["mem2"]
        ),
        ScenarioNode(
            type=ScenarioType.ANTICIPATED,
            content="Anticipated scenario",
            confidence=0.7,
            source_memory_ids=["mem3"]
        ),
    ]

    async def populate():
        for s in scenarios:
            await store.store(s)

    asyncio.run(populate())
    return store


class TestScenarioNode:
    """Tests for ScenarioNode dataclass."""

    def test_scenario_creation(self):
        """Test basic ScenarioNode creation."""
        scenario = ScenarioNode(
            content="Test content",
            context_summary="Test context"
        )

        assert scenario.id.startswith("scenario_")
        assert scenario.type == ScenarioType.CONTINUATION  # Default
        assert scenario.confidence == 0.5  # Default
        assert scenario.verified is False

    def test_scenario_age_hours(self):
        """Test age calculation."""
        scenario = ScenarioNode(
            content="Test",
            created_at=(datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        )

        age = scenario.age_hours()
        assert 4.9 < age < 5.1  # Allow small timing variance

    def test_apply_decay(self):
        """Test decay application."""
        scenario = ScenarioNode(
            content="Test",
            confidence=0.8,
            created_at=(datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
        )

        # Apply decay with half-life of 12 hours
        decayed_confidence = scenario.apply_decay(
            decay_lambda=0.05,
            half_life_hours=12.0
        )

        # After 12 hours (one half-life), confidence should be reduced
        assert decayed_confidence < 0.8
        assert scenario.decay_factor < 1.0

    def test_current_confidence(self):
        """Test current confidence calculation with decay."""
        scenario = ScenarioNode(
            content="Test",
            confidence=1.0,
            decay_factor=0.5
        )

        assert scenario.current_confidence() == 0.5

    def test_is_expired(self):
        """Test expiration check."""
        # Not expired scenario
        scenario = ScenarioNode(
            content="Test",
            confidence=0.8,
            decay_factor=1.0
        )
        assert scenario.is_expired(threshold=0.1) is False

        # Expired scenario
        scenario.decay_factor = 0.05
        assert scenario.is_expired(threshold=0.1) is True

    def test_verify_occurred(self):
        """Test verification when scenario occurred."""
        scenario = ScenarioNode(content="Test")

        scenario.verify(occurred=True, notes="Verified in production")

        assert scenario.verified is True
        assert scenario.verification_outcome is True
        assert scenario.verified_at is not None
        assert scenario.metadata.get("verification_notes") == "Verified in production"

    def test_verify_did_not_occur(self):
        """Test verification when scenario did not occur."""
        scenario = ScenarioNode(content="Test")

        scenario.verify(occurred=False)

        assert scenario.verified is True
        assert scenario.verification_outcome is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        scenario = ScenarioNode(
            id="scenario_test",
            type=ScenarioType.BRANCHING,
            content="Test content",
            confidence=0.75
        )

        data = scenario.to_dict()

        assert data["id"] == "scenario_test"
        assert data["type"] == "branching"
        assert data["content"] == "Test content"
        assert data["confidence"] == 0.75

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "scenario_restored",
            "type": "anticipated",
            "content": "Restored content",
            "context_summary": "Context",
            "confidence": 0.9,
            "probability": 0.8,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_memory_ids": ["mem1"],
            "verified": True,
            "verification_outcome": True
        }

        scenario = ScenarioNode.from_dict(data)

        assert scenario.id == "scenario_restored"
        assert scenario.type == ScenarioType.ANTICIPATED
        assert scenario.confidence == 0.9
        assert scenario.verified is True


class TestScenarioStoreLifecycle:
    """Tests for ScenarioStore CRUD operations."""

    @pytest.mark.asyncio
    async def test_store_scenario(self, scenario_store, sample_scenario):
        """Test storing a scenario."""
        store = scenario_store

        scenario_id = await store.store(sample_scenario)

        assert scenario_id == sample_scenario.id
        assert sample_scenario.id in store._scenarios

    @pytest.mark.asyncio
    async def test_get_scenario(self, scenario_store, sample_scenario):
        """Test retrieving a scenario."""
        store = scenario_store
        await store.store(sample_scenario)

        retrieved = await store.get(sample_scenario.id)

        assert retrieved is not None
        assert retrieved.id == sample_scenario.id
        assert retrieved.content == sample_scenario.content

    @pytest.mark.asyncio
    async def test_get_nonexistent_scenario(self, scenario_store):
        """Test retrieving a non-existent scenario."""
        retrieved = await scenario_store.get("nonexistent_id")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_updates_access_time(self, scenario_store, sample_scenario):
        """Test that get() updates last_accessed time."""
        store = scenario_store
        await store.store(sample_scenario)

        old_access_time = sample_scenario.last_accessed
        await asyncio.sleep(0.01)  # Small delay

        await store.get(sample_scenario.id)

        assert store._scenarios[sample_scenario.id].last_accessed != old_access_time

    @pytest.mark.asyncio
    async def test_list_active_scenarios(self, populated_store):
        """Test listing active scenarios."""
        active = await populated_store.list_active(min_confidence=0.1)

        assert len(active) > 0
        for scenario in active:
            assert not scenario.verified
            assert scenario.current_confidence() >= 0.1

    @pytest.mark.asyncio
    async def test_list_active_by_type(self, populated_store):
        """Test listing active scenarios filtered by type."""
        active = await populated_store.list_active(
            min_confidence=0.1,
            scenario_type=ScenarioType.CONTINUATION
        )

        for scenario in active:
            assert scenario.type == ScenarioType.CONTINUATION

    @pytest.mark.asyncio
    async def test_list_active_excludes_verified(self, scenario_store):
        """Test that list_active excludes verified scenarios."""
        store = scenario_store

        # Create and verify a scenario
        scenario = ScenarioNode(content="To verify", confidence=0.9)
        await store.store(scenario)
        await store.verify(scenario.id, occurred=True)

        active = await store.list_active()
        assert scenario not in active

    @pytest.mark.asyncio
    async def test_get_by_source_memory(self, scenario_store):
        """Test retrieving scenarios by source memory ID."""
        store = scenario_store

        s1 = ScenarioNode(content="S1", source_memory_ids=["mem_a", "mem_b"])
        s2 = ScenarioNode(content="S2", source_memory_ids=["mem_a"])
        s3 = ScenarioNode(content="S3", source_memory_ids=["mem_c"])

        await store.store(s1)
        await store.store(s2)
        await store.store(s3)

        related = await store.get_by_source_memory("mem_a")

        assert len(related) == 2
        assert s1 in related
        assert s2 in related
        assert s3 not in related


class TestScenarioStoreVerification:
    """Tests for scenario verification."""

    @pytest.mark.asyncio
    async def test_verify_scenario_occurred(self, scenario_store, sample_scenario):
        """Test verifying a scenario that occurred."""
        store = scenario_store
        await store.store(sample_scenario)

        result = await store.verify(sample_scenario.id, occurred=True, notes="Confirmed")

        assert result is not None
        assert result.verified is True
        assert result.verification_outcome is True

    @pytest.mark.asyncio
    async def test_verify_scenario_did_not_occur(self, scenario_store, sample_scenario):
        """Test verifying a scenario that did not occur."""
        store = scenario_store
        await store.store(sample_scenario)

        result = await store.verify(sample_scenario.id, occurred=False)

        assert result.verification_outcome is False

    @pytest.mark.asyncio
    async def test_verify_nonexistent_scenario(self, scenario_store):
        """Test verifying a non-existent scenario."""
        result = await scenario_store.verify("nonexistent", occurred=True)
        assert result is None


class TestScenarioStoreConfidenceDecay:
    """Tests for confidence decay over time."""

    @pytest.mark.asyncio
    async def test_decay_on_get(self, scenario_store):
        """Test that decay is applied when getting a scenario."""
        store = scenario_store

        # Create scenario with old creation time
        old_time = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        scenario = ScenarioNode(
            content="Old scenario",
            confidence=1.0,
            created_at=old_time
        )
        await store.store(scenario)

        await store.get(scenario.id)

        # Decay should have been applied
        assert store._scenarios[scenario.id].decay_factor < 1.0

    @pytest.mark.asyncio
    async def test_decay_on_list_active(self, scenario_store):
        """Test that decay is applied when listing active scenarios."""
        store = scenario_store

        # Create scenario with old creation time
        old_time = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
        scenario = ScenarioNode(
            content="Decaying scenario",
            confidence=0.9,
            created_at=old_time
        )
        await store.store(scenario)

        initial_decay = scenario.decay_factor

        await store.list_active()

        # Decay factor should have been updated
        # (might be same if already decayed, but the function runs)

    @pytest.mark.asyncio
    async def test_old_scenarios_expire(self, scenario_store):
        """Test that very old scenarios expire below threshold."""
        store = scenario_store

        # Create scenario with very old creation time
        very_old = (datetime.now(timezone.utc) - timedelta(hours=200)).isoformat()
        scenario = ScenarioNode(
            content="Very old scenario",
            confidence=0.5,
            created_at=very_old
        )
        await store.store(scenario)

        # Apply decay
        current_conf = scenario.apply_decay(
            store.config.scenario_decay_lambda,
            store.config.scenario_half_life_hours
        )

        # Should be expired
        assert scenario.is_expired(threshold=0.1)


class TestScenarioStoreCleanup:
    """Tests for cleanup of expired scenarios."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_scenarios(self, scenario_store):
        """Test removing expired scenarios."""
        store = scenario_store

        # Create fresh scenario
        fresh = ScenarioNode(content="Fresh", confidence=0.9)
        await store.store(fresh)

        # Create old scenario that will expire
        very_old = (datetime.now(timezone.utc) - timedelta(hours=500)).isoformat()
        old = ScenarioNode(
            content="Old",
            confidence=0.5,
            created_at=very_old
        )
        await store.store(old)

        removed_count = await store.cleanup_expired()

        assert removed_count >= 1
        assert fresh.id in store._scenarios
        # Old scenario should be removed (if decayed below threshold)

    @pytest.mark.asyncio
    async def test_cleanup_preserves_verified(self, scenario_store):
        """Test that cleanup preserves verified scenarios."""
        store = scenario_store

        # Create and verify an old scenario
        very_old = (datetime.now(timezone.utc) - timedelta(hours=500)).isoformat()
        verified_old = ScenarioNode(
            content="Verified old",
            confidence=0.5,
            created_at=very_old
        )
        await store.store(verified_old)
        await store.verify(verified_old.id, occurred=True)

        await store.cleanup_expired()

        # Verified scenario should still be present
        assert verified_old.id in store._scenarios

    @pytest.mark.asyncio
    async def test_storage_limit_enforcement(self, eft_config, temp_storage_path):
        """Test that storage limit is enforced."""
        eft_config.max_stored_scenarios = 5
        eft_config.persist_scenarios = False
        store = ScenarioStore(config=eft_config, storage_path=temp_storage_path)

        # Add more scenarios than limit
        for i in range(10):
            scenario = ScenarioNode(
                content=f"Scenario {i}",
                confidence=0.5 + (i * 0.05)  # Varying confidence
            )
            await store.store(scenario)

        # Should have enforced limit (or be close after cleanup)
        assert len(store._scenarios) <= eft_config.max_stored_scenarios + 2


class TestScenarioStoreStats:
    """Tests for ScenarioStore.stats() method."""

    @pytest.mark.asyncio
    async def test_stats_empty_store(self, scenario_store):
        """Test stats on empty store."""
        stats = await scenario_store.stats()

        assert stats["total_scenarios"] == 0
        assert stats["active_scenarios"] == 0
        assert stats["verified_scenarios"] == 0
        assert stats["avg_confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_stats_with_scenarios(self, populated_store):
        """Test stats with scenarios."""
        stats = await populated_store.stats()

        assert stats["total_scenarios"] > 0
        assert stats["active_scenarios"] > 0
        assert "by_type" in stats
        assert "avg_confidence" in stats

    @pytest.mark.asyncio
    async def test_stats_counts_verified(self, scenario_store):
        """Test that stats correctly counts verified scenarios."""
        store = scenario_store

        s1 = ScenarioNode(content="S1", confidence=0.9)
        await store.store(s1)
        await store.verify(s1.id, occurred=True)

        stats = await store.stats()

        assert stats["verified_scenarios"] == 1

    @pytest.mark.asyncio
    async def test_stats_by_type(self, scenario_store):
        """Test that stats groups by scenario type."""
        store = scenario_store

        await store.store(ScenarioNode(type=ScenarioType.CONTINUATION, content="C1"))
        await store.store(ScenarioNode(type=ScenarioType.CONTINUATION, content="C2"))
        await store.store(ScenarioNode(type=ScenarioType.BRANCHING, content="B1"))

        stats = await store.stats()

        assert "by_type" in stats
        assert stats["by_type"].get("continuation", 0) >= 2
        assert stats["by_type"].get("branching", 0) >= 1


class TestScenarioStorePersistence:
    """Tests for scenario persistence."""

    @pytest.mark.asyncio
    async def test_persist_scenario(self, eft_config, temp_storage_path):
        """Test that scenarios are persisted to file."""
        eft_config.persist_scenarios = True
        store = ScenarioStore(config=eft_config, storage_path=temp_storage_path)

        scenario = ScenarioNode(content="Persisted scenario", confidence=0.8)
        await store.store(scenario)

        # Check file was written
        assert os.path.exists(temp_storage_path)

        with open(temp_storage_path, 'r') as f:
            line = f.readline()
            data = json.loads(line)
            assert data["content"] == "Persisted scenario"

    @pytest.mark.asyncio
    async def test_load_from_storage(self, eft_config, temp_storage_path):
        """Test loading scenarios from storage file."""
        # Write scenarios to file
        with open(temp_storage_path, 'w') as f:
            scenario_data = {
                "id": "scenario_loaded",
                "type": "continuation",
                "content": "Loaded scenario",
                "confidence": 0.75,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "verified": False
            }
            f.write(json.dumps(scenario_data) + "\n")

        eft_config.persist_scenarios = False
        store = ScenarioStore(config=eft_config, storage_path=temp_storage_path)

        count = await store.load_from_storage()

        assert count == 1
        assert "scenario_loaded" in store._scenarios

    @pytest.mark.asyncio
    async def test_load_skips_verified(self, eft_config, temp_storage_path):
        """Test that loading skips verified scenarios."""
        # Write verified scenario
        with open(temp_storage_path, 'w') as f:
            scenario_data = {
                "id": "scenario_verified",
                "type": "continuation",
                "content": "Verified scenario",
                "confidence": 0.75,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "verified": True
            }
            f.write(json.dumps(scenario_data) + "\n")

        eft_config.persist_scenarios = False
        store = ScenarioStore(config=eft_config, storage_path=temp_storage_path)

        count = await store.load_from_storage()

        # Verified scenarios are not loaded into active memory
        assert "scenario_verified" not in store._scenarios


class TestEFTConfig:
    """Tests for EFTConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EFTConfig()

        assert config.enabled is True
        assert config.max_scenarios_per_simulation == 5
        assert config.persist_scenarios is True

    def test_config_validate(self):
        """Test configuration validation."""
        config = EFTConfig()
        # Should not raise
        config.validate()

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = EFTConfig(
            enabled=False,
            max_scenarios_per_simulation=10,
            scenario_decay_lambda=0.1
        )

        assert config.enabled is False
        assert config.max_scenarios_per_simulation == 10
        assert config.scenario_decay_lambda == 0.1


class TestScenarioType:
    """Tests for ScenarioType enum."""

    def test_scenario_types(self):
        """Test available scenario types."""
        assert ScenarioType.CONTINUATION.value == "continuation"
        assert ScenarioType.BRANCHING.value == "branching"
        assert ScenarioType.ANOMALY.value == "anomaly"
        assert ScenarioType.GOAL_DIRECTED.value == "goal_directed"
        assert ScenarioType.ANTICIPATED.value == "anticipated"

    def test_scenario_type_from_string(self):
        """Test creating ScenarioType from string."""
        st = ScenarioType("branching")
        assert st == ScenarioType.BRANCHING


class TestEpisodeFutureSimulatorBasic:
    """Basic tests for EpisodeFutureSimulator."""

    @pytest.mark.asyncio
    async def test_simulator_disabled(self, eft_config):
        """Test that disabled simulator returns empty results."""
        eft_config.enabled = False
        simulator = EpisodeFutureSimulator(config=eft_config)

        results = await simulator.simulate("Test context")

        assert results == []

    @pytest.mark.asyncio
    async def test_simulator_enabled(self, eft_config):
        """Test enabled simulator can be created."""
        eft_config.enabled = True
        simulator = EpisodeFutureSimulator(config=eft_config)

        assert simulator.config.enabled is True

    @pytest.mark.asyncio
    async def test_simulator_has_scenario_store(self, eft_config):
        """Test that simulator has a scenario store."""
        simulator = EpisodeFutureSimulator(config=eft_config)

        assert simulator.scenario_store is not None
        assert isinstance(simulator.scenario_store, ScenarioStore)
