"""
Comprehensive Tests for Immunology Module
========================================

Tests the vector immune system including:
- ImmunologyConfig
- Bit entropy calculation
- ImmunologyLoop lifecycle
- Node assessment
- Drift detection
- Corruption quarantine
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from mnemocore.core.immunology import (
    ImmunologyConfig,
    ImmunologyLoop,
    _bit_entropy,
)
from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.node import MemoryNode


class TestImmunologyConfig:
    """Test ImmunologyConfig dataclass."""

    def test_default_config(self):
        """Default configuration should have sensible values."""
        config = ImmunologyConfig()
        assert config.sweep_interval_seconds == 300.0
        assert config.drift_threshold == 0.40
        assert config.entropy_threshold == 0.48
        assert config.min_ltp_to_keep == 0.05
        assert config.attractor_k == 5
        assert config.attractor_enabled is True
        assert config.re_encode_drifted is True
        assert config.quarantine_corrupted is True
        assert config.enabled is True

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = ImmunologyConfig(
            sweep_interval_seconds=600.0,
            drift_threshold=0.30,
            entropy_threshold=0.50,
            min_ltp_to_keep=0.1,
            attractor_k=10,
            attractor_enabled=False,
            re_encode_drifted=False,
            quarantine_corrupted=False,
            enabled=False,
        )
        assert config.sweep_interval_seconds == 600.0
        assert config.drift_threshold == 0.30
        assert config.entropy_threshold == 0.50
        assert config.attractor_enabled is False


class TestBitEntropy:
    """Test bit entropy calculation."""

    def test_entropy_balanced_vector(self):
        """Balanced vector (~50% ones) should have entropy ~1.0."""
        hdv = BinaryHDV.random(1024)
        entropy = _bit_entropy(hdv)
        # Random vector should have high entropy
        assert entropy > 0.9

    def test_entropy_zero_vector(self):
        """Zero vector should have entropy 0.0."""
        hdv = BinaryHDV.zeros(1024)
        entropy = _bit_entropy(hdv)
        assert entropy == 0.0

    def test_entropy_ones_vector(self):
        """All-ones vector should have entropy 0.0."""
        hdv = BinaryHDV.ones(1024)
        entropy = _bit_entropy(hdv)
        # All ones = p=1 = entropy 0
        assert entropy == 0.0

    def test_entropy_in_range(self):
        """Entropy should always be in [0, 1]."""
        for _ in range(10):
            hdv = BinaryHDV.random(1024)
            entropy = _bit_entropy(hdv)
            assert 0.0 <= entropy <= 1.0

    def test_entropy_symmetric(self):
        """Vectors with complementary bit patterns should have same entropy."""
        hdv1 = BinaryHDV.random(1024)
        # Create complement by XOR with all-ones
        hdv2 = hdv1.invert()

        entropy1 = _bit_entropy(hdv1)
        entropy2 = _bit_entropy(hdv2)

        assert abs(entropy1 - entropy2) < 0.01


class TestImmunologyLoopInit:
    """Test ImmunologyLoop initialization."""

    def test_initialization(self):
        """Should initialize with engine and config."""
        mock_engine = MagicMock()
        config = ImmunologyConfig(sweep_interval_seconds=100)

        loop = ImmunologyLoop(mock_engine, config)

        assert loop.engine == mock_engine
        assert loop.cfg is config
        assert loop._running is False
        assert loop.last_sweep is None

    def test_initialization_default_config(self):
        """Should use default config when none provided."""
        mock_engine = MagicMock()

        loop = ImmunologyLoop(mock_engine)

        assert loop.cfg is not None
        assert loop.cfg.sweep_interval_seconds == 300.0

    def test_initializes_stats(self):
        """Should initialize stats dict."""
        mock_engine = MagicMock()

        loop = ImmunologyLoop(mock_engine)

        assert "sweeps" in loop.cumulative_stats
        assert "drifted_corrected" in loop.cumulative_stats
        assert "corrupted_quarantined" in loop.cumulative_stats
        assert loop.cumulative_stats["sweeps"] == 0


class TestImmunologyLoopLifecycle:
    """Test ImmunologyLoop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_when_disabled(self):
        """Start should return immediately when disabled."""
        mock_engine = MagicMock()
        config = ImmunologyConfig(enabled=False)

        loop = ImmunologyLoop(mock_engine, config)

        await loop.start()

        assert loop._running is False

    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        """Start should create background task."""
        mock_engine = MagicMock()
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=[])

        loop = ImmunologyLoop(mock_engine)

        await loop.start()

        assert loop._running is True
        assert loop._task is not None

        # Cleanup
        await loop.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        """Stop should cancel background task."""
        mock_engine = MagicMock()
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=[])

        loop = ImmunologyLoop(mock_engine)
        await loop.start()

        await loop.stop()

        assert loop._running is False

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        """Stop should be safe to call multiple times."""
        mock_engine = MagicMock()
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=[])

        loop = ImmunologyLoop(mock_engine)
        await loop.start()

        await loop.stop()
        await loop.stop()  # Should not raise

        assert loop._running is False


class TestImmunologyLoopSweep:
    """Test ImmunologyLoop sweep functionality."""

    @pytest.mark.asyncio
    async def test_sweep_empty_tier(self):
        """Sweep should handle empty HOT tier."""
        mock_engine = MagicMock()
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=[])
        mock_engine.cleanup_decay = AsyncMock()

        loop = ImmunologyLoop(mock_engine)

        stats = await loop.sweep()

        assert stats == {}

    @pytest.mark.asyncio
    async def test_sweep_calls_cleanup_decay(self):
        """Sweep should delegate stale synapse cleanup."""
        mock_engine = MagicMock()
        node = MagicMock()
        node.id = "n1"
        node.hdv = BinaryHDV.random(1024)
        node.ltp_strength = 0.5
        node.content = "test"
        node.metadata = {}
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=[node])
        mock_engine.cleanup_decay = AsyncMock()

        loop = ImmunologyLoop(mock_engine)
        await loop.sweep()

        mock_engine.cleanup_decay.assert_called_once_with(threshold=0.05)

    @pytest.mark.asyncio
    async def test_sweep_returns_stats(self):
        """Sweep should return statistics."""
        mock_engine = MagicMock()
        mock_engine.tier_manager = MagicMock()
        mock_engine.cleanup_decay = AsyncMock()

        # Create mock nodes
        nodes = []
        for i in range(5):
            node = MagicMock()
            node.id = f"node{i}"
            node.hdv = BinaryHDV.random(1024)
            node.ltp_strength = 0.5
            node.content = f"content {i}"
            node.metadata = {}
            nodes.append(node)

        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=nodes)
        mock_engine.encode_content = MagicMock(return_value=BinaryHDV.random(1024))
        mock_engine.cleanup_decay = AsyncMock()

        loop = ImmunologyLoop(mock_engine)

        stats = await loop.sweep()

        assert "nodes_scanned" in stats
        assert stats["nodes_scanned"] == 5
        assert "elapsed_seconds" in stats

    @pytest.mark.asyncio
    async def test_sweep_updates_cumulative_stats(self):
        """Sweep should update cumulative statistics."""
        mock_engine = MagicMock()
        mock_engine.tier_manager = MagicMock()
        mock_engine.cleanup_decay = AsyncMock()

        nodes = [MagicMock(id="n1", hdv=BinaryHDV.random(1024),
                         ltp_strength=0.5, content="test", metadata={})]
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=nodes)

        loop = ImmunologyLoop(mock_engine)
        await loop.sweep()

        assert loop.cumulative_stats["sweeps"] == 1
        assert loop.last_sweep is not None


class TestImmunologyLoopAssessNode:
    """Test node assessment logic."""

    @pytest.mark.asyncio
    async def test_assess_node_ok(self):
        """Healthy node should return 'ok'."""
        mock_engine = MagicMock()
        mock_engine.encode_content = MagicMock(return_value=BinaryHDV.random(1024))

        loop = ImmunologyLoop(mock_engine)

        node = MagicMock()
        node.id = "node1"
        node.hdv = BinaryHDV.random(1024)
        node.ltp_strength = 0.7
        node.content = "healthy content"
        node.metadata = {}

        all_nodes = [node]
        vecs = np.array([node.hdv.data])

        result = await loop._assess_node(node, 0, all_nodes, vecs)

        assert result == "ok"

    @pytest.mark.asyncio
    async def test_assess_node_quarantines_corrupted(self):
        """Should quarantine corrupted low-LTP nodes."""
        mock_engine = MagicMock()
        mock_engine.encode_content = MagicMock(return_value=BinaryHDV.random(1024))
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.delete_memory = AsyncMock()

        config = ImmunologyConfig(
            entropy_threshold=0.6,  # Higher threshold
            min_ltp_to_keep=0.5,
            quarantine_corrupted=True
        )
        loop = ImmunologyLoop(mock_engine, config)

        # Create low-entropy (corrupted) node
        node = MagicMock()
        node.id = "corrupted"
        node.hdv = BinaryHDV.zeros(1024)  # Low entropy
        node.ltp_strength = 0.3  # Below threshold
        node.content = "corrupted"
        node.metadata = {}

        all_nodes = [node]
        vecs = np.array([node.hdv.data])

        result = await loop._assess_node(node, 0, all_nodes, vecs)

        assert result == "quarantined"
        mock_engine.tier_manager.delete_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_assess_node_re_encodes_drifted(self):
        """Should re-encode drifted nodes."""
        mock_engine = MagicMock()

        new_hdv = BinaryHDV.from_seed("corrected", 1024)
        mock_engine.encode_content = MagicMock(return_value=new_hdv)

        config = ImmunologyConfig(
            drift_threshold=0.1,  # Very low - almost anything will be drifted
            re_encode_drifted=True
        )
        loop = ImmunologyLoop(mock_engine, config)

        # Create two very different vectors
        node = MagicMock()
        node.id = "drifted"
        node.hdv = BinaryHDV.from_seed("original", 1024)
        node.ltp_strength = 0.7
        node.content = "content"
        node.metadata = {}

        # Very different neighbor
        other = MagicMock()
        other.hdv = BinaryHDV.random(1024)

        all_nodes = [node, other]
        vecs = np.array([n.hdv.data for n in all_nodes])

        result = await loop._assess_node(node, 0, all_nodes, vecs)

        assert result == "corrected"
        assert node.hdv == new_hdv
        assert "immune_re_encoded_at" in node.metadata

    @pytest.mark.asyncio
    async def test_assess_node_attractor_convergence(self):
        """Should apply attractor convergence when enabled."""
        mock_engine = MagicMock()
        mock_engine.encode_content = MagicMock(return_value=BinaryHDV.random(1024))

        config = ImmunologyConfig(
            drift_threshold=0.1,
            re_encode_drifted=False,  # Don't re-encode
            attractor_enabled=True  # Use attractor
        )
        loop = ImmunologyLoop(mock_engine, config)

        node = MagicMock()
        node.id = "node1"
        node.hdv = BinaryHDV.from_seed("drifted", 1024)
        node.ltp_strength = 0.7
        node.content = "content"
        node.metadata = {}

        # Create similar neighbors for attractor
        neighbors = []
        for i in range(5):
            n = MagicMock()
            n.hdv = BinaryHDV.from_seed("similar", 1024)
            neighbors.append(n)

        all_nodes = [node] + neighbors
        vecs = np.array([n.hdv.data for n in all_nodes])

        result = await loop._assess_node(node, 0, all_nodes, vecs)

        # Should converge to attractor (bundled neighbors)
        assert result == "ok" or result == "corrected"


class TestImmunologyLoopBackground:
    """Test background loop behavior."""

    @pytest.mark.asyncio
    async def test_loop_runs_periodically(self):
        """Loop should run at configured interval."""
        mock_engine = MagicMock()
        mock_engine.tier_manager = MagicMock()
        node = MagicMock()
        node.id = "n1"
        node.hdv = BinaryHDV.random(1024)
        node.ltp_strength = 0.5
        node.content = "test"
        node.metadata = {}
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=[node])
        mock_engine.encode_content = MagicMock(return_value=BinaryHDV.random(1024))
        mock_engine.cleanup_decay = AsyncMock()

        config = ImmunologyConfig(sweep_interval_seconds=0.1)
        loop = ImmunologyLoop(mock_engine, config)

        await loop.start()

        # Wait for multiple sweeps
        await asyncio.sleep(0.35)

        await loop.stop()

        # Should have run at least 2 sweeps
        assert loop.cumulative_stats["sweeps"] >= 2

    @pytest.mark.asyncio
    async def test_loop_handles_errors_gracefully(self):
        """Loop should continue after errors."""
        mock_engine = MagicMock()
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(side_effect=RuntimeError("test error"))

        config = ImmunologyConfig(sweep_interval_seconds=0.1)
        loop = ImmunologyLoop(mock_engine, config)

        await loop.start()
        await asyncio.sleep(0.2)
        await loop.stop()

        # Should have attempted sweeps
        # Loop continues despite errors


class TestImmunologyLoopStats:
    """Test statistics reporting."""

    def test_stats_property(self):
        """Stats property should return cumulative stats."""
        mock_engine = MagicMock()

        loop = ImmunologyLoop(mock_engine)

        stats = loop.stats

        assert "sweeps" in stats
        assert "drifted_corrected" in stats
        assert "corrupted_quarantined" in stats
        assert "last_sweep" in stats

    def test_stats_includes_last_sweep_time(self):
        """Should include last sweep timestamp."""
        mock_engine = MagicMock()
        mock_engine.tier_manager = MagicMock()
        node = MagicMock()
        node.id = "n1"
        node.hdv = BinaryHDV.random(1024)
        node.ltp_strength = 0.5
        node.content = "test"
        node.metadata = {}
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=[node])
        mock_engine.encode_content = MagicMock(return_value=BinaryHDV.random(1024))
        mock_engine.cleanup_decay = AsyncMock()

        loop = ImmunologyLoop(mock_engine)

        # Run a sweep synchronously in this test
        asyncio.run(loop.sweep())

        stats = loop.stats
        assert stats["last_sweep"] is not None


class TestImmunologyConfigEdgeCases:
    """Test edge cases in configuration."""

    def test_zero_attractor_k(self):
        """Should handle zero attractor_k."""
        config = ImmunologyConfig(attractor_k=0)
        assert config.attractor_k == 0

    def test_zero_drift_threshold(self):
        """Should handle zero drift threshold."""
        config = ImmunologyConfig(drift_threshold=0.0)
        assert config.drift_threshold == 0.0

    def test_zero_sweep_interval(self):
        """Should handle zero sweep interval."""
        config = ImmunologyConfig(sweep_interval_seconds=0.0)
        assert config.sweep_interval_seconds == 0.0


class TestImmunologyPropertyBased:
    """Property-based tests using Hypothesis."""

    from hypothesis import given, strategies as st

    @given(st.integers(min_value=64, max_value=2048).map(lambda x: x * 8))
    def test_entropy_always_valid(self, dimension):
        """Entropy should always be in valid range."""
        hdv = BinaryHDV.random(dimension)
        entropy = _bit_entropy(hdv)
        assert 0.0 <= entropy <= 1.0

    @pytest.mark.asyncio
    @given(st.integers(min_value=0, max_value=100))
    async def test_sweep_handles_various_node_counts(self, node_count):
        """Sweep should handle various numbers of nodes."""
        mock_engine = MagicMock()
        mock_engine.cleanup_decay = AsyncMock()

        nodes = []
        for i in range(node_count):
            node = MagicMock()
            node.id = f"node{i}"
            node.hdv = BinaryHDV.random(1024)
            node.ltp_strength = 0.5
            node.content = f"content {i}"
            node.metadata = {}
            nodes.append(node)

        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_hot_snapshot = AsyncMock(return_value=nodes)
        mock_engine.encode_content = MagicMock(return_value=BinaryHDV.random(1024))

        loop = ImmunologyLoop(mock_engine)

        stats = await loop.sweep()

        if node_count == 0:
            assert stats == {}
        else:
            assert stats["nodes_scanned"] == node_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
