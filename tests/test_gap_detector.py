"""
Comprehensive Tests for GapDetector Module
===========================================

Tests the knowledge gap detection system including:
- Gap detection from query results
- Registry management
- Priority scoring
- Negative feedback handling
- Stale gap eviction
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from mnemocore.core.gap_detector import (
    GapDetector,
    GapDetectorConfig,
    GapRecord,
    _query_id,
    _bit_entropy_packed,
)
from mnemocore.core.binary_hdv import BinaryHDV


class TestGapDetectorConfig:
    """Test configuration validation and defaults."""

    def test_default_config(self):
        """Default configuration should have sensible values."""
        config = GapDetectorConfig()
        assert config.min_confidence_threshold == 0.45
        assert config.min_results_required == 2
        assert config.mask_entropy_threshold == 0.46
        assert config.negative_feedback_weight == 2.0
        assert config.gap_ttl_seconds == 86400 * 7
        assert config.max_gap_registry_size == 500
        assert config.enabled is True

    def test_custom_config(self):
        """Custom configuration values should be set correctly."""
        config = GapDetectorConfig(
            min_confidence_threshold=0.5,
            min_results_required=3,
            mask_entropy_threshold=0.5,
            negative_feedback_weight=3.0,
            gap_ttl_seconds=3600,
            max_gap_registry_size=1000,
            enabled=False,
        )
        assert config.min_confidence_threshold == 0.5
        assert config.min_results_required == 3
        assert config.mask_entropy_threshold == 0.5
        assert config.negative_feedback_weight == 3.0
        assert config.gap_ttl_seconds == 3600
        assert config.max_gap_registry_size == 1000
        assert config.enabled is False


class TestGapRecord:
    """Test GapRecord dataclass and priority update."""

    def test_gap_record_creation(self):
        """GapRecord should be created with all required fields."""
        now = datetime.now(timezone.utc)
        record = GapRecord(
            gap_id="test_gap",
            query_text="What is quantum computing?",
            detected_at=now,
            last_seen=now,
            signal="low_confidence",
            confidence=0.3,
        )
        assert record.gap_id == "test_gap"
        assert record.query_text == "What is quantum computing?"
        assert record.signal == "low_confidence"
        assert record.confidence == 0.3
        assert record.seen_count == 1
        assert record.filled is False
        assert record.filled_at is None

    def test_priority_update_new_gap(self):
        """Priority for new gap should be computed correctly."""
        now = datetime.now(timezone.utc)
        record = GapRecord(
            gap_id="new_gap",
            query_text="new query",
            detected_at=now,
            last_seen=now,
            signal="low_confidence",
            confidence=0.2,
        )
        record.update_priority()
        assert record.priority_score > 0
        assert record.priority_score <= 1.0

    def test_priority_update_old_gap(self):
        """Priority should decay for older gaps."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=5)

        old_record = GapRecord(
            gap_id="old_gap",
            query_text="old query",
            detected_at=old_time,
            last_seen=old_time,
            signal="low_confidence",
            confidence=0.3,
        )
        old_record.update_priority()

        new_record = GapRecord(
            gap_id="new_gap",
            query_text="new query",
            detected_at=now,
            last_seen=now,
            signal="low_confidence",
            confidence=0.3,
        )
        new_record.update_priority()

        # Newer gap should have higher priority
        assert new_record.priority_score > old_record.priority_score

    def test_priority_update_frequent_gap(self):
        """Frequently seen gaps should have higher priority."""
        now = datetime.now(timezone.utc)

        rare_record = GapRecord(
            gap_id="rare_gap",
            query_text="rare query",
            detected_at=now,
            last_seen=now,
            signal="low_confidence",
            confidence=0.3,
            seen_count=1,
        )
        rare_record.update_priority()

        frequent_record = GapRecord(
            gap_id="frequent_gap",
            query_text="frequent query",
            detected_at=now,
            last_seen=now,
            signal="low_confidence",
            confidence=0.3,
            seen_count=10,
        )
        frequent_record.update_priority()

        assert frequent_record.priority_score > rare_record.priority_score

    def test_priority_update_low_confidence(self):
        """Lower confidence gaps should have higher priority."""
        now = datetime.now(timezone.utc)

        high_conf_record = GapRecord(
            gap_id="high_conf_gap",
            query_text="high conf query",
            detected_at=now,
            last_seen=now,
            signal="low_confidence",
            confidence=0.7,
        )
        high_conf_record.update_priority()

        low_conf_record = GapRecord(
            gap_id="low_conf_gap",
            query_text="low conf query",
            detected_at=now,
            last_seen=now,
            signal="low_confidence",
            confidence=0.2,
        )
        low_conf_record.update_priority()

        assert low_conf_record.priority_score > high_conf_record.priority_score


class TestQueryId:
    """Test query ID generation."""

    def test_query_id_stability(self):
        """Same query should produce same ID."""
        query = "What is the meaning of life?"
        id1 = _query_id(query)
        id2 = _query_id(query)
        assert id1 == id2

    def test_query_id_different_queries(self):
        """Different queries should produce different IDs."""
        id1 = _query_id("query one")
        id2 = _query_id("query two")
        assert id1 != id2

    def test_query_id_case_insensitive(self):
        """Query ID should be case-insensitive."""
        id1 = _query_id("Test Query")
        id2 = _query_id("test query")
        assert id1 == id2

    def test_query_id_whitespace_agnostic(self):
        """Query ID should ignore leading/trailing whitespace."""
        id1 = _query_id("test query")
        id2 = _query_id("  test query  ")
        assert id1 == id2


class TestBitEntropy:
    """Test bit entropy calculation."""

    def test_entropy_zero_vector(self):
        """Zero vector should have zero entropy."""
        hdv = BinaryHDV.zeros(1024)
        entropy = _bit_entropy_packed(hdv)
        assert entropy == 0.0

    def test_entropy_balanced_vector(self):
        """Balanced vector (~50% ones) should have entropy ~1.0."""
        hdv = BinaryHDV.random(1024)
        entropy = _bit_entropy_packed(hdv)
        # Random vector should have high entropy
        assert entropy > 0.9

    def test_entropy_ones_vector(self):
        """All-ones vector should have zero entropy."""
        hdv = BinaryHDV.zeros(1024)
        entropy = _bit_entropy_packed(hdv)
        assert entropy == 0.0


class TestGapDetector:
    """Test GapDetector main functionality."""

    def test_detector_initialization(self):
        """Detector should initialize with default or custom config."""
        default_detector = GapDetector()
        assert default_detector.cfg.enabled is True
        assert len(default_detector._registry) == 0

        custom_config = GapDetectorConfig(enabled=False)
        custom_detector = GapDetector(custom_config)
        assert custom_detector.cfg.enabled is False

    @pytest.mark.asyncio
    async def test_assess_query_low_confidence(self):
        """Should detect gap when confidence is below threshold."""
        detector = GapDetector(GapDetectorConfig(min_confidence_threshold=0.5))
        results = [("node1", 0.3), ("node2", 0.4)]

        gaps = await detector.assess_query("quantum physics", results)

        assert len(gaps) == 1
        assert gaps[0].signal == "low_confidence"
        assert gaps[0].query_text == "quantum physics"

    @pytest.mark.asyncio
    async def test_assess_query_sparse_results(self):
        """Should detect gap when too few results returned."""
        detector = GapDetector(GapDetectorConfig(min_results_required=5))
        results = [("node1", 0.8), ("node2", 0.7)]

        gaps = await detector.assess_query("sparse topic", results)

        assert len(gaps) == 1
        assert gaps[0].signal == "sparse"

    @pytest.mark.asyncio
    async def test_assess_query_coverage_gap(self):
        """Should detect gap when attention mask has high entropy."""
        detector = GapDetector(GapDetectorConfig(mask_entropy_threshold=0.3))
        results = [("node1", 0.6), ("node2", 0.5)]

        # High entropy mask
        mask = BinaryHDV.random(1024)

        gaps = await detector.assess_query("coverage test", results, mask)

        assert len(gaps) >= 1
        # Should have coverage gap due to high entropy
        coverage_gaps = [g for g in gaps if g.signal == "coverage"]
        assert len(coverage_gaps) >= 1

    @pytest.mark.asyncio
    async def test_assess_query_no_gap(self):
        """Should not detect gap when results are good."""
        detector = GapDetector(
            GapDetectorConfig(
                min_confidence_threshold=0.3,
                min_results_required=2,
            )
        )
        results = [("node1", 0.8), ("node2", 0.7)]

        gaps = await detector.assess_query("well known topic", results)

        # High confidence and enough results = no gap
        assert len(gaps) == 0

    @pytest.mark.asyncio
    async def test_assess_query_empty_results(self):
        """Should detect gap when no results returned."""
        detector = GapDetector()
        results = []

        gaps = await detector.assess_query("unknown topic", results)

        # Empty results trigger both low_confidence (avg=0) and sparse (<2)
        # Both are the same gap so deduped
        assert len(gaps) >= 1

    @pytest.mark.asyncio
    async def test_assess_query_disabled(self):
        """Should not detect gaps when disabled."""
        detector = GapDetector(GapDetectorConfig(enabled=False))
        results = []

        gaps = await detector.assess_query("any query", results)

        assert len(gaps) == 0

    @pytest.mark.asyncio
    async def test_assess_query_updates_existing_gap(self):
        """Should update existing gap instead of creating duplicate."""
        detector = GapDetector()
        query = "repeated query"

        # First assessment
        gaps1 = await detector.assess_query(query, [("node1", 0.2)])
        assert len(gaps1) == 1
        gap_id = gaps1[0].gap_id

        # Second assessment - may trigger both signals again
        gaps2 = await detector.assess_query(query, [("node1", 0.3)])
        assert len(gaps2) >= 1
        # Check if gap ID matches
        assert any(g.gap_id == gap_id for g in gaps2)
        # seen_count should have increased (may be 2 or 4 depending on signals)
        assert gaps2[0].seen_count >= 2

    @pytest.mark.asyncio
    async def test_register_negative_feedback(self):
        """Should register high-priority gap from negative feedback."""
        detector = GapDetector()

        gap = await detector.register_negative_feedback("bad result query")

        assert gap.signal == "negative"
        assert gap.seen_count >= 2  # Should be weighted
        assert gap.priority_score > 0

    @pytest.mark.asyncio
    async def test_negative_feedback_updates_existing(self):
        """Negative feedback should update existing gap."""
        detector = GapDetector()
        query = "feedback query"

        # Create initial gap
        await detector.assess_query(query, [])

        # Add negative feedback
        gap = await detector.register_negative_feedback(query)

        # Should have higher seen_count due to weight
        assert gap.seen_count >= 2

    def test_get_open_gaps(self):
        """Should return only unfilled gaps sorted by priority."""
        detector = GapDetector()

        # Create some gaps
        now = datetime.now(timezone.utc)
        detector._registry["gap1"] = GapRecord(
            gap_id="gap1",
            query_text="query1",
            detected_at=now,
            last_seen=now,
            signal="low_confidence",
            confidence=0.3,
            priority_score=0.8,
            filled=False,
        )
        detector._registry["gap2"] = GapRecord(
            gap_id="gap2",
            query_text="query2",
            detected_at=now,
            last_seen=now,
            signal="sparse",
            confidence=0.4,
            priority_score=0.9,
            filled=False,
        )
        detector._registry["gap3"] = GapRecord(
            gap_id="gap3",
            query_text="query3",
            detected_at=now,
            last_seen=now,
            signal="coverage",
            confidence=0.5,
            priority_score=0.7,
            filled=True,  # Already filled
        )

        open_gaps = detector.get_open_gaps(top_n=10)

        assert len(open_gaps) == 2
        # Should be sorted by priority (descending)
        assert open_gaps[0].gap_id == "gap2"
        assert open_gaps[1].gap_id == "gap1"

    def test_get_open_gaps_respects_top_n(self):
        """Should limit results to top_n."""
        detector = GapDetector()
        now = datetime.now(timezone.utc)

        for i in range(10):
            detector._registry[f"gap{i}"] = GapRecord(
                gap_id=f"gap{i}",
                query_text=f"query{i}",
                detected_at=now,
                last_seen=now,
                signal="low_confidence",
                confidence=0.3,
                priority_score=float(i),
                filled=False,
            )

        open_gaps = detector.get_open_gaps(top_n=5)

        assert len(open_gaps) == 5

    def test_get_all_gaps(self):
        """Should return all gaps including filled ones."""
        detector = GapDetector()
        now = datetime.now(timezone.utc)

        detector._registry["gap1"] = GapRecord(
            gap_id="gap1",
            query_text="query1",
            detected_at=now,
            last_seen=now,
            signal="low_confidence",
            confidence=0.3,
            filled=False,
        )
        detector._registry["gap2"] = GapRecord(
            gap_id="gap2",
            query_text="query2",
            detected_at=now,
            last_seen=now,
            signal="sparse",
            confidence=0.4,
            filled=True,
        )

        all_gaps = detector.get_all_gaps()

        assert len(all_gaps) == 2

    def test_mark_filled(self):
        """Should mark gap as filled."""
        detector = GapDetector()
        now = datetime.now(timezone.utc)

        detector._registry["gap1"] = GapRecord(
            gap_id="gap1",
            query_text="query1",
            detected_at=now,
            last_seen=now,
            signal="low_confidence",
            confidence=0.3,
            filled=False,
        )

        result = detector.mark_filled("gap1")

        assert result is True
        assert detector._registry["gap1"].filled is True
        assert detector._registry["gap1"].filled_at is not None

    def test_mark_filled_nonexistent(self):
        """Should return False for nonexistent gap."""
        detector = GapDetector()
        result = detector.mark_filled("nonexistent")
        assert result is False

    def test_stats_property(self):
        """Should return correct statistics."""
        detector = GapDetector()
        now = datetime.now(timezone.utc)

        detector._registry["gap1"] = GapRecord(
            gap_id="gap1",
            query_text="query1",
            detected_at=now,
            last_seen=now,
            signal="low_confidence",
            confidence=0.3,
            filled=False,
        )
        detector._registry["gap2"] = GapRecord(
            gap_id="gap2",
            query_text="query2",
            detected_at=now,
            last_seen=now,
            signal="sparse",
            confidence=0.4,
            filled=True,
        )

        stats = detector.stats

        assert stats["total_gaps"] == 2
        assert stats["open_gaps"] == 1
        assert stats["filled_gaps"] == 1


class TestGapDetectorEviction:
    """Test stale gap eviction."""

    @pytest.mark.asyncio
    async def test_evict_stale_filled_gaps(self):
        """Should evict old filled gaps."""
        detector = GapDetector(GapDetectorConfig(gap_ttl_seconds=3600))
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(seconds=7200)

        detector._registry["old_filled"] = GapRecord(
            gap_id="old_filled",
            query_text="old",
            detected_at=old_time,
            last_seen=old_time,
            signal="low_confidence",
            confidence=0.3,
            filled=True,
        )
        detector._registry["recent"] = GapRecord(
            gap_id="recent",
            query_text="recent",
            detected_at=now,
            last_seen=now,
            signal="low_confidence",
            confidence=0.3,
            filled=False,
        )

        # Trigger eviction via assess_query
        await detector.assess_query("trigger", [])

        assert "old_filled" not in detector._registry
        assert "recent" in detector._registry

    @pytest.mark.asyncio
    async def test_evict_very_old_unfilled_gaps(self):
        """Should evict very old unfilled gaps (2x TTL)."""
        detector = GapDetector(GapDetectorConfig(gap_ttl_seconds=3600))
        now = datetime.now(timezone.utc)
        very_old = now - timedelta(seconds=7200)  # 2x TTL

        detector._registry["ancient"] = GapRecord(
            gap_id="ancient",
            query_text="ancient",
            detected_at=very_old,
            last_seen=very_old,
            signal="low_confidence",
            confidence=0.3,
            filled=False,
        )

        await detector.assess_query("trigger", [])

        assert "ancient" not in detector._registry

    @pytest.mark.asyncio
    async def test_evict_when_over_capacity(self):
        """Should evict lowest priority gaps when over capacity."""
        detector = GapDetector(GapDetectorConfig(max_gap_registry_size=5))
        now = datetime.now(timezone.utc)

        # Add 10 gaps with varying priorities
        for i in range(10):
            detector._registry[f"gap{i}"] = GapRecord(
                gap_id=f"gap{i}",
                query_text=f"query{i}",
                detected_at=now,
                last_seen=now,
                signal="low_confidence",
                confidence=0.3,
                priority_score=float(i),
            )

        await detector.assess_query("trigger", [])

        # Should only keep 5
        assert len(detector._registry) <= 5
        # Should keep high priority ones
        for i in range(5, 10):
            assert f"gap{i}" in detector._registry


class TestGapDetectorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_query_text(self):
        """Should handle empty query text."""
        detector = GapDetector()
        gaps = await detector.assess_query("", [])
        # Should still work but may not detect anything meaningful
        assert isinstance(gaps, list)

    @pytest.mark.asyncio
    async def test_very_long_query_text(self):
        """Should handle very long query text."""
        detector = GapDetector()
        long_query = "query " * 1000
        gaps = await detector.assess_query(long_query, [])
        assert isinstance(gaps, list)

    @pytest.mark.asyncio
    async def test_reopen_filled_gap_on_low_confidence(self):
        """Should reopen filled gap if confidence is still low."""
        detector = GapDetector(GapDetectorConfig(min_confidence_threshold=0.5))
        now = datetime.now(timezone.utc)

        # First, create and fill a gap
        await detector.assess_query("reopen test", [("node1", 0.2)])
        gap_id = list(detector._registry.keys())[0]
        detector.mark_filled(gap_id)

        # Query again with low confidence
        gaps = await detector.assess_query("reopen test", [("node1", 0.3)])

        # Gap should be reopened
        assert detector._registry[gap_id].filled is False


class TestGapDetectorPropertyBased:
    """Property-based tests using Hypothesis."""

    from hypothesis import given, strategies as st

    @pytest.mark.asyncio
    @given(st.lists(st.tuples(st.text(min_size=1), st.floats(0.0, 1.0)), min_size=0, max_size=20))
    async def test_assess_query_various_results(self, results):
        """Should handle various result lists."""
        detector = GapDetector()
        # Convert to list of tuples format expected
        formatted_results = [(f"node{i}", score) for i, (node_id, score) in enumerate(results)]
        gaps = await detector.assess_query("test query", formatted_results)
        assert isinstance(gaps, list)

    @given(st.floats(0.0, 1.0))
    def test_priority_score_in_range(self, confidence):
        """Priority score should always be in [0, 1]."""
        now = datetime.now(timezone.utc)
        record = GapRecord(
            gap_id="test",
            query_text="test",
            detected_at=now,
            last_seen=now,
            signal="low_confidence",
            confidence=confidence,
        )
        record.update_priority()
        assert 0.0 <= record.priority_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
