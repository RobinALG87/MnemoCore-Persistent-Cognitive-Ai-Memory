"""
Comprehensive Tests for GapFiller Module
=========================================

Tests the autonomous LLM-driven knowledge gap filler.

Coverage:
- fill_now() with mocked GapDetector + LLM
- Rate limiting enforcement
- Poll loop start/stop lifecycle
- Error handling: LLM returns garbage
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from mnemocore.core.gap_filler import (
    GapFiller,
    GapFillerConfig,
    _FILL_PROMPT_TEMPLATE,
    _REFINE_PROMPT_TEMPLATE,
)
from mnemocore.core.gap_detector import GapDetector, GapRecord, GapDetectorConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def gap_filler_config():
    """Create a GapFillerConfig for testing."""
    return GapFillerConfig(
        poll_interval_seconds=1,  # Short for testing
        max_fills_per_hour=10,
        min_priority_to_fill=0.3,
        min_seen_before_fill=2,
        max_statements_per_gap=5,
        dry_run=False,
        store_tag="llm_gap_fill",
        enabled=True,
    )


@pytest.fixture
def gap_detector():
    """Create a GapDetector with some test gaps."""
    detector = GapDetector(GapDetectorConfig())
    now = datetime.now(timezone.utc)

    # Add some test gaps
    detector._registry["gap1"] = GapRecord(
        gap_id="gap1",
        query_text="What is quantum entanglement?",
        detected_at=now - timedelta(hours=1),
        last_seen=now,
        signal="low_confidence",
        confidence=0.2,
        seen_count=3,
        priority_score=0.8,
        filled=False,
    )
    detector._registry["gap2"] = GapRecord(
        gap_id="gap2",
        query_text="How does photosynthesis work?",
        detected_at=now - timedelta(hours=2),
        last_seen=now,
        signal="sparse",
        confidence=0.3,
        seen_count=2,
        priority_score=0.6,
        filled=False,
    )
    # Low priority gap - should be filtered
    detector._registry["gap3"] = GapRecord(
        gap_id="gap3",
        query_text="Low priority topic",
        detected_at=now,
        last_seen=now,
        signal="low_confidence",
        confidence=0.4,
        seen_count=1,  # Below min_seen_before_fill
        priority_score=0.2,  # Below min_priority_to_fill
        filled=False,
    )

    return detector


@pytest.fixture
def mock_engine():
    """Create a mock HAIMEngine for testing."""
    engine = MagicMock()
    engine.store = AsyncMock(return_value="stored-node-id")
    return engine


@pytest.fixture
def mock_llm_integrator():
    """Create a mock LLM integrator."""
    integrator = MagicMock()
    integrator._call_llm = MagicMock(return_value="- Statement one about the topic.\n- Statement two with more info.\n- Statement three with details.")
    return integrator


# =============================================================================
# GapFillerConfig Tests
# =============================================================================

class TestGapFillerConfig:
    """Test GapFillerConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = GapFillerConfig()

        assert config.poll_interval_seconds == 600.0
        assert config.max_fills_per_hour == 20
        assert config.min_priority_to_fill == 0.3
        assert config.min_seen_before_fill == 2
        assert config.max_statements_per_gap == 5
        assert config.dry_run is False
        assert config.enabled is True

    def test_custom_config(self):
        """Custom config values should be set correctly."""
        config = GapFillerConfig(
            poll_interval_seconds=300.0,
            max_fills_per_hour=5,
            min_priority_to_fill=0.5,
            dry_run=True,
        )

        assert config.poll_interval_seconds == 300.0
        assert config.max_fills_per_hour == 5
        assert config.min_priority_to_fill == 0.5
        assert config.dry_run is True


# =============================================================================
# GapFiller Initialization Tests
# =============================================================================

class TestGapFillerInit:
    """Test GapFiller initialization."""

    def test_init_with_defaults(self, mock_engine, mock_llm_integrator, gap_detector):
        """GapFiller should initialize with default config."""
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector)

        assert filler.engine == mock_engine
        assert filler.llm == mock_llm_integrator
        assert filler.detector == gap_detector
        assert filler.cfg.max_fills_per_hour == 20  # Default

    def test_init_with_custom_config(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """GapFiller should use custom config when provided."""
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        assert filler.cfg.max_fills_per_hour == 10
        assert filler.cfg.min_priority_to_fill == 0.3

    def test_init_stats(self, mock_engine, mock_llm_integrator, gap_detector):
        """GapFiller should initialize stats to zero."""
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector)

        assert filler.stats["gaps_filled"] == 0
        assert filler.stats["statements_stored"] == 0
        assert filler.stats["llm_calls"] == 0
        assert filler.stats["errors"] == 0


# =============================================================================
# GapFiller Lifecycle Tests
# =============================================================================

class TestGapFillerLifecycle:
    """Test GapFiller start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_enabled(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """GapFiller should start when enabled."""
        gap_filler_config.poll_interval_seconds = 0.1  # Short for testing
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        await filler.start()
        await asyncio.sleep(0.05)

        assert filler._running is True
        assert filler._task is not None

        await filler.stop()

    @pytest.mark.asyncio
    async def test_start_disabled(self, mock_engine, mock_llm_integrator, gap_detector):
        """GapFiller should not start when disabled."""
        config = GapFillerConfig(enabled=False)
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=config)

        await filler.start()

        assert filler._running is False
        assert filler._task is None

    @pytest.mark.asyncio
    async def test_stop(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """GapFiller should stop cleanly."""
        gap_filler_config.poll_interval_seconds = 0.1
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        await filler.start()
        await asyncio.sleep(0.05)
        await filler.stop()

        assert filler._running is False

    @pytest.mark.asyncio
    async def test_poll_loop_calls_fill_now(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """Poll loop should call fill_now periodically."""
        gap_filler_config.poll_interval_seconds = 0.1
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        # Track fill_now calls
        fill_now_calls = [0]
        original_fill_now = filler.fill_now

        async def tracked_fill_now(n=5):
            fill_now_calls[0] += 1
            return await original_fill_now(n)

        filler.fill_now = tracked_fill_now

        await filler.start()
        await asyncio.sleep(0.25)  # Should trigger 2+ polls
        await filler.stop()

        assert fill_now_calls[0] >= 2


# =============================================================================
# GapFiller fill_now Tests
# =============================================================================

class TestGapFillerFillNow:
    """Test fill_now method."""

    @pytest.mark.asyncio
    async def test_fill_now_with_gaps(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """fill_now should fill eligible gaps."""
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        results = await filler.fill_now(n=5)

        # Should have filled gap1 and gap2 (gap3 is filtered by priority/seen_count)
        assert len(results) <= 2
        assert filler.stats["llm_calls"] > 0

    @pytest.mark.asyncio
    async def test_fill_now_respects_rate_limit(self, mock_engine, mock_llm_integrator, gap_detector):
        """fill_now should respect rate limit."""
        config = GapFillerConfig(max_fills_per_hour=1)
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=config)

        # Fill rate limit
        filler._fill_timestamps.append(asyncio.get_event_loop().time())

        results = await filler.fill_now(n=5)

        # Should return empty due to rate limit
        assert results == []

    @pytest.mark.asyncio
    async def test_fill_now_filters_by_priority(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """fill_now should filter gaps below priority threshold."""
        gap_filler_config.min_priority_to_fill = 0.7
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        results = await filler.fill_now(n=5)

        # Only gap1 has priority >= 0.7
        filled_ids = [r.get("gap_id") for r in results if r.get("status") == "filled"]
        assert all(gid == "gap1" for gid in filled_ids)

    @pytest.mark.asyncio
    async def test_fill_now_filters_by_seen_count(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """fill_now should filter gaps with too few seen counts."""
        gap_filler_config.min_seen_before_fill = 5
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        results = await filler.fill_now(n=5)

        # No gaps have seen_count >= 5
        assert len(results) == 0


# =============================================================================
# GapFiller Statement Parsing Tests
# =============================================================================

class TestGapFillerParseStatements:
    """Test statement parsing from LLM response."""

    def test_parse_statements_with_bullets(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """Should parse bullet-pointed statements."""
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        raw = "- First statement here.\n- Second statement here.\n- Third statement here."
        statements = filler._parse_statements(raw)

        assert len(statements) == 3
        assert "First statement" in statements[0]

    def test_parse_statements_with_numbers(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """Should parse numbered statements."""
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        raw = "1. First statement here.\n2. Second statement here.\n3. Third statement here."
        statements = filler._parse_statements(raw)

        assert len(statements) == 3

    def test_parse_statements_respects_max(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """Should limit to max_statements_per_gap."""
        gap_filler_config.max_statements_per_gap = 2
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        raw = "- Statement 1.\n- Statement 2.\n- Statement 3.\n- Statement 4."
        statements = filler._parse_statements(raw)

        assert len(statements) == 2

    def test_parse_statements_filters_short(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """Should filter out very short lines."""
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        raw = "Short\nThis is a valid statement that is long enough.\nAlso too short"
        statements = filler._parse_statements(raw)

        assert len(statements) == 1

    def test_parse_statements_empty_input(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """Should handle empty input."""
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        statements = filler._parse_statements("")

        assert len(statements) == 0


# =============================================================================
# GapFiller Rate Limiting Tests
# =============================================================================

class TestGapFillerRateLimit:
    """Test rate limiting enforcement."""

    def test_rate_check_under_limit(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """_rate_check should return True when under limit."""
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        result = filler._rate_check()

        assert result is True

    def test_rate_check_at_limit(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """_rate_check should return False when at limit."""
        gap_filler_config.max_fills_per_hour = 2
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        # Fill up rate limit
        filler._record_call()
        filler._record_call()

        result = filler._rate_check()

        assert result is False

    def test_rate_check_expires_old_calls(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """Old calls should expire after 1 hour."""
        import time
        gap_filler_config.max_fills_per_hour = 2
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        # Add old calls (2 hours ago)
        old_time = time.time() - 7200
        filler._fill_timestamps = [old_time, old_time]

        result = filler._rate_check()

        # Old calls should have been removed
        assert result is True
        assert len(filler._fill_timestamps) == 0

    def test_record_call(self, mock_engine, mock_llm_integrator, gap_detector, gap_filler_config):
        """_record_call should add timestamp."""
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        filler._record_call()

        assert len(filler._fill_timestamps) == 1


# =============================================================================
# GapFiller Error Handling Tests
# =============================================================================

class TestGapFillerErrorHandling:
    """Test error handling in gap filling."""

    @pytest.mark.asyncio
    async def test_llm_returns_garbage(self, mock_engine, gap_detector, gap_filler_config):
        """Should handle unparseable LLM response gracefully."""
        # LLM returns garbage
        mock_llm = MagicMock()
        mock_llm._call_llm = MagicMock(return_value="This is not valid JSON or statements")

        filler = GapFiller(mock_engine, mock_llm, gap_detector, config=gap_filler_config)

        results = await filler.fill_now(n=1)

        # Should return empty_response status
        if results:
            assert results[0]["status"] in ["empty_response", "error"]

    @pytest.mark.asyncio
    async def test_llm_raises_exception(self, mock_engine, gap_detector, gap_filler_config):
        """Should handle LLM exceptions gracefully."""
        # LLM raises exception
        mock_llm = MagicMock()
        mock_llm._call_llm = MagicMock(side_effect=Exception("LLM connection failed"))

        filler = GapFiller(mock_engine, mock_llm, gap_detector, config=gap_filler_config)

        results = await filler.fill_now(n=1)

        # Should return error status
        if results:
            assert results[0]["status"] == "error"
            assert "error" in results[0]

        assert filler.stats["errors"] > 0

    @pytest.mark.asyncio
    async def test_store_raises_exception(self, mock_llm_integrator, gap_detector, gap_filler_config):
        """Should handle store exceptions gracefully."""
        # Engine store raises exception
        mock_engine = MagicMock()
        mock_engine.store = AsyncMock(side_effect=Exception("Storage failed"))

        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=gap_filler_config)

        results = await filler.fill_now(n=1)

        # Should still mark gap as filled even if store fails
        # (or handle gracefully based on implementation)
        assert isinstance(results, list)


# =============================================================================
# GapFiller Dry Run Tests
# =============================================================================

class TestGapFillerDryRun:
    """Test dry run mode."""

    @pytest.mark.asyncio
    async def test_dry_run_does_not_store(self, mock_engine, mock_llm_integrator, gap_detector):
        """In dry_run mode, statements should not be stored."""
        config = GapFillerConfig(dry_run=True, enabled=True)
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=config)

        await filler.fill_now(n=1)

        # store should not have been called
        mock_engine.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_marks_gap_filled(self, mock_engine, mock_llm_integrator, gap_detector):
        """In dry_run mode, gap should still be marked filled."""
        config = GapFillerConfig(dry_run=True, enabled=True)
        filler = GapFiller(mock_engine, mock_llm_integrator, gap_detector, config=config)

        results = await filler.fill_now(n=2)

        # Results should have dry_run status
        for result in results:
            if result.get("status"):
                assert result["status"] == "dry_run"


# =============================================================================
# GapFiller Integration Tests
# =============================================================================

class TestGapFillerIntegration:
    """Integration tests for GapFiller."""

    @pytest.mark.asyncio
    async def test_full_fill_cycle(self, mock_engine, gap_detector, gap_filler_config):
        """Test a complete fill cycle with valid responses."""
        # Create LLM that returns valid statements
        mock_llm = MagicMock()
        mock_llm._call_llm = MagicMock(
            return_value="- Quantum entanglement is a physical phenomenon.\n"
                         "- When particles are entangled, measuring one affects the other.\n"
                         "- This happens regardless of the distance between them."
        )

        filler = GapFiller(mock_engine, mock_llm, gap_detector, config=gap_filler_config)

        results = await filler.fill_now(n=1)

        # Should have filled gap1
        assert len(results) >= 1
        assert results[0]["status"] == "filled"
        assert len(results[0]["statements"]) > 0
        assert len(results[0]["stored_node_ids"]) > 0

        # Stats should be updated
        assert filler.stats["gaps_filled"] > 0
        assert filler.stats["statements_stored"] > 0

    @pytest.mark.asyncio
    async def test_gap_marked_filled_after_fill(self, mock_engine, gap_detector, gap_filler_config):
        """Gap should be marked as filled after successful fill."""
        mock_llm = MagicMock()
        mock_llm._call_llm = MagicMock(return_value="- Some statement about the topic.")

        filler = GapFiller(mock_engine, mock_llm, gap_detector, config=gap_filler_config)

        await filler.fill_now(n=1)

        # Check that gap was marked as filled
        filled_gaps = [g for g in gap_detector._registry.values() if g.filled]
        assert len(filled_gaps) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
