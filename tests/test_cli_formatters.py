"""
Tests for CLI Output Formatters
===============================
Tests for src/mnemocore/cli/formatters.py covering table formatting,
truncation, timestamp formatting, and ANSI color handling.
"""

import pytest
from datetime import datetime, timezone

from mnemocore.cli.formatters import (
    Colors,
    MemoryResult,
    format_memory_table,
    format_stats,
    format_health,
    format_dream_report,
    format_export_summary,
    truncate,
    format_timestamp,
    get_terminal_width,
    format_progress,
)


class TestColors:
    """Tests for ANSI color handling."""

    def test_green_color(self):
        """Test green color wrapping."""
        result = Colors.green("test")
        assert result.startswith("\033[92m")
        assert result.endswith("\033[0m")
        assert "test" in result

    def test_red_color(self):
        """Test red color wrapping."""
        result = Colors.red("error")
        assert result.startswith("\033[91m")
        assert result.endswith("\033[0m")
        assert "error" in result

    def test_yellow_color(self):
        """Test yellow color wrapping."""
        result = Colors.yellow("warning")
        assert result.startswith("\033[93m")
        assert result.endswith("\033[0m")
        assert "warning" in result

    def test_blue_color(self):
        """Test blue color wrapping."""
        result = Colors.blue("info")
        assert result.startswith("\033[94m")
        assert result.endswith("\033[0m")
        assert "info" in result

    def test_bold(self):
        """Test bold formatting."""
        result = Colors.bold("important")
        assert result.startswith("\033[1m")
        assert result.endswith("\033[0m")
        assert "important" in result


class TestMemoryResult:
    """Tests for MemoryResult dataclass."""

    def test_to_table_row_short_content(self):
        """Test table row generation with short content."""
        result = MemoryResult(
            id="mem_12345678901234567890",
            content="Short content",
            score=0.85,
            tier="hot",
            created_at="2024-01-15T10:30:00Z",
            tags=["tag1", "tag2"],
        )
        row = result.to_table_row()
        assert len(row) == 6
        assert "Short content" in row[1]
        assert "0.85" in row[2]
        assert "HOT" in row[3]

    def test_to_table_row_long_content_truncation(self):
        """Test that long content is truncated in table row."""
        long_content = "A" * 100
        result = MemoryResult(
            id="mem_short",
            content=long_content,
            score=0.95,
            tier="warm",
            created_at="2024-01-15T10:30:00Z",
            tags=[],
        )
        row = result.to_table_row()
        # Content should be truncated to 60 chars + "..."
        assert len(row[1]) <= 63  # 60 chars + "..."

    def test_to_table_row_long_id_truncation(self):
        """Test that long IDs are truncated."""
        result = MemoryResult(
            id="mem_" + "x" * 50,
            content="Test",
            score=0.5,
            tier="cold",
            created_at=None,
            tags=[],
        )
        row = result.to_table_row()
        # ID should be truncated to 12 chars + "..."
        assert row[0].endswith("...")

    def test_to_table_row_no_created_at(self):
        """Test table row with no created_at timestamp."""
        result = MemoryResult(
            id="mem_123",
            content="Test",
            score=0.5,
            tier="hot",
            created_at=None,
            tags=[],
        )
        row = result.to_table_row()
        assert row[4] == "N/A"

    def test_to_table_row_tags_limit(self):
        """Test that only first 3 tags are shown."""
        result = MemoryResult(
            id="mem_123",
            content="Test",
            score=0.5,
            tier="hot",
            created_at=None,
            tags=["tag1", "tag2", "tag3", "tag4", "tag5"],
        )
        row = result.to_table_row()
        # Should only show first 3 tags
        assert "tag1" in row[5]
        assert "tag4" not in row[5]

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = MemoryResult(
            id="mem_123",
            content="Test content",
            score=0.85,
            tier="hot",
            created_at="2024-01-15T10:30:00Z",
            tags=["tag1"],
            metadata={"key": "value"},
        )
        d = result.to_dict()
        assert d["id"] == "mem_123"
        assert d["content"] == "Test content"
        assert d["score"] == 0.85
        assert d["tier"] == "hot"
        assert d["tags"] == ["tag1"]
        assert d["metadata"] == {"key": "value"}


class TestFormatMemoryTable:
    """Tests for format_memory_table function."""

    def test_empty_memories(self):
        """Test table formatting with no memories."""
        result = format_memory_table([])
        assert result == "No memories found."

    def test_single_memory(self):
        """Test table formatting with single memory."""
        memory = MemoryResult(
            id="mem_123",
            content="Test content",
            score=0.85,
            tier="hot",
            created_at="2024-01-15T10:30:00Z",
            tags=[],
        )
        result = format_memory_table([memory])
        assert "mem_123" in result
        assert "Test content" in result

    def test_multiple_memories(self):
        """Test table formatting with multiple memories."""
        memories = [
            MemoryResult(
                id=f"mem_{i}",
                content=f"Content {i}",
                score=0.5 + i * 0.1,
                tier="hot",
                created_at=None,
                tags=[],
            )
            for i in range(5)
        ]
        result = format_memory_table(memories)
        assert "mem_0" in result
        assert "mem_4" in result

    def test_without_headers(self):
        """Test table formatting without headers."""
        memory = MemoryResult(
            id="mem_123",
            content="Test",
            score=0.85,
            tier="hot",
            created_at=None,
            tags=[],
        )
        result = format_memory_table([memory], show_headers=False)
        assert "ID" not in result


class TestTruncate:
    """Tests for truncate function."""

    def test_short_text_no_truncation(self):
        """Test that short text is not truncated."""
        result = truncate("Short text", max_length=20)
        assert result == "Short text"

    def test_long_text_truncation(self):
        """Test that long text is truncated."""
        long_text = "A" * 100
        result = truncate(long_text, max_length=50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_custom_suffix(self):
        """Test truncation with custom suffix."""
        long_text = "A" * 100
        result = truncate(long_text, max_length=50, suffix="...")
        assert len(result) == 50
        assert result.endswith("...")

    def test_exact_length(self):
        """Test text that is exactly max_length."""
        text = "A" * 50
        result = truncate(text, max_length=50)
        assert result == text
        assert "..." not in result


class TestFormatTimestamp:
    """Tests for format_timestamp function."""

    def test_valid_iso_timestamp(self):
        """Test formatting a valid ISO timestamp."""
        result = format_timestamp("2024-01-15T10:30:00Z")
        assert "2024-01-15" in result

    def test_valid_iso_timestamp_with_timezone(self):
        """Test formatting ISO timestamp with timezone offset."""
        result = format_timestamp("2024-01-15T10:30:00+00:00")
        assert "2024-01-15" in result

    def test_none_timestamp(self):
        """Test formatting None timestamp returns N/A."""
        result = format_timestamp(None)
        assert result == "N/A"

    def test_empty_string_timestamp(self):
        """Test formatting empty string timestamp returns N/A."""
        result = format_timestamp("")
        assert result == "N/A"

    def test_invalid_timestamp(self):
        """Test formatting invalid timestamp returns original string."""
        result = format_timestamp("not-a-timestamp")
        assert result == "not-a-timestamp"

    def test_custom_format(self):
        """Test formatting with custom output format."""
        result = format_timestamp("2024-01-15T10:30:00Z", format="%Y/%m/%d")
        assert result == "2024/01/15"


class TestFormatStats:
    """Tests for format_stats function."""

    def test_basic_stats(self):
        """Test formatting basic statistics."""
        stats = {
            "engine_version": "5.0.0",
            "dimension": 16384,
            "encoding": "binary_hdv",
        }
        result = format_stats(stats)
        assert "5.0.0" in result
        assert "16384" in result
        assert "binary_hdv" in result

    def test_stats_with_tiers(self):
        """Test formatting statistics with tier information."""
        stats = {
            "engine_version": "5.0.0",
            "tiers": {
                "hot": {"count": 100, "max": 1000},
                "warm": {"count": 500, "max": 10000},
                "cold": {"count": 2000, "max": "unlimited"},
            },
        }
        result = format_stats(stats)
        assert "HOT" in result
        assert "WARM" in result
        assert "COLD" in result
        assert "100" in result
        assert "TOTAL" in result

    def test_stats_with_background_workers(self):
        """Test formatting statistics with background worker status."""
        stats = {
            "engine_version": "5.0.0",
            "background_workers": {
                "consolidation": {"running": True},
                "dream_loop": {"running": False},
            },
        }
        result = format_stats(stats)
        assert "consolidation" in result
        assert "dream_loop" in result


class TestFormatHealth:
    """Tests for format_health function."""

    def test_healthy_status(self):
        """Test formatting healthy status."""
        health = {"status": "healthy", "initialized": True}
        result = format_health(health)
        assert "HEALTHY" in result

    def test_degraded_status(self):
        """Test formatting degraded status."""
        health = {"status": "degraded", "initialized": True}
        result = format_health(health)
        assert "DEGRADED" in result

    def test_unhealthy_status(self):
        """Test formatting unhealthy status."""
        health = {"status": "unhealthy", "initialized": False}
        result = format_health(health)
        assert "UNHEALTHY" in result

    def test_health_with_tiers(self):
        """Test formatting health with tier counts."""
        health = {
            "status": "healthy",
            "tiers": {
                "hot": {"count": 100},
                "warm": {"count": 500},
            },
        }
        result = format_health(health)
        assert "Total" in result

    def test_health_with_qdrant_connected(self):
        """Test formatting health with Qdrant connected."""
        health = {
            "status": "healthy",
            "qdrant": {"connected": True},
        }
        result = format_health(health)
        assert "Connected" in result

    def test_health_with_qdrant_error(self):
        """Test formatting health with Qdrant error."""
        health = {
            "status": "degraded",
            "qdrant": {"error": "Connection refused"},
        }
        result = format_health(health)
        assert "Error" in result


class TestFormatDreamReport:
    """Tests for format_dream_report function."""

    def test_basic_dream_report(self):
        """Test formatting basic dream report."""
        report = {
            "session_duration_seconds": 10.5,
            "summary": {
                "episodic_clusters_found": 5,
                "patterns_discovered": 10,
                "synthesis_insights": 3,
                "contradictions_found": 2,
                "contradictions_resolved": 1,
                "memories_promoted": 4,
            },
        }
        result = format_dream_report(report)
        assert "10.5" in result
        assert "5" in result

    def test_dream_report_with_patterns(self):
        """Test formatting dream report with patterns."""
        report = {
            "session_duration_seconds": 5.0,
            "summary": {},
            "pattern_discoveries": {
                "top_patterns": [
                    {"type": "temporal", "value": "morning routine", "frequency": 10},
                    {"type": "semantic", "value": "project work", "frequency": 8},
                ]
            },
        }
        result = format_dream_report(report)
        assert "temporal" in result
        assert "morning routine" in result

    def test_dream_report_with_recommendations(self):
        """Test formatting dream report with recommendations."""
        report = {
            "session_duration_seconds": 5.0,
            "summary": {},
            "recommendations": [
                "Consider consolidating old memories",
                "Review contradictory beliefs",
            ],
        }
        result = format_dream_report(report)
        assert "consolidating" in result.lower()
        assert "contradictory" in result.lower()


class TestFormatExportSummary:
    """Tests for format_export_summary function."""

    def test_successful_export(self):
        """Test formatting successful export."""
        result_data = {
            "success": True,
            "records_exported": 1000,
            "size_bytes": 1024 * 1024 * 5,  # 5 MB
            "duration_seconds": 2.5,
            "output_path": "/tmp/export.json",
        }
        result = format_export_summary(result_data)
        assert "successfully" in result.lower()
        assert "1000" in result
        assert "5.00 MB" in result

    def test_failed_export(self):
        """Test formatting failed export."""
        result_data = {
            "success": False,
            "error_message": "Disk full",
        }
        result = format_export_summary(result_data)
        assert "failed" in result.lower()
        assert "Disk full" in result


class TestGetTerminalWidth:
    """Tests for get_terminal_width function."""

    def test_returns_positive_integer(self):
        """Test that terminal width returns a positive integer."""
        result = get_terminal_width()
        assert isinstance(result, int)
        assert result > 0

    def test_default_on_failure(self):
        """Test that default is returned when detection fails."""
        # This should always return at least the default
        result = get_terminal_width(default=120)
        assert isinstance(result, int)


class TestFormatProgress:
    """Tests for format_progress function."""

    def test_zero_progress(self):
        """Test progress bar at zero."""
        result = format_progress(0, 100)
        assert "0%" in result
        assert "[" in result
        assert "]" in result

    def test_half_progress(self):
        """Test progress bar at 50%."""
        result = format_progress(50, 100)
        assert "50%" in result

    def test_full_progress(self):
        """Test progress bar at 100%."""
        result = format_progress(100, 100)
        assert "100%" in result

    def test_zero_total(self):
        """Test progress bar with zero total."""
        result = format_progress(0, 0)
        # Should return empty bar
        assert "[" in result
        assert "]" in result

    def test_custom_width(self):
        """Test progress bar with custom width."""
        result = format_progress(50, 100, width=20)
        assert len(result) > 20  # Includes percentage text
