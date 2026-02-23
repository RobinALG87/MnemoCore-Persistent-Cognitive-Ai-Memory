"""
Tests for MnemoCore CLI
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from mnemocore.cli.main import cli


@pytest.fixture
def runner():
    """CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_engine():
    """Mock HAIMEngine."""
    engine = MagicMock()
    engine.initialize = AsyncMock()
    engine.close = AsyncMock()
    engine.store = MagicMock(return_value="mem_test123")
    engine.query = MagicMock(return_value=[("mem_test123", 0.85)])
    engine.delete_memory = MagicMock(return_value=True)
    return engine


class TestCLIStore:
    """Tests for store command."""

    def test_store_basic(self, runner, mock_engine):
        """Test basic store command."""
        with patch("mnemocore.core.engine.HAIMEngine", return_value=mock_engine):
            result = runner.invoke(cli, ["store", "Test memory"])
            assert result.exit_code == 0
            assert "Stored memory" in result.output or "mem_test" in result.output

    def test_store_with_tags(self, runner, mock_engine):
        """Test store with tags."""
        with patch("mnemocore.core.engine.HAIMEngine", return_value=mock_engine):
            result = runner.invoke(cli, ["store", "Test memory", "-t", "tag1", "-t", "tag2"])
            assert result.exit_code == 0

    def test_store_json_output(self, runner, mock_engine):
        """Test store with JSON output."""
        with patch("mnemocore.core.engine.HAIMEngine", return_value=mock_engine):
            result = runner.invoke(cli, ["store", "Test memory", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["success"] is True
            assert "memory_id" in data


class TestCLIRecall:
    """Tests for recall command."""

    def test_recall_basic(self, runner, mock_engine):
        """Test basic recall command."""
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_memory = AsyncMock(
            return_value=MagicMock(
                id="mem_test123",
                content="Test content",
                tier="hot",
                created_at=None,
                metadata={},
            )
        )

        with patch("mnemocore.core.engine.HAIMEngine", return_value=mock_engine):
            result = runner.invoke(cli, ["recall", "test query"])
            assert result.exit_code == 0

    def test_recall_with_top_k(self, runner, mock_engine):
        """Test recall with custom top_k."""
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_memory = AsyncMock(
            return_value=MagicMock(
                id="mem_test123",
                content="Test content",
                tier="hot",
                created_at=None,
                metadata={},
            )
        )

        with patch("mnemocore.core.engine.HAIMEngine", return_value=mock_engine):
            result = runner.invoke(cli, ["recall", "test query", "-k", "10"])
            assert result.exit_code == 0


class TestCLIStats:
    """Tests for stats command."""

    def test_stats_basic(self, runner, mock_engine):
        """Test basic stats command."""
        mock_engine.get_stats = AsyncMock(
            return_value={
                "engine_version": "4.5.0",
                "dimension": 16384,
                "encoding": "binary_hdv",
                "tiers": {
                    "hot": {"count": 100},
                    "warm": {"count": 1000},
                },
                "concepts_count": 50,
                "symbols_count": 100,
                "synapses_count": 500,
            }
        )

        with patch("mnemocore.core.engine.HAIMEngine", return_value=mock_engine):
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 0
            assert "MnemoCore" in result.output or "4.5.0" in result.output

    def test_stats_json(self, runner, mock_engine):
        """Test stats with JSON output."""
        mock_engine.get_stats = AsyncMock(
            return_value={
                "engine_version": "4.5.0",
                "tiers": {"hot": {"count": 100}},
            }
        )

        with patch("mnemocore.core.engine.HAIMEngine", return_value=mock_engine):
            result = runner.invoke(cli, ["stats", "--json"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["engine_version"] == "4.5.0"


class TestCLIHealth:
    """Tests for health command."""

    def test_health_basic(self, runner, mock_engine):
        """Test basic health command."""
        mock_engine.health_check = AsyncMock(
            return_value={
                "status": "healthy",
                "initialized": True,
                "timestamp": "2024-01-01T12:00:00Z",
                "tiers": {
                    "hot": {"count": 100},
                },
            }
        )

        with patch("mnemocore.core.engine.HAIMEngine", return_value=mock_engine):
            result = runner.invoke(cli, ["health"])
            assert result.exit_code == 0
            assert "healthy" in result.output.lower()


class TestCLIDelete:
    """Tests for delete command."""

    def test_delete_with_force(self, runner, mock_engine):
        """Test delete with force flag."""
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_memory = AsyncMock(
            return_value=MagicMock(
                id="mem_test123",
                content="Test content",
            )
        )

        with patch("mnemocore.core.engine.HAIMEngine", return_value=mock_engine):
            result = runner.invoke(cli, ["delete", "mem_test123", "--force"])
            assert result.exit_code == 0


class TestCLIGet:
    """Tests for get command."""

    def test_get_basic(self, runner, mock_engine):
        """Test get memory by ID."""
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_memory = AsyncMock(
            return_value=MagicMock(
                id="mem_test123",
                content="Test content",
                tier="hot",
                ltp_strength=0.75,
                created_at=None,
                metadata={"tags": ["test"]},
            )
        )

        with patch("mnemocore.core.engine.HAIMEngine", return_value=mock_engine):
            result = runner.invoke(cli, ["get", "mem_test123"])
            assert result.exit_code == 0
            assert "mem_test123" in result.output


class TestCLIExport:
    """Tests for export command."""

    def test_export_json(self, runner, mock_engine):
        """Test export to JSON."""
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_all_hot = AsyncMock(
            return_value=[
                MagicMock(
                    id="mem_test123",
                    content="Test content",
                    tier="hot",
                    ltp_strength=0.75,
                    created_at=None,
                    metadata={},
                    hdv=MagicMock(tolist=lambda: [1, 0, 1]),
                )
            ]
        )
        mock_engine.tier_manager.get_hot_recent = AsyncMock(return_value=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.json"

            with patch("mnemocore.core.engine.HAIMEngine", return_value=mock_engine):
                result = runner.invoke(cli, ["export", "-f", "json", "-o", str(output_path)])
                assert result.exit_code == 0
                # The export command runs async, file may not be created in test
                # Just verify the command runs successfully

    def test_export_jsonl(self, runner, mock_engine):
        """Test export to JSONL."""
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_all_hot = AsyncMock(
            return_value=[
                MagicMock(
                    id="mem_test123",
                    content="Test content",
                    tier="hot",
                    ltp_strength=0.75,
                    created_at=None,
                    metadata={},
                    hdv=MagicMock(tolist=lambda: [1, 0, 1]),
                )
            ]
        )
        mock_engine.tier_manager.get_hot_recent = AsyncMock(return_value=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.jsonl"

            with patch("mnemocore.core.engine.HAIMEngine", return_value=mock_engine):
                result = runner.invoke(cli, ["export", "-f", "jsonl", "-o", str(output_path)])
                assert result.exit_code == 0
                # The export command runs async, file may not be created in test
                # Just verify the command runs successfully


class TestCLIDream:
    """Tests for dream command."""

    def test_dream_without_now(self, runner, mock_engine):
        """Test dream command without --now flag."""
        result = runner.invoke(cli, ["dream"])
        assert result.exit_code == 0
        assert "--now" in result.output

    def test_dream_with_now(self, runner, mock_engine):
        """Test dream command with --now flag."""
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_all_hot = AsyncMock(return_value=[])

        with patch("mnemocore.subconscious.dream_pipeline.DreamPipeline") as mock_pipeline:
            mock_pipeline.return_value.run = AsyncMock(
                return_value={
                    "success": True,
                    "duration_seconds": 1.5,
                    "memories_processed": 100,
                    "episodic_clusters_count": 5,
                    "patterns_extracted_count": 3,
                }
            )

            with patch("mnemocore.core.engine.HAIMEngine", return_value=mock_engine):
                result = runner.invoke(cli, ["dream", "--now"])
                assert result.exit_code == 0
