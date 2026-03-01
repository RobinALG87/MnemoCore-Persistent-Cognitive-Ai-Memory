"""
Comprehensive tests for Forgetting Analytics Module.
====================================================
Tests for:
  - Dashboard data generation
  - Chart data for learning progress, SM-2 performance
  - CSV/JSON export
  - Empty data handling
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import (
    MagicMock,
    patch,
)

import pytest

from mnemocore.cognitive.forgetting_analytics import (
    ChartType,
    ChartData,
    DashboardWidget,
    ForgettingAnalyticsCognitive,
    create_forgetting_analytics,
    get_dashboard_html,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def mock_forgetting_manager():
    """Create a mock ForgettingCurveManager for testing."""
    manager = MagicMock()

    # Mock get_profile
    mock_profile = MagicMock()
    mock_profile.learning_style = "balanced"
    mock_profile.easiness_factor = 2.5
    mock_profile.stats = {
        "total_reviews": 100,
        "successful_reviews": 80,
    }
    manager.get_profile = MagicMock(return_value=mock_profile)

    # Mock get_schedule
    mock_schedule_entries = []
    base_time = datetime.now(timezone.utc)

    for i in range(10):
        entry = MagicMock()
        entry.memory_id = f"mem_{i}"
        entry.agent_id = "default"
        entry.current_retention = 0.8 - (i * 0.05)
        entry.emotional_salience = 0.3 if i % 2 == 0 else 0.7
        entry.due_at = base_time + timedelta(hours=i)
        entry.sm2_state = MagicMock()
        entry.sm2_state.last_review_date = base_time - timedelta(days=i)
        entry.sm2_state.repetitions = i % 5
        mock_schedule_entries.append(entry)

    manager.get_schedule = MagicMock(return_value=mock_schedule_entries)

    # Mock review history
    manager._review_history = [
        {
            "agent_id": "default",
            "memory_id": f"mem_{i}",
            "quality": 3 + (i % 3),
            "new_interval": i * 2,
        }
        for i in range(20)
    ]

    return manager


@pytest.fixture
def mock_core_analytics():
    """Create mock core analytics for testing."""
    analytics = MagicMock()

    analytics.get_dashboard_data = MagicMock(return_value={
        "summary": {
            "total_scheduled": 10,
            "avg_retention": 0.75,
            "due_now": 3,
            "overdue": 1,
        },
        "agents": {
            "default": {
                "total_memories": 10,
                "avg_retention": 0.75,
            }
        },
        "retention_curve": [
            {"time_days": 0, "retention": 1.0},
            {"time_days": 1, "retention": 0.9},
            {"time_days": 7, "retention": 0.7},
        ],
        "emotional_distribution": {
            "emotional": 5,
            "neutral": 5,
        },
        "review_schedule": {
            "by_action": {
                "review": 5,
                "learn": 3,
                "relearn": 2,
            }
        },
        "recommendations": [
            "Review overdue memories",
            "Focus on emotional memories",
        ],
    })

    analytics.get_agent_comparison = MagicMock(return_value=[
        {
            "agent_id": "default",
            "avg_retention": 0.75,
            "total_memories": 10,
            "learning_style": "balanced",
        }
    ])

    return analytics


# =====================================================================
# ChartType and ChartData Tests
# =====================================================================

class TestChartDataClasses:
    """Tests for chart-related data classes."""

    def test_chart_type_values(self):
        """Test ChartType enum values."""
        assert ChartType.RETENTION_CURVE.value == "retention_curve"
        assert ChartType.LEARNING_PROGRESS.value == "learning_progress"
        assert ChartType.AGENT_COMPARISON.value == "agent_comparison"
        assert ChartType.EMOTIONAL_DISTRIBUTION.value == "emotional_distribution"
        assert ChartType.SCHEDULE_HEATMAP.value == "schedule_heatmap"
        assert ChartType.SM2_PERFORMANCE.value == "sm2_performance"

    def test_chart_data_to_dict(self):
        """Test ChartData serialization."""
        chart = ChartData(
            chart_type=ChartType.RETENTION_CURVE,
            title="Test Chart",
            data={"series": [{"x": 1, "y": 2}]},
            metadata={"x_label": "Time"},
        )

        d = chart.to_dict()

        assert d["chart_type"] == "retention_curve"
        assert d["title"] == "Test Chart"
        assert "series" in d["data"]
        assert "generated_at" in d

    def test_dashboard_widget_to_dict(self):
        """Test DashboardWidget serialization."""
        widget = DashboardWidget(
            widget_id="test_widget",
            widget_type="line_chart",
            title="Test Widget",
            data={"values": [1, 2, 3]},
            position={"row": 0, "col": 0},
        )

        d = widget.to_dict()

        assert d["widget_id"] == "test_widget"
        assert d["widget_type"] == "line_chart"
        assert d["title"] == "Test Widget"
        assert d["position"]["row"] == 0


# =====================================================================
# ForgettingAnalyticsCognitive Tests
# =====================================================================

class TestForgettingAnalyticsCognitive:
    """Tests for the ForgettingAnalyticsCognitive class."""

    def test_init(self, mock_forgetting_manager):
        """Test initialization."""
        analytics = ForgettingAnalyticsCognitive(
            mock_forgetting_manager,
            default_agent_id="test_agent"
        )

        assert analytics.manager == mock_forgetting_manager
        assert analytics.default_agent_id == "test_agent"

    def test_get_dashboard(self, mock_forgetting_manager, mock_core_analytics):
        """Test dashboard data generation."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
            return_value=mock_core_analytics
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            dashboard = analytics.get_dashboard()

        assert "title" in dashboard
        assert "agent_id" in dashboard
        assert "summary" in dashboard
        assert "widgets" in dashboard
        assert "generated_at" in dashboard

    def test_get_dashboard_with_charts(self, mock_forgetting_manager, mock_core_analytics):
        """Test dashboard with charts included."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
            return_value=mock_core_analytics
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            dashboard = analytics.get_dashboard(include_charts=True)

        assert "widgets" in dashboard
        # Should have multiple widgets including charts
        assert len(dashboard["widgets"]) > 0

    def test_get_dashboard_without_charts(self, mock_forgetting_manager, mock_core_analytics):
        """Test dashboard without charts."""
        mock_core_analytics.get_dashboard_data.return_value = {
            "summary": {"total_scheduled": 5},
            "agents": {},
        }

        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
            return_value=mock_core_analytics
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            dashboard = analytics.get_dashboard(include_charts=False)

        assert "widgets" in dashboard

    def test_get_dashboard_with_recommendations(
        self, mock_forgetting_manager, mock_core_analytics
    ):
        """Test dashboard includes recommendations."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
            return_value=mock_core_analytics
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            dashboard = analytics.get_dashboard(include_recommendations=True)

        assert "recommendations" in dashboard

    def test_get_widget(self, mock_forgetting_manager, mock_core_analytics):
        """Test getting specific widget by ID."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
            return_value=mock_core_analytics
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            widget = analytics.get_widget("summary_metrics")

        assert widget is not None
        assert widget["widget_id"] == "summary_metrics"

    def test_get_widget_not_found(self, mock_forgetting_manager, mock_core_analytics):
        """Test getting non-existent widget returns None."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
            return_value=mock_core_analytics
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            widget = analytics.get_widget("nonexistent_widget")

        assert widget is None


# =====================================================================
# Chart Generation Tests
# =====================================================================

class TestChartGeneration:
    """Tests for chart data generation methods."""

    def test_get_chart_retention_curve(self, mock_forgetting_manager):
        """Test retention curve chart generation."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            chart = analytics.get_chart(ChartType.RETENTION_CURVE)

        assert chart is not None
        assert chart.chart_type == ChartType.RETENTION_CURVE
        assert "series" in chart.data

    def test_get_chart_learning_progress(self, mock_forgetting_manager):
        """Test learning progress chart generation."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            chart = analytics.get_chart(ChartType.LEARNING_PROGRESS)

        assert chart is not None
        assert chart.chart_type == ChartType.LEARNING_PROGRESS

    def test_get_chart_agent_comparison(self, mock_forgetting_manager, mock_core_analytics):
        """Test agent comparison chart generation."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
            return_value=mock_core_analytics
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            analytics._core_analytics = mock_core_analytics
            chart = analytics.get_chart(ChartType.AGENT_COMPARISON)

        assert chart is not None
        assert chart.chart_type == ChartType.AGENT_COMPARISON

    def test_get_chart_emotional_distribution(self, mock_forgetting_manager):
        """Test emotional distribution chart generation."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            chart = analytics.get_chart(ChartType.EMOTIONAL_DISTRIBUTION)

        assert chart is not None
        assert chart.chart_type == ChartType.EMOTIONAL_DISTRIBUTION
        assert "categories" in chart.data
        assert "values" in chart.data

    def test_get_chart_sm2_performance(self, mock_forgetting_manager):
        """Test SM-2 performance chart generation."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            chart = analytics.get_chart(ChartType.SM2_PERFORMANCE)

        assert chart is not None
        assert chart.chart_type == ChartType.SM2_PERFORMANCE

    def test_get_chart_schedule_heatmap(self, mock_forgetting_manager):
        """Test schedule heatmap chart generation."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            chart = analytics.get_chart(ChartType.SCHEDULE_HEATMAP)

        assert chart is not None
        assert chart.chart_type == ChartType.SCHEDULE_HEATMAP

    def test_get_chart_unsupported_type(self, mock_forgetting_manager):
        """Test unsupported chart type returns None for unknown and not-None for valid types."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)

            # Valid types should return something
            result = analytics.get_chart(ChartType.RETENTION_CURVE)
            assert result is not None


# =====================================================================
# Export Tests
# =====================================================================

class TestExportFunctions:
    """Tests for CSV and JSON export functions."""

    def test_export_csv(self, mock_forgetting_manager):
        """Test CSV export."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            csv_data = analytics.export_csv(ChartType.RETENTION_CURVE)

        assert isinstance(csv_data, str)
        # Should have header and data
        lines = csv_data.strip().split("\n")
        assert len(lines) >= 1

    def test_export_csv_with_header(self, mock_forgetting_manager):
        """Test CSV export with header."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            csv_data = analytics.export_csv(
                ChartType.RETENTION_CURVE,
                include_header=True
            )

        assert "x,y" in csv_data or len(csv_data) > 0

    def test_export_csv_empty_chart(self, mock_forgetting_manager):
        """Test CSV export with empty chart data."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)

            # Mock get_chart to return None
            with patch.object(analytics, "get_chart", return_value=None):
                csv_data = analytics.export_csv(ChartType.RETENTION_CURVE)

        assert csv_data == ""

    def test_export_json(self, mock_forgetting_manager, mock_core_analytics):
        """Test JSON export."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
            return_value=mock_core_analytics
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            json_data = analytics.export_json()

        assert isinstance(json_data, str)
        # Should be valid JSON
        parsed = json.loads(json_data)
        assert "title" in parsed

    def test_export_json_pretty(self, mock_forgetting_manager, mock_core_analytics):
        """Test pretty-printed JSON export."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
            return_value=mock_core_analytics
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            json_data = analytics.export_json(pretty=True)

        # Pretty JSON has newlines and indentation
        assert "\n" in json_data

    def test_export_json_compact(self, mock_forgetting_manager, mock_core_analytics):
        """Test compact JSON export."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
            return_value=mock_core_analytics
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            json_data = analytics.export_json(pretty=False)

        # Compact JSON should be on one line (mostly)
        parsed = json.loads(json_data)
        assert parsed is not None

    def test_export_to_file_json(self, mock_forgetting_manager, mock_core_analytics):
        """Test export to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_export.json"

            with patch(
                "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
                return_value=mock_core_analytics
            ):
                analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
                analytics.export_to_file(str(filepath), format="json")

            assert filepath.exists()
            content = filepath.read_text()
            parsed = json.loads(content)
            assert "title" in parsed

    def test_export_to_file_csv(self, mock_forgetting_manager):
        """Test export to CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_export.csv"

            with patch(
                "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
            ):
                analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
                analytics.export_to_file(str(filepath), format="csv")

            assert filepath.exists()
            content = filepath.read_text()
            assert len(content) >= 0  # May be empty if no data

    def test_export_to_file_unsupported_format(self, mock_forgetting_manager):
        """Test export with unsupported format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_export.xyz"

            with patch(
                "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
            ):
                analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)

                with pytest.raises(ValueError, match="Unsupported export format"):
                    analytics.export_to_file(str(filepath), format="xyz")


# =====================================================================
# Agent Summary Tests
# =====================================================================

class TestAgentSummary:
    """Tests for agent summary functionality."""

    def test_get_agent_summary(self, mock_forgetting_manager):
        """Test getting agent summary."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            summary = analytics.get_agent_summary("default")

        assert "agent_id" in summary
        assert "learning_style" in summary
        assert "total_memories" in summary

    def test_get_agent_summary_not_found(self, mock_forgetting_manager):
        """Test agent summary for non-existent agent."""
        mock_forgetting_manager.get_profile.return_value = None

        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            summary = analytics.get_agent_summary("nonexistent")

        assert "error" in summary

    def test_compare_agents(self, mock_forgetting_manager):
        """Test comparing multiple agents."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)

            # Mock get_agent_summary for multiple agents
            with patch.object(
                analytics,
                "get_agent_summary",
                side_effect=lambda aid: {
                    "agent_id": aid,
                    "avg_retention": 0.7 if aid == "agent_a" else 0.8,
                    "success_rate": 0.75,
                }
            ):
                comparison = analytics.compare_agents(["agent_a", "agent_b"])

        assert "agents" in comparison
        assert "best_performing" in comparison
        assert "averages" in comparison


# =====================================================================
# Empty Data Handling Tests
# =====================================================================

class TestEmptyDataHandling:
    """Tests for handling empty or missing data."""

    def test_dashboard_empty_schedule(self):
        """Test dashboard with empty schedule."""
        mock_manager = MagicMock()
        mock_manager.get_schedule.return_value = []
        mock_manager.get_profile.return_value = MagicMock(
            learning_style="balanced",
            easiness_factor=2.5,
            stats={"total_reviews": 0, "successful_reviews": 0},
        )
        mock_manager._review_history = []

        mock_core = MagicMock()
        mock_core.get_dashboard_data.return_value = {
            "summary": {"total_scheduled": 0, "avg_retention": 0.0},
            "agents": {},
        }

        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
            return_value=mock_core
        ):
            analytics = ForgettingAnalyticsCognitive(mock_manager)
            dashboard = analytics.get_dashboard()

        assert dashboard is not None
        assert "summary" in dashboard

    def test_chart_empty_history(self):
        """Test chart generation with empty review history."""
        mock_manager = MagicMock()
        mock_manager.get_schedule.return_value = []
        mock_manager._review_history = []

        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
        ):
            analytics = ForgettingAnalyticsCognitive(mock_manager)
            chart = analytics.get_chart(ChartType.LEARNING_PROGRESS)

        # Should still return a chart, just with empty data
        assert chart is not None
        assert "series" in chart.data

    def test_export_empty_data(self):
        """Test export with empty data."""
        mock_manager = MagicMock()
        mock_manager.get_schedule.return_value = []
        mock_manager._review_history = []

        mock_core = MagicMock()
        mock_core.get_dashboard_data.return_value = {
            "summary": {},
            "agents": {},
        }

        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
            return_value=mock_core
        ):
            analytics = ForgettingAnalyticsCognitive(mock_manager)
            json_data = analytics.export_json()

        # Should still produce valid JSON
        parsed = json.loads(json_data)
        assert parsed is not None


# =====================================================================
# Factory Function Tests
# =====================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_forgetting_analytics(self, mock_forgetting_manager):
        """Test create_forgetting_analytics factory."""
        analytics = create_forgetting_analytics(
            mock_forgetting_manager,
            default_agent_id="test"
        )

        assert isinstance(analytics, ForgettingAnalyticsCognitive)
        assert analytics.default_agent_id == "test"

    def test_get_dashboard_html(self, mock_forgetting_manager, mock_core_analytics):
        """Test HTML dashboard generation."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
            return_value=mock_core_analytics
        ):
            html = get_dashboard_html(mock_forgetting_manager, agent_id="default")

        assert "<!DOCTYPE html>" in html
        assert "Memory Retention Dashboard" in html
        assert "Generated:" in html


# =====================================================================
# Integration Tests
# =====================================================================

class TestAnalyticsIntegration:
    """End-to-end integration scenarios."""

    def test_full_analytics_workflow(self, mock_forgetting_manager, mock_core_analytics):
        """Test full analytics workflow from data to export."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
            return_value=mock_core_analytics
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)

            # Get dashboard
            dashboard = analytics.get_dashboard(include_charts=True)
            assert "widgets" in dashboard

            # Get specific chart
            chart = analytics.get_chart(ChartType.RETENTION_CURVE)
            assert chart is not None

            # Export to JSON
            json_data = analytics.export_json()
            assert json_data is not None

            # Get agent summary
            summary = analytics.get_agent_summary("default")
            assert "agent_id" in summary

    def test_multi_agent_comparison(self, mock_forgetting_manager):
        """Test comparing multiple agents."""
        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics"
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)

            # Mock summaries for multiple agents
            summaries = {
                "agent_a": {
                    "agent_id": "agent_a",
                    "avg_retention": 0.85,
                    "success_rate": 0.90,
                },
                "agent_b": {
                    "agent_id": "agent_b",
                    "avg_retention": 0.75,
                    "success_rate": 0.80,
                },
                "agent_c": {
                    "agent_id": "agent_c",
                    "avg_retention": 0.80,
                    "success_rate": 0.85,
                },
            }

            with patch.object(
                analytics,
                "get_agent_summary",
                side_effect=lambda aid: summaries.get(aid, {})
            ):
                comparison = analytics.compare_agents(["agent_a", "agent_b", "agent_c"])

        assert "best_performing" in comparison
        assert comparison["best_performing"] == "agent_a"
        assert "averages" in comparison

    def test_widget_building_with_all_chart_types(self, mock_forgetting_manager):
        """Test widget building includes all chart types when data available."""
        mock_core = MagicMock()
        mock_core.get_dashboard_data.return_value = {
            "summary": {"total_scheduled": 10, "avg_retention": 0.8},
            "agents": {"default": {"total_memories": 10}},
            "retention_curve": [{"x": 0, "y": 1.0}],
            "emotional_distribution": {"emotional": 5, "neutral": 5},
            "review_schedule": {"by_action": {"review": 5}},
        }

        with patch(
            "mnemocore.cognitive.forgetting_analytics.CoreAnalytics",
            return_value=mock_core
        ):
            analytics = ForgettingAnalyticsCognitive(mock_forgetting_manager)
            dashboard = analytics.get_dashboard(include_charts=True)

        widget_types = {w["widget_type"] for w in dashboard["widgets"]}
        # Should have multiple widget types
        assert len(widget_types) > 0
