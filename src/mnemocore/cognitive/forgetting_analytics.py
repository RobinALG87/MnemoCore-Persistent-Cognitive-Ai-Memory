"""
Forgetting Analytics Module (Cognitive Layer)
==============================================
Provides visualization and analytics capabilities for the forgetting curve system.

This module serves as the cognitive layer interface to the forgetting curve analytics,
offering both programmatic APIs and export capabilities for dashboard integration.

Key Features:
  1. Dashboard data generation for web UIs
  2. CSV/JSON export for external tools
  3. Agent comparison and learning progress tracking
  4. Retention curve visualization data
  5. Emotional memory distribution analytics

Public API:
    analytics = ForgettingAnalyticsCognitive(forgetting_manager)
    dashboard = analytics.get_dashboard()
    csv_data = analytics.export_csv()
    chart_data = analytics.get_progress_chart(agent_id)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pathlib import Path
from enum import Enum

from loguru import logger

if TYPE_CHECKING:
    from ..core.forgetting_curve import (
        ForgettingCurveManager,
        LearningProfile,
        SM2State,
        ReviewEntry,
    )


# ------------------------------------------------------------------ #
#  Visualization Data Structures                                      #
# ------------------------------------------------------------------ #

class ChartType(Enum):
    """Types of charts supported by the analytics module."""
    RETENTION_CURVE = "retention_curve"
    LEARNING_PROGRESS = "learning_progress"
    AGENT_COMPARISON = "agent_comparison"
    EMOTIONAL_DISTRIBUTION = "emotional_distribution"
    SCHEDULE_HEATMAP = "schedule_heatmap"
    SM2_PERFORMANCE = "sm2_performance"


@dataclass
class ChartData:
    """
    Container for chart visualization data.

    Attributes:
        chart_type: Type of chart
        title: Human-readable title
        data: The actual chart data (structure depends on chart_type)
        metadata: Additional metadata for rendering
        generated_at: When this data was generated
    """
    chart_type: ChartType
    title: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "chart_type": self.chart_type.value,
            "title": self.title,
            "data": self.data,
            "metadata": self.metadata,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class DashboardWidget:
    """
    A single dashboard widget configuration.

    Widgets can be combined to form a complete dashboard layout.
    """
    widget_id: str
    widget_type: str  # "chart", "metric", "table", etc.
    title: str
    data: Dict[str, Any]
    position: Optional[Dict[str, int]] = None  # {"row": 0, "col": 0, "rowspan": 1, "colspan": 2}
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "widget_id": self.widget_id,
            "widget_type": self.widget_type,
            "title": self.title,
            "data": self.data,
            "position": self.position,
            "config": self.config,
        }


# ------------------------------------------------------------------ #
#  Main Analytics Class                                               #
# ------------------------------------------------------------------ #

class ForgettingAnalyticsCognitive:
    """
    Cognitive layer analytics for forgetting curve visualization.

    This class wraps the core ForgettingAnalytics with additional
    cognitive-specific features and export capabilities.
    """

    def __init__(
        self,
        manager: "ForgettingCurveManager",
        default_agent_id: str = "default"
    ):
        """
        Initialize analytics with a forgetting curve manager.

        Args:
            manager: The ForgettingCurveManager to analyze
            default_agent_id: Default agent for filtered queries
        """
        # Import here to avoid circular dependency
        from ..core.forgetting_curve import ForgettingAnalytics as CoreAnalytics

        self.manager = manager
        self.default_agent_id = default_agent_id
        self._core_analytics = CoreAnalytics(manager)

    # ------------------------------------------------------------------
    #  Dashboard Generation
    # ------------------------------------------------------------------

    def get_dashboard(
        self,
        agent_id: Optional[str] = None,
        include_charts: bool = True,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Get complete dashboard data for visualization.

        Returns a structured dashboard ready for rendering.

        Args:
            agent_id: Filter by specific agent (None = all agents)
            include_charts: Include chart data
            include_recommendations: Include AI-generated recommendations

        Returns:
            Complete dashboard data dictionary
        """
        agent = agent_id or self.default_agent_id
        core_data = self._core_analytics.get_dashboard_data(agent)

        dashboard = {
            "title": f"Memory Retention Dashboard - {agent}",
            "agent_id": agent,
            "summary": core_data["summary"],
            "widgets": self._build_widgets(core_data, include_charts),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        if include_recommendations:
            dashboard["recommendations"] = core_data.get("recommendations", [])

        return dashboard

    def _build_widgets(
        self,
        core_data: Dict[str, Any],
        include_charts: bool
    ) -> List[Dict[str, Any]]:
        """Build dashboard widgets from core analytics data."""
        widgets = []

        # Summary metrics widget
        widgets.append(DashboardWidget(
            widget_id="summary_metrics",
            widget_type="metric_grid",
            title="Summary Metrics",
            data=core_data["summary"],
            position={"row": 0, "col": 0, "rowspan": 1, "colspan": 2},
        ).to_dict())

        # Agent analytics table
        if core_data.get("agents"):
            agents_data = []
            for aid, analytics in core_data["agents"].items():
                entry = {"agent_id": aid}
                if hasattr(analytics, "__dict__"):
                    entry.update(analytics.__dict__)
                elif isinstance(analytics, dict):
                    entry.update(analytics)
                else:
                    entry["data"] = str(analytics)
                agents_data.append(entry)
            widgets.append(DashboardWidget(
                widget_id="agent_table",
                widget_type="table",
                title="Agent Performance",
                data={"columns": ["agent_id", "total_memories", "avg_retention"], "rows": agents_data},
                position={"row": 1, "col": 0, "rowspan": 2, "colspan": 1},
            ).to_dict())

        if include_charts:
            # Retention curve chart
            if core_data.get("retention_curve"):
                widgets.append(DashboardWidget(
                    widget_id="retention_curve",
                    widget_type="line_chart",
                    title="Retention Curve",
                    data={
                        "x": "time_days",
                        "y": "retention",
                        "series": core_data["retention_curve"],
                    },
                    position={"row": 0, "col": 2, "rowspan": 2, "colspan": 2},
                    config={"x_label": "Time (days)", "y_label": "Retention", "y_range": [0, 1]},
                ).to_dict())

            # Emotional distribution chart
            if core_data.get("emotional_distribution"):
                widgets.append(DashboardWidget(
                    widget_id="emotional_pie",
                    widget_type="pie_chart",
                    title="Emotional vs Neutral Memories",
                    data={
                        "categories": ["Emotional", "Neutral"],
                        "values": [
                            core_data["emotional_distribution"]["emotional"],
                            core_data["emotional_distribution"]["neutral"],
                        ],
                    },
                    position={"row": 2, "col": 2, "rowspan": 1, "colspan": 1},
                ).to_dict())

            # Schedule breakdown
            if core_data.get("review_schedule"):
                widgets.append(DashboardWidget(
                    widget_id="schedule_breakdown",
                    widget_type="bar_chart",
                    title="Review Schedule by Action",
                    data={
                        "categories": list(core_data["review_schedule"]["by_action"].keys()),
                        "values": list(core_data["review_schedule"]["by_action"].values()),
                    },
                    position={"row": 2, "col": 3, "rowspan": 1, "colspan": 1},
                ).to_dict())

        return widgets

    def get_widget(self, widget_id: str, agent_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a specific dashboard widget by ID.

        Args:
            widget_id: The widget identifier
            agent_id: Optional agent filter

        Returns:
            Widget data or None if not found
        """
        dashboard = self.get_dashboard(agent_id)
        for widget in dashboard.get("widgets", []):
            if widget.get("widget_id") == widget_id:
                return widget
        return None

    # ------------------------------------------------------------------
    #  Chart Data Generation
    # ------------------------------------------------------------------

    def get_chart(
        self,
        chart_type: ChartType,
        agent_id: Optional[str] = None,
        memory_id: Optional[str] = None
    ) -> Optional[ChartData]:
        """
        Get data for a specific chart type.

        Args:
            chart_type: The type of chart to generate
            agent_id: Optional agent filter
            memory_id: Optional specific memory (for some chart types)

        Returns:
            ChartData object or None if unsupported
        """
        agent = agent_id or self.default_agent_id

        if chart_type == ChartType.RETENTION_CURVE:
            return self._get_retention_curve_chart(agent)
        elif chart_type == ChartType.LEARNING_PROGRESS:
            return self._get_learning_progress_chart(agent, memory_id)
        elif chart_type == ChartType.AGENT_COMPARISON:
            return self._get_agent_comparison_chart()
        elif chart_type == ChartType.EMOTIONAL_DISTRIBUTION:
            return self._get_emotional_distribution_chart(agent)
        elif chart_type == ChartType.SM2_PERFORMANCE:
            return self._get_sm2_performance_chart(agent)
        elif chart_type == ChartType.SCHEDULE_HEATMAP:
            return self._get_schedule_heatmap(agent)

        logger.warning(f"Unsupported chart type: {chart_type}")
        return None

    def _get_retention_curve_chart(self, agent_id: str) -> ChartData:
        """Generate retention curve chart data."""
        from ..core.forgetting_curve import HIGH_SALIENCE_THRESHOLD

        schedule = self.manager.get_schedule(agent_id)

        # Group by time buckets
        time_buckets: Dict[float, List[Dict]] = {}
        now = datetime.now(timezone.utc)

        for entry in schedule:
            if entry.sm2_state and entry.sm2_state.last_review_date:
                age = (now - entry.sm2_state.last_review_date).total_seconds() / 86400.0
            else:
                age = 0.0

            bucket = round(age, 1)
            if bucket not in time_buckets:
                time_buckets[bucket] = []

            time_buckets[bucket].append({
                "retention": entry.current_retention,
                "is_emotional": entry.emotional_salience >= HIGH_SALIENCE_THRESHOLD,
            })

        # Build series data
        all_data = []
        emotional_data = []

        for bucket in sorted(time_buckets.keys()):
            entries = time_buckets[bucket]
            avg_retention = sum(e["retention"] for e in entries) / len(entries)
            all_data.append({"x": bucket, "y": avg_retention})

            emotional_entries = [e for e in entries if e["is_emotional"]]
            if emotional_entries:
                avg_emotional = sum(e["retention"] for e in emotional_entries) / len(emotional_entries)
                emotional_data.append({"x": bucket, "y": avg_emotional})

        return ChartData(
            chart_type=ChartType.RETENTION_CURVE,
            title="Memory Retention Over Time",
            data={
                "series": [
                    {"name": "All Memories", "data": all_data},
                    {"name": "Emotional Memories", "data": emotional_data},
                ],
            },
            metadata={
                "x_label": "Time Since Last Review (days)",
                "y_label": "Retention Probability",
                "y_range": [0, 1],
            }
        )

    def _get_learning_progress_chart(
        self,
        agent_id: str,
        memory_id: Optional[str] = None
    ) -> ChartData:
        """Generate learning progress chart data."""
        history = self.manager._review_history
        history = [h for h in history if h.get("agent_id") == agent_id]

        if memory_id:
            history = [h for h in history if h.get("memory_id") == memory_id]
            title = f"Learning Progress: {memory_id[:8]}"
        else:
            title = f"Learning Progress: {agent_id}"

        chart_data = []
        for i, entry in enumerate(history):
            chart_data.append({
                "x": i + 1,
                "y": entry.get("quality", 0),
                "interval": entry.get("new_interval", 0),
            })

        return ChartData(
            chart_type=ChartType.LEARNING_PROGRESS,
            title=title,
            data={
                "series": [{"name": "Review Quality", "data": chart_data}],
            },
            metadata={
                "x_label": "Review Number",
                "y_label": "Quality (0-5)",
                "y_range": [0, 5],
            }
        )

    def _get_agent_comparison_chart(self) -> ChartData:
        """Generate agent comparison chart data."""
        comparison = self._core_analytics.get_agent_comparison()

        chart_data = [
            {
                "x": comp["agent_id"],
                "y": comp["avg_retention"],
                "memories": comp["total_memories"],
                "style": comp.get("learning_style", "unknown"),
            }
            for comp in comparison
        ]

        return ChartData(
            chart_type=ChartType.AGENT_COMPARISON,
            title="Agent Performance Comparison",
            data={
                "series": [{"name": "Average Retention", "data": chart_data}],
            },
            metadata={
                "x_label": "Agent ID",
                "y_label": "Average Retention",
                "y_range": [0, 1],
            }
        )

    def _get_emotional_distribution_chart(self, agent_id: str) -> ChartData:
        """Generate emotional distribution chart data."""
        schedule = self.manager.get_schedule(agent_id)
        from ..core.forgetting_curve import HIGH_SALIENCE_THRESHOLD

        quartiles = {"emotional": 0, "neutral": 0, "mixed": 0}

        for entry in schedule:
            if entry.emotional_salience >= HIGH_SALIENCE_THRESHOLD:
                quartiles["emotional"] += 1
            elif entry.emotional_salience >= 0.2:
                quartiles["mixed"] += 1
            else:
                quartiles["neutral"] += 1

        chart_data = [
            {"label": k, "value": v}
            for k, v in quartiles.items()
            if v > 0
        ]

        return ChartData(
            chart_type=ChartType.EMOTIONAL_DISTRIBUTION,
            title=f"Emotional Distribution: {agent_id}",
            data={
                "categories": [d["label"] for d in chart_data],
                "values": [d["value"] for d in chart_data],
            },
            metadata={
                "chart_type": "pie",
            }
        )

    def _get_sm2_performance_chart(self, agent_id: str) -> ChartData:
        """Generate SM-2 algorithm performance chart."""
        history = self.manager._review_history
        history = [h for h in history if h.get("agent_id") == agent_id]

        # Calculate success rate over time (moving window)
        window_size = 10
        success_rate_over_time = []

        for i in range(window_size, len(history)):
            window = history[i - window_size:i]
            successful = sum(1 for h in window if h.get("quality", 0) >= 3)
            rate = successful / window_size

            success_rate_over_time.append({
                "x": i,
                "y": rate,
            })

        return ChartData(
            chart_type=ChartType.SM2_PERFORMANCE,
            title=f"SM-2 Performance: {agent_id}",
            data={
                "series": [{"name": "Success Rate", "data": success_rate_over_time}],
            },
            metadata={
                "x_label": "Review Number",
                "y_label": "Success Rate",
                "y_range": [0, 1],
                "window_size": window_size,
            }
        )

    def _get_schedule_heatmap(self, agent_id: str) -> ChartData:
        """Generate review schedule heatmap data."""
        schedule = self.manager.get_schedule(agent_id)
        now = datetime.now(timezone.utc)

        # Group by day of week and hour
        heatmap_data: Dict[str, Dict[int, int]] = {}

        for entry in schedule:
            due = entry.due_at
            day_name = due.strftime("%A")
            hour = due.hour

            if day_name not in heatmap_data:
                heatmap_data[day_name] = {}

            heatmap_data[day_name][hour] = heatmap_data[day_name].get(hour, 0) + 1

        # Convert to flat array for visualization
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        flat_data = []

        for day in days_of_week:
            for hour in range(24):
                count = heatmap_data.get(day, {}).get(hour, 0)
                flat_data.append({
                    "x": hour,
                    "y": day,
                    "v": count,
                })

        return ChartData(
            chart_type=ChartType.SCHEDULE_HEATMAP,
            title=f"Review Schedule Heatmap: {agent_id}",
            data={
                "series": [{"name": "Scheduled Reviews", "data": flat_data}],
            },
            metadata={
                "x_label": "Hour of Day",
                "y_label": "Day of Week",
                "color_scale": "heatmap",
            }
        )

    # ------------------------------------------------------------------
    #  Export Functions
    # ------------------------------------------------------------------

    def export_csv(
        self,
        chart_type: ChartType,
        agent_id: Optional[str] = None,
        include_header: bool = True
    ) -> str:
        """
        Export chart data as CSV string.

        Args:
            chart_type: Type of chart to export
            agent_id: Optional agent filter
            include_header: Include CSV header row

        Returns:
            CSV-formatted string
        """
        chart = self.get_chart(chart_type, agent_id)
        if not chart:
            return ""

        lines = []

        if include_header:
            lines.append("x,y")

        for series in chart.data.get("series", []):
            for point in series.get("data", []):
                if isinstance(point, dict):
                    x = point.get("x", "")
                    y = point.get("y", "")
                    lines.append(f"{x},{y}")
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    lines.append(f"{point[0]},{point[1]}")

        return "\n".join(lines)

    def export_json(
        self,
        agent_id: Optional[str] = None,
        pretty: bool = True
    ) -> str:
        """
        Export full dashboard as JSON string.

        Args:
            agent_id: Optional agent filter
            pretty: Pretty-print JSON

        Returns:
            JSON-formatted string
        """
        dashboard = self.get_dashboard(agent_id)

        if pretty:
            return json.dumps(dashboard, indent=2, default=str)
        return json.dumps(dashboard, default=str)

    def export_to_file(
        self,
        filepath: str,
        format: str = "json",
        agent_id: Optional[str] = None
    ) -> None:
        """
        Export analytics data to a file.

        Args:
            filepath: Path to output file
            format: Export format ("json" or "csv")
            agent_id: Optional agent filter
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            content = self.export_json(agent_id, pretty=True)
        elif format == "csv":
            # For CSV, export retention curve as default
            content = self.export_csv(ChartType.RETENTION_CURVE, agent_id)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Exported analytics to {filepath}")

    # ------------------------------------------------------------------
    #  Analysis Utilities
    # ------------------------------------------------------------------

    def get_agent_summary(self, agent_id: str) -> Dict[str, Any]:
        """
        Get a summary of an agent's learning state.

        Returns key metrics and insights for a single agent.
        """
        profile = self.manager.get_profile(agent_id)
        if not profile:
            return {"error": f"Agent '{agent_id}' not found"}

        schedule = self.manager.get_schedule(agent_id)
        agent_entries = [e for e in schedule if e.agent_id == agent_id]

        now = datetime.now(timezone.utc)
        due_count = sum(1 for e in agent_entries if e.due_at <= now)
        overdue_count = sum(1 for e in agent_entries if e.due_at < now)

        # Calculate next review window
        next_24h = sum(1 for e in agent_entries if 0 <= (e.due_at - now).total_seconds() / 86400.0 <= 1)
        next_week = sum(1 for e in agent_entries if 0 <= (e.due_at - now).total_seconds() / 86400.0 <= 7)

        # SM-2 statistics
        sm2_states = [
            e.sm2_state for e in agent_entries
            if e.sm2_state and e.sm2_state.repetitions > 0
        ]

        if sm2_states:
            avg_repetitions = sum(s.repetitions for s in sm2_states) / len(sm2_states)
            mature_items = sum(1 for s in sm2_states if s.repetitions >= 3)
        else:
            avg_repetitions = 0
            mature_items = 0

        return {
            "agent_id": agent_id,
            "learning_style": profile.learning_style,
            "easiness_factor": round(profile.easiness_factor, 3),
            "total_memories": len(agent_entries),
            "due_now": due_count,
            "overdue": overdue_count,
            "next_24h": next_24h,
            "next_week": next_week,
            "avg_retention": round(
                sum(e.current_retention for e in agent_entries) / len(agent_entries), 4
            ) if agent_entries else 0.0,
            "sm2_active_items": len(sm2_states),
            "sm2_avg_repetitions": round(avg_repetitions, 2),
            "sm2_mature_items": mature_items,
            "total_reviews": profile.stats.get("total_reviews", 0),
            "success_rate": round(
                profile.stats.get("successful_reviews", 0) /
                max(profile.stats.get("total_reviews", 1), 1), 4
            ),
        }

    def compare_agents(self, agent_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple agents side by side.

        Args:
            agent_ids: List of agent IDs to compare

        Returns:
            Comparison data with metrics for each agent
        """
        summaries = {
            agent_id: self.get_agent_summary(agent_id)
            for agent_id in agent_ids
        }

        # Find best performing agent
        best_agent = max(
            agent_ids,
            key=lambda aid: summaries[aid].get("avg_retention", 0)
        )

        # Calculate averages
        avg_retention = sum(s.get("avg_retention", 0) for s in summaries.values()) / len(agent_ids)
        avg_success_rate = sum(s.get("success_rate", 0) for s in summaries.values()) / len(agent_ids)

        return {
            "agents": summaries,
            "best_performing": best_agent,
            "averages": {
                "retention": round(avg_retention, 4),
                "success_rate": round(avg_success_rate, 4),
            },
            "comparison_at": datetime.now(timezone.utc).isoformat(),
        }


# ------------------------------------------------------------------ #
#  Convenience Functions                                               #
# ------------------------------------------------------------------ #

def create_forgetting_analytics(
    manager: "ForgettingCurveManager",
    default_agent_id: str = "default"
) -> ForgettingAnalyticsCognitive:
    """
    Factory function to create a ForgettingAnalyticsCognitive instance.

    Args:
        manager: The ForgettingCurveManager to analyze
        default_agent_id: Default agent for queries

    Returns:
        Configured ForgettingAnalyticsCognitive instance
    """
    return ForgettingAnalyticsCognitive(manager, default_agent_id)


def get_dashboard_html(
    manager: "ForgettingCurveManager",
    agent_id: str = "default"
) -> str:
    """
    Generate a basic HTML dashboard for viewing in a browser.

    Args:
        manager: The ForgettingCurveManager to visualize
        agent_id: Agent to display data for

    Returns:
        HTML string for the dashboard
    """
    analytics = create_forgetting_analytics(manager, agent_id)
    dashboard = analytics.get_dashboard(agent_id)

    # Simple HTML template
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Memory Retention Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .metric {{ background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        .section {{ margin: 30px 0; }}
        .section h2 {{ color: #555; }}
        pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{dashboard['title']}</h1>
        <p>Generated: {dashboard['generated_at']}</p>

        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{dashboard['summary'].get('total_scheduled', 0)}</div>
                <div class="metric-label">Total Memories</div>
            </div>
            <div class="metric">
                <div class="metric-value">{dashboard['summary'].get('avg_retention', 0):.2%}</div>
                <div class="metric-label">Avg Retention</div>
            </div>
            <div class="metric">
                <div class="metric-value">{dashboard['summary'].get('due_now', 0)}</div>
                <div class="metric-label">Due for Review</div>
            </div>
            <div class="metric">
                <div class="metric-value">{dashboard['summary'].get('overdue', 0)}</div>
                <div class="metric-label">Overdue</div>
            </div>
        </div>

        <div class="section">
            <h2>Dashboard Data (JSON)</h2>
            <pre>{json.dumps(dashboard, indent=2, default=str)}</pre>
        </div>
    </div>
</body>
</html>
"""

    return html
