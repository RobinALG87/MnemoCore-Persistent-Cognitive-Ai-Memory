"""
Dream Report Generator – Stage 6 of Dream Pipeline
===================================================
Generates a comprehensive report from dream session results.

The report includes:
- Session metadata (timing, trigger)
- Cluster analysis summary
- Pattern discoveries
- Contradiction resolution status
- Promotion actions
- Insights and recommendations
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

from loguru import logger

if TYPE_CHECKING:
    from .pipeline import DreamPipelineConfig


@dataclass
class DreamReport:
    """Generated dream report."""
    report_generated_at: str
    session_duration_seconds: float
    pipeline_config: Dict[str, bool]
    summary: Dict[str, Any]
    episodic_analysis: Dict[str, Any]
    pattern_discoveries: Dict[str, Any]
    synthesis_insights: Dict[str, Any]
    contradiction_status: Dict[str, Any]
    promotion_summary: Dict[str, Any]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_generated_at": self.report_generated_at,
            "session_duration_seconds": self.session_duration_seconds,
            "pipeline_config": self.pipeline_config,
            "summary": self.summary,
            "episodic_analysis": self.episodic_analysis,
            "pattern_discoveries": self.pattern_discoveries,
            "synthesis_insights": self.synthesis_insights,
            "contradiction_status": self.contradiction_status,
            "promotion_summary": self.promotion_summary,
            "recommendations": self.recommendations,
        }


class DreamReportGenerator:
    """
    Generates a comprehensive report from dream session results.

    The report includes:
    - Session metadata (timing, trigger)
    - Cluster analysis summary
    - Pattern discoveries
    - Contradiction resolution status
    - Promotion actions
    - Insights and recommendations
    """

    def __init__(
        self,
        include_memory_details: bool = False,
        max_insights: int = 20,
    ):
        self.include_memory_details = include_memory_details
        self.max_insights = max_insights

    def generate(
        self,
        config: "DreamPipelineConfig",
        clusters: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]],
        synthesis: List[Dict[str, Any]],
        contradictions: Dict[str, Any],
        promotions: Dict[str, Any],
        duration_seconds: float,
    ) -> Dict[str, Any]:
        """Generate the dream report."""
        report = {
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
            "session_duration_seconds": round(duration_seconds, 2),
            "pipeline_config": {
                "episodic_clustering": config.enable_episodic_clustering,
                "pattern_extraction": config.enable_pattern_extraction,
                "recursive_synthesis": config.enable_recursive_synthesis,
                "contradiction_resolution": config.enable_contradiction_resolution,
                "semantic_promotion": config.enable_semantic_promotion,
            },
            "summary": self._generate_summary(
                clusters, patterns, synthesis, contradictions, promotions
            ),
            "episodic_analysis": self._analyze_clusters(clusters),
            "pattern_discoveries": self._analyze_patterns(patterns),
            "synthesis_insights": self._analyze_synthesis(synthesis),
            "contradiction_status": self._analyze_contradictions(contradictions),
            "promotion_summary": promotions,
            "recommendations": self._generate_recommendations(
                clusters, patterns, contradictions
            ),
        }

        return report

    def _generate_summary(
        self,
        clusters: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]],
        synthesis: List[Dict[str, Any]],
        contradictions: Dict[str, Any],
        promotions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate executive summary."""
        return {
            "episodic_clusters_found": len(clusters),
            "patterns_discovered": len(patterns),
            "synthesis_insights": len(synthesis),
            "contradictions_found": contradictions.get("contradictions_found", 0),
            "contradictions_resolved": contradictions.get("contradictions_resolved", 0),
            "memories_promoted": promotions.get("promoted_count", 0),
            "overall_health": self._calculate_health_score(
                clusters, patterns, contradictions
            ),
        }

    def _analyze_clusters(self, clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze episodic clusters."""
        if not clusters:
            return {"total": 0, "insights": []}

        # Calculate cluster statistics
        total_memories = sum(c.get("memory_count", 0) for c in clusters)
        avg_cluster_size = total_memories / len(clusters) if clusters else 0

        # Find largest cluster
        largest = max(clusters, key=lambda c: c.get("memory_count", 0), default={})

        return {
            "total": len(clusters),
            "total_memories_in_clusters": total_memories,
            "avg_cluster_size": round(avg_cluster_size, 1),
            "largest_cluster": {
                "id": largest.get("cluster_id"),
                "size": largest.get("memory_count", 0),
                "duration_hours": largest.get("duration_hours", 0),
            },
            "categories": self._extract_cluster_categories(clusters),
        }

    def _extract_cluster_categories(self, clusters: List[Dict[str, Any]]) -> List[str]:
        """Extract common categories from clusters."""
        category_counts: Dict[str, int] = defaultdict(int)
        for cluster in clusters:
            for cat in cluster.get("categories", []):
                category_counts[cat] += 1

        return [cat for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:5]]

    def _analyze_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze extracted patterns."""
        # Group by type
        by_type: Dict[str, List[Dict]] = defaultdict(list)
        for pattern in patterns:
            ptype = pattern.get("pattern_type", "unknown")
            by_type[ptype].append(pattern)

        return {
            "total": len(patterns),
            "by_type": {
                ptype: len(pts) for ptype, pts in by_type.items()
            },
            "top_patterns": [
                {
                    "type": p.get("pattern_type"),
                    "value": p.get("pattern_value"),
                    "frequency": p.get("frequency", 0),
                }
                for p in sorted(patterns, key=lambda x: x.get("frequency", 0), reverse=True)[:10]
            ],
        }

    def _analyze_synthesis(self, synthesis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze synthesis results."""
        return {
            "total_syntheses": len(synthesis),
            "avg_results_per_synthesis": round(
                sum(s.get("results_count", 0) for s in synthesis) / max(len(synthesis), 1),
                1
            ),
            "dream_memories_created": [
                s.get("dream_memory_id") for s in synthesis if s.get("dream_memory_id")
            ],
        }

    def _analyze_contradictions(self, contradictions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contradiction status."""
        found = contradictions.get("contradictions_found", 0)
        resolved = contradictions.get("contradictions_resolved", 0)

        return {
            "found": found,
            "resolved": resolved,
            "unresolved": found - resolved,
            "resolution_rate": round(resolved / max(found, 1) * 100, 1),
            "unresolved_ids": contradictions.get("unresolved_ids", [])[:10],
        }

    def _generate_recommendations(
        self,
        clusters: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]],
        contradictions: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Contradiction recommendations
        if contradictions.get("contradictions_found", 0) > 0:
            unresolved = contradictions.get("contradictions_found", 0) - contradictions.get("contradictions_resolved", 0)
            if unresolved > 0:
                recommendations.append(
                    f"Review {unresolved} unresolved contradictions for manual resolution"
                )

        # Pattern recommendations
        high_freq_patterns = [p for p in patterns if p.get("frequency", 0) >= 5]
        if high_freq_patterns:
            recommendations.append(
                f"Investigate {len(high_freq_patterns)} high-frequency patterns for deeper analysis"
            )

        # Cluster recommendations
        if clusters:
            avg_size = sum(c.get("memory_count", 0) for c in clusters) / max(len(clusters), 1)
            if avg_size > 10:
                recommendations.append(
                    "Consider creating concept nodes from large episodic clusters"
                )

        return recommendations[:5]

    def _calculate_health_score(
        self,
        clusters: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]],
        contradictions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate overall memory system health score."""
        score = 100.0

        # Deduct for unresolved contradictions
        unresolved = contradictions.get("contradictions_found", 0) - contradictions.get("contradictions_resolved", 0)
        score -= min(unresolved * 5, 30)  # Max -30 for contradictions

        # Bonus for healthy clustering
        if clusters:
            score += min(len(clusters), 10)  # Max +10 for clusters

        # Bonus for pattern diversity
        pattern_types = set(p.get("pattern_type") for p in patterns)
        score += min(len(pattern_types) * 2, 10)  # Max +10 for diversity

        return {
            "score": round(max(0, min(100, score)), 1),
            "status": "healthy" if score >= 70 else "attention_needed" if score >= 40 else "critical",
        }


__all__ = ["DreamReportGenerator", "DreamReport"]
