"""
Dream Pipeline – Phase 6.0: Offline Consolidation Pipeline
==========================================================
Multi-stage pipeline for offline memory consolidation during idle periods.

Pipeline Stages:
    1. [Episodic Cluster]      – Group memories by temporal proximity
    2. [Pattern Extractor]     – Identify recurring patterns
    3. [Recursive Synthesizer] – Build higher-level abstractions
    4. [Contradiction Resolver] – Detect and resolve contradictions
    5. [Semantic Promoter]     – Promote important memories
    6. [Dream Report Generator] – Generate summary report

The pipeline operates on:
- HOT tier memories (frequent access)
- WARM tier memories (archived but accessible)
- Contradiction registry entries
- Synaptic connections

Integration:
    - Uses RecursiveSynthesizer from core.recursive_synthesizer
    - Uses ContradictionDetector from core.contradiction
    - Integrates with SubconsciousAI for LLM-powered analysis
    - Publishes events to the subconscious bus

Usage:
    pipeline = DreamPipeline(engine, config)
    result = await pipeline.run()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

# Events integration
try:
    from ...events import integration as event_integration
    EVENTS_AVAILABLE = True
except ImportError:
    EVENTS_AVAILABLE = False
    event_integration = None  # type: ignore

# Import stage components
from .clusterer import EpisodicClusterer
from .patterns import PatternExtractor
from .synthesizer import DreamSynthesizer
from .contradictions import ContradictionResolver
from .promoter import SemanticPromoter
from .report import DreamReportGenerator

if TYPE_CHECKING:
    from ...core.engine import HAIMEngine
    from ...core.node import MemoryNode


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DreamPipelineConfig:
    """Configuration for the dream pipeline."""
    # Stage 1: Episodic Clustering
    enable_episodic_clustering: bool = True
    cluster_time_window_hours: float = 24.0  # Group memories within 24h
    min_cluster_size: int = 3

    # Stage 2: Pattern Extraction
    enable_pattern_extraction: bool = True
    pattern_min_frequency: int = 2  # Min occurrences to be a pattern
    pattern_similarity_threshold: float = 0.75

    # Stage 3: Recursive Synthesis
    enable_recursive_synthesis: bool = True
    synthesis_max_depth: int = 3
    synthesis_max_patterns: int = 10

    # Stage 4: Contradiction Resolution
    enable_contradiction_resolution: bool = True
    contradiction_similarity_threshold: float = 0.80
    auto_resolve_contradictions: bool = False  # Require manual review

    # Stage 5: Semantic Promotion
    enable_semantic_promotion: bool = True
    promotion_ltp_threshold: float = 0.7
    promotion_access_threshold: int = 5
    auto_promote_to_warm: bool = True

    # Stage 6: Dream Report
    enable_dream_report: bool = True
    report_include_memory_details: bool = False  # Privacy: don't include full content
    report_max_insights: int = 20

    # Resource limits
    max_memories_per_tier: int = 10000
    timeout_seconds: float = 600.0


# =============================================================================
# Pipeline Result
# =============================================================================

@dataclass
class DreamPipelineResult:
    """Result from running the dream pipeline."""
    success: bool
    duration_seconds: float
    started_at: datetime
    completed_at: datetime

    # Stage results
    episodic_clusters: List[Dict[str, Any]] = field(default_factory=list)
    patterns_extracted: List[Dict[str, Any]] = field(default_factory=list)
    synthesis_results: List[Dict[str, Any]] = field(default_factory=list)
    contradictions_resolved: List[Dict[str, Any]] = field(default_factory=list)
    promoted_memories: List[Dict[str, Any]] = field(default_factory=list)

    # Aggregated metrics
    memories_processed: int = 0
    memories_consolidated: int = 0
    contradictions_found: int = 0
    contradictions_resolved_count: int = 0
    patterns_extracted_count: int = 0
    semantic_promotions: int = 0
    recursive_synthesis_calls: int = 0

    # Dream report
    dream_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "duration_seconds": round(self.duration_seconds, 2),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "episodic_clusters_count": len(self.episodic_clusters),
            "patterns_extracted_count": self.patterns_extracted_count,
            "synthesis_results_count": len(self.synthesis_results),
            "contradictions_resolved_count": self.contradictions_resolved_count,
            "promoted_memories_count": len(self.promoted_memories),
            "memories_processed": self.memories_processed,
            "memories_consolidated": self.memories_consolidated,
            "contradictions_found": self.contradictions_found,
            "contradictions_resolved": self.contradictions_resolved_count,
            "patterns_extracted": self.patterns_extracted,
            "semantic_promotions": self.semantic_promotions,
            "recursive_synthesis_calls": self.recursive_synthesis_calls,
            "dream_report": self.dream_report,
        }


# =============================================================================
# Main Pipeline
# =============================================================================

class DreamPipeline:
    """
    Main dream pipeline orchestrator.

    Runs all stages in sequence:
    1. Fetch memories from HOT/WARM tiers
    2. Episodic clustering
    3. Pattern extraction
    4. Recursive synthesis
    5. Contradiction resolution
    6. Semantic promotion
    7. Dream report generation

    The pipeline is designed to run during idle periods and produces
    a comprehensive report of consolidation activities.
    """

    def __init__(
        self,
        engine: "HAIMEngine",
        config: Optional[DreamPipelineConfig] = None,
        event_bus: Optional[Any] = None,
    ):
        self.engine = engine
        self.cfg = config or DreamPipelineConfig()
        self.event_bus = event_bus

        # Pipeline components
        self.episodic_clusterer = EpisodicClusterer(
            time_window_hours=self.cfg.cluster_time_window_hours,
            min_cluster_size=self.cfg.min_cluster_size,
        )
        self.pattern_extractor = PatternExtractor(
            min_frequency=self.cfg.pattern_min_frequency,
            similarity_threshold=self.cfg.pattern_similarity_threshold,
        )
        self.dream_synthesizer = DreamSynthesizer(
            engine=self.engine,
            max_depth=self.cfg.synthesis_max_depth,
            max_patterns=self.cfg.synthesis_max_patterns,
        )
        self.contradiction_resolver = ContradictionResolver(
            engine=self.engine,
            similarity_threshold=self.cfg.contradiction_similarity_threshold,
            auto_resolve=self.cfg.auto_resolve_contradictions,
        )
        self.semantic_promoter = SemanticPromoter(
            engine=self.engine,
            ltp_threshold=self.cfg.promotion_ltp_threshold,
            access_threshold=self.cfg.promotion_access_threshold,
            auto_promote=self.cfg.auto_promote_to_warm,
        )
        self.report_generator = DreamReportGenerator(
            include_memory_details=self.cfg.report_include_memory_details,
            max_insights=self.cfg.report_max_insights,
        )

    async def run(self) -> Dict[str, Any]:
        """
        Execute the full dream pipeline.

        Returns:
            Dict with pipeline results and metrics.
        """
        # Generate session ID
        session_id = f"dream_{uuid.uuid4().hex[:12]}"

        started_at = datetime.now(timezone.utc)
        logger.info("[DreamPipeline] Starting offline consolidation pipeline")

        # Emit dream.started event
        if EVENTS_AVAILABLE and event_integration:
            await event_integration.emit_dream_started(
                event_bus=self.event_bus,
                session_id=session_id,
                trigger="idle",
                max_memories=self.cfg.max_memories_per_tier,
                stages_enabled=self._get_enabled_stages(),
            )

        try:
            # Fetch memories to process
            memories = await self._fetch_memories()

            if not memories:
                logger.info("[DreamPipeline] No memories to process")
                return self._empty_result(started_at)

            # Stage 1: Episodic Clustering
            clusters = []
            if self.cfg.enable_episodic_clustering:
                clusters = await self.episodic_clusterer.cluster(memories)
                logger.info(f"[DreamPipeline] Stage 1 complete: {len(clusters)} clusters")

            # Stage 2: Pattern Extraction
            patterns = []
            if self.cfg.enable_pattern_extraction:
                llm_client = getattr(self.engine, "subconscious_ai", None)
                llm_client = getattr(llm_client, "_model_client", None) if llm_client else None
                patterns = await self.pattern_extractor.extract(memories, llm_client)
                logger.info(f"[DreamPipeline] Stage 2 complete: {len(patterns)} patterns")

            # Stage 3: Recursive Synthesis
            synthesis_results = []
            if self.cfg.enable_recursive_synthesis and patterns:
                synthesis_results = await self.dream_synthesizer.synthesize_patterns(patterns)
                logger.info(f"[DreamPipeline] Stage 3 complete: {len(synthesis_results)} syntheses")

            # Stage 4: Contradiction Resolution
            contradiction_result = {}
            if self.cfg.enable_contradiction_resolution:
                contradiction_result = await self.contradiction_resolver.scan_and_resolve(memories)
                logger.info(
                    f"[DreamPipeline] Stage 4 complete: "
                    f"{contradiction_result['contradictions_found']} contradictions, "
                    f"{contradiction_result['contradictions_resolved']} resolved"
                )

            # Stage 5: Semantic Promotion
            promotion_result = {}
            if self.cfg.enable_semantic_promotion:
                promotion_result = await self.semantic_promoter.promote(memories, clusters)
                logger.info(
                    f"[DreamPipeline] Stage 5 complete: "
                    f"{promotion_result['promoted_count']} memories promoted"
                )

            # Stage 6: Dream Report
            dream_report = None
            if self.cfg.enable_dream_report:
                completed_at = datetime.now(timezone.utc)
                duration = (completed_at - started_at).total_seconds()
                dream_report = self.report_generator.generate(
                    self.cfg,
                    clusters,
                    patterns,
                    synthesis_results,
                    contradiction_result,
                    promotion_result,
                    duration,
                )
                logger.info("[DreamPipeline] Stage 6 complete: dream report generated")

            # Build result
            result = DreamPipelineResult(
                success=True,
                duration_seconds=(datetime.now(timezone.utc) - started_at).total_seconds(),
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                episodic_clusters=clusters,
                patterns_extracted=patterns,
                synthesis_results=synthesis_results,
                contradictions_resolved=contradiction_result.get("resolved_ids", []),
                promoted_memories=promotion_result.get("promoted_ids", []),
                memories_processed=len(memories),
                memories_consolidated=sum(c.get("memory_count", 0) for c in clusters),
                contradictions_found=contradiction_result.get("contradictions_found", 0),
                contradictions_resolved_count=contradiction_result.get("contradictions_resolved", 0),
                patterns_extracted_count=len(patterns),
                semantic_promotions=promotion_result.get("promoted_count", 0),
                recursive_synthesis_calls=len(synthesis_results),
                dream_report=dream_report,
            )

            logger.info(
                f"[DreamPipeline] Complete - processed={result.memories_processed}, "
                f"consolidated={result.memories_consolidated}, "
                f"duration={result.duration_seconds:.1f}s"
            )

            # Emit dream.completed event
            if EVENTS_AVAILABLE and event_integration:
                await event_integration.emit_dream_completed(
                    event_bus=self.event_bus,
                    session_id=session_id,
                    duration_seconds=result.duration_seconds,
                    memories_processed=result.memories_processed,
                    clusters_found=len(result.episodic_clusters),
                    patterns_extracted=result.patterns_extracted_count,
                    contradictions_resolved=result.contradictions_resolved_count,
                    memories_promoted=result.semantic_promotions,
                )

            return result.to_dict()

        except Exception as e:
            logger.error(f"[DreamPipeline] Error: {e}", exc_info=True)

            # Emit dream.failed event
            if EVENTS_AVAILABLE and event_integration:
                await event_integration.emit_dream_failed(
                    event_bus=self.event_bus,
                    session_id=session_id,
                    error=str(e),
                    duration_seconds=(datetime.now(timezone.utc) - started_at).total_seconds(),
                )

            return {
                "success": False,
                "error": str(e),
                "duration_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
                "started_at": started_at.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

    async def _fetch_memories(self) -> List["MemoryNode"]:
        """Fetch memories from tiers for processing."""
        memories: List["MemoryNode"] = []

        # Fetch from HOT tier
        try:
            hot_memories = await self.engine.tier_manager.get_all_hot()
            memories.extend(hot_memories[:self.cfg.max_memories_per_tier])
        except Exception as e:
            logger.warning(f"[DreamPipeline] Failed to fetch HOT memories: {e}")

        # Optionally fetch from WARM tier
        # (Disabled by default to avoid processing large archives)
        # if self.cfg.process_warm_tier:
        #     try:
        #         warm_memories = await self.engine.tier_manager.get_warm_recent(limit=self.cfg.max_memories_per_tier)
        #         memories.extend(warm_memories)
        #     except Exception as e:
        #         logger.warning(f"[DreamPipeline] Failed to fetch WARM memories: {e}")

        return memories

    def _empty_result(self, started_at: datetime) -> Dict[str, Any]:
        """Return empty result when no memories to process."""
        return {
            "success": True,
            "duration_seconds": 0.0,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "memories_processed": 0,
            "message": "No memories to process",
        }

    def _get_enabled_stages(self) -> List[str]:
        """Get list of enabled pipeline stages."""
        stages = []
        if self.cfg.enable_episodic_clustering:
            stages.append("episodic_clustering")
        if self.cfg.enable_pattern_extraction:
            stages.append("pattern_extraction")
        if self.cfg.enable_recursive_synthesis:
            stages.append("recursive_synthesis")
        if self.cfg.enable_contradiction_resolution:
            stages.append("contradiction_resolution")
        if self.cfg.enable_semantic_promotion:
            stages.append("semantic_promotion")
        if self.cfg.enable_dream_report:
            stages.append("dream_report")
        return stages


__all__ = [
    "DreamPipeline",
    "DreamPipelineConfig",
    "DreamPipelineResult",
]
