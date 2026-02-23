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

import asyncio
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from loguru import logger

# Events integration
try:
    from ..events import integration as event_integration
    EVENTS_AVAILABLE = True
except ImportError:
    EVENTS_AVAILABLE = False
    event_integration = None  # type: ignore

if TYPE_CHECKING:
    from ..core.engine import HAIMEngine
    from ..core.node import MemoryNode
    from ..core.recursive_synthesizer import RecursiveSynthesizer
    from ..core.contradiction import ContradictionDetector


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
            "patterns_extracted_count": len(self.patterns_extracted),
            "synthesis_results_count": len(self.synthesis_results),
            "contradictions_resolved_count": len(self.contradictions_resolved),
            "promoted_memories_count": len(self.promoted_memories),
            "memories_processed": self.memories_processed,
            "memories_consolidated": self.memories_consolidated,
            "contradictions_found": self.contradictions_found,
            "contradictions_resolved": self.contradictions_resolved_count,
            "patterns_extracted": self.patterns_extracted_count,
            "semantic_promotions": self.semantic_promotions,
            "recursive_synthesis_calls": self.recursive_synthesis_calls,
            "dream_report": self.dream_report,
        }


# =============================================================================
# Stage 1: Episodic Clustering
# =============================================================================

class EpisodicClusterer:
    """
    Groups memories into episodic clusters based on temporal proximity.

    Memories created within a time window are grouped together as
    potential episodes. This mimics how the brain organizes related
    experiences.

    Algorithm:
    1. Sort memories by creation time
    2. Group memories within time_window_hours
    3. Filter clusters by min_cluster_size
    4. Optionally boost synaptic connections within clusters
    """

    def __init__(
        self,
        time_window_hours: float = 24.0,
        min_cluster_size: int = 3,
    ):
        self.time_window = timedelta(hours=time_window_hours)
        self.min_cluster_size = min_cluster_size

    async def cluster(
        self,
        memories: List["MemoryNode"],
        boost_connections: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Cluster memories by temporal proximity.

        Args:
            memories: List of memory nodes to cluster.
            boost_connections: If True, strengthen connections within clusters.

        Returns:
            List of cluster dicts with metadata.
        """
        if not memories:
            return []

        # Sort by creation time
        sorted_memories = sorted(memories, key=lambda m: m.created_at)

        clusters: List[Dict[str, Any]] = []
        current_cluster: List["MemoryNode"] = []
        cluster_start: Optional[datetime] = None

        for memory in sorted_memories:
            if cluster_start is None:
                cluster_start = memory.created_at
                current_cluster = [memory]
            elif memory.created_at - cluster_start <= self.time_window:
                current_cluster.append(memory)
            else:
                # Finalize current cluster if large enough
                if len(current_cluster) >= self.min_cluster_size:
                    clusters.append(self._create_cluster(current_cluster))
                # Start new cluster
                cluster_start = memory.created_at
                current_cluster = [memory]

        # Don't forget the last cluster
        if len(current_cluster) >= self.min_cluster_size:
            clusters.append(self._create_cluster(current_cluster))

        logger.info(f"[EpisodicClusterer] Found {len(clusters)} clusters from {len(memories)} memories")

        return clusters

    def _create_cluster(self, memories: List["MemoryNode"]) -> Dict[str, Any]:
        """Create a cluster dict from a group of memories."""
        if not memories:
            return {}

        # Calculate cluster metadata
        start_time = min(m.created_at for m in memories)
        end_time = max(m.created_at for m in memories)
        duration = end_time - start_time

        # Extract common themes from metadata
        categories = set()
        for m in memories:
            if cat := m.metadata.get("category"):
                categories.add(cat)

        return {
            "cluster_id": f"cluster_{start_time.strftime('%Y%m%d_%H%M%S')}",
            "memory_count": len(memories),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_hours": duration.total_seconds() / 3600,
            "memory_ids": [m.id for m in memories],
            "categories": list(categories),
            "avg_ltp": sum(m.ltp_strength for m in memories) / len(memories),
        }


# =============================================================================
# Stage 2: Pattern Extractor
# =============================================================================

class PatternExtractor:
    """
    Extracts recurring patterns from memory content and metadata.

    Patterns include:
    - Recurring keywords/topics
    - Temporal patterns (e.g., weekly activities)
    - Semantic similarities across unrelated memories
    - Temporal sequences (A often follows B)

    Uses both heuristic analysis and optional LLM-based pattern detection.
    """

    def __init__(
        self,
        min_frequency: int = 2,
        similarity_threshold: float = 0.75,
    ):
        self.min_frequency = min_frequency
        self.similarity_threshold = similarity_threshold

        # Common stopwords for filtering
        self._stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "as", "is", "was",
            "are", "were", "been", "be", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might",
            # Swedish stopwords
            "och", "eller", "men", "för", "av", "på", "i", "med", "till",
            "från", "som", "är", "var", "varit", "blir", "blev", "ha",
            "har", "hade", "kommer", "skulle", "kunde", "måste",
        }

    async def extract(
        self,
        memories: List["MemoryNode"],
        llm_client: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract patterns from memories.

        Args:
            memories: List of memory nodes to analyze.
            llm_client: Optional LLM client for semantic pattern extraction.

        Returns:
            List of pattern dicts with metadata.
        """
        patterns = []

        # 1. Keyword frequency patterns
        keyword_patterns = self._extract_keyword_patterns(memories)
        patterns.extend(keyword_patterns)

        # 2. Temporal patterns
        temporal_patterns = self._extract_temporal_patterns(memories)
        patterns.extend(temporal_patterns)

        # 3. Metadata patterns
        metadata_patterns = self._extract_metadata_patterns(memories)
        patterns.extend(metadata_patterns)

        # 4. LLM-based semantic patterns (if available)
        if llm_client:
            semantic_patterns = await self._extract_semantic_patterns(
                memories, llm_client
            )
            patterns.extend(semantic_patterns)

        logger.info(f"[PatternExtractor] Extracted {len(patterns)} patterns")

        return patterns[:50]  # Limit to top patterns

    def _extract_keyword_patterns(self, memories: List["MemoryNode"]) -> List[Dict[str, Any]]:
        """Extract recurring keyword patterns."""
        word_counts = defaultdict(int)
        word_memories: Dict[str, Set[str]] = defaultdict(set)

        for memory in memories:
            # Tokenize and clean content
            words = self._tokenize(memory.content)
            for word in words:
                if word not in self._stopwords and len(word) > 3:
                    word_counts[word] += 1
                    word_memories[word].add(memory.id)

        # Filter by frequency
        patterns = []
        for word, count in word_counts.items():
            if count >= self.min_frequency:
                patterns.append({
                    "pattern_type": "keyword",
                    "pattern_value": word,
                    "frequency": count,
                    "memory_ids": list(word_memories[word])[:20],  # Limit
                })

        # Sort by frequency
        patterns.sort(key=lambda p: p["frequency"], reverse=True)
        return patterns[:20]

    def _extract_temporal_patterns(self, memories: List["MemoryNode"]) -> List[Dict[str, Any]]:
        """Extract temporal patterns (hour of day, day of week)."""
        hour_counts = defaultdict(int)
        dow_counts = defaultdict(int)

        for memory in memories:
            hour_counts[memory.created_at.hour] += 1
            dow_counts[memory.created_at.weekday()] += 1

        patterns = []

        # Hour patterns
        peak_hour = max(hour_counts.items(), key=lambda x: x[1])
        if peak_hour[1] >= self.min_frequency:
            patterns.append({
                "pattern_type": "temporal_hour",
                "pattern_value": f"hour_{peak_hour[0]}",
                "frequency": peak_hour[1],
                "description": f"Most memories created around {peak_hour[0]}:00",
            })

        # Day of week patterns
        peak_dow = max(dow_counts.items(), key=lambda x: x[1])
        if peak_dow[1] >= self.min_frequency:
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            patterns.append({
                "pattern_type": "temporal_dow",
                "pattern_value": f"dow_{peak_dow[0]}",
                "frequency": peak_dow[1],
                "description": f"Most memories on {days[peak_dow[0]]}",
            })

        return patterns

    def _extract_metadata_patterns(self, memories: List["MemoryNode"]) -> List[Dict[str, Any]]:
        """Extract patterns from metadata fields."""
        category_counts = defaultdict(int)
        tag_counts: Dict[str, int] = defaultdict(int)

        for memory in memories:
            if cat := memory.metadata.get("category"):
                category_counts[cat] += 1

            tags = memory.metadata.get("tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    tag_counts[tag] += 1

        patterns = []

        # Category patterns
        for cat, count in category_counts.items():
            if count >= self.min_frequency:
                patterns.append({
                    "pattern_type": "category",
                    "pattern_value": cat,
                    "frequency": count,
                })

        # Tag patterns
        for tag, count in tag_counts.items():
            if count >= self.min_frequency:
                patterns.append({
                    "pattern_type": "tag",
                    "pattern_value": tag,
                    "frequency": count,
                })

        return patterns[:10]

    async def _extract_semantic_patterns(
        self,
        memories: List["MemoryNode"],
        llm_client: Any,
    ) -> List[Dict[str, Any]]:
        """Use LLM to extract semantic patterns."""
        if len(memories) < 5:
            return []

        # Sample memories for LLM analysis
        sample = memories[:20]
        contents = [f"{i+1}. {m.content[:150]}" for i, m in enumerate(sample)]

        prompt = f"""Analyze these memory fragments and identify 3-5 recurring themes or patterns.
Output ONLY a valid JSON array with this format:
[{{"theme": "pattern name", "description": "brief description", "evidence_count": N}}]

Memories:
{chr(10).join(contents)}
"""

        try:
            response = await llm_client.generate(prompt, max_tokens=300)

            # Parse JSON response
            if "[" in response:
                start = response.index("[")
                end = response.rindex("]") + 1
                from ..utils import json_compat as json
                parsed = json.loads(response[start:end])

                return [
                    {
                        "pattern_type": "semantic_theme",
                        "pattern_value": p.get("theme", "unknown"),
                        "description": p.get("description", ""),
                        "frequency": p.get("evidence_count", self.min_frequency),
                    }
                    for p in parsed
                ]
        except Exception as e:
            logger.debug(f"[PatternExtractor] LLM pattern extraction failed: {e}")

        return []

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for pattern extraction."""
        # Lowercase and extract words
        words = re.findall(r'\b[a-zA-ZåäöÅÄÖ]{3,}\b', text.lower())
        return words


# =============================================================================
# Stage 3: Recursive Synthesizer Wrapper
# =============================================================================

class DreamSynthesizer:
    """
    Wrapper around RecursiveSynthesizer for dream-time consolidation.

    Uses the existing RecursiveSynthesizer to:
    1. Synthesize patterns into higher-level abstractions
    2. Create semantic bridges between related concepts
    3. Generate "dream" memories from synthesis results

    The synthesizer operates on pattern clusters rather than user queries.
    """

    def __init__(
        self,
        engine: "HAIMEngine",
        max_depth: int = 3,
        max_patterns: int = 10,
    ):
        self.engine = engine
        self.max_depth = max_depth
        self.max_patterns = max_patterns

        # Lazy load synthesizer
        self._synthesizer: Optional["RecursiveSynthesizer"] = None

    def _get_synthesizer(self) -> "RecursiveSynthesizer":
        """Get or create the synthesizer instance."""
        if self._synthesizer is None:
            from ..core.recursive_synthesizer import (
                RecursiveSynthesizer,
                SynthesizerConfig,
            )

            # Check if engine has LLM capability
            llm_call = getattr(self.engine, "subconscious_ai", None)
            if llm_call:
                llm_call = llm_call._model_client.generate

            self._synthesizer = RecursiveSynthesizer(
                engine=self.engine,
                config=SynthesizerConfig(
                    max_depth=self.max_depth,
                    max_sub_queries=5,
                    final_top_k=10,
                ),
                llm_call=llm_call,
            )
        return self._synthesizer

    async def synthesize_patterns(
        self,
        patterns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Synthesize patterns into higher-level abstractions.

        Args:
            patterns: List of extracted patterns.

        Returns:
            List of synthesis results.
        """
        if not patterns:
            return []

        synthesizer = self._get_synthesizer()
        results = []

        # Process top patterns
        top_patterns = patterns[:self.max_patterns]

        for pattern in top_patterns:
            # Build a query from the pattern
            query = self._pattern_to_query(pattern)

            try:
                synthesis = await synthesizer.synthesize(query)

                # Store synthesis as a "dream memory" if significant
                if synthesis.results and synthesis.synthesis:
                    dream_memory_id = await self._store_dream_memory(
                        pattern, synthesis
                    )

                    results.append({
                        "pattern": pattern,
                        "query": query,
                        "results_count": len(synthesis.results),
                        "synthesis": synthesis.synthesis[:500],  # Truncate
                        "dream_memory_id": dream_memory_id,
                    })

            except Exception as e:
                logger.warning(f"[DreamSynthesizer] Failed to synthesize pattern: {e}")

        return results

    def _pattern_to_query(self, pattern: Dict[str, Any]) -> str:
        """Convert a pattern into a search query."""
        ptype = pattern.get("pattern_type", "")
        value = pattern.get("pattern_value", "")

        if ptype == "keyword":
            return f"Memories related to {value}"
        elif ptype == "semantic_theme":
            desc = pattern.get("description", "")
            return f"{value}: {desc}"
        elif ptype == "category":
            return f"Memories in category: {value}"
        elif ptype == "temporal_hour":
            return f"Memories from around {value}"
        else:
            return str(value)

    async def _store_dream_memory(
        self,
        pattern: Dict[str, Any],
        synthesis: Any,
    ) -> Optional[str]:
        """Store synthesis result as a dream memory."""
        content = f"[DREAM SYNTHESIS] {pattern.get('pattern_value', 'unknown')}\n{synthesis.synthesis}"

        metadata = {
            "type": "dream_synthesis",
            "pattern_type": pattern.get("pattern_type"),
            "pattern_value": pattern.get("pattern_value"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "results_count": len(synthesis.results),
        }

        try:
            # Store using engine's async-friendly method
            mem_id = await asyncio.to_thread(
                self.engine.store,
                content,
                metadata=metadata,
            )
            return mem_id
        except Exception as e:
            logger.debug(f"[DreamSynthesizer] Failed to store dream memory: {e}")
            return None


# =============================================================================
# Stage 4: Contradiction Resolver
# =============================================================================

class ContradictionResolver:
    """
    Detects and resolves contradictions using the existing ContradictionDetector.

    During dream sessions, we:
    1. Scan memories for contradictions
    2. Attempt auto-resolution for simple cases
    3. Flag complex cases for manual review
    4. Track resolution status across sessions
    """

    def __init__(
        self,
        engine: "HAIMEngine",
        similarity_threshold: float = 0.80,
        auto_resolve: bool = False,
    ):
        self.engine = engine
        self.similarity_threshold = similarity_threshold
        self.auto_resolve = auto_resolve

        # Lazy load detector
        self._detector: Optional["ContradictionDetector"] = None

    def _get_detector(self) -> "ContradictionDetector":
        """Get or create the contradiction detector."""
        if self._detector is None:
            from ..core.contradiction import get_contradiction_detector
            self._detector = get_contradiction_detector(self.engine)
        return self._detector

    async def scan_and_resolve(
        self,
        memories: List["MemoryNode"],
    ) -> Dict[str, Any]:
        """
        Scan memories for contradictions and attempt resolution.

        Args:
            memories: List of memory nodes to scan.

        Returns:
            Dict with scan results and resolution info.
        """
        if not memories:
            return {
                "contradictions_found": 0,
                "contradictions_resolved": 0,
                "resolved_ids": [],
                "unresolved_ids": [],
            }

        detector = self._get_detector()

        # Run background scan
        found = await detector.scan(memories)

        contradictions = []
        resolved = []

        for record in found:
            if not record.resolved:
                contradictions.append(record)

                # Attempt auto-resolution if enabled
                if self.auto_resolve and self._is_simple_contradiction(record):
                    resolution = await self._auto_resolve(record)
                    if resolution:
                        resolved.append(resolution)

        logger.info(
            f"[ContradictionResolver] Found {len(contradictions)} contradictions, "
            f"resolved {len(resolved)}"
        )

        return {
            "contradictions_found": len(contradictions),
            "contradictions_resolved": len(resolved),
            "resolved_ids": [r.group_id for r in resolved],
            "unresolved_ids": [r.group_id for r in contradictions if r not in resolved],
        }

    def _is_simple_contradiction(self, record: Any) -> bool:
        """Check if a contradiction is simple enough to auto-resolve."""
        # Simple contradictions have high similarity and clear resolution
        return (
            record.similarity_score > 0.95 and
            not record.llm_confirmed  # Only auto-resolve non-LLM confirmed
        )

    async def _auto_resolve(self, record: Any) -> Optional[Any]:
        """Attempt automatic resolution of a contradiction."""
        try:
            # Mark as resolved with a note
            self._detector.registry.resolve(
                record.group_id,
                note="Auto-resolved by DreamSession - high similarity"
            )
            return record
        except Exception as e:
            logger.debug(f"[ContradictionResolver] Auto-resolve failed: {e}")
            return None


# =============================================================================
# Stage 5: Semantic Promoter
# =============================================================================

class SemanticPromoter:
    """
    Promotes important memories from HOT to WARM tier.

    Promotion criteria:
    - High LTP strength (consolidated memory)
    - High access count (frequently retrieved)
    - Part of important episodic clusters
    - Tagged as important by user or system

    Ensures important memories are properly consolidated and archived.
    """

    def __init__(
        self,
        engine: "HAIMEngine",
        ltp_threshold: float = 0.7,
        access_threshold: int = 5,
        auto_promote: bool = True,
    ):
        self.engine = engine
        self.ltp_threshold = ltp_threshold
        self.access_threshold = access_threshold
        self.auto_promote = auto_promote

    async def promote(
        self,
        memories: List["MemoryNode"],
        clusters: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Identify and promote memories to WARM tier.

        Args:
            memories: List of memory nodes to evaluate.
            clusters: Optional episodic clusters for context-aware promotion.

        Returns:
            Dict with promotion results.
        """
        promoted = []
        candidates = []

        # Find candidates
        for memory in memories:
            if memory.tier != "hot":
                continue

            # Check promotion criteria
            if self._should_promote(memory):
                candidates.append(memory)

        # Promote if auto-promote is enabled
        if self.auto_promote and candidates:
            for memory in candidates:
                try:
                    await self._promote_to_warm(memory)
                    promoted.append(memory.id)
                except Exception as e:
                    logger.debug(f"[SemanticPromoter] Failed to promote {memory.id}: {e}")

        logger.info(
            f"[SemanticPromoter] Promoted {len(promoted)} memories "
            f"from {len(candidates)} candidates"
        )

        return {
            "candidates_count": len(candidates),
            "promoted_count": len(promoted),
            "promoted_ids": promoted,
        }

    def _should_promote(self, memory: "MemoryNode") -> bool:
        """Check if a memory should be promoted."""
        # High LTP strength
        if memory.ltp_strength >= self.ltp_threshold:
            return True

        # High access count
        if memory.access_count >= self.access_threshold:
            return True

        # Manually tagged as important
        if memory.metadata.get("important"):
            return True

        return False

    async def _promote_to_warm(self, memory: "MemoryNode") -> None:
        """Promote a memory to WARM tier."""
        # Update tier
        memory.tier = "warm"

        # Use tier manager's promotion logic
        await asyncio.to_thread(
            self.engine.tier_manager.promote_to_warm,
            memory.id,
        )


# =============================================================================
# Stage 6: Dream Report Generator
# =============================================================================

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
        config: DreamPipelineConfig,
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
        import uuid
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
