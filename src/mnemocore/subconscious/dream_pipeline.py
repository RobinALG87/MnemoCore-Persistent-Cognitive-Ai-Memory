"""
Dream Pipeline – Phase 6.0: Offline Consolidation Pipeline
==========================================================
Multi-stage pipeline for offline memory consolidation during idle periods.

This module provides backward-compatible imports from the new dream package.
All implementation has been moved to the subconscious/dream/ package.

Pipeline Stages:
    1. [Episodic Cluster]      – Group memories by temporal proximity
    2. [Pattern Extractor]     – Identify recurring patterns
    3. [Recursive Synthesizer] – Build higher-level abstractions
    4. [Contradiction Resolver] – Detect and resolve contradictions
    5. [Semantic Promoter]     – Promote important memories
    6. [Dream Report Generator] – Generate summary report

Usage:
    pipeline = DreamPipeline(engine, config)
    result = await pipeline.run()

For direct access to stage components:
    from mnemocore.subconscious.dream import (
        EpisodicClusterer,
        PatternExtractor,
        DreamSynthesizer,
        ContradictionResolver,
        SemanticPromoter,
        DreamReportGenerator,
    )
"""

# Re-export all components from the dream package for backward compatibility
from .dream import (
    # Configuration and Result
    DreamPipelineConfig,
    DreamPipelineResult,
    # Stage components
    EpisodicClusterer,
    EpisodicCluster,
    PatternExtractor,
    DreamSynthesizer,
    SynthesisResult,
    ContradictionResolver,
    ContradictionScanResult,
    SemanticPromoter,
    PromotionResult,
    DreamReportGenerator,
    DreamReport,
    # Main pipeline
    DreamPipeline,
)


__all__ = [
    # Configuration and Result
    "DreamPipelineConfig",
    "DreamPipelineResult",
    # Stage components
    "EpisodicClusterer",
    "EpisodicCluster",
    "PatternExtractor",
    "DreamSynthesizer",
    "SynthesisResult",
    "ContradictionResolver",
    "ContradictionScanResult",
    "SemanticPromoter",
    "PromotionResult",
    "DreamReportGenerator",
    "DreamReport",
    # Main pipeline
    "DreamPipeline",
]
