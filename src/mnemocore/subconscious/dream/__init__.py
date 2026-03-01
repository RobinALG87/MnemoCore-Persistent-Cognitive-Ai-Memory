"""
Dream Package – Phase 6.0: Offline Consolidation Pipeline
=========================================================
Multi-stage pipeline for offline memory consolidation during idle periods.

This package provides a modular dream pipeline with per-stage components:

Stage Components:
    - EpisodicClusterer: Groups memories by temporal proximity
    - PatternExtractor: Identifies recurring patterns
    - DreamSynthesizer: Builds higher-level abstractions
    - ContradictionResolver: Detects and resolves contradictions
    - SemanticPromoter: Promotes important memories
    - DreamReportGenerator: Generates summary report

Orchestrator:
    - DreamPipeline: Main pipeline that coordinates all stages

Configuration:
    - DreamPipelineConfig: Pipeline configuration dataclass

Result:
    - DreamPipelineResult: Result dataclass from running the pipeline

Usage:
    from mnemocore.subconscious.dream import DreamPipeline, DreamPipelineConfig

    config = DreamPipelineConfig(
        enable_episodic_clustering=True,
        enable_pattern_extraction=True,
        ...
    )
    pipeline = DreamPipeline(engine, config)
    result = await pipeline.run()
"""

from .clusterer import EpisodicClusterer, EpisodicCluster
from .patterns import PatternExtractor
from .synthesizer import DreamSynthesizer, SynthesisResult
from .contradictions import ContradictionResolver, ContradictionScanResult
from .promoter import SemanticPromoter, PromotionResult
from .report import DreamReportGenerator, DreamReport
from .pipeline import (
    DreamPipeline,
    DreamPipelineConfig,
    DreamPipelineResult,
)


__all__ = [
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
    "DreamPipelineConfig",
    "DreamPipelineResult",
]
