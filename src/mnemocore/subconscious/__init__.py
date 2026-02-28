"""Subconscious module for background processing and dreaming."""

__all__ = [
    "SubconsciousDaemon",
    "DreamScheduler",
    "DreamSession",
    "DreamPipeline",
    "IdleDetector",
    "SelfImprovementWorker",
    "create_self_improvement_worker",
]

from .daemon import SubconsciousDaemon
from .dream_scheduler import (
    DreamScheduler,
    DreamSession,
    IdleDetector,
    DreamSchedulerConfig,
    DreamSessionConfig,
    DreamTriggerReason,
)
from .dream_pipeline import (
    DreamPipeline,
    DreamPipelineConfig,
)
from .self_improvement_worker import (
    SelfImprovementWorker,
    create_self_improvement_worker,
)
