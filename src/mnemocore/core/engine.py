"""
Holographic Active Inference Memory Engine (HAIM) - Phase 6 Refactored

This module provides HAIMEngine as a facade that composes functionality from:
- engine_core.py: Core memory operations (store, query, delete)
- engine_lifecycle.py: Initialization, shutdown, health checks, garbage collection
- engine_coordinator.py: Orchestration between sub-systems and event routing

The facade pattern maintains backward compatibility while improving code organization
and keeping each module under 800 lines.
"""

from typing import List, Tuple, Dict, Optional, Any, TYPE_CHECKING, Deque
from collections import deque
import asyncio
import uuid
import numpy as np
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

from .config import get_config, HAIMConfig
from .binary_hdv import BinaryHDV, TextEncoder
from .node import MemoryNode
from .synapse import SynapticConnection
from .tier_manager import TierManager

# Import the mixin modules
from .engine_core import EngineCoreOperations
from .engine_lifecycle import EngineLifecycleManager
from .engine_coordinator import EngineCoordinator

# Phase 4.0 imports
from .attention import XORAttentionMasker, AttentionConfig, XORIsolationMask, IsolationConfig
from .gap_detector import GapDetector, GapDetectorConfig
from .gap_filler import GapFiller
from .synapse_index import SynapseIndex

# Phase 5 AGI Stores
from .working_memory import WorkingMemoryService
from .episodic_store import EpisodicStoreService
from .semantic_store import SemanticStoreService
from .procedural_store import ProceduralStoreService
from .meta_memory import MetaMemoryService

# Phase 4.0 workers
from .semantic_consolidation import SemanticConsolidationWorker
from .immunology import ImmunologyLoop


class HAIMEngine(EngineCoreOperations, EngineLifecycleManager, EngineCoordinator):
    """
    Holographic Active Inference Memory Engine (Phase 6 Refactored)

    This class is a facade that composes functionality from three mixin modules:
    - EngineCoreOperations: store(), query(), delete(), and related operations
    - EngineLifecycleManager: initialize(), close(), health_check(), cleanup
    - EngineCoordinator: gap filling, recursive synthesis, OR operations

    Uses Binary HDV and Tiered Storage for efficient cognitive memory.

    All public interfaces from the original engine.py are preserved for
    backward compatibility.
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        persist_path: Optional[str] = None,
        config: Optional[HAIMConfig] = None,
        tier_manager: Optional[TierManager] = None,
        working_memory: Optional[WorkingMemoryService] = None,
        episodic_store: Optional[EpisodicStoreService] = None,
        semantic_store: Optional[SemanticStoreService] = None,
        procedural_store: Optional[ProceduralStoreService] = None,
        meta_memory: Optional[MetaMemoryService] = None,
    ):
        """
        Initialize HAIMEngine with optional dependency injection.

        Args:
            dimension: Vector dimensionality (default 16384).
            persist_path: Path to memory persistence file.
            config: Configuration object. If None, uses global get_config().
            tier_manager: TierManager instance. If None, creates a new one.
            working_memory: Optional Phase 5 WM service.
            episodic_store: Optional Phase 5 EM service.
            semantic_store: Optional Phase 5 Semantic service.
            procedural_store: Optional Phase 5 Procedural service.
            meta_memory: Optional Phase 5 Meta-Memory service.
        """
        base_config = config or get_config()
        if dimension is not None and dimension != base_config.dimensionality:
            self.config = replace(base_config, dimensionality=dimension)
        else:
            self.config = base_config
        self.dimension = self.config.dimensionality

        # Persistence paths
        self.persist_path = persist_path or self.config.paths.memory_file
        self.synapse_path = self.config.paths.synapses_file

        # Core Components
        self.tier_manager = tier_manager or TierManager(config=self.config)
        self.binary_encoder = TextEncoder(self.dimension)

        # Phase 5 Components
        self.working_memory = working_memory
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store
        self.procedural_store = procedural_store
        self.meta_memory = meta_memory

        # Initialize core state via EngineCoreOperations
        self._initialize_core_state(
            config=self.config,
            tier_manager=self.tier_manager,
            dimension=self.dimension,
            persist_path=self.persist_path,
            synapse_path=self.synapse_path,
        )

        # ── Phase 4.0: hardened O(1) synapse adjacency index ──────────
        self._synapse_index = SynapseIndex()

        # ── Phase 4.0: XOR attention masker ───────────────────────────
        self.attention_masker = XORAttentionMasker(AttentionConfig())

        # ── Phase 4.1: XOR project isolation masker ───────────────────
        isolation_enabled = getattr(self.config, 'attention_masking', None)
        isolation_enabled = isolation_enabled.enabled if isolation_enabled else True
        self.isolation_masker = XORIsolationMask(IsolationConfig(
            enabled=isolation_enabled,
            dimension=self.dimension,
        ))

        # ── Phase 4.0: gap detector & filler ──
        self.gap_detector = GapDetector(GapDetectorConfig())
        self._gap_filler: Optional[GapFiller] = None

        # ── Phase 4.0: semantic consolidation worker ───────────────────
        self._semantic_worker: Optional[SemanticConsolidationWorker] = None

        # ── Phase 4.0: immunology loop ─────────────────────────────────
        self._immunology: Optional[ImmunologyLoop] = None

        # ── Phase 4.4: subconscious AI worker (BETA) ───────────────────
        self._subconscious_ai: Optional[Any] = None  # SubconsciousAIWorker

        # ── Phase 4.5: recursive synthesizer ───────────────────────────
        self._recursive_synthesizer: Optional[Any] = None  # RecursiveSynthesizer

        # ── Phase 12.2: Contextual Topic Tracker ───────────────────────
        from .topic_tracker import TopicTracker
        self.topic_tracker = TopicTracker(self.config.context, self.dimension)

        # ── Phase 12.3: Preference Learning ────────────────────────────
        from .preference_store import PreferenceStore
        self.preference_store = PreferenceStore(self.config.preference, self.dimension)

        # ── Phase 13.2: Anticipatory Memory ────────────────────────────
        from .anticipatory import AnticipatoryEngine
        self.anticipatory_engine = AnticipatoryEngine(
            self.config.anticipatory,
            self._synapse_index,
            self.tier_manager,
            self.topic_tracker
        )

        data_dir = Path(self.config.paths.data_dir)

        # ── Phase 6.0: Association Network ─────────────────────────────
        # Initialize the graph-based association tracking system
        from ..cognitive.associations import (
            AssociationsNetwork,
            AssociationConfig,
            AssociationRecallIntegrator,
        )
        associations_config = AssociationConfig(
            persist_path=str(data_dir / "associations.json"),
            auto_save=getattr(self.config, 'associations_auto_save', True),
            decay_enabled=getattr(self.config, 'associations_decay', True),
        )
        self.associations = AssociationsNetwork(
            config=associations_config,
            storage_dir=str(data_dir),
        )
        self.associations_integrator = AssociationRecallIntegrator(
            network=self.associations,
            auto_strengthen=True,
            strengthen_threshold=2,
        )

        # Conceptual Layer (VSA Soul)
        self.soul: Any  # ConceptualMemory
        from .holographic import ConceptualMemory
        self.soul = ConceptualMemory(dimension=self.dimension, storage_dir=str(data_dir))

        # ── Phase 3.x: synapse raw dicts (kept for backward compat) ──
        self.synapses: Dict[Tuple[str, str], SynapticConnection] = {}
        self.synapse_adjacency: Dict[str, List[SynapticConnection]] = {}

        # Pass Phase 4+ components to core operations
        self._set_phase4_components(
            synapse_index=self._synapse_index,
            attention_masker=self.attention_masker,
            isolation_masker=self.isolation_masker,
            gap_detector=self.gap_detector,
            soul=self.soul,
            topic_tracker=self.topic_tracker,
            preference_store=self.preference_store,
            anticipatory_engine=self.anticipatory_engine,
        )

    # ==========================================================================
    # Backward Compatibility Properties
    # ==========================================================================

    @property
    def subconscious(self) -> Optional[Any]:
        """Access to the SubconsciousAI worker, if enabled."""
        return getattr(self, '_subconscious_ai', None)

    @subconscious.setter
    def subconscious(self, value: Optional[Any]):
        """Set the SubconsciousAI worker."""
        self._subconscious_ai = value

    # ==========================================================================
    # Phase 6.0: Reconstructive Memory Integration
    # ==========================================================================

    async def reconstructive_recall(
        self,
        query: str,
        top_k: int = 10,
        enable_synthesis: bool = True,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform reconstructive recall - synthesize memories from fragments.

        This method integrates with the cognitive.ReconstructiveRecall module
        to provide intelligent memory reconstruction when direct matches are
        insufficient.

        Args:
            query: The query text to recall memories for.
            top_k: Number of candidate fragments to retrieve.
            enable_synthesis: Whether to synthesize from fragments.
            project_id: Optional project ID for isolation masking.

        Returns:
            Dictionary containing:
                - content: Primary content (reconstructed or direct match)
                - is_reconstructed: Whether the result was synthesized
                - confidence: Confidence score
                - fragments: List of fragments used
                - direct_matches: List of (node_id, similarity) tuples
        """
        from ..cognitive.memory_reconstructor import ReconstructiveRecall, ReconstructionConfig

        # Lazy load the reconstructor
        if not hasattr(self, '_reconstructive_recall'):
            config = ReconstructionConfig(
                enable_gap_detection=True,
                enable_persistent_storage=False,  # Don't auto-store by default
            )
            self._reconstructive_recall = ReconstructiveRecall(
                engine=self,
                config=config,
                gap_detector=self.gap_detector,
            )

        # Perform reconstructive recall
        result = await self._reconstructive_recall.recall(
            query=query,
            top_k=top_k,
            enable_synthesis=enable_synthesis,
            project_id=project_id,
        )

        # Return structured result
        return {
            "content": result.get_primary_content(),
            "is_reconstructed": result.is_reconstructed,
            "confidence": result.confidence_breakdown.get("overall_confidence", 0.0),
            "fragments": [f.to_dict() for f in result.fragments],
            "direct_matches": result.direct_matches,
            "reconstructed": result.reconstructed.to_dict() if result.reconstructed else None,
            "confidence_breakdown": result.confidence_breakdown,
            "gap_detected": len(result.gap_records) > 0,
        }

    async def enable_reconstructive_memory(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Enable and configure the reconstructive memory module.

        Args:
            config: Optional configuration overrides as dictionary.
        """
        from ..cognitive.memory_reconstructor import ReconstructiveRecall, ReconstructionConfig

        cfg = ReconstructionConfig()
        if config:
            for key, value in config.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)

        self._reconstructive_recall = ReconstructiveRecall(
            engine=self,
            config=cfg,
            gap_detector=self.gap_detector,
        )

        logger.info("Phase 6.0 Reconstructive Memory enabled.")

    # ==========================================================================
    # Phase 5.1: Cognitive Module Access (lazy-loaded)
    # ==========================================================================

    def get_context_prioritizer(self):
        """
        Get or create the ContextWindowPrioritizer for LLM context optimization.

        Returns a prioritizer that can rank and filter memories to fit within
        a model's context window, preserving the most relevant information.
        """
        if not hasattr(self, '_context_prioritizer'):
            from ..cognitive.context_optimizer import create_prioritizer
            self._context_prioritizer = create_prioritizer()
        return self._context_prioritizer

    def get_future_simulator(self):
        """
        Get or create the EpisodeFutureSimulator for scenario generation.

        Uses episodic memories to simulate plausible future scenarios,
        supporting anticipatory reasoning and planning.
        """
        if not hasattr(self, '_future_simulator'):
            from ..cognitive.future_thinking import create_future_thinking_pipeline
            self._future_simulator = create_future_thinking_pipeline(
                config=self.config.eft,
                tier_manager=self.tier_manager,
                synapse_index=self._synapse_index,
                attention_masker=self.attention_masker,
            )
        return self._future_simulator

    def get_forgetting_analytics(self):
        """
        Get or create the ForgettingAnalytics module for SM-2 spaced
        repetition analytics and memory health dashboards.
        """
        if not hasattr(self, '_forgetting_analytics'):
            from ..cognitive.forgetting_analytics import create_forgetting_analytics
            self._forgetting_analytics = create_forgetting_analytics()
        return self._forgetting_analytics


# All public methods are inherited from the mixin classes:
#
# From EngineCoreOperations:
#   - store()
#   - query()
#   - delete_memory()
#   - get_memory()
#   - bind_memories()
#   - get_context_nodes()
#   - encode_content()
#   - define_concept()
#   - reason_by_analogy()
#   - cross_domain_inference()
#   - inspect_concept()
#   - generate_subtle_thoughts()
#
# From EngineLifecycleManager:
#   - initialize()
#   - close()
#   - health_check()
#   - is_ready()
#   - cleanup_decay()
#   - garbage_collect_memories()
#   - get_stats()
#   - get_component_status()
#   - log_decision()
#
# From EngineCoordinator:
#   - enable_gap_filling()
#   - enable_recursive_synthesis()
#   - record_retrieval_feedback()
#   - register_negative_feedback()
#   - get_node_boost()
#   - orchestrate_orch_or()
#   - associative_query()
#   - temporal_query()
#   - find_related_by_content()
#   - batch_store()
#   - batch_get_memories()
#   - export_memories()
#   - get_synaptic_path()
#   - get_memory_connections()
#   - analyze_memory_clusters()


# Export for backward compatibility
__all__ = ["HAIMEngine"]
