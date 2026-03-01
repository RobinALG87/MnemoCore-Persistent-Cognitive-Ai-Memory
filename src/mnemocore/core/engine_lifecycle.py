"""
HAIMEngine Lifecycle Management Module (Phase 6 Refactor)

Contains initialization, shutdown, health checks, and garbage collection.
Handles the setup and teardown of engine components and background workers.

Separated from engine_core.py and engine_coordinator.py to maintain
clear separation of concerns and keep files under 800 lines.
"""

from typing import Dict, Any, Optional, List, TYPE_CHECKING
import os
import re
import hashlib
from datetime import datetime, timezone
from loguru import logger

# Version management - use importlib.metadata as single source of truth
try:
    from importlib.metadata import version as get_version
    _ENGINE_VERSION = get_version("mnemocore")
except Exception:
    _ENGINE_VERSION = "2.0.0"  # Fallback for development environments

import numpy as np
import asyncio

# Core imports
from .config import HAIMConfig, SubconsciousAIConfig
from .binary_hdv import BinaryHDV, TextEncoder
from .node import MemoryNode
from .synapse import SynapticConnection
from .tier_manager import TierManager

# Phase 4.0 imports
from .attention import XORAttentionMasker, AttentionConfig, XORIsolationMask, IsolationConfig
from .bayesian_ltp import get_bayesian_updater
from .semantic_consolidation import SemanticConsolidationWorker, SemanticConsolidationConfig
from .immunology import ImmunologyLoop, ImmunologyConfig
from .gap_detector import GapDetector, GapDetectorConfig
from .gap_filler import GapFiller, GapFillerConfig
from .synapse_index import SynapseIndex

# Phase 4.5 imports
from .recursive_synthesizer import RecursiveSynthesizer, SynthesizerConfig

# Observability imports
from .metrics import update_memory_count, record_error

if TYPE_CHECKING:
    from .container import Container
    from .qdrant_store import QdrantStore
    from .holographic import ConceptualMemory
    from .topic_tracker import TopicTracker
    from .preference_store import PreferenceStore
    from .anticipatory import AnticipatoryEngine
    from .working_memory import WorkingMemoryService
    from .episodic_store import EpisodicStoreService
    from .semantic_store import SemanticStoreService


class EngineLifecycleManager:
    """
    Lifecycle management for HAIMEngine.

    This mixin class provides initialization, shutdown, health checks,
    and garbage collection functionality.

    Responsibilities:
    - Async initialization of all engine components
    - Graceful shutdown and cleanup
    - Health check monitoring
    - Garbage collection (synapse decay cleanup)
    - Legacy data loading and migration
    - Statistics aggregation
    """

    # ==========================================================================
    # Initialization (Task 4.4: use shared utils)
    # ==========================================================================

    # Token vector cache for instance-level caching
    _token_vector_cache: Dict[str, np.ndarray] = {}

    def _get_token_vector(self, token: str, dimension: int) -> np.ndarray:
        """Cached generation of deterministic token vectors (legacy compatibility)."""
        from ._utils import get_token_vector
        cache_key = (token, dimension)
        if cache_key not in self._token_vector_cache:
            self._token_vector_cache[cache_key] = get_token_vector(token, dimension)
        return self._token_vector_cache[cache_key]

    async def initialize(self):
        """
        Async initialization of all engine components.

        This method must be called before using any store/query operations.
        It initializes:
        - TierManager (HOT/WARM/COLD storage)
        - Legacy memory loading
        - Synapse index loading
        - Background workers (semantic consolidation, immunology, subconscious AI)
        """
        if self._initialized:
            return

        try:
            await self.tier_manager.initialize()
            await self._load_legacy_if_needed()
            await self._load_synapses()
            self._initialized = True

            # Phase 4.0: start background workers
            self._semantic_worker = SemanticConsolidationWorker(self)
            await self._semantic_worker.start()

            self._immunology = ImmunologyLoop(self)
            await self._immunology.start()

            # Phase 4.4: start subconscious AI worker (Phase 5.4: lazy loaded to save RAM)
            if self.config.subconscious_ai.enabled:
                # Lazy load heavy SubconsciousAI modules only when enabled
                from .subconscious_ai import SubconsciousAIWorker
                self.subconscious = SubconsciousAIWorker(self, self.config.subconscious_ai)
                await self.subconscious.start()
                logger.info("Phase 4.4 SubconsciousAI worker started (BETA).")
            else:
                self.subconscious = None

            logger.info("Phase 4.0 background workers started (consolidation + immunology).")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self._initialized = False
            # Attempt to stop any workers that may have started
            await self._cleanup_on_init_failure()
            raise

    async def _cleanup_on_init_failure(self):
        """Stop any workers that started before initialization failed."""
        if hasattr(self, '_semantic_worker') and self._semantic_worker:
            try:
                await self._semantic_worker.stop()
            except Exception as stop_err:
                logger.warning(f"Failed to stop semantic_worker during cleanup: {stop_err}")

        if hasattr(self, '_immunology') and self._immunology:
            try:
                await self._immunology.stop()
            except Exception as stop_err:
                logger.warning(f"Failed to stop immunology during cleanup: {stop_err}")

        if hasattr(self, 'subconscious') and self.subconscious:
            try:
                await self.subconscious.stop()
            except Exception as stop_err:
                logger.warning(f"Failed to stop subconscious during cleanup: {stop_err}")

    async def _load_legacy_if_needed(self):
        """
        Load from memory.jsonl into TierManager, converting to BinaryHDV.

        Handles migration from legacy memory format to the new tiered storage.
        """
        from mnemocore.utils import json_compat as json
        from ._utils import run_in_thread  # Task 4.4: use shared utility

        if not os.path.exists(self.persist_path):
            return

        logger.info(f"Loading legacy memory from {self.persist_path}")

        def _load():
            try:
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    return f.readlines()
            except (IOError, OSError) as e:
                logger.warning(f"Failed to read legacy memory file {self.persist_path}: {e}")
                return []
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in legacy memory file {self.persist_path}: {e}")
                return []

        lines = await run_in_thread(_load)

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                content = rec.get('content', '')
                if not content:
                    continue

                node_id = rec.get('id')

                # Always convert to BinaryHDV
                hdv = self.binary_encoder.encode(content)

                node = MemoryNode(
                    id=node_id,
                    hdv=hdv,
                    content=content,
                    metadata=rec.get('metadata') or {}
                )

                # Restore timestamps if available
                if 'created_at' in rec:
                    node.created_at = datetime.fromisoformat(rec['created_at'])

                # Phase 4.3: Restore episodic chain link
                if 'previous_id' in rec:
                    node.previous_id = rec['previous_id']

                # Add to TierManager
                await self.tier_manager.add_memory(node)

            except Exception as e:
                logger.warning(f"Failed to load record: {e}")

    async def _load_synapses(self):
        """
        Load synapses from disk.

        Phase 4.0: uses SynapseIndex.load_from_file() which restores Bayesian state.
        Task 4.4: use shared run_in_thread utility.
        """
        from ._utils import run_in_thread

        if not os.path.exists(self.synapse_path):
            return

        await self._synapse_index.load_from_file(self.synapse_path)

    # ==========================================================================
    # Shutdown
    # ==========================================================================

    async def close(self):
        """
        Perform graceful shutdown of engine components.

        Stops all background workers, saves state, and closes connections.
        Each worker is stopped in its own try/except to ensure all workers
        are attempted to stop even if one fails.
        """
        logger.info("Shutting down HAIMEngine...")

        # Phase 4.0: stop background workers (each wrapped to ensure all are attempted)
        if self._semantic_worker:
            try:
                await self._semantic_worker.stop()
            except Exception as e:
                logger.warning(f"Failed to stop semantic_worker: {e}")

        if self._immunology:
            try:
                await self._immunology.stop()
            except Exception as e:
                logger.warning(f"Failed to stop immunology: {e}")

        if self._gap_filler:
            try:
                await self._gap_filler.stop()
            except Exception as e:
                logger.warning(f"Failed to stop gap_filler: {e}")

        if self._subconscious_ai:
            try:
                await self._subconscious_ai.stop()
            except Exception as e:
                logger.warning(f"Failed to stop subconscious_ai: {e}")

        try:
            await self._save_synapses()
        except Exception as e:
            logger.warning(f"Failed to save synapses during shutdown: {e}")

        if self.tier_manager.use_qdrant and self.tier_manager.qdrant:
            try:
                await self.tier_manager.qdrant.close()
            except Exception as e:
                logger.warning(f"Failed to close qdrant connection: {e}")

    async def _save_synapses(self):
        """
        Save synapses to disk in JSONL format.

        Phase 4.0: uses SynapseIndex.save_to_file() which includes Bayesian state.
        A dedicated _write_lock serialises concurrent callers so the file is never
        written by two coroutines at the same time. Does NOT acquire synapse_lock.

        Task 4.4: use shared run_in_thread utility.
        """
        from ._utils import run_in_thread

        path_snapshot = self.synapse_path

        async with self._write_lock:
            await self._synapse_index.save_to_file(path_snapshot)

    # ==========================================================================
    # Health Checks
    # ==========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the engine and its components.

        Returns a dictionary with health status information including:
        - initialized: Whether the engine has been properly initialized
        - components: Health status of individual components
        - storage: Storage tier statistics
        - background_workers: Status of background workers
        """
        health = {
            "initialized": self._initialized,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if not self._initialized:
            health["status"] = "not_initialized"
            return health

        # Check TierManager
        try:
            tier_stats = await self.tier_manager.get_stats()
            health["tiers"] = tier_stats
        except Exception as e:
            health["tiers"] = {"error": str(e)}

        # Check background workers
        workers = {}
        if self._semantic_worker:
            workers["semantic_consolidation"] = {
                "running": self._semantic_worker.running,
                "stats": self._semantic_worker.stats if hasattr(self._semantic_worker, 'stats') else {},
            }
        if self._immunology:
            workers["immunology"] = {
                "running": self._immunology.running,
                "stats": self._immunology.stats if hasattr(self._immunology, 'stats') else {},
            }
        if self._gap_filler:
            workers["gap_filler"] = {
                "running": self._gap_filler.running,
            }
        if self._subconscious_ai:
            workers["subconscious_ai"] = {
                "running": self._subconscious_ai.running,
                "stats": self._subconscious_ai.stats if hasattr(self._subconscious_ai, 'stats') else {},
            }
        health["background_workers"] = workers

        # Check Qdrant connection if used
        if self.tier_manager.use_qdrant and self.tier_manager.qdrant:
            try:
                qdrant_health = await self.tier_manager.qdrant.health_check()
                health["qdrant"] = qdrant_health
            except Exception as e:
                health["qdrant"] = {"error": str(e)}

        # Phase 5.1: Cognitive services health
        cognitive = {}
        has_degraded = False

        if hasattr(self, 'working_memory') and self.working_memory:
            cognitive["working_memory"] = {"status": "active"}
        if hasattr(self, 'episodic_store') and self.episodic_store:
            try:
                cognitive["episodic_store"] = self.episodic_store.get_stats()
                cognitive["episodic_store"]["status"] = "active"
            except Exception as e:
                cognitive["episodic_store"] = {"status": "degraded", "error": str(e)}
                has_degraded = True
        if hasattr(self, 'semantic_store') and self.semantic_store:
            try:
                cognitive["semantic_store"] = self.semantic_store.get_stats()
                cognitive["semantic_store"]["status"] = "active"
            except Exception as e:
                cognitive["semantic_store"] = {"status": "degraded", "error": str(e)}
                has_degraded = True
        if hasattr(self, 'procedural_store') and self.procedural_store:
            try:
                cognitive["procedural_store"] = self.procedural_store.get_stats()
                cognitive["procedural_store"]["status"] = "active"
            except Exception as e:
                cognitive["procedural_store"] = {"status": "degraded", "error": str(e)}
                has_degraded = True
        if hasattr(self, 'meta_memory') and self.meta_memory:
            cognitive["meta_memory"] = {"status": "active"}
        health["cognitive_services"] = cognitive

        # Check for tier errors
        if "error" in health.get("tiers", {}):
            has_degraded = True

        # Check for qdrant errors
        if "error" in health.get("qdrant", {}):
            has_degraded = True

        # Overall status - degraded if any component fails or workers not running
        workers_healthy = all(w.get("running", True) for w in workers.values())
        if not workers_healthy:
            health["status"] = "degraded"
        elif has_degraded:
            health["status"] = "degraded"
        else:
            health["status"] = "healthy"

        return health

    async def is_ready(self) -> bool:
        """
        Quick check if the engine is ready to process requests.

        Returns True if initialized and all critical components are operational.
        """
        return self._initialized

    # ==========================================================================
    # Garbage Collection
    # ==========================================================================

    async def cleanup_decay(self, threshold: float = 0.1):
        """
        Remove synapses that have decayed below the threshold.

        Phase 4.0: O(E) via SynapseIndex.compact(), no lock required for the index itself.
        Also syncs any legacy dict entries into the index before compacting.

        Args:
            threshold: The minimum strength threshold for synapses to retain.
                      Synapses with strength below this value will be removed.
        """
        async with self.synapse_lock:
            # Retain legacy->index sync so tests that write to self.synapses directly
            # still get their entries registered.
            for syn in list(self.synapses.values()):
                if await self._synapse_index.get(syn.neuron_a_id, syn.neuron_b_id) is None:
                    await self._synapse_index.register(syn)

            removed = await self._synapse_index.compact(threshold)

        if removed:
            logger.info(f"cleanup_decay: pruned {removed} synapses below {threshold}")
            await self._save_synapses()

    async def garbage_collect_memories(self, max_age_days: int = 365, min_access_count: int = 1):
        """
        Garbage collect old, rarely accessed memories from COLD storage.

        This is a more aggressive cleanup than decay-based pruning.
        Use with caution as deleted memories cannot be recovered.

        Args:
            max_age_days: Maximum age in days for memories to keep.
            min_access_count: Minimum access count for memories to keep.

        Returns:
            Number of memories removed.
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        removed = 0

        # Get all memories from COLD tier
        cold_memories = await self.tier_manager.get_cold_memories()

        for mem in cold_memories:
            if mem.created_at < cutoff and getattr(mem, 'access_count', 0) < min_access_count:
                # Check if memory has important tags
                tags = mem.metadata.get('tags', []) if mem.metadata else []
                if 'important' in tags or 'persistent' in tags:
                    continue

                # Delete the memory
                if await self.delete_memory(mem.id):
                    removed += 1

        logger.info(f"Garbage collected {removed} memories older than {max_age_days} days")
        return removed

    # ==========================================================================
    # Statistics
    # ==========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """
        Aggregate statistics from engine components.

        Returns a comprehensive dictionary with statistics including:
        - Engine version and configuration
        - Tier statistics (HOT/WARM/COLD counts)
        - Concept and symbol counts
        - Synapse statistics
        - Background worker statistics
        - Subconscious queue backlog
        """
        tier_stats = await self.tier_manager.get_stats()

        async with self.synapse_lock:
            syn_count = await self._synapse_index.__len__()

        stats = {
            "engine_version": _ENGINE_VERSION,
            "dimension": self.dimension,
            "encoding": "binary_hdv",
            "tiers": tier_stats,
            "concepts_count": len(self.soul.concepts),
            "symbols_count": len(self.soul.symbols),
            "synapses_count": syn_count,
            "synapse_index": await self._synapse_index.stats,
            "subconscious_backlog": len(self.subconscious_queue),
            # Phase 4.0
            "gap_detector": self.gap_detector.stats,
            "immunology": self._immunology.stats if self._immunology else {},
            "semantic_consolidation": (
                self._semantic_worker.stats if self._semantic_worker else {}
            ),
            # Phase 4.4: Subconscious AI worker stats (BETA)
            "subconscious_ai": (
                self._subconscious_ai.stats if self._subconscious_ai else {}
            ),
            # Phase 4.5: RecursiveSynthesizer stats
            "recursive_synthesizer": (
                self._recursive_synthesizer.stats if self._recursive_synthesizer else {}
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return stats

    # ==========================================================================
    # Legacy Helpers (for migration compatibility)
    # ==========================================================================

    def _legacy_encode_content_numpy(self, content: str) -> np.ndarray:
        """
        Original localized encoding logic for backward compatibility.
        Used only for migrating legacy data.
        """
        tokens = re.findall(r'\w+', content.lower())
        if not tokens:
            seed_bytes = hashlib.shake_256(content.encode()).digest(4)
            seed = int.from_bytes(seed_bytes, 'little')
            return np.random.RandomState(seed).choice([-1, 1], size=self.dimension)

        combined = np.zeros(self.dimension)
        for t in tokens:
            t_vec = self._get_token_vector(t, self.dimension)
            combined += t_vec

        v = np.sign(combined)
        v[v == 0] = np.random.RandomState(42).choice([-1, 1], size=np.sum(v == 0))
        return v.astype(int)

    # ==========================================================================
    # Component Status
    # ==========================================================================

    def get_component_status(self) -> Dict[str, Any]:
        """
        Get status information about all engine components.

        Returns a dictionary with component status including:
        - Which workers are active
        - Which features are enabled
        - Component configuration
        """
        return {
            "semantic_worker": {
                "enabled": True,
                "running": self._semantic_worker.running if self._semantic_worker else False,
            },
            "immunology": {
                "enabled": True,
                "running": self._immunology.running if self._immunology else False,
            },
            "gap_filler": {
                "enabled": self._gap_filler is not None,
                "running": self._gap_filler.running if self._gap_filler else False,
            },
            "subconscious_ai": {
                "enabled": self.config.subconscious_ai.enabled,
                "running": self._subconscious_ai.running if self._subconscious_ai else False,
            },
            "recursive_synthesizer": {
                "enabled": self._recursive_synthesizer is not None,
            },
            "anticipatory": {
                "enabled": self.config.anticipatory.enabled,
            },
            "preference_learning": {
                "enabled": self.preference_store.config.enabled,
            },
            "context_tracking": {
                "enabled": self.topic_tracker.config.enabled,
            },
        }

    # ==========================================================================
    # Preference Learning (Phase 12.3)
    # ==========================================================================

    async def log_decision(self, context_text: str, outcome: float) -> None:
        """
        Phase 12.3: Logs a user decision or feedback context to update preference vector.
        Outcome should be positive (e.g. 1.0) or negative (e.g. -1.0).
        """
        import functools

        async def _run_in_thread(func, *args, **kwargs):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

        vec = await _run_in_thread(self.binary_encoder.encode, context_text)
        self.preference_store.log_decision(vec, outcome)
