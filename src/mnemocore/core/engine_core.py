"""
HAIMEngine Core Operations Module (Phase 6 Refactor)

Contains fundamental memory operations: store, retrieve, query, delete.
These are the core data-path methods that handle memory persistence and retrieval.

Separated from engine_lifecycle.py and engine_coordinator.py to maintain
clear separation of concerns and keep files under 800 lines.
"""

from typing import List, Tuple, Dict, Optional, Any, TYPE_CHECKING, Deque
from collections import deque
from datetime import datetime, timezone
import functools
import uuid
import asyncio
from loguru import logger

import numpy as np

# Core imports
from .config import HAIMConfig
from .binary_hdv import BinaryHDV, TextEncoder, majority_bundle
from .node import MemoryNode
from .synapse import SynapticConnection
from .tier_manager import TierManager

# Shared utilities (Task 4.4: centralized helpers)
from ._utils import run_in_thread, get_token_vector, safe_ensure_future, log_task_exception

# Observability imports
from .metrics import (
    timer, traced, STORE_DURATION_SECONDS, QUERY_DURATION_SECONDS,
    update_queue_length
)

# Events integration
try:
    from ..events import get_event_bus, EventBus
    from ..events import integration as event_integration
    EVENTS_AVAILABLE = True
except ImportError:
    EVENTS_AVAILABLE = False
    EventBus = None  # type: ignore
    event_integration = None  # type: ignore

if TYPE_CHECKING:
    from .holographic import ConceptualMemory
    from .synapse_index import SynapseIndex
    from .attention import XORAttentionMasker, AttentionConfig, XORIsolationMask, IsolationConfig
    from .gap_detector import GapDetector, GapDetectorConfig
    from .working_memory import WorkingMemoryService
    from .episodic_store import EpisodicStoreService
    from .semantic_store import SemanticStoreService
    from .procedural_store import ProceduralStoreService
    from .meta_memory import MetaMemoryService
    from .topic_tracker import TopicTracker
    from .preference_store import PreferenceStore
    from .anticipatory import AnticipatoryEngine


class EngineCoreOperations:
    """
    Core memory operations for HAIMEngine.

    This mixin class provides the fundamental store/retrieve/query/forget operations.
    Designed to be mixed into HAIMEngine without circular dependencies.

    Responsibilities:
    - Memory storage with encoding and persistence
    - Memory query with semantic search
    - Memory deletion
    - Epistemic valuation (EIG calculation)
    - Working memory integration
    """

    # Maximum allowed content length (input validation)
    _MAX_CONTENT_LENGTH: int = 100_000

    # ==========================================================================
    # Initialization Support
    # ==========================================================================

    def _initialize_core_state(
        self,
        config: HAIMConfig,
        tier_manager: TierManager,
        dimension: int,
        persist_path: Optional[str],
        synapse_path: str,
    ):
        """
        Initialize core state attributes. Called from HAIMEngine.__init__.

        This is a factory method that sets up the core state without
        creating circular dependencies.
        """
        # Core configuration
        self.config = config
        self.dimension = dimension
        self.persist_path = persist_path
        self.synapse_path = synapse_path

        # Core components
        self.tier_manager = tier_manager
        self.binary_encoder = TextEncoder(dimension)

        # Initialization guard
        self._initialized: bool = False

        # Legacy synapse storage (kept for backward compatibility)
        self.synapses: Dict[Tuple[str, str], SynapticConnection] = {}
        self.synapse_adjacency: Dict[str, List[SynapticConnection]] = {}

        # Async locks
        self.synapse_lock: asyncio.Lock = asyncio.Lock()
        self._write_lock: asyncio.Lock = asyncio.Lock()
        self._store_lock: asyncio.Lock = asyncio.Lock()
        self._dream_sem: asyncio.Semaphore = asyncio.Semaphore(1)

        # Passive Subconscious Layer
        queue_maxlen = config.dream_loop.subconscious_queue_maxlen
        self.subconscious_queue: Deque[str] = deque(maxlen=queue_maxlen)
        self._last_stored_id: Optional[str] = None

        # Epistemic Drive
        self.epistemic_drive_active = True
        self.surprise_threshold = 0.7

        # Event system integration (lazy init)
        self._event_bus: Optional[EventBus] = None
        self._event_bus_initialized = False

    def _set_phase5_components(
        self,
        working_memory: Optional["WorkingMemoryService"],
        episodic_store: Optional["EpisodicStoreService"],
        semantic_store: Optional["SemanticStoreService"],
    ):
        """Set Phase 5 AGI store components."""
        self.working_memory = working_memory
        self.episodic_store = episodic_store
        self.semantic_store = semantic_store

    def _set_phase4_components(
        self,
        synapse_index: "SynapseIndex",
        attention_masker: "XORAttentionMasker",
        isolation_masker: "XORIsolationMask",
        gap_detector: "GapDetector",
        soul: "ConceptualMemory",
        topic_tracker: "TopicTracker",
        preference_store: "PreferenceStore",
        anticipatory_engine: "AnticipatoryEngine",
    ):
        """Set Phase 4+ components that core operations depend on."""
        self._synapse_index = synapse_index
        self.attention_masker = attention_masker
        self.isolation_masker = isolation_masker
        self.gap_detector = gap_detector
        self.soul = soul
        self.topic_tracker = topic_tracker
        self.preference_store = preference_store
        self.anticipatory_engine = anticipatory_engine

    # ==========================================================================
    # Helper Methods (imported from _utils.py for deduplication)
    # ==========================================================================

    # Use run_in_thread from _utils module (Task 4.4)
    async def _run_in_thread(self, func, *args, **kwargs):
        """Run blocking function in thread pool. Delegates to shared utility."""
        return await run_in_thread(func, *args, **kwargs)

    # Use get_token_vector from _utils module (Task 4.4)
    # Cached at instance level to avoid memory leaks from unbounded static cache
    _token_vector_cache: Dict[Tuple[str, int], np.ndarray] = {}

    def _get_token_vector(self, token: str, dimension: int) -> np.ndarray:
        """Cached generation of deterministic token vectors (legacy compatibility)."""
        cache_key = (token, dimension)
        if cache_key not in self._token_vector_cache:
            self._token_vector_cache[cache_key] = get_token_vector(token, dimension)
        return self._token_vector_cache[cache_key]

    # ==========================================================================
    # Event System Integration
    # ==========================================================================

    @property
    def event_bus(self) -> Optional[EventBus]:
        """
        Get the EventBus instance for this engine.

        Lazy-loads the EventBus on first access if enabled in config.
        """
        if not EVENTS_AVAILABLE:
            return None

        if not self._event_bus_initialized:
            if self.config.events.enabled:
                self._event_bus = get_event_bus()
                self._event_bus_initialized = True
                logger.debug("[EngineCore] EventBus initialized")
            else:
                self._event_bus_initialized = True

        return self._event_bus

    async def emit_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Emit an event to the EventBus.

        Convenience method for components to emit events without
        directly accessing the event_bus property.

        Args:
            event_type: Type of event to emit
            data: Event payload data
            metadata: Optional metadata
        """
        if EVENTS_AVAILABLE and event_integration:
            await event_integration.emit_event(
                event_bus=self.event_bus,
                event_type=event_type,
                data=data,
                metadata=metadata,
            )

    def calculate_eig(self, candidate: BinaryHDV, context: BinaryHDV) -> float:
        """
        Calculate Expected Information Gain (EIG).
        Proportional to novelty (distance) against the context.

        Returns value in [0.0, 1.0] where:
        - 0.0 = candidate is identical to context (no new information)
        - 1.0 = candidate is maximally different from context (max information)
        """
        return candidate.normalized_distance(context)

    async def _current_context_vector(self, sample_n: int = 50) -> BinaryHDV:
        """Superpose a slice of working memory (HOT tier) into a single context vector."""
        recent_nodes = await self.tier_manager.get_hot_recent(sample_n)

        if not recent_nodes:
            return BinaryHDV.zeros(self.dimension)

        vectors = [n.hdv for n in recent_nodes]
        if not vectors:
            return BinaryHDV.zeros(self.dimension)

        return majority_bundle(vectors)

    # ==========================================================================
    # Store Operations - Pipeline Stages (Task 4.1 refactoring)
    # ==========================================================================

    def _validate_content(self, content: str) -> str:
        """
        Validate memory content before storage.

        Stage 1 of the store pipeline.

        Args:
            content: The text content to validate.

        Returns:
            The validated and stripped content.

        Raises:
            ValueError: If content is empty or exceeds maximum length.
            RuntimeError: If engine is not initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                "HAIMEngine.initialize() must be awaited before calling store()."
            )
        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty or whitespace-only.")
        if len(content) > self._MAX_CONTENT_LENGTH:
            raise ValueError(
                f"Memory content is too long ({len(content):,} chars). "
                f"Maximum: {self._MAX_CONTENT_LENGTH:,}."
            )
        return content.strip()

    async def _encode_content(self, content: str) -> BinaryHDV:
        """
        Encode text content to BinaryHDV.

        Stage 2 of the store pipeline.

        Args:
            content: The validated text content.

        Returns:
            BinaryHDV representation of the content.
        """
        return await self._run_in_thread(self.binary_encoder.encode, content)

    async def _post_store_hooks(
        self,
        node: MemoryNode,
        content: str,
        encoded_vec: BinaryHDV,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Execute all post-store hooks: WM push, episodic logging, semantic consolidation,
        event emission, association update, and meta-memory recording.

        Stage 5 of the store pipeline - runs after persistence.

        Args:
            node: The stored MemoryNode.
            content: Original text content.
            encoded_vec: The encoded BinaryHDV.
            metadata: The metadata dictionary.
        """
        # Phase 5.1: Working Memory and Episodic Store integration
        agent_id = metadata.get("agent_id")
        if agent_id:
            if self.working_memory:
                from .memory_model import WorkingMemoryItem
                await self.working_memory.push_item(
                    agent_id,
                    WorkingMemoryItem(
                        id=f"wm_{node.id[:8]}",
                        agent_id=agent_id,
                        created_at=datetime.now(timezone.utc),
                        ttl_seconds=3600,
                        content=content,
                        kind="observation",
                        importance=node.epistemic_value or 0.5,
                        tags=metadata.get("tags", []),
                        hdv=encoded_vec
                    )
                )

            episode_id = metadata.get("episode_id")
            if episode_id and self.episodic_store:
                self.episodic_store.append_event(
                    episode_id=episode_id,
                    kind="observation",
                    content=content,
                    metadata=metadata
                )

        # Emit memory.created event
        if EVENTS_AVAILABLE and event_integration:
            await event_integration.emit_memory_created(
                event_bus=self.event_bus,
                memory_id=node.id,
                content=content,
                tier=node.tier or "hot",
                ltp_strength=node.ltp_strength,
                eig=metadata.get("eig"),
                tags=metadata.get("tags"),
                category=metadata.get("category"),
                agent_id=metadata.get("agent_id"),
                episode_id=metadata.get("episode_id"),
                previous_id=node.previous_id,
            )

        # Phase 6.0: Association network integration
        if hasattr(self, 'associations_integrator'):
            try:
                context_nodes = []
                if self._last_stored_id:
                    last_mem = await self.tier_manager.get_memory(self._last_stored_id)
                    if last_mem:
                        context_nodes.append(last_mem)
                self.associations_integrator.on_store(node, context_nodes)
                logger.debug(f"Added memory {node.id} to association network")
            except Exception as e:
                logger.warning(f"Failed to add to association network: {e}")

        # Phase 5.1: Auto-consolidate into semantic store
        if hasattr(self, 'semantic_store') and self.semantic_store and content:
            try:
                episode_ids = []
                episode_id = metadata.get("episode_id")
                if episode_id:
                    episode_ids.append(episode_id)
                self.semantic_store.consolidate_from_content(
                    content=content,
                    hdv=encoded_vec,
                    episode_ids=episode_ids,
                    tags=metadata.get("tags", []),
                    agent_id=metadata.get("agent_id"),
                )
            except Exception as e:
                logger.warning(f"Semantic consolidation on store skipped: {e}")

        # Phase 5.1: Record store metric for meta-memory
        if hasattr(self, 'meta_memory') and self.meta_memory:
            try:
                self.meta_memory.record_metric("store_count", 1.0, "per_call")
            except Exception as e:
                logger.warning(f"Failed to record meta-memory metric: {e}")

        # Update queue length metric
        update_queue_length(len(self.subconscious_queue))

    async def _encode_input(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        goal_id: Optional[str] = None,
    ) -> Tuple[BinaryHDV, Dict[str, Any]]:
        """
        Encode input content to BinaryHDV and bind goal context if present.

        Args:
            content: The text content to encode.
            metadata: Optional metadata dictionary (will be mutated if goal_id present).
            goal_id: Optional goal identifier to bind as context.

        Returns:
            Tuple of (encoded BinaryHDV, updated metadata dict).
        """
        # Encode content (CPU bound operation)
        content_vec = await self._run_in_thread(self.binary_encoder.encode, content)

        # Initialize metadata if needed
        if metadata is None:
            metadata = {}

        final_vec = content_vec

        # Bind goal context if provided
        if goal_id:
            goal_vec = await self._run_in_thread(
                self.binary_encoder.encode, f"GOAL_CONTEXT_{goal_id}"
            )
            final_vec = content_vec.xor_bind(goal_vec)
            metadata['goal_context'] = goal_id

        return final_vec, metadata

    async def _evaluate_tier(
        self,
        encoded_vec: BinaryHDV,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate epistemic valuation (EIG) and update metadata accordingly.

        Args:
            encoded_vec: The encoded BinaryHDV to evaluate.
            metadata: Metadata dictionary to update with EIG values.

        Returns:
            Updated metadata dictionary with EIG information.
        """
        if self.epistemic_drive_active:
            ctx_vec = await self._current_context_vector(sample_n=50)
            eig = self.calculate_eig(encoded_vec, ctx_vec)
            metadata["eig"] = float(eig)

            if eig >= self.surprise_threshold:
                metadata.setdefault("tags", [])
                if isinstance(metadata["tags"], list):
                    metadata["tags"].append("epistemic_high")
        else:
            metadata.setdefault("eig", 0.0)

        return metadata

    async def _persist_memory(
        self,
        content: str,
        encoded_vec: BinaryHDV,
        metadata: Dict[str, Any],
    ) -> MemoryNode:
        """
        Create MemoryNode and persist to tier manager and disk.

        Phase 4.3: Automatically sets previous_id for episodic chaining.

        Args:
            content: Original text content.
            encoded_vec: Encoded BinaryHDV for the content.
            metadata: Metadata dictionary for the node.

        Returns:
            The created and persisted MemoryNode.
        """
        async with self._store_lock:
            previous_id = self._last_stored_id

            # Create node with unique ID
            node_id = str(uuid.uuid4())
            node = MemoryNode(
                id=node_id,
                hdv=encoded_vec,
                content=content,
                metadata=metadata,
                previous_id=previous_id,  # Phase 4.3: Episodic chaining
            )

            # Map EIG/Importance
            node.epistemic_value = float(metadata.get("eig", 0.0))
            node.calculate_ltp()

            # Store in Tier Manager (starts in HOT)
            await self.tier_manager.add_memory(node)

            # Append to persistence log (Legacy/Backup)
            await self._append_persisted(node)

            # Update linear episodic chain head only after successful persistence.
            self._last_stored_id = node.id

            return node

    async def _trigger_post_store(
        self,
        node: MemoryNode,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Execute post-store triggers: subconscious queue and background dream.

        Gap-filled memories must NOT re-enter the dream/gap loop to prevent
        an indefinite store -> dream -> detect -> fill -> store cycle.

        Args:
            node: The MemoryNode that was stored.
            metadata: Metadata dictionary (checked for gap fill source).
        """
        _is_gap_fill = metadata.get("source") == "llm_gap_fill"

        # Phase 12.1: Aggressive Synapse Formation (Auto-bind).
        # Fix 4: collect all bindings first, persist synapses only once at the end.
        if hasattr(self.config, 'synapse') and self.config.synapse.auto_bind_on_store:
            # Import here to avoid circular dependency
            from .engine_coordinator import EngineCoordinator
            similar_nodes = await self.query(
                node.content,
                top_k=3,
                associative_jump=False,
                track_gaps=False,
            )
            bind_pairs = [
                (node.id, neighbor_id)
                for neighbor_id, similarity in similar_nodes
                if neighbor_id != node.id
                and similarity >= self.config.synapse.similarity_threshold
            ]
            if bind_pairs:
                await self._auto_bind_batch(bind_pairs)

        self.subconscious_queue.append(node.id)

        if not _is_gap_fill:
            await self._background_dream(depth=1)

    @timer(STORE_DURATION_SECONDS, labels={"tier": "hot"})
    @traced("store_memory")
    async def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        goal_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> str:
        """
        Store new memory with holographic encoding.

        This method orchestrates the memory storage pipeline (Task 4.1 refactored):
        1. _validate_content: Validate input
        2. _encode_content: Encode input to BinaryHDV
        3. _evaluate_tier: Calculate EIG for tier placement
        4. _persist_memory: Create and persist MemoryNode
        5. _trigger_post_store + _post_store_hooks: Post-processing

        Args:
            content: The text content to store. Must be non-empty and <=100 000 chars.
            metadata: Optional metadata dictionary.
            goal_id: Optional goal identifier for context binding.
            project_id: Optional project identifier for isolation masking (Phase 4.1).

        Returns:
            The unique identifier of the stored memory node.

        Raises:
            ValueError: If content is empty or exceeds the maximum allowed length.
            RuntimeError: If the engine has not been initialized via initialize().
        """
        # Stage 1: Validate content
        content = self._validate_content(content)

        # Stage 2: Encode input and bind goal context
        encoded_vec, updated_metadata = await self._encode_input(content, metadata, goal_id)

        # Stage 2b: Apply project isolation mask (Phase 4.1)
        if project_id:
            encoded_vec = self.isolation_masker.apply_mask(encoded_vec, project_id)
            updated_metadata['project_id'] = project_id

        # Stage 3: Calculate EIG and evaluate tier placement
        updated_metadata = await self._evaluate_tier(encoded_vec, updated_metadata)

        # Stage 4: Create and persist memory node
        node = await self._persist_memory(content, encoded_vec, updated_metadata)

        # Stage 5a: Trigger post-store processing (subconscious queue, synapse binding)
        await self._trigger_post_store(node, updated_metadata)

        # Stage 5b: Execute all post-store hooks (WM, episodic, events, associations, etc.)
        await self._post_store_hooks(node, content, encoded_vec, updated_metadata)

        logger.info(f"Stored memory {node.id} (EIG: {updated_metadata.get('eig', 0.0):.4f})")
        return node.id

    # ==========================================================================
    # Query Operations - Pipeline Stages (Task 4.2 refactoring)
    # ==========================================================================

    async def _encode_query(
        self,
        query_text: str,
        project_id: Optional[str] = None,
    ) -> BinaryHDV:
        """
        Encode query text to BinaryHDV with optional project isolation.

        Stage 1 of the query pipeline.

        Args:
            query_text: The query text to encode.
            project_id: Optional project ID for isolation masking.

        Returns:
            Encoded BinaryHDV for the query.
        """
        query_vec = await self._run_in_thread(self.binary_encoder.encode, query_text)

        # Phase 12.2: Context Tracking
        is_shift, sim = self.topic_tracker.add_query(query_vec)
        if is_shift:
            logger.info(f"Context shifted during query. (sim {sim:.3f})")

        # Phase 4.1: Apply project isolation mask
        if project_id:
            query_vec = self.isolation_masker.apply_mask(query_vec, project_id)

        return query_vec

    async def _search_tiers(
        self,
        query_vec: BinaryHDV,
        top_k: int,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_cold: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Perform primary search across tiers.

        Stage 2 of the query pipeline.

        Args:
            query_vec: The encoded query vector.
            top_k: Number of results to return.
            time_range: Optional time range filter.
            metadata_filter: Optional metadata filter.
            include_cold: Whether to include cold tier.

        Returns:
            List of (node_id, similarity) tuples.
        """
        return await self.tier_manager.search(
            query_vec,
            top_k=top_k * 2,
            time_range=time_range,
            metadata_filter=metadata_filter,
            include_cold=include_cold,
        )

    async def _apply_temporal_weighting(
        self,
        scores: Dict[str, float],
        search_results: List[Tuple[str, float]],
        chrono_weight: bool,
        chrono_lambda: float,
    ) -> Dict[str, MemoryNode]:
        """
        Apply temporal decay weighting to scores.

        Stage 3 of the query pipeline.

        Args:
            scores: Score dictionary to update.
            search_results: Original search results.
            chrono_weight: Whether to apply temporal weighting.
            chrono_lambda: Decay rate.

        Returns:
            Dictionary of loaded MemoryNodes for reuse.
        """
        now_ts = datetime.now(timezone.utc).timestamp()
        mem_map: Dict[str, MemoryNode] = {}

        if chrono_weight and search_results:
            mems = await self.tier_manager.get_memories_batch(
                [nid for nid, _ in search_results]
            )
            mem_map = {m.id: m for m in mems if m}

        for nid, base_sim in search_results:
            # Boost by synaptic health
            boost = await self._synapse_index.boost(nid)
            score = base_sim * boost

            # Chrono-weighting (temporal decay)
            if chrono_weight and score > 0:
                mem = mem_map.get(nid)
                if mem:
                    time_delta = max(0.0, now_ts - mem.created_at.timestamp())
                    decay_factor = 1.0 / (1.0 + chrono_lambda * time_delta)
                    score = score * decay_factor

            scores[nid] = score

        return mem_map

    async def _apply_preference_bias(
        self,
        scores: Dict[str, float],
        mem_map: Dict[str, MemoryNode],
    ) -> None:
        """
        Apply preference learning bias to scores.

        Stage 4 of the query pipeline.

        Args:
            scores: Score dictionary to update in-place.
            mem_map: Memory node cache for lookup.
        """
        if not (self.preference_store.config.enabled and
                self.preference_store.preference_vector is not None):
            return

        for nid in list(scores.keys()):
            mem = mem_map.get(nid)
            if not mem:
                mem = await self.tier_manager.get_memory(nid)
                if mem and mem.id not in mem_map:
                    mem_map[mem.id] = mem
            if mem:
                scores[nid] = self.preference_store.bias_score(mem.hdv, scores[nid])

    async def _apply_wm_boost(
        self,
        scores: Dict[str, float],
        query_text: str,
        metadata_filter: Optional[Dict[str, Any]],
        mem_map: Dict[str, MemoryNode],
    ) -> None:
        """
        Apply working memory context boost to scores.

        Stage 5 of the query pipeline.

        Args:
            scores: Score dictionary to update in-place.
            query_text: The original query text.
            metadata_filter: Optional metadata filter containing agent_id.
            mem_map: Memory node cache for lookup.
        """
        agent_id = metadata_filter.get("agent_id") if metadata_filter else None
        if not (agent_id and self.working_memory):
            return

        wm_state = await self.working_memory.get_state(agent_id)
        if not wm_state or not wm_state.items:
            return

        wm_texts = [item.content for item in wm_state.items]
        if not wm_texts:
            return

        q_lower = query_text.lower()
        for nid in scores:
            mem = mem_map.get(nid)
            if mem and mem.content:
                if any(w_text.lower() in mem.content.lower() for w_text in wm_texts):
                    scores[nid] *= 1.15  # 15% boost for WM overlap

    async def _apply_attention_rerank(
        self,
        scores: Dict[str, float],
        query_vec: BinaryHDV,
        top_k: int,
    ) -> Tuple[List[Tuple[str, float]], Optional[BinaryHDV]]:
        """
        Apply XOR attention re-ranking to results.

        Stage 6 of the query pipeline.

        Args:
            scores: Score dictionary.
            query_vec: The encoded query vector.
            top_k: Number of results to return.

        Returns:
            Tuple of (sorted results, attention_mask or None).
        """
        attention_mask = None
        top_results: List[Tuple[str, float]] = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        if not scores:
            return top_results, None

        # Build context key from recent HOT nodes
        recent_nodes = await self.tier_manager.get_hot_recent(
            self.attention_masker.config.context_sample_n
        )
        if not recent_nodes:
            return top_results, None

        ctx_vecs = [n.hdv for n in recent_nodes]
        ctx_key = self.attention_masker.build_context_key(ctx_vecs)
        attention_mask = self.attention_masker.build_attention_mask(query_vec, ctx_key)

        # Collect HDVs for re-ranking
        mem_vecs: Dict[str, BinaryHDV] = {}
        async with self.tier_manager.lock:
            for nid in list(scores.keys()):
                node = self.tier_manager.hot.get(nid)
                if node:
                    mem_vecs[nid] = node.hdv

        ranked = self.attention_masker.rerank(scores, mem_vecs, attention_mask)
        top_results = self.attention_masker.extract_scores(ranked)[:top_k]

        return top_results, attention_mask

    async def _trigger_post_query(
        self,
        top_results: List[Tuple[str, float]],
        query_text: str,
        query_vec: BinaryHDV,
        attention_mask: Optional[BinaryHDV],
        track_gaps: bool,
        metadata_filter: Optional[Dict[str, Any]],
        scores: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        """
        Execute post-query triggers: gap detection, anticipatory preloading,
        association strengthening, and meta-memory recording.

        Stage 7 of the query pipeline (fire-and-forget async hooks).

        Args:
            top_results: The query results.
            query_text: Original query text.
            query_vec: Encoded query vector.
            attention_mask: The attention mask used (or None).
            track_gaps: Whether to track knowledge gaps.
            metadata_filter: Optional metadata filter.
            scores: The score dictionary.

        Returns:
            Final results (potentially with temporal neighbors added).
        """
        # Phase 4.0: Knowledge gap detection (Task 4.3: added exception callback)
        if track_gaps:
            safe_ensure_future(
                self.gap_detector.assess_query(query_text, top_results, attention_mask),
                name="gap_detection"
            )

        # Phase 4.3: Sequential Context Window (fetch temporal neighbors)
        # Note: This is done synchronously as it modifies results
        # The actual neighbor fetching could be moved to a separate method if needed

        # Phase 13.2: Anticipatory preloading (Task 4.3: added exception callback)
        if top_results and self._initialized and self.config.anticipatory.enabled:
            safe_ensure_future(
                self.anticipatory_engine.predict_and_preload(top_results[0][0]),
                name="anticipatory_preload"
            )

        # Phase 6.0: Association network integration (Task 4.3: added exception callback)
        if top_results and hasattr(self, 'associations_integrator'):
            retrieved_ids = [nid for nid, _ in top_results]
            safe_ensure_future(
                self._strengthen_recall_associations(retrieved_ids, query_text),
                name="association_strengthen"
            )

        # Phase 5.1: Record query metrics for meta-memory
        if hasattr(self, 'meta_memory') and self.meta_memory:
            try:
                hit_rate = 1.0 if top_results else 0.0
                best_score = top_results[0][1] if top_results else 0.0
                self.meta_memory.record_metric("query_hit_rate", hit_rate, "per_call")
                self.meta_memory.record_metric("query_best_score", best_score, "per_call")
                self.meta_memory.record_metric("query_result_count", float(len(top_results)), "per_call")
            except Exception as e:
                logger.warning(f"Failed to record query metrics in meta-memory: {e}")

        return top_results

    # ==========================================================================
    # Query Operations
    # ==========================================================================

    @timer(QUERY_DURATION_SECONDS)
    @traced("query_memory")
    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        associative_jump: bool = True,
        track_gaps: bool = True,
        project_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        chrono_weight: bool = True,
        chrono_lambda: float = 0.0001,
        include_neighbors: bool = False,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_cold: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Query memories using Hamming distance.
        Searches HOT tier and limited WARM tier.

        Phase 4.0 additions:
          - XOR attention masking re-ranks results for novelty.
          - Gap detection runs on low-confidence results (disabled when
            track_gaps=False to prevent dream-loop feedback).

        Phase 4.1 additions:
          - project_id applies isolation mask to query for project-scoped search.

        Phase 4.3 additions (Temporal Recall):
          - time_range: Filter to memories within (start, end) datetime range.
          - chrono_weight: Apply temporal decay to boost newer memories.
            Formula: Final_Score = Semantic_Similarity * (1 / (1 + lambda * Time_Delta))
          - chrono_lambda: Decay rate in seconds^-1 (default: 0.0001 ~ 2.7h half-life).
          - include_neighbors: Also fetch temporal neighbors (previous/next) for top results.
          - include_cold: Include COLD tier in the search (bounded linear scan, default False).

        Phase 13.2: Triggers anticipatory preloading as fire-and-forget after returning.
        """
        # Encode Query
        query_vec = await self._run_in_thread(self.binary_encoder.encode, query_text)

        # Phase 12.2: Context Tracking
        is_shift, sim = self.topic_tracker.add_query(query_vec)
        if is_shift:
            logger.info(f"Context shifted during query. (sim {sim:.3f})")

        # Phase 4.1: Apply project isolation mask to query
        if project_id:
            query_vec = self.isolation_masker.apply_mask(query_vec, project_id)

        # 1. Primary Search (Accelerated FAISS/HNSW + Qdrant)
        # Phase 4.3: Pass time_range to tier_manager if filtering needed
        search_results = await self.tier_manager.search(
            query_vec,
            top_k=top_k * 2,
            time_range=time_range,
            metadata_filter=metadata_filter,
            include_cold=include_cold,
        )

        scores: Dict[str, float] = {}
        now_ts = datetime.now(timezone.utc).timestamp()
        mem_map: Dict[str, MemoryNode] = {}

        if chrono_weight and search_results:
            mems = await self.tier_manager.get_memories_batch(
                [nid for nid, _ in search_results]
            )
            mem_map = {m.id: m for m in mems if m}

        for nid, base_sim in search_results:
            # Boost by synaptic health (Phase 4.0: use SynapseIndex.boost for O(k))
            boost = await self._synapse_index.boost(nid)
            score = base_sim * boost

            # Phase 4.3: Chrono-weighting (temporal decay)
            if chrono_weight and score > 0:
                mem = mem_map.get(nid)
                if mem:
                    time_delta = max(0.0, now_ts - mem.created_at.timestamp())
                    decay_factor = 1.0 / (1.0 + chrono_lambda * time_delta)
                    score = score * decay_factor

            # Phase 12.3: Preference Learning Bias
            if self.preference_store.config.enabled and self.preference_store.preference_vector is not None:
                mem = mem_map.get(nid)
                if not mem:
                    mem = await self.tier_manager.get_memory(nid)
                    if mem and mem.id not in mem_map:
                        mem_map[mem.id] = mem
                if mem:
                    score = self.preference_store.bias_score(mem.hdv, score)

            scores[nid] = score

        # Phase 5.1: Boost context matching Working Memory
        agent_id = metadata_filter.get("agent_id") if metadata_filter else None
        if agent_id and self.working_memory:
            wm_state = await self.working_memory.get_state(agent_id)
            if wm_state:
                wm_texts = [item.content for item in wm_state.items]
                if wm_texts:
                    q_lower = query_text.lower()
                    for nid in scores:
                        mem = mem_map.get(nid)
                        if mem and mem.content:
                            if any(w_text.lower() in mem.content.lower() for w_text in wm_texts):
                                scores[nid] *= 1.15  # 15% boost for WM overlap

        # 2. Associative Spreading (via SynapseIndex for O(1) adjacency lookup)
        if associative_jump and self._synapse_index is not None:
            top_seeds = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            augmented_scores = scores.copy()

            for seed_id, seed_score in top_seeds:
                if seed_score <= 0:
                    continue

                neighbour_synapses = await self._synapse_index.neighbours(seed_id)

                for syn in neighbour_synapses:
                    neighbor = (
                        syn.neuron_b_id if syn.neuron_a_id == seed_id else syn.neuron_a_id
                    )
                    if neighbor not in augmented_scores:
                        mem = await self.tier_manager.get_memory(neighbor)
                        if mem:
                            if metadata_filter:
                                match = True
                                node_meta = mem.metadata or {}
                                for k, v in metadata_filter.items():
                                    if node_meta.get(k) != v:
                                        match = False
                                        break
                                if not match:
                                    continue
                            augmented_scores[neighbor] = query_vec.similarity(mem.hdv)

                    if neighbor in augmented_scores:
                        spread = seed_score * syn.get_current_strength() * 0.3
                        augmented_scores[neighbor] += spread

            scores = augmented_scores

        # Phase 4.0: XOR attention re-ranking
        attention_mask = None
        top_results: List[Tuple[str, float]] = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        if scores:
            # Build context key from recent HOT nodes
            recent_nodes = await self.tier_manager.get_hot_recent(
                self.attention_masker.config.context_sample_n
            )
            if recent_nodes:
                ctx_vecs = [n.hdv for n in recent_nodes]
                ctx_key = self.attention_masker.build_context_key(ctx_vecs)
                attention_mask = self.attention_masker.build_attention_mask(query_vec, ctx_key)

                # Collect HDVs for re-ranking (only HOT nodes available synchronously)
                mem_vecs: Dict[str, BinaryHDV] = {}
                async with self.tier_manager.lock:
                    for nid in list(scores.keys()):
                        node = self.tier_manager.hot.get(nid)
                        if node:
                            mem_vecs[nid] = node.hdv

                ranked = self.attention_masker.rerank(scores, mem_vecs, attention_mask)
                top_results = self.attention_masker.extract_scores(ranked)[:top_k]

        # Phase 4.0: Knowledge gap detection
        # Disabled during dream cycles to break the store->dream->gap->fill->store loop.
        # Task 4.3: Use safe_ensure_future for exception logging
        if track_gaps:
            safe_ensure_future(
                self.gap_detector.assess_query(query_text, top_results, attention_mask),
                name="gap_detection"
            )

        # Phase 4.3: Sequential Context Window
        # Fetch temporal neighbors (previous_id chain and next in chain)
        if include_neighbors and top_results:
            neighbor_ids: set = set()
            for result_id, _ in top_results[:3]:  # Only for top 3 results
                mem = await self.tier_manager.get_memory(result_id)
                if not mem:
                    continue

                # Get the memory that came before this one (if episodic chain exists)
                if mem.previous_id:
                    prev_mem = await self.tier_manager.get_memory(mem.previous_id)
                    if prev_mem and prev_mem.id not in scores:
                        if metadata_filter:
                            match = True
                            p_meta = prev_mem.metadata or {}
                            for k, v in metadata_filter.items():
                                if p_meta.get(k) != v:
                                    match = False
                                    break
                            if not match:
                                continue
                        neighbor_ids.add(prev_mem.id)

                # Try to find the memory that follows this one (has this as previous_id).
                next_mem = await self.tier_manager.get_next_in_chain(result_id)
                if next_mem and next_mem.id not in scores:
                    neighbor_ids.add(next_mem.id)

            # Add neighbors with their semantic scores (no chrono boost for context)
            for neighbor_id in neighbor_ids:
                mem = await self.tier_manager.get_memory(neighbor_id)
                if mem:
                    neighbor_score = query_vec.similarity(mem.hdv)
                    top_results.append((neighbor_id, neighbor_score * 0.8))  # Slightly discounted

            # Re-sort after adding neighbors, but preserve query() top_k contract.
            top_results = sorted(top_results, key=lambda x: x[1], reverse=True)[:top_k]

        # Phase 13.2: Anticipatory preloading — fire-and-forget so it
        # never blocks the caller. Only activated when the engine is fully warm.
        # Task 4.3: Use safe_ensure_future for exception logging
        if top_results and self._initialized and self.config.anticipatory.enabled:
            safe_ensure_future(
                self.anticipatory_engine.predict_and_preload(top_results[0][0]),
                name="anticipatory_preload"
            )

        # Phase 6.0: Association network integration
        # Fire-and-forget: strengthen associations between co-retrieved memories
        # Task 4.3: Use safe_ensure_future for exception logging
        if top_results and hasattr(self, 'associations_integrator'):
            retrieved_ids = [nid for nid, _ in top_results]
            safe_ensure_future(
                self._strengthen_recall_associations(retrieved_ids, query_text),
                name="association_strengthen"
            )

        # Phase 5.1: Record query metrics for meta-memory
        if hasattr(self, 'meta_memory') and self.meta_memory:
            try:
                hit_rate = 1.0 if top_results else 0.0
                best_score = top_results[0][1] if top_results else 0.0
                self.meta_memory.record_metric("query_hit_rate", hit_rate, "per_call")
                self.meta_memory.record_metric("query_best_score", best_score, "per_call")
                self.meta_memory.record_metric("query_result_count", float(len(top_results)), "per_call")
            except Exception as e:
                logger.warning(f"Failed to record query metrics in meta-memory: {e}")

        return top_results

    async def _strengthen_recall_associations(
        self,
        retrieved_ids: List[str],
        query_text: str
    ) -> None:
        """
        Phase 6.0: Strengthen associations between co-retrieved memories.

        This is called asynchronously after each query to implement
        Hebbian learning: memories that fire together wire together.

        Args:
            retrieved_ids: List of memory IDs that were retrieved together
            query_text: The query text (for context)
        """
        if not hasattr(self, 'associations') or not hasattr(self, 'associations_integrator'):
            return

        try:
            # Fetch the actual memory nodes
            memories = []
            for nid in retrieved_ids:
                mem = await self.tier_manager.get_memory(nid)
                if mem:
                    memories.append(mem)

            # Add nodes to association network if not present
            for mem in memories:
                if not self.associations.has_node(mem.id):
                    self.associations.add_node(mem)

            # Strengthen associations between all pairs
            self.associations_integrator.on_recall(memories, query_text)

            logger.debug(f"Strengthened associations for {len(memories)} co-retrieved memories")

        except Exception as e:
            logger.warning(f"Failed to strengthen associations: {e}")

    async def get_context_nodes(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Phase 12.2: Contextual Awareness
        Retrieves the top_k most relevant nodes relating to the current topic context vector.
        Should be explicitly used by prompt builders before LLM logic injection.
        """
        if not self.topic_tracker.config.enabled:
            return []

        ctx = self.topic_tracker.get_context()
        if ctx is None:
            return []

        results = await self.tier_manager.search(
            ctx,
            top_k=top_k,
            time_range=None,
            metadata_filter=None,
        )
        return results

    async def get_memory(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve memory via TierManager."""
        return await self.tier_manager.get_memory(node_id)

    async def get_associated_memories(
        self,
        node_id: str,
        max_results: int = 10,
        min_strength: float = 0.1,
        include_content: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Phase 6.0: Get memories associated with a given node.

        Uses the association network to find related memories based on
        co-retrieval history and explicit associations.

        Args:
            node_id: The ID of the starting memory node
            max_results: Maximum number of associated memories to return
            min_strength: Minimum association strength threshold
            include_content: Whether to include memory content in results

        Returns:
            List of dicts containing associated memory information
        """
        if not hasattr(self, 'associations'):
            return []

        try:
            associations = self.associations.get_associations(
                node_id,
                min_strength=min_strength,
                limit=max_results,
            )

            results = []
            for edge in associations:
                other_id = edge.target_id if edge.source_id == node_id else edge.source_id
                mem = await self.tier_manager.get_memory(other_id)

                if mem:
                    result = {
                        "id": mem.id,
                        "strength": edge.strength,
                        "association_type": edge.association_type.value,
                        "fire_count": edge.fire_count,
                    }
                    if include_content:
                        result["content"] = mem.content
                        result["metadata"] = mem.metadata
                    results.append(result)

            return results

        except Exception as e:
            logger.warning(f"Failed to get associated memories: {e}")
            return []

    # ==========================================================================
    # Delete Operations
    # ==========================================================================

    async def delete_memory(self, node_id: str) -> bool:
        """
        Delete a memory from all internal states and storage tiers.
        Returns True if something was deleted.

        Phase 4.0: uses SynapseIndex.remove_node() for O(k) removal.
        Phase 6.0: also removes from association network.
        """
        logger.info(f"Deleting memory {node_id}")

        # Get memory info before deletion for event
        mem = await self.tier_manager.get_memory(node_id)
        mem_tier = mem.tier if mem else "unknown"

        # 1. Remove from TierManager (HOT/WARM/COLD-pending)
        deleted = await self.tier_manager.delete_memory(node_id)

        # 2. Remove from subconscious queue if present
        if node_id in self.subconscious_queue:
            self.subconscious_queue.remove(node_id)

        # 3. Phase 4.0: clean up via SynapseIndex (O(k)).
        async with self.synapse_lock:
            removed_count = await self._synapse_index.remove_node(node_id)

        if removed_count:
            await self._save_synapses()

        # 4. Phase 6.0: Remove from association network
        if hasattr(self, 'associations'):
            try:
                self.associations.remove_node(node_id)
                logger.debug(f"Removed memory {node_id} from association network")
            except Exception as e:
                logger.warning(f"Failed to remove from association network: {e}")

        # 5. Emit memory.deleted event
        if deleted and EVENTS_AVAILABLE and event_integration:
            await event_integration.emit_memory_deleted(
                event_bus=self.event_bus,
                memory_id=node_id,
                tier=mem_tier,
                reason="manual",
            )

        return deleted

    # ==========================================================================
    # Background Dream Processing
    # ==========================================================================

    async def _background_dream(self, depth: int = 2):
        """
        Passive Subconscious – strengthen synapses in idle cycles.

        Uses a semaphore so at most one dream task runs concurrently,
        and passes track_gaps=False so dream queries cannot feed the
        gap detector (breaking the store->dream->gap->fill->store loop).
        """
        if not self.subconscious_queue:
            return

        # Non-blocking: if a dream is already in progress, skip this cycle.
        if self._dream_sem.locked():
            return

        async with self._dream_sem:
            stim_id = self.subconscious_queue.popleft()
            stim_node = await self.tier_manager.get_memory(stim_id)
            if not stim_node:
                return

            potential_connections = await self.query(
                stim_node.content,
                top_k=depth + 1,
                associative_jump=False,
                track_gaps=False,   # no gap detection inside dream
            )

            for neighbor_id, similarity in potential_connections:
                if neighbor_id != stim_id and similarity > 0.15:
                    await self.bind_memories(stim_id, neighbor_id, success=True)

    # ==========================================================================
    # Persistence Helpers
    # ==========================================================================

    async def _append_persisted(self, node: MemoryNode):
        """Append-only log with Phase 4.3 temporal metadata."""
        from mnemocore.utils import json_compat as json

        def _append():
            try:
                with open(self.persist_path, 'a', encoding='utf-8') as f:
                    rec = {
                        'id': node.id,
                        'content': node.content,
                        'metadata': node.metadata,
                        'created_at': node.created_at.isoformat(),
                        # Phase 4.3: Temporal metadata for indexing
                        'unix_timestamp': node.unix_timestamp,
                        'iso_date': node.iso_date,
                        'previous_id': node.previous_id,
                    }
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                logger.error(f"Failed to persist memory: {e}")

        await self._run_in_thread(_append)

    async def persist_memory_snapshot(self, node: MemoryNode) -> None:
        """Persist a current snapshot of a memory node to the append-only log."""
        await self._append_persisted(node)

    # ==========================================================================
    # Synapse Operations
    # ==========================================================================

    async def _auto_bind_batch(
        self,
        pairs: List[Tuple[str, str]],
        success: bool = True,
        weight: float = 1.0,
    ) -> None:
        """
        Bind multiple (id_a, id_b) pairs in one pass, saving synapses once.

        Used by auto-bind in _trigger_post_store() to avoid N disk writes per store.
        """
        async with self.synapse_lock:
            for id_a, id_b in pairs:
                mem_a = await self.tier_manager.get_memory(id_a)
                mem_b = await self.tier_manager.get_memory(id_b)
                if mem_a and mem_b:
                    await self._synapse_index.add_or_fire(id_a, id_b, success=success, weight=weight)
        await self._save_synapses()

    async def bind_memories(self, id_a: str, id_b: str, success: bool = True, weight: float = 1.0):
        """
        Bind two memories by ID.

        Delegates exclusively to SynapseIndex — legacy dict sync removed.
        The legacy self.synapses / self.synapse_adjacency attributes remain for
        backward compatibility but are only populated at startup from disk.
        """
        mem_a = await self.tier_manager.get_memory(id_a)
        mem_b = await self.tier_manager.get_memory(id_b)

        if not mem_a or not mem_b:
            return

        async with self.synapse_lock:
            await self._synapse_index.add_or_fire(id_a, id_b, success=success, weight=weight)

        await self._save_synapses()

    async def _save_synapses(self):
        """
        Save synapses to disk in JSONL format.

        Uses SynapseIndex.save_to_file() which includes Bayesian state.
        A dedicated _write_lock serialises concurrent callers so the file is never
        written by two coroutines at the same time. Does NOT acquire synapse_lock.
        """
        path_snapshot = self.synapse_path

        async with self._write_lock:
            await self._synapse_index.save_to_file(path_snapshot)

    # ==========================================================================
    # Encoding Helper
    # ==========================================================================

    def encode_content(self, content: str) -> BinaryHDV:
        """Encode text to Binary HDV."""
        return self.binary_encoder.encode(content)
