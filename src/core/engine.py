"""
Holographic Active Inference Memory Engine (HAIM) - Phase 3.5+
Uses Binary HDV for efficient storage and computation.
"""

from typing import List, Tuple, Dict, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .container import Container
    from .qdrant_store import QdrantStore
import heapq
from itertools import islice
import numpy as np
import hashlib
import os
import json
import asyncio
import functools
import uuid
import re
from datetime import datetime, timezone
from loguru import logger

from .config import get_config, HAIMConfig
from .binary_hdv import BinaryHDV, TextEncoder, majority_bundle
from .node import MemoryNode
from .synapse import SynapticConnection
from .holographic import ConceptualMemory
from .tier_manager import TierManager

# Phase 4.0 imports
from .attention import XORAttentionMasker, AttentionConfig, XORIsolationMask, IsolationConfig
from .bayesian_ltp import get_bayesian_updater
from .semantic_consolidation import SemanticConsolidationWorker, SemanticConsolidationConfig
from .immunology import ImmunologyLoop, ImmunologyConfig
from .gap_detector import GapDetector, GapDetectorConfig
from .gap_filler import GapFiller, GapFillerConfig
from .synapse_index import SynapseIndex

# Observability imports (Phase 4.1)
from .metrics import (
    timer, traced, get_trace_id, set_trace_id,
    STORE_DURATION_SECONDS, QUERY_DURATION_SECONDS,
    MEMORY_COUNT_TOTAL, QUEUE_LENGTH, ERROR_TOTAL,
    update_memory_count, update_queue_length, record_error
)


class HAIMEngine:
    """
    Holographic Active Inference Memory Engine (Phase 3.5+)
    Uses Binary HDV and Tiered Storage for efficient cognitive memory.
    """

    @staticmethod
    @functools.lru_cache(maxsize=10000)
    def _get_token_vector(token: str, dimension: int) -> np.ndarray:
        """Cached generation of deterministic token vectors (legacy compatibility)."""
        seed_bytes = hashlib.shake_256(token.encode()).digest(4)
        seed = int.from_bytes(seed_bytes, 'little')
        return np.random.RandomState(seed).choice([-1, 1], size=dimension)

    def __init__(
        self,
        dimension: int = 16384,
        persist_path: Optional[str] = None,
        config: Optional[HAIMConfig] = None,
        tier_manager: Optional[TierManager] = None,
    ):
        """
        Initialize HAIMEngine with optional dependency injection.

        Args:
            dimension: Vector dimensionality (default 16384).
            persist_path: Path to memory persistence file.
            config: Configuration object. If None, uses global get_config().
            tier_manager: TierManager instance. If None, creates a new one.
        """
        self.config = config or get_config()
        self.dimension = self.config.dimensionality

        # Initialization guard
        self._initialized: bool = False

        # Core Components
        self.tier_manager = tier_manager or TierManager(config=self.config)
        self.binary_encoder = TextEncoder(self.dimension)

        # ── Phase 3.x: synapse raw dicts (kept for backward compat) ──
        self.synapses: Dict[Tuple[str, str], SynapticConnection] = {}
        self.synapse_adjacency: Dict[str, List[SynapticConnection]] = {}
        # Async locks – safe to create here in Python 3.10+
        self.synapse_lock: asyncio.Lock = asyncio.Lock()
        # Serialises concurrent _save_synapses disk writes
        self._write_lock: asyncio.Lock = asyncio.Lock()
        # Semaphore: only one dream cycle at a time (rate limiting)
        self._dream_sem: asyncio.Semaphore = asyncio.Semaphore(1)

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

        # ── Phase 4.0: gap detector & filler (wired in initialize()) ──
        self.gap_detector = GapDetector(GapDetectorConfig())
        self._gap_filler: Optional[GapFiller] = None

        # ── Phase 4.0: semantic consolidation worker ───────────────────
        self._semantic_worker: Optional[SemanticConsolidationWorker] = None

        # ── Phase 4.0: immunology loop ─────────────────────────────────
        self._immunology: Optional[ImmunologyLoop] = None

        # Conceptual Layer (VSA Soul)
        data_dir = self.config.paths.data_dir
        self.soul = ConceptualMemory(dimension=self.dimension, storage_dir=data_dir)

        # Persistence paths
        self.persist_path = persist_path or self.config.paths.memory_file
        self.synapse_path = self.config.paths.synapses_file

        # Passive Subconscious Layer
        self.subconscious_queue: List[str] = []

        # Epistemic Drive
        self.epistemic_drive_active = True
        self.surprise_threshold = 0.7

    async def initialize(self):
        """Async initialization."""
        if self._initialized:
            return

        await self.tier_manager.initialize()
        await self._load_legacy_if_needed()
        await self._load_synapses()
        self._initialized = True

        # ── Phase 4.0: start background workers ───────────────────────
        self._semantic_worker = SemanticConsolidationWorker(self)
        await self._semantic_worker.start()

        self._immunology = ImmunologyLoop(self)
        await self._immunology.start()

        logger.info("Phase 4.0 background workers started (consolidation + immunology).")

    async def _run_in_thread(self, func, *args, **kwargs):
        """Run blocking function in thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

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
    # Private Helper Methods for store() - Extracted for maintainability
    # ==========================================================================

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

        Args:
            content: Original text content.
            encoded_vec: Encoded BinaryHDV for the content.
            metadata: Metadata dictionary for the node.

        Returns:
            The created and persisted MemoryNode.
        """
        # Create node with unique ID
        node_id = str(uuid.uuid4())
        node = MemoryNode(
            id=node_id,
            hdv=encoded_vec,
            content=content,
            metadata=metadata,
        )

        # Map EIG/Importance
        node.epistemic_value = float(metadata.get("eig", 0.0))
        node.calculate_ltp()

        # Store in Tier Manager (starts in HOT)
        await self.tier_manager.add_memory(node)

        # Append to persistence log (Legacy/Backup)
        await self._append_persisted(node)

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

        self.subconscious_queue.append(node.id)

        if not _is_gap_fill:
            await self._background_dream(depth=1)

    # ==========================================================================
    # Main store() method - Orchestration only
    # ==========================================================================

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

        This method orchestrates the memory storage pipeline:
        1. Encode input content
        2. Evaluate tier placement via EIG
        3. Persist to storage
        4. Trigger post-store processing

        Args:
            content: The text content to store.
            metadata: Optional metadata dictionary.
            goal_id: Optional goal identifier for context binding.
            project_id: Optional project identifier for isolation masking (Phase 4.1).

        Returns:
            The unique identifier of the stored memory node.
        """
        # 1. Encode input and bind goal context
        encoded_vec, updated_metadata = await self._encode_input(content, metadata, goal_id)

        # 1b. Apply project isolation mask (Phase 4.1)
        if project_id:
            encoded_vec = self.isolation_masker.apply_mask(encoded_vec, project_id)
            updated_metadata['project_id'] = project_id

        # 2. Calculate EIG and evaluate tier placement
        updated_metadata = await self._evaluate_tier(encoded_vec, updated_metadata)

        # 3. Create and persist memory node
        node = await self._persist_memory(content, encoded_vec, updated_metadata)

        # 4. Trigger post-store processing
        await self._trigger_post_store(node, updated_metadata)

        # 5. Update queue length metric
        update_queue_length(len(self.subconscious_queue))

        logger.info(f"Stored memory {node.id} (EIG: {updated_metadata.get('eig', 0.0):.4f})")
        return node.id

    async def delete_memory(self, node_id: str) -> bool:
        """
        Delete a memory from all internal states and storage tiers.
        Returns True if something was deleted.

        Phase 4.0: uses SynapseIndex.remove_node() for O(k) removal.
        """
        logger.info(f"Deleting memory {node_id}")

        # 1. Remove from TierManager (HOT/WARM/COLD-pending)
        deleted = await self.tier_manager.delete_memory(node_id)

        # 2. Remove from subconscious queue if present
        if node_id in self.subconscious_queue:
            self.subconscious_queue.remove(node_id)

        # 3. Phase 4.0: clean up via SynapseIndex (O(k))
        async with self.synapse_lock:
            removed_count = self._synapse_index.remove_node(node_id)

            # Rebuild legacy dicts
            self.synapses = dict(self._synapse_index.items())
            self.synapse_adjacency = {}
            for syn in self._synapse_index.values():
                self.synapse_adjacency.setdefault(syn.neuron_a_id, [])
                self.synapse_adjacency.setdefault(syn.neuron_b_id, [])
                self.synapse_adjacency[syn.neuron_a_id].append(syn)
                self.synapse_adjacency[syn.neuron_b_id].append(syn)

        if removed_count:
            await self._save_synapses()

        return deleted

    async def close(self):
        """Perform graceful shutdown of engine components."""
        logger.info("Shutting down HAIMEngine...")

        # Phase 4.0: stop background workers
        if self._semantic_worker:
            await self._semantic_worker.stop()
        if self._immunology:
            await self._immunology.stop()
        if self._gap_filler:
            await self._gap_filler.stop()

        await self._save_synapses()
        if self.tier_manager.use_qdrant and self.tier_manager.qdrant:
            await self.tier_manager.qdrant.close()

    @timer(QUERY_DURATION_SECONDS)
    @traced("query_memory")
    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        associative_jump: bool = True,
        track_gaps: bool = True,
        project_id: Optional[str] = None,
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
        """
        # Encode Query
        query_vec = await self._run_in_thread(self.binary_encoder.encode, query_text)

        # Phase 4.1: Apply project isolation mask to query
        if project_id:
            query_vec = self.isolation_masker.apply_mask(query_vec, project_id)

        # 1. Primary Search (Accelerated FAISS/HNSW + Qdrant)
        search_results = await self.tier_manager.search(query_vec, top_k=top_k * 2)

        scores: Dict[str, float] = {}
        for nid, base_sim in search_results:
            # Boost by synaptic health (Phase 4.0: use SynapseIndex.boost for O(k))
            boost = self._synapse_index.boost(nid)
            scores[nid] = base_sim * boost

        # 2. Associative Spreading (via SynapseIndex for O(1) adjacency lookup)
        if associative_jump and self._synapse_index:
            top_seeds = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            augmented_scores = scores.copy()

            for seed_id, seed_score in top_seeds:
                if seed_score <= 0:
                    continue

                neighbour_synapses = self._synapse_index.neighbours(seed_id)

                for syn in neighbour_synapses:
                    neighbor = (
                        syn.neuron_b_id if syn.neuron_a_id == seed_id else syn.neuron_a_id
                    )
                    if neighbor not in augmented_scores:
                        mem = await self.tier_manager.get_memory(neighbor)
                        if mem:
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
        if track_gaps:
            asyncio.ensure_future(
                self.gap_detector.assess_query(query_text, top_results, attention_mask)
            )

        return top_results

    async def _background_dream(self, depth: int = 2):
        """
        Passive Subconscious – strengthen synapses in idle cycles.

        Uses a semaphore so at most one dream task runs concurrently,
        and passes track_gaps=False so dream queries cannot feed the
        gap detector (breaking the store→dream→gap→fill→store loop).
        """
        if not self.subconscious_queue:
            return

        # Non-blocking: if a dream is already in progress, skip this cycle.
        if not self._dream_sem._value:  # noqa: SLF001
            return

        async with self._dream_sem:
            stim_id = self.subconscious_queue.pop(0)
            stim_node = await self.tier_manager.get_memory(stim_id)
            if not stim_node:
                return

            potential_connections = await self.query(
                stim_node.content,
                top_k=depth + 1,
                associative_jump=False,
                track_gaps=False,   # ← no gap detection inside dream
            )

            for neighbor_id, similarity in potential_connections:
                if neighbor_id != stim_id and similarity > 0.15:
                    await self.bind_memories(stim_id, neighbor_id, success=True)

    async def bind_memories(self, id_a: str, id_b: str, success: bool = True):
        """
        Bind two memories by ID.

        Phase 4.0: delegates to SynapseIndex for O(1) insert/fire.
        Also syncs legacy dicts for backward-compat.
        """
        mem_a = await self.tier_manager.get_memory(id_a)
        mem_b = await self.tier_manager.get_memory(id_b)

        if not mem_a or not mem_b:
            return

        async with self.synapse_lock:
            syn = self._synapse_index.add_or_fire(id_a, id_b, success=success)

            # Keep legacy dict in sync for any external code still using it
            synapse_key = tuple(sorted([id_a, id_b]))
            self.synapses[synapse_key] = syn
            self.synapse_adjacency.setdefault(synapse_key[0], [])
            self.synapse_adjacency.setdefault(synapse_key[1], [])
            if syn not in self.synapse_adjacency[synapse_key[0]]:
                self.synapse_adjacency[synapse_key[0]].append(syn)
            if syn not in self.synapse_adjacency[synapse_key[1]]:
                self.synapse_adjacency[synapse_key[1]].append(syn)

        await self._save_synapses()

    async def get_node_boost(self, node_id: str) -> float:
        """
        Compute synaptic boost for scoring.

        Phase 4.0: O(k) via SynapseIndex (was O(k) before but with lock overhead).
        """
        return self._synapse_index.boost(node_id)

    async def cleanup_decay(self, threshold: float = 0.1):
        """
        Remove synapses that have decayed below the threshold.

        Phase 4.0: O(E) via SynapseIndex.compact(), no lock required for the index itself.
        Also syncs any legacy dict entries into the index before compacting.
        """
        async with self.synapse_lock:
            # Sync legacy dict → SynapseIndex via the public register() API
            # (handles tests / external code that injects into self.synapses directly)
            for key, syn in list(self.synapses.items()):
                if self._synapse_index.get(syn.neuron_a_id, syn.neuron_b_id) is None:
                    self._synapse_index.register(syn)

            removed = self._synapse_index.compact(threshold)

            if removed:
                # Rebuild legacy dicts from the index
                self.synapses = dict(self._synapse_index.items())
                self.synapse_adjacency = {}
                for syn in self._synapse_index.values():
                    self.synapse_adjacency.setdefault(syn.neuron_a_id, [])
                    self.synapse_adjacency.setdefault(syn.neuron_b_id, [])
                    self.synapse_adjacency[syn.neuron_a_id].append(syn)
                    self.synapse_adjacency[syn.neuron_b_id].append(syn)

                logger.info(f"cleanup_decay: pruned {removed} synapses below {threshold}")
                await self._save_synapses()

    async def get_stats(self) -> Dict[str, Any]:
        """Aggregate statistics from engine components."""
        tier_stats = await self.tier_manager.get_stats()

        async with self.synapse_lock:
            syn_count = len(self._synapse_index)

        stats = {
            "engine_version": "4.0.0",
            "dimension": self.dimension,
            "encoding": "binary_hdv",
            "tiers": tier_stats,
            "concepts_count": len(self.soul.concepts),
            "symbols_count": len(self.soul.symbols),
            "synapses_count": syn_count,
            "synapse_index": self._synapse_index.stats,
            "subconscious_backlog": len(self.subconscious_queue),
            # Phase 4.0
            "gap_detector": self.gap_detector.stats,
            "immunology": self._immunology.stats if self._immunology else {},
            "semantic_consolidation": (
                self._semantic_worker.stats if self._semantic_worker else {}
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return stats

    def encode_content(self, content: str) -> BinaryHDV:
        """Encode text to Binary HDV."""
        return self.binary_encoder.encode(content)

    # ── Phase 4.0: Gap filling ─────────────────────────────────────

    async def enable_gap_filling(
        self,
        llm_integrator,
        config: Optional["GapFillerConfig"] = None,
    ) -> None:
        """
        Attach an LLM integrator to autonomously fill knowledge gaps.

        Args:
            llm_integrator: HAIMLLMIntegrator instance.
            config: Optional GapFillerConfig overrides.
        """
        if self._gap_filler:
            await self._gap_filler.stop()

        self._gap_filler = GapFiller(
            engine=self,
            llm_integrator=llm_integrator,
            gap_detector=self.gap_detector,
            config=config or GapFillerConfig(),
        )
        await self._gap_filler.start()
        logger.info("Phase 4.0 GapFiller started.")

    async def record_retrieval_feedback(
        self,
        node_id: str,
        helpful: bool,
        eig_signal: float = 1.0,
    ) -> None:
        """
        Record whether a retrieved memory was useful.

        Phase 4.0: feeds the Bayesian LTP updater for the node.

        Args:
            node_id: The memory node that was retrieved.
            helpful: Was the retrieval actually useful?
            eig_signal: Strength of evidence (0–1).
        """
        node = await self.tier_manager.get_memory(node_id)
        if node:
            updater = get_bayesian_updater()
            updater.observe_node_retrieval(node, helpful=helpful, eig_signal=eig_signal)

    async def register_negative_feedback(self, query_text: str) -> None:
        """
        Signal that a recent query was not adequately answered.
        Creates a high-priority gap record for LLM gap-filling.
        """
        await self.gap_detector.register_negative_feedback(query_text)

    async def get_memory(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve memory via TierManager."""
        return await self.tier_manager.get_memory(node_id)

    # --- Legacy Helpers (for migration compatibility) ---

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

    async def _load_legacy_if_needed(self):
        """Load from memory.jsonl into TierManager, converting to BinaryHDV."""
        if not os.path.exists(self.persist_path):
            return

        logger.info(f"Loading legacy memory from {self.persist_path}")

        def _load():
            try:
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    return f.readlines()
            except Exception:
                return []

        lines = await self._run_in_thread(_load)

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

                # Add to TierManager
                await self.tier_manager.add_memory(node)

            except Exception as e:
                logger.warning(f"Failed to load record: {e}")

    async def _load_synapses(self):
        """
        Load synapses from disk.

        Phase 4.0: uses SynapseIndex.load_from_file() which restores Bayesian state.
        """
        if not os.path.exists(self.synapse_path):
            return

        def _load():
            self._synapse_index.load_from_file(self.synapse_path)

        await self._run_in_thread(_load)

        # Rebuild legacy dicts from SynapseIndex for backward compat
        async with self.synapse_lock:
            self.synapses = dict(self._synapse_index.items())
            self.synapse_adjacency = {}
            for syn in self._synapse_index.values():
                self.synapse_adjacency.setdefault(syn.neuron_a_id, [])
                self.synapse_adjacency.setdefault(syn.neuron_b_id, [])
                self.synapse_adjacency[syn.neuron_a_id].append(syn)
                self.synapse_adjacency[syn.neuron_b_id].append(syn)

    async def _save_synapses(self):
        """
        Save synapses to disk in JSONL format.

        Phase 4.0: uses SynapseIndex.save_to_file() which includes Bayesian state.
        A dedicated _write_lock serialises concurrent callers so the file is never
        written by two coroutines at the same time.  Does NOT acquire synapse_lock.
        """
        path_snapshot = self.synapse_path

        def _save():
            self._synapse_index.save_to_file(path_snapshot)

        async with self._write_lock:
            await self._run_in_thread(_save)

    async def _append_persisted(self, node: MemoryNode):
        """Append-only log."""

        def _append():
            try:
                with open(self.persist_path, 'a', encoding='utf-8') as f:
                    rec = {
                        'id': node.id,
                        'content': node.content,
                        'metadata': node.metadata,
                        'created_at': node.created_at.isoformat()
                    }
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                logger.error(f"Failed to persist memory: {e}")

        await self._run_in_thread(_append)

    # --- Conceptual Proxy ---

    async def define_concept(self, name: str, attributes: Dict[str, str]):
        await self._run_in_thread(self.soul.store_concept, name, attributes)

    async def reason_by_analogy(self, src: str, val: str, tgt: str):
        return await self._run_in_thread(self.soul.solve_analogy, src, val, tgt)

    async def cross_domain_inference(self, src: str, tgt: str, pat: str):
        return await self._run_in_thread(self.soul.solve_analogy, src, pat, tgt)

    async def inspect_concept(self, name: str, attr: str):
        return await self._run_in_thread(self.soul.extract_attribute, name, attr)
