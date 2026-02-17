from typing import List, Tuple, Dict, Optional, Union, Any
import heapq
from itertools import islice
import numpy as np
import hashlib
import os
import json
import logging
import asyncio
import functools
import uuid
import re
from datetime import datetime, timezone

from .config import get_config
from .hdv import HDV
from .binary_hdv import BinaryHDV, TextEncoder, majority_bundle
from .node import MemoryNode
from .synapse import SynapticConnection
from .holographic import ConceptualMemory
from .tier_manager import TierManager

logger = logging.getLogger(__name__)

class HAIMEngine:
    """
    Holographic Active Inference Memory Engine (Phase 3.0+)
    Supports Binary HDV and Tiered Storage.
    """

    @staticmethod
    @functools.lru_cache(maxsize=10000)
    def _get_token_vector(token: str, dimension: int) -> np.ndarray:
        """Cached generation of deterministic token vectors."""
        # Use SHAKE-256 to generate a 32-bit seed (4 bytes)
        seed_bytes = hashlib.shake_256(token.encode()).digest(4)
        seed = int.from_bytes(seed_bytes, 'little')
        return np.random.RandomState(seed).choice([-1, 1], size=dimension)

    def __init__(self, dimension: int = 16384, persist_path: Optional[str] = None):
        self.config = get_config()
        # Allow override from init, but prefer config
        self.dimension = self.config.dimensionality
        
        # Core Components
        self.tier_manager = TierManager()
        self.binary_encoder = TextEncoder(self.dimension)
        
        self.synapses: Dict[Tuple[str, str], SynapticConnection] = {}
        self.synapse_adjacency: Dict[str, List[SynapticConnection]] = {}
        self.synapse_lock = asyncio.Lock()
        
        # Buffers for vector operations
        self._legacy_context_buffer = np.zeros(self.dimension)
        # We don't easily buffer binary majority vote without modifying binary_hdv heavily,
        # but we can reuse the legacy buffer if we are careful? No, types differ.

        # Conceptual Layer (VSA Soul)
        data_dir = self.config.paths.data_dir
        self.soul = ConceptualMemory(dimension=self.dimension, storage_dir=data_dir)

        # Legacy persistence path (for backward compat / migration)
        self.persist_path = persist_path or self.config.paths.memory_file
        self.synapse_path = self.config.paths.synapses_file

        # Passive Subconscious Layer
        self.subconscious_queue: List[str] = []

        # Epistemic Drive
        self.epistemic_drive_active = True
        self.surprise_threshold = 0.7 

    async def initialize(self):
        """Async initialization."""
        await self.tier_manager.initialize()
        await self._load_legacy_if_needed()
        await self._load_synapses()

    async def _run_in_thread(self, func, *args, **kwargs):
        """Run blocking function in thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

    def calculate_eig(self, candidate: Union[HDV, BinaryHDV], context: Union[HDV, BinaryHDV]) -> float:
        """
        Calculate Expected Information Gain (EIG).
        Proportional to novelty (distance) against the context.
        """
        # Proprietary novelty detection logic - Adaptive EIG calculation
        if isinstance(candidate, BinaryHDV) and isinstance(context, BinaryHDV):
            # Normalized Hamming distance: 0.0 (identical) to 1.0 (opposite)
            # Novelty = distance (0 = no info, 1 = max info)
            return candidate.normalized_distance(context)
        else:
            # Legacy Float HDV
            # Cosine similarity: 1.0 (identical) to -1.0 (opposite)
            # Novelty = 1.0 - similarity (0 = identical, 2 = opposite)
            # We saturate at 0 and 1 for EIG probability
            sim = candidate.cosine_similarity(context)
            return max(0.0, min(1.0, 1.0 - sim))

    async def _current_context_vector(self, sample_n: int = 50) -> Union[HDV, BinaryHDV]:
        """Superpose a slice of working memory (HOT tier) into a single context vector."""
        # Get recent nodes efficiently
        recent_nodes = await self.tier_manager.get_hot_recent(sample_n)
        
        if not recent_nodes:
            if self.config.encoding.mode == "binary":
                return BinaryHDV.zeros(self.dimension)
            else:
                return HDV(dimension=self.dimension)

        # Check type conformity
        is_binary = isinstance(recent_nodes[0].hdv, BinaryHDV)
        
        if is_binary:
            vectors = [n.hdv for n in recent_nodes if isinstance(n.hdv, BinaryHDV)]
            if not vectors: return BinaryHDV.zeros(self.dimension)
            # Computationally heavy?
            return majority_bundle(vectors)
        else:
            # Legacy addition optimization: reuse buffer
            # We must zero it out first.
            self._legacy_context_buffer.fill(0)
            combined = self._legacy_context_buffer
            
            count = 0
            for n in recent_nodes:
                if isinstance(n.hdv, HDV):
                    combined += n.hdv.vector
                    count += 1
            
            if count == 0:
                 return HDV(dimension=self.dimension)
                 
            hdv = HDV(dimension=self.dimension)
            hdv.vector = np.sign(combined)
            hdv.vector[hdv.vector == 0] = 1
            return hdv

    async def store(self, content: str, metadata: dict = None, goal_id: str = None) -> str:
        """Store new memory with holographic encoding."""
        
        # 1. Encode Content (CPU Bound)
        def _encode():
            if self.config.encoding.mode == "binary":
                return self.binary_encoder.encode(content)
            else:
                legacy_hdv = HDV(dimension=self.dimension)
                legacy_hdv.vector = self._legacy_encode_content_numpy(content)
                return legacy_hdv
        
        content_vec = await self._run_in_thread(_encode)

        # 2. Bind Goal Context (if any)
        # This is fast enough to run in loop usually, or move to encoder
        final_vec = content_vec
        if goal_id:
            # Simplified for async safety, strictly simple CPU ops
            if isinstance(content_vec, BinaryHDV):
                # Binary Goal Binding
                goal_vec = await self._run_in_thread(self.binary_encoder.encode, f"GOAL_CONTEXT_{goal_id}")
                final_vec = content_vec.xor_bind(goal_vec)
            else:
                # Legacy Goal Binding
                goal_vec = HDV(dimension=self.dimension)
                goal_vec.vector = self._legacy_encode_content_numpy(f"GOAL_CONTEXT_{goal_id}")
                final_vec.vector = content_vec.vector * goal_vec.vector
            
            if metadata is None: metadata = {}
            metadata['goal_context'] = goal_id

        # 3. Epistemic Valuation (EIG)
        if metadata is None: metadata = {}
        
        if self.epistemic_drive_active:
            ctx_vec = await self._current_context_vector(sample_n=50)
            # Ensure types match for EIG calc
            if isinstance(final_vec, type(ctx_vec)):
                eig = self.calculate_eig(final_vec, ctx_vec)
                metadata["eig"] = float(eig)
                if eig >= self.surprise_threshold:
                    metadata.setdefault("tags", [])
                    if isinstance(metadata["tags"], list):
                        metadata["tags"].append("epistemic_high")
        else:
             metadata.setdefault("eig", 0.0)

        # 4. Create Node
        node_id = str(uuid.uuid4())
        node = MemoryNode(
            id=node_id,
            hdv=final_vec,
            content=content,
            metadata=metadata
        )
        
        # Map EIG/Importance
        node.epistemic_value = float(metadata.get("eig", 0.0))
        node.calculate_ltp() # Initial LTP

        # 5. Store in Tier Manager (starts in HOT)
        await self.tier_manager.add_memory(node)

        # 6. Append to persistence log (Legacy/Backup)
        await self._append_persisted(node)

        # 7. Subconscious Trigger
        self.subconscious_queue.append(node_id)
        # Fire and forget background dream? Or await?
        # Awaiting ensures consistency but adds latency.
        # For now, await basic depth 1 dream.
        await self._background_dream(depth=1)

        logger.info(f"Stored memory {node_id} (EIG: {metadata.get('eig', 0.0):.4f})")
        return node_id

    async def delete_memory(self, node_id: str) -> bool:
        """
        Delete a memory from all internal states and storage tiers.
        Returns True if something was deleted.
        """
        logger.info(f"Deleting memory {node_id}")
        
        # 1. Remove from TierManager (HOT/WARM/COLD-pending)
        deleted = await self.tier_manager.delete_memory(node_id)
        
        # 2. Remove from subconscious queue if present
        if node_id in self.subconscious_queue:
            self.subconscious_queue.remove(node_id)
            
        # 3. Clean up synapses safely
        async with self.synapse_lock:
            keys_to_remove = [k for k in self.synapses.keys() if node_id in k]
            for k in keys_to_remove:
                syn = self.synapses[k]
                del self.synapses[k]
                # Remove from adjacency
                if syn.neuron_a_id in self.synapse_adjacency:
                    if syn in self.synapse_adjacency[syn.neuron_a_id]:
                        self.synapse_adjacency[syn.neuron_a_id].remove(syn)
                if syn.neuron_b_id in self.synapse_adjacency:
                    if syn in self.synapse_adjacency[syn.neuron_b_id]:
                        self.synapse_adjacency[syn.neuron_b_id].remove(syn)
            
            # Also ensure key itself is removed from adjacency dict if present (for node_id)
            if node_id in self.synapse_adjacency:
                del self.synapse_adjacency[node_id]
        
        if keys_to_remove:
            await self._save_synapses()
            
        return deleted

    async def close(self):
        """Perform graceful shutdown of engine components."""
        logger.info("Shutting down HAIMEngine...")
        await self._save_synapses()
        if self.tier_manager.use_qdrant and self.tier_manager.qdrant:
            await self.tier_manager.qdrant.close()

    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        associative_jump: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Query memories using Hamming distance (Binary) or Cosine (Legacy).
        Searches HOT tier and limited WARM tier.
        """
        # Encode Query
        def _encode():
            if self.config.encoding.mode == "binary":
                return self.binary_encoder.encode(query_text)
            else:
                qd = HDV(dimension=self.dimension)
                qd.vector = self._legacy_encode_content_numpy(query_text)
                return qd

        query_vec = await self._run_in_thread(_encode)

        # 1. Primary Search (Accelerated FAISS + Qdrant)
        # Replacing O(N) linear scan with O(log N) ANN search
        search_results = await self.tier_manager.search(query_vec, top_k=top_k * 2) 
        
        scores: Dict[str, float] = {}
        for nid, base_sim in search_results:
            # Boost by synaptic health
            boost = await self.get_node_boost(nid)
            scores[nid] = base_sim * boost

        # 2. Associative Spreading
        if associative_jump:
            # Snapshot synapses to avoid holding lock during computation
            async with self.synapse_lock:
                synapses_snap = list(self.synapses.items())

            if synapses_snap:
                augmented_scores = scores.copy()
                # Top seeds
                top_seeds = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                
                # Check synapses - Loop over snapshot
                # Optimization: In real system, use an adjacency list for O(1) lookup
                # Phase 3.0.1: Adjacency list optimization applied.
                
                for seed_id, seed_score in top_seeds:
                    if seed_score <= 0: continue
                    
                    # Use adjacency list
                    # Create a local list of neighbors from adjacency if available, else fall back?
                    # Since we are inside snapshot context (sort of), but adjacency isn't snapshotted.
                    # We should acquire lock or rely on eventual consistency?
                    # The outer method queried snapshot of flat dict.
                    # Let's use the adjacency list on self directly, but we need to be careful about concurrency if not locked.
                    # This method `query` doesn't hold `synapse_lock` during this loop.
                    # Ideally we should have snapshotted the relevant adjacency.
                    # For now, to be safe and fast, we can re-acquire lock briefly or iterate over copy.
                    # Or simpler: The flat snapshot was for the O(N) scan. 
                    # Now we can just use the adjacency list with a lock around the specific lookup?
                    # But that might be too much locking.
                    # Let's just trust Python's GIL for list iteration safety or make a copy.
                    
                    # Better approach: We already have 'synapses_snap' which is a list of ALL synapses.
                    # Filtering that is O(N).
                    # We want O(K).
                    # We can build a temporary mini-adjacency from the snapshot? No, that's O(N).
                    # We must use self.synapse_adjacency.
                    
                    neighbors_conns = []
                    async with self.synapse_lock:
                         if seed_id in self.synapse_adjacency:
                             neighbors_conns = list(self.synapse_adjacency[seed_id])
                    
                    for synapse in neighbors_conns:
                         neighbor = synapse.neuron_b_id if synapse.neuron_a_id == seed_id else synapse.neuron_a_id
                         
                         # Retrieve neighbor if not in scores (could be in WARM)
                         if neighbor not in augmented_scores:
                             # Try to fetch from TierManager (might promote)
                             mem = await self.tier_manager.get_memory(neighbor)
                             if mem:
                                 # Compute base sim
                                 if isinstance(query_vec, BinaryHDV) and isinstance(mem.hdv, BinaryHDV):
                                     n_sim = query_vec.similarity(mem.hdv)
                                 else:
                                     n_sim = 0.0
                                 augmented_scores[neighbor] = n_sim
                         
                         if neighbor in augmented_scores:
                             spread = seed_score * synapse.get_current_strength() * 0.3
                             augmented_scores[neighbor] += spread
                scores = augmented_scores

        # Sort
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def _background_dream(self, depth: int = 2):
        """Passive Subconscious - Strengthening synapses in idle cycles"""
        if not self.subconscious_queue:
            return
        
        stim_id = self.subconscious_queue.pop(0)
        stim_node = await self.tier_manager.get_memory(stim_id) 
        if not stim_node: return

        potential_connections = await self.query(stim_node.content, top_k=depth+1, associative_jump=False)
        
        for neighbor_id, similarity in potential_connections:
            if neighbor_id != stim_id and similarity > 0.15:
                await self.bind_memories(stim_id, neighbor_id, success=True)

    async def bind_memories(self, id_a: str, id_b: str, success: bool = True):
        """Bind two memories by ID."""
        # Validation
        mem_a = await self.tier_manager.get_memory(id_a)
        mem_b = await self.tier_manager.get_memory(id_b)
        
        if not mem_a or not mem_b:
            return

        synapse_key = tuple(sorted([id_a, id_b]))
        
        async with self.synapse_lock:
            if synapse_key not in self.synapses:
                syn = SynapticConnection(synapse_key[0], synapse_key[1])
                self.synapses[synapse_key] = syn
                # Update adjacency
                if synapse_key[0] not in self.synapse_adjacency: self.synapse_adjacency[synapse_key[0]] = []
                if synapse_key[1] not in self.synapse_adjacency: self.synapse_adjacency[synapse_key[1]] = []
                self.synapse_adjacency[synapse_key[0]].append(syn)
                self.synapse_adjacency[synapse_key[1]].append(syn)
                
            self.synapses[synapse_key].fire(success=success)
            
        await self._save_synapses()

    async def get_node_boost(self, node_id: str) -> float:
        boost = 1.0
        # O(K) lookup
        connections = []
        async with self.synapse_lock:
            if node_id in self.synapse_adjacency:
                connections = list(self.synapse_adjacency[node_id])

        for synapse in connections:
            boost *= (1.0 + synapse.get_current_strength())
        return boost

    async def cleanup_decay(self, threshold: float = 0.1):
        """Remove synapses that have decayed below the threshold."""
        async with self.synapse_lock:
            to_remove = []
            for key, synapse in self.synapses.items():
                if synapse.get_current_strength() < threshold:
                    to_remove.append(key)
            
            for key in to_remove:
                syn = self.synapses[key]
                del self.synapses[key]
                # Remove from adjacency
                if syn.neuron_a_id in self.synapse_adjacency:
                    if syn in self.synapse_adjacency[syn.neuron_a_id]:
                        self.synapse_adjacency[syn.neuron_a_id].remove(syn)
                if syn.neuron_b_id in self.synapse_adjacency:
                    if syn in self.synapse_adjacency[syn.neuron_b_id]:
                        self.synapse_adjacency[syn.neuron_b_id].remove(syn)
            
            should_save = len(to_remove) > 0
            count = len(to_remove)

        if should_save:
            logger.info(f"Cleaning up {count} decayed synapses")
            await self._save_synapses()

    async def get_stats(self) -> Dict[str, Any]:
        """Aggregate statistics from engine components."""
        # Check if tier_manager has get_stats
        tier_stats = await self.tier_manager.get_stats()
        
        async with self.synapse_lock:
            syn_count = len(self.synapses)
            
        return {
            "engine_version": "3.5.1",
            "dimension": self.dimension,
            "tiers": tier_stats,
            "concepts_count": len(self.soul.concepts),
            "symbols_count": len(self.soul.symbols),
            "synapses_count": syn_count,
            "subconscious_backlog": len(self.subconscious_queue),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def encode_content(self, content: str) -> Union[HDV, BinaryHDV]:
        """Encode text to HDV (Binary or Legacy Float)."""
        # This one is sync because it's used by helpers or tests mostly.
        # But if it's CPU bound, should be careful.
        if self.config.encoding.mode == "binary":
            return self.binary_encoder.encode(content)
        else:
            hdv = HDV(dimension=self.dimension)
            hdv.vector = self._legacy_encode_content_numpy(content)
            return hdv

    async def get_memory(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve memory via TierManager."""
        return await self.tier_manager.get_memory(node_id)

    # --- Legacy Helpers ---

    def _legacy_encode_content_numpy(self, content: str) -> np.ndarray:
        """Original localized encoding logic for backward compatibility."""
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
        # Use a fixed seed for tie-breaking to maintain determinism
        v[v == 0] = np.random.RandomState(42).choice([-1, 1], size=np.sum(v == 0))
        return v.astype(int)

    async def _load_legacy_if_needed(self):
        """Load from memory.jsonl into TierManager."""
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
            if not line: continue
            try:
                rec = json.loads(line)
                content = rec.get('content', '')
                if not content: continue
                
                node_id = rec.get('id')
                
                # Convert to current encoding
                if self.config.encoding.mode == "binary":
                    hdv = self.binary_encoder.encode(content)
                else:
                    hdv = HDV(dimension=self.dimension)
                    hdv.vector = self._legacy_encode_content_numpy(content)
                    
                node = MemoryNode(
                    id=node_id,
                    hdv=hdv,
                    content=content,
                    metadata=rec.get('metadata') or {}
                )
                
                # Restore timestamps if available
                if 'created_at' in rec:
                    node.created_at = datetime.fromisoformat(rec['created_at'])
                    
                # Add to TierManager (will handle eviction if full)
                await self.tier_manager.add_memory(node)
                
            except Exception as e:
                logger.warning(f"Failed to load record: {e}")

    async def _load_synapses(self):
        if not os.path.exists(self.synapse_path): return
        
        def _load():
            res = {}
            try:
                with open(self.synapse_path, 'r') as f:
                    for line in f:
                        if not line.strip(): continue
                        rec = json.loads(line)
                        syn = SynapticConnection(rec['neuron_a_id'], rec['neuron_b_id'], rec['strength'])
                        syn.fire_count = rec.get('fire_count', 0)
                        syn.success_count = rec.get('success_count', 0)
                        if 'last_fired' in rec:
                            syn.last_fired = datetime.fromisoformat(rec['last_fired'])
                        res[tuple(sorted([syn.neuron_a_id, syn.neuron_b_id]))] = syn
            except Exception as e:
                logger.error(f"Error loading synapses: {e}")
            return res
            
        async with self.synapse_lock:
            loaded_synapses = await self._run_in_thread(_load)
            self.synapses = loaded_synapses
            # Rebuild adjacency
            self.synapse_adjacency = {}
            for syn in self.synapses.values():
                if syn.neuron_a_id not in self.synapse_adjacency: self.synapse_adjacency[syn.neuron_a_id] = []
                if syn.neuron_b_id not in self.synapse_adjacency: self.synapse_adjacency[syn.neuron_b_id] = []
                self.synapse_adjacency[syn.neuron_a_id].append(syn)
                self.synapse_adjacency[syn.neuron_b_id].append(syn)

    async def _save_synapses(self):
        """Save synapses to disk in JSONL format."""
        
        # 1. Snapshot securely
        async with self.synapse_lock:
            # We must serialise the items to pass to thread, 
            # OR pass the dict and iterate carefully?
            # Safer to take values() likely.
            snapshot = list(self.synapses.values())
            
        def _save(syn_list):
            try:
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(self.synapse_path), exist_ok=True)
                
                with open(self.synapse_path, 'w') as f:
                    for synapse in syn_list:
                        rec = {
                            'neuron_a_id': synapse.neuron_a_id,
                            'neuron_b_id': synapse.neuron_b_id,
                            'strength': synapse.strength,
                            'fire_count': synapse.fire_count,
                            'success_count': synapse.success_count,
                        }
                        if synapse.last_fired:
                            rec['last_fired'] = synapse.last_fired.isoformat()
                        f.write(json.dumps(rec) + "\n")
            except Exception as e:
                logger.error(f"Error saving synapses: {e}")
                
        await self._run_in_thread(_save, snapshot)

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
