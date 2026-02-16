from typing import List, Tuple, Dict, Optional, Union
import heapq
import numpy as np
import hashlib
import os
import json
import logging
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

    def __init__(self, dimension: int = 16384, persist_path: Optional[str] = None):
        self.config = get_config()
        # Allow override from init, but prefer config
        self.dimension = self.config.dimensionality
        
        # Core Components
        self.tier_manager = TierManager()
        self.binary_encoder = TextEncoder(self.dimension)
        
        self.synapses: Dict[Tuple[str, str], SynapticConnection] = {}

        # Conceptual Layer (VSA Soul)
        data_dir = self.config.paths.data_dir
        self.soul = ConceptualMemory(dimension=self.dimension, storage_dir=data_dir)

        # Legacy persistence path (for backward compat / migration)
        self.persist_path = persist_path or self.config.paths.memory_file
        self.synapse_path = self.config.paths.synapses_file

        # Load initial state
        self._load_legacy_if_needed()
        self._load_synapses()

        # Passive Subconscious Layer
        self.subconscious_queue: List[str] = []

        # Epistemic Drive
        self.epistemic_drive_active = True
        self.surprise_threshold = 0.7 

    def calculate_eig(self, candidate: Union[HDV, BinaryHDV], context: Union[HDV, BinaryHDV]) -> float:
        """
        Calculate Expected Information Gain (EIG).
        Proportional to novelty (distance) against the context.
        """
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

    def _current_context_vector(self, sample_n: int = 50) -> Union[HDV, BinaryHDV]:
        """Superpose a slice of working memory (HOT tier) into a single context vector."""
        # Get recent nodes from HOT tier
        recent_nodes = list(self.tier_manager.hot.values())[-sample_n:]
        
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
            return majority_bundle(vectors)
        else:
            # Legacy addition
            combined = np.zeros(self.dimension)
            for n in recent_nodes:
                if isinstance(n.hdv, HDV):
                    combined += n.hdv.vector
            hdv = HDV(dimension=self.dimension)
            hdv.vector = np.sign(combined)
            hdv.vector[hdv.vector == 0] = 1
            return hdv

    def store(self, content: str, metadata: dict = None, goal_id: str = None) -> str:
        """Store new memory with holographic encoding."""
        
        # 1. Encode Content
        if self.config.encoding.mode == "binary":
            content_vec = self.binary_encoder.encode(content)
        else:
            # Legacy encoding
            legacy_hdv = HDV(dimension=self.dimension)
            legacy_hdv.vector = self._legacy_encode_content_numpy(content)
            content_vec = legacy_hdv

        # 2. Bind Goal Context (if any)
        final_vec = content_vec
        if goal_id:
            if isinstance(content_vec, BinaryHDV):
                # Binary Goal Binding
                goal_vec = self.binary_encoder.encode(f"GOAL_CONTEXT_{goal_id}")
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
            ctx_vec = self._current_context_vector(sample_n=50)
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
        node_id = f"mem_{int(datetime.now().timestamp() * 1000)}"
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
        self.tier_manager.add_memory(node)

        # 6. Append to persistence log (Legacy/Backup)
        self._append_persisted(node)

        # 7. Subconscious Trigger
        self.subconscious_queue.append(node_id)
        self._background_dream(depth=1)

        return node_id

    def query(
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
        if self.config.encoding.mode == "binary":
            query_vec = self.binary_encoder.encode(query_text)
        else:
            query_vec = HDV(dimension=self.dimension)
            query_vec.vector = self._legacy_encode_content_numpy(query_text)

        # 1. Primary Search (Scan HOT + cached WARM?)
        # For Phase 3.0, we iterate over HOT nodes.
        nodes_to_search = list(self.tier_manager.hot.values())
        
        # TODO: Phase 3.5 Qdrant search for WARM/COLD
        
        scores: Dict[str, float] = {}
        for node in nodes_to_search:
            if isinstance(query_vec, BinaryHDV) and isinstance(node.hdv, BinaryHDV):
                sim = query_vec.similarity(node.hdv)
            elif isinstance(query_vec, HDV) and isinstance(node.hdv, HDV):
                sim = query_vec.cosine_similarity(node.hdv)
            else:
                sim = 0.0 # Mismatched types
            
            # Boost by synaptic health
            sim *= self.get_node_boost(node.id)
            scores[node.id] = sim

        # 2. Associative Spreading
        if associative_jump and self.synapses:
            augmented_scores = scores.copy()
            # Top seeds
            top_seeds = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            for seed_id, seed_score in top_seeds:
                if seed_score <= 0: continue
                
                # Check synapses
                for synapse_key, synapse in self.synapses.items():
                    if seed_id in synapse_key:
                        neighbor = synapse_key[1] if synapse_key[0] == seed_id else synapse_key[0]
                        # Retrieve neighbor if not in scores (could be in WARM)
                        if neighbor not in augmented_scores:
                             # Try to fetch from TierManager (might promote)
                             mem = self.tier_manager.get_memory(neighbor)
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

    def _background_dream(self, depth: int = 2):
        """Passive Subconscious - Strengthening synapses in idle cycles"""
        if not self.subconscious_queue:
            return
        
        stim_id = self.subconscious_queue.pop(0)
        stim_node = self.tier_manager.get_memory(stim_id) 
        if not stim_node: return

        potential_connections = self.query(stim_node.content, top_k=depth+1, associative_jump=False)
        
        for neighbor_id, similarity in potential_connections:
            if neighbor_id != stim_id and similarity > 0.15:
                self.bind_memories(stim_id, neighbor_id, success=True)

    def bind_memories(self, id_a: str, id_b: str, success: bool = True):
        """Bind two memories by ID."""
        # Validation
        if not self.tier_manager.get_memory(id_a) or not self.tier_manager.get_memory(id_b):
            return

        synapse_key = tuple(sorted([id_a, id_b]))
        if synapse_key not in self.synapses:
            self.synapses[synapse_key] = SynapticConnection(synapse_key[0], synapse_key[1])

        self.synapses[synapse_key].fire(success=success)
        self._save_synapses()

    def get_node_boost(self, node_id: str) -> float:
        boost = 1.0
        # Iterate over synapses is inefficient, but okay for Phase 3.0 prototype
        # Correct approach: Adjacency list
        for synapse_key, synapse in self.synapses.items():
            if node_id in synapse_key:
                boost *= (1.0 + synapse.get_current_strength())
        return boost

    def encode_content(self, content: str) -> Union[HDV, BinaryHDV]:
        """Encode text to HDV (Binary or Legacy Float)."""
        if self.config.encoding.mode == "binary":
            return self.binary_encoder.encode(content)
        else:
            hdv = HDV(dimension=self.dimension)
            hdv.vector = self._legacy_encode_content_numpy(content)
            return hdv

    def get_memory(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve memory via TierManager."""
        return self.tier_manager.get_memory(node_id)

    # --- Legacy Helpers ---

    def _legacy_encode_content_numpy(self, content: str) -> np.ndarray:
        """Original localized encoding logic for backward compatibility."""
        import re
        tokens = re.findall(r'\w+', content.lower())
        if not tokens:
            hash_val = int(hashlib.md5(content.encode('utf-8')).hexdigest(), 16)
            seed = hash_val % (2**32)
            return np.random.RandomState(seed).choice([-1, 1], size=self.dimension)

        combined = np.zeros(self.dimension)
        for t in tokens:
            t_hash = int(hashlib.md5(t.encode('utf-8')).hexdigest(), 16)
            t_seed = t_hash % (2**32)
            t_vec = np.random.RandomState(t_seed).choice([-1, 1], size=self.dimension)
            combined += t_vec

        v = np.sign(combined)
        v[v == 0] = np.random.RandomState(42).choice([-1, 1], size=np.sum(v == 0))
        return v.astype(int)

    def _load_legacy_if_needed(self):
        """Load from memory.jsonl into TierManager."""
        if not os.path.exists(self.persist_path):
            return
            
        logger.info(f"Loading legacy memory from {self.persist_path}")
        with open(self.persist_path, 'r', encoding='utf-8') as f:
            for line in f:
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
                    self.tier_manager.add_memory(node)
                    
                except Exception as e:
                    logger.warning(f"Failed to load record: {e}")

    def _load_synapses(self):
        if not os.path.exists(self.synapse_path): return
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
                    self.synapses[tuple(sorted([syn.neuron_a_id, syn.neuron_b_id]))] = syn
        except Exception as e:
            logger.error(f"Error loading synapses: {e}")

    def _save_synapses(self):
        """Save synapses to disk in JSONL format."""
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(self.synapse_path), exist_ok=True)
            
            with open(self.synapse_path, 'w') as f:
                for synapse_key, synapse in self.synapses.items():
                    rec = {
                        'neuron_a_id': synapse.neuron_a_id,
                        'neuron_b_id': synapse.neuron_b_id,
                        'strength': synapse.base_strength,
                        'fire_count': synapse.fire_count,
                        'success_count': synapse.success_count,
                    }
                    if synapse.last_fired:
                        rec['last_fired'] = synapse.last_fired.isoformat()
                    f.write(json.dumps(rec) + "\n")
        except Exception as e:
            logger.error(f"Error saving synapses: {e}")

    def _append_persisted(self, node: MemoryNode):
        """Append-only log."""
        try:
            with open(self.persist_path, 'a', encoding='utf-8') as f:
                rec = {
                    'id': node.id,
                    'content': node.content,
                    'metadata': node.metadata,
                    'created_at': node.created_at.isoformat()
                }
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    # --- Conceptual Proxy ---
    def define_concept(self, name: str, attributes: Dict[str, str]):
        self.soul.store_concept(name, attributes)
        
    def reason_by_analogy(self, src: str, val: str, tgt: str):
        return self.soul.solve_analogy(src, val, tgt)
        
    def cross_domain_inference(self, src: str, tgt: str, pat: str):
        return self.soul.solve_analogy(src, pat, tgt)
        
    def inspect_concept(self, name: str, attr: str):
        return self.soul.extract_attribute(name, attr)
