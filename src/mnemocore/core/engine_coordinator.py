"""
HAIMEngine Coordinator Module (Phase 6 Refactor)

Contains orchestration between sub-systems and event routing.
Handles advanced features like gap filling, recursive synthesis,
and retrieval feedback.

Separated from engine_core.py and engine_lifecycle.py to maintain
clear separation of concerns and keep files under 800 lines.
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timezone
import uuid
import numpy as np
from loguru import logger

# Core imports
from .config import HAIMConfig
from .binary_hdv import BinaryHDV
from .node import MemoryNode

# Phase 4.0 imports
from .bayesian_ltp import get_bayesian_updater
from .gap_detector import GapDetector
from .gap_filler import GapFiller, GapFillerConfig
from .synapse_index import SynapseIndex

# Phase 4.5 imports
from .recursive_synthesizer import RecursiveSynthesizer, SynthesizerConfig

if TYPE_CHECKING:
    from .tier_manager import TierManager


class EngineCoordinator:
    """
    Orchestration and event routing for HAIMEngine.

    This mixin class provides coordination between engine subsystems
    and handles advanced features.

    Responsibilities:
    - Gap filling orchestration
    - Recursive synthesis coordination
    - Retrieval feedback processing
    - Synaptic boost calculation
    - OR (Orchestration/Recall) operations
    - Negative feedback handling
    """

    # ==========================================================================
    # Gap Filling (Phase 4.0)
    # ==========================================================================

    async def enable_gap_filling(
        self,
        llm_integrator,
        config: Optional["GapFillerConfig"] = None,
    ) -> None:
        """
        Attach an LLM integrator to autonomously fill knowledge gaps.

        When enabled, the engine will detect gaps in knowledge during queries
        and use the LLM to generate appropriate fill-in content.

        Args:
            llm_integrator: HAIMLLMIntegrator instance with LLM access.
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

    # ==========================================================================
    # Recursive Synthesis (Phase 4.5)
    # ==========================================================================

    async def enable_recursive_synthesis(
        self,
        llm_call: Optional[Any] = None,
        config: Optional["SynthesizerConfig"] = None,
    ) -> None:
        """
        Enable Phase 4.5 Recursive Synthesis Engine.

        The recursive synthesizer can decompose complex queries and
        synthesize comprehensive answers from multiple memory sources.

        Args:
            llm_call: Optional callable for LLM-powered decomposition and synthesis.
                     Signature: (prompt: str) -> str.
            config: Optional SynthesizerConfig overrides.
        """
        self._recursive_synthesizer = RecursiveSynthesizer(
            engine=self,
            config=config or SynthesizerConfig(),
            llm_call=llm_call,
        )
        logger.info("Phase 4.5 RecursiveSynthesizer enabled.")

    # ==========================================================================
    # Retrieval Feedback (Phase 4.0)
    # ==========================================================================

    async def record_retrieval_feedback(
        self,
        node_id: str,
        helpful: bool,
        eig_signal: float = 1.0,
    ) -> None:
        """
        Record whether a retrieved memory was useful.

        Phase 4.0: feeds the Bayesian LTP updater for the node.
        This feedback loop helps the engine learn which memories are
        actually useful in practice.

        Args:
            node_id: The memory node that was retrieved.
            helpful: Was the retrieval actually useful?
            eig_signal: Strength of evidence (0-1).
        """
        node = await self.tier_manager.get_memory(node_id)
        if node:
            updater = get_bayesian_updater()
            updater.observe_node_retrieval(node, helpful=helpful, eig_signal=eig_signal)

    async def register_negative_feedback(self, query_text: str) -> None:
        """
        Signal that a recent query was not adequately answered.
        Creates a high-priority gap record for LLM gap-filling.

        This is used when the system recognizes that a query failed
        to retrieve useful information, creating an opportunity for
        proactive knowledge acquisition.
        """
        await self.gap_detector.register_negative_feedback(query_text)

    # ==========================================================================
    # Synaptic Operations
    # ==========================================================================

    async def get_node_boost(self, node_id: str) -> float:
        """
        Compute synaptic boost for scoring.

        Phase 4.0: O(k) via SynapseIndex (was O(k) before but with lock overhead).

        The boost factor represents how well-connected a memory is,
        which influences its relevance score during queries.

        Args:
            node_id: The memory node ID to compute boost for.

        Returns:
            The boost factor (>= 1.0, where higher means more connected).
        """
        return self._synapse_index.boost(node_id)

    # ==========================================================================
    # OR (Orchestration/Recall) Operations
    # ==========================================================================

    async def orchestrate_orch_or(self, max_collapse: int = 3) -> List[MemoryNode]:
        """
        Collapse active HOT-tier superposition by a free-energy proxy.

        This method implements a simplified version of the Active Inference
        Free Energy Principle, selecting nodes that should be consolidated
        based on their stability and importance.

        The score combines:
        - LTP (long-term stability): 60% weight
        - Epistemic value (novelty): 30% weight
        - Access count (usage evidence): 10% weight (log-scaled)

        Args:
            max_collapse: Maximum number of nodes to return.

        Returns:
            List of top MemoryNodes selected for consolidation.
        """
        async with self.tier_manager.lock:
            active_nodes = list(self.tier_manager.hot.values())
        if not active_nodes or max_collapse <= 0:
            return []

        def score(node: MemoryNode) -> float:
            ltp = float(getattr(node, "ltp_strength", 0.0))
            epistemic = float(getattr(node, "epistemic_value", 0.0))
            access = float(getattr(node, "access_count", 0))
            return (0.6 * ltp) + (0.3 * epistemic) + (0.1 * np.log1p(access))

        return sorted(active_nodes, key=score, reverse=True)[:max_collapse]

    # ==========================================================================
    # Advanced Query Operations
    # ==========================================================================

    async def associative_query(
        self,
        seed_text: str,
        depth: int = 2,
        top_k: int = 5,
        min_similarity: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """
        Perform associative spreading activation query.

        This method starts with a seed query and then follows synaptic
        connections to find related memories, performing a breadth-first
        search through the memory network.

        Args:
            seed_text: The initial query text.
            depth: How many hops to follow in the association graph.
            top_k: How many results to return per level.
            min_similarity: Minimum similarity threshold for associations.

        Returns:
            List of dictionaries containing node_id, similarity, and hop_level.
        """
        # First, get seed results
        seed_results = await self.query(
            seed_text,
            top_k=top_k,
            associative_jump=False,
            track_gaps=False,
        )

        results = []
        visited = set()

        # Add seed results (hop 0)
        for node_id, similarity in seed_results:
            if similarity >= min_similarity and node_id not in visited:
                visited.add(node_id)
                results.append({
                    "node_id": node_id,
                    "similarity": float(similarity),
                    "hop_level": 0,
                })

        # Follow associations
        current_level_ids = [node_id for node_id, _ in seed_results]

        for hop in range(1, depth + 1):
            if not current_level_ids:
                break

            next_level_ids = []

            for seed_id in current_level_ids:
                # Get neighbors via synapse index
                neighbour_synapses = self._synapse_index.neighbours(seed_id)

                for syn in neighbour_synapses:
                    neighbor = (
                        syn.neuron_b_id if syn.neuron_a_id == seed_id else syn.neuron_a_id
                    )

                    if neighbor in visited:
                        continue

                    visited.add(neighbor)

                    # Get the memory to compute similarity
                    mem = await self.tier_manager.get_memory(neighbor)
                    if mem:
                        # Compute similarity from seed
                        seed_mem = await self.tier_manager.get_memory(seed_id)
                        if seed_mem:
                            similarity = seed_mem.hdv.similarity(mem.hdv)

                            if similarity >= min_similarity:
                                results.append({
                                    "node_id": neighbor,
                                    "similarity": float(similarity),
                                    "hop_level": hop,
                                })
                                next_level_ids.append(neighbor)

            current_level_ids = next_level_ids

        # Sort by combined score (similarity decreases with hop level)
        for r in results:
            r["score"] = r["similarity"] * (0.7 ** r["hop_level"])

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k * depth]

    async def temporal_query(
        self,
        query_text: str,
        time_range: tuple[datetime, datetime],
        top_k: int = 5,
    ) -> List[tuple[str, float]]:
        """
        Query memories within a specific time range.

        This is useful for finding memories from a specific period,
        such as "what did I work on last week?"

        Args:
            query_text: The query text.
            time_range: (start, end) datetime tuple defining the time window.
            top_k: Maximum number of results to return.

        Returns:
            List of (node_id, similarity) tuples.
        """
        return await self.query(
            query_text,
            top_k=top_k,
            time_range=time_range,
            associative_jump=False,
            track_gaps=False,
        )

    async def find_related_by_content(
        self,
        node_id: str,
        top_k: int = 5,
    ) -> List[tuple[str, float]]:
        """
        Find memories related to a specific memory by content similarity.

        This is useful for finding "what else is like this?" given a
        specific memory node.

        Args:
            node_id: The ID of the reference memory.
            top_k: Maximum number of results to return.

        Returns:
            List of (node_id, similarity) tuples.
        """
        node = await self.tier_manager.get_memory(node_id)
        if not node:
            return []

        # Use the node's content as the query
        return await self.query(
            node.content,
            top_k=top_k + 1,  # +1 because the node itself will match
            associative_jump=False,
            track_gaps=False,
        )

    # ==========================================================================
    # Batch Operations
    # ==========================================================================

    async def batch_store(
        self,
        contents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        goal_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[str]:
        """
        Store multiple memories in a batch.

        This is more efficient than calling store() multiple times
        as it can batch certain operations.

        Args:
            contents: List of content strings to store.
            metadatas: Optional list of metadata dictionaries (one per content).
            goal_id: Optional goal identifier for all memories.
            project_id: Optional project identifier for all memories.

        Returns:
            List of stored memory node IDs.
        """
        if not contents:
            return []

        if metadatas is None:
            metadatas = [None] * len(contents)
        elif len(metadatas) != len(contents):
            raise ValueError("metadatas list length must match contents list length")

        node_ids = []
        for content, metadata in zip(contents, metadatas):
            node_id = await self.store(
                content,
                metadata=metadata,
                goal_id=goal_id,
                project_id=project_id,
            )
            node_ids.append(node_id)

        return node_ids

    async def batch_get_memories(
        self,
        node_ids: List[str],
    ) -> List[Optional[MemoryNode]]:
        """
        Retrieve multiple memories in a batch.

        Args:
            node_ids: List of memory node IDs to retrieve.

        Returns:
            List of MemoryNode objects (None for missing nodes).
        """
        return await self.tier_manager.get_memories_batch(node_ids)

    # ==========================================================================
    # Memory Export/Import
    # ==========================================================================

    async def export_memories(
        self,
        output_path: str,
        time_range: Optional[tuple[datetime, datetime]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Export memories to a JSONL file.

        Args:
            output_path: Path to the output file.
            time_range: Optional time range to filter memories.
            metadata_filter: Optional metadata filter.

        Returns:
            Number of memories exported.
        """
        import json
        import functools

        # Get memories matching filters
        # This is a simplified export - in production you'd want streaming
        exported = 0

        # Query for matching memories
        # For simplicity, we use a broad query and filter results
        all_ids = []
        async with self.tier_manager.lock:
            all_ids.extend(list(self.tier_manager.hot.keys()))

        # Add WARM tier IDs if available
        if hasattr(self.tier_manager, 'warm') and self.tier_manager.warm:
            async with self.tier_manager.warm.lock:
                all_ids.extend(list(self.tier_manager.warm.cache.keys()))

        def _write_export():
            nonlocal exported
            with open(output_path, 'w', encoding='utf-8') as f:
                for node_id in all_ids:
                    mem = asyncio.run(self.tier_manager.get_memory(node_id))
                    if not mem:
                        continue

                    # Apply time range filter
                    if time_range:
                        start, end = time_range
                        if mem.created_at < start or mem.created_at > end:
                            continue

                    # Apply metadata filter
                    if metadata_filter:
                        mem_meta = mem.metadata or {}
                        match = True
                        for k, v in metadata_filter.items():
                            if mem_meta.get(k) != v:
                                match = False
                                break
                        if not match:
                            continue

                    # Write record
                    record = {
                        'id': mem.id,
                        'content': mem.content,
                        'metadata': mem.metadata,
                        'created_at': mem.created_at.isoformat(),
                        'previous_id': mem.previous_id,
                        'ltp_strength': getattr(mem, 'ltp_strength', None),
                        'epistemic_value': getattr(mem, 'epistemic_value', None),
                    }
                    f.write(json.dumps(record) + '\n')
                    exported += 1

        await functools._run_in_thread(self._run_in_thread, _write_export)
        logger.info(f"Exported {exported} memories to {output_path}")
        return exported

    # ==========================================================================
    # Debugging and Diagnostics
    # ==========================================================================

    async def get_synaptic_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5,
    ) -> Optional[List[str]]:
        """
        Find a synaptic path between two memories using BFS.

        This is useful for understanding how two memories are connected.

        Args:
            start_id: Starting memory node ID.
            end_id: Target memory node ID.
            max_depth: Maximum path length to search.

        Returns:
            List of node IDs forming the path, or None if no path found.
        """
        if start_id == end_id:
            return [start_id]

        from collections import deque

        queue = deque([(start_id, [start_id])])
        visited = {start_id}

        while queue:
            current_id, path = queue.popleft()

            if len(path) > max_depth:
                continue

            # Get neighbors
            neighbour_synapses = self._synapse_index.neighbours(current_id)

            for syn in neighbour_synapses:
                neighbor = (
                    syn.neuron_b_id if syn.neuron_a_id == current_id else syn.neuron_a_id
                )

                if neighbor == end_id:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    async def get_memory_connections(
        self,
        node_id: str,
    ) -> Dict[str, Any]:
        """
        Get detailed information about a memory's synaptic connections.

        Args:
            node_id: The memory node ID.

        Returns:
            Dictionary with connection details including
            inbound, outbound, and connection strengths.
        """
        node = await self.tier_manager.get_memory(node_id)
        if not node:
            return {"error": "Memory not found"}

        neighbour_synapses = self._synapse_index.neighbours(node_id)

        connections = {
            "node_id": node_id,
            "total_connections": len(neighbour_synapses),
            "inbound": [],
            "outbound": [],
            "bidirectional": [],
        }

        for syn in neighbour_synapses:
            strength = syn.get_current_strength()
            if syn.neuron_a_id == node_id:
                if syn.neuron_b_id == node_id:
                    connections["bidirectional"].append({
                        "node_id": syn.neuron_b_id,
                        "strength": strength,
                    })
                else:
                    connections["outbound"].append({
                        "node_id": syn.neuron_b_id,
                        "strength": strength,
                    })
            else:
                connections["inbound"].append({
                    "node_id": syn.neuron_a_id,
                    "strength": strength,
                })

        return connections

    # ==========================================================================
    # Memory Analysis
    # ==========================================================================

    async def analyze_memory_clusters(
        self,
        sample_size: int = 100,
        min_cluster_size: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Analyze memories to find clusters of related content.

        This uses a simple clustering approach based on synaptic
        connectivity density.

        Args:
            sample_size: Maximum number of HOT memories to analyze.
            min_cluster_size: Minimum cluster size to report.

        Returns:
            List of cluster descriptions with member IDs.
        """
        # Get sample of HOT memories
        async with self.tier_manager.lock:
            hot_ids = list(self.tier_manager.hot.keys())[:sample_size]

        if not hot_ids:
            return []

        # Build adjacency map
        adjacency = {nid: set() for nid in hot_ids}
        for nid in hot_ids:
            neighbour_synapses = self._synapse_index.neighbours(nid)
            for syn in neighbour_synapses:
                neighbor = (
                    syn.neuron_b_id if syn.neuron_a_id == nid else syn.neuron_a_id
                )
                if neighbor in adjacency:
                    adjacency[nid].add(neighbor)
                    adjacency[neighbor].add(nid)

        # Find connected components (clusters)
        visited = set()
        clusters = []

        def dfs(node_id, cluster):
            cluster.append(node_id)
            visited.add(node_id)
            for neighbor in adjacency[node_id]:
                if neighbor not in visited:
                    dfs(neighbor, cluster)

        for nid in hot_ids:
            if nid not in visited:
                cluster = []
                dfs(nid, cluster)
                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)

        # Get representative content for each cluster
        results = []
        for cluster in clusters:
            # Get a sample memory from the cluster
            sample_mem = await self.tier_manager.get_memory(cluster[0])
            results.append({
                "size": len(cluster),
                "member_ids": cluster,
                "sample_content": sample_mem.content[:100] if sample_mem else None,
            })

        # Sort by size descending
        results.sort(key=lambda x: x["size"], reverse=True)
        return results

    # ==========================================================================
    # Phase 5.2: Association Engine ("Subtle Thoughts")
    # ==========================================================================

    async def generate_subtle_thoughts(self, agent_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Generate "subtle thoughts" (associations) for an agent based on their active
        working memory context and recent episodic history.

        Returns a list of CandidateAssociation dictionaries.
        """
        if not self.episodic_store and not self.working_memory:
            return []

        context_texts = []
        source_episode_ids = []

        # 1. Gather context from WM
        if self.working_memory:
            wm_state = self.working_memory.get_state(agent_id)
            if wm_state and wm_state.items:
                # Use top 3 most important WM items as seed
                top_items = sorted(wm_state.items, key=lambda x: x.importance, reverse=True)[:3]
                for item in top_items:
                    context_texts.append(item.content)

        # 2. Gather context from recent episodes
        if self.episodic_store:
            recent_episodes = self.episodic_store.get_recent(agent_id, limit=2)
            for ep in recent_episodes:
                source_episode_ids.append(ep.id)
                if ep.goal:
                    context_texts.append(f"Goal: {ep.goal}")
                # Add content from last 2 events
                for ev in ep.events[-2:]:
                    context_texts.append(ev.content)

        if not context_texts:
            return []

        # Combine context to form a query
        unified_query = " ".join(context_texts)

        # 3. Perform a semantic query (k-NN) to find related concepts/memories
        search_results = await self.query(unified_query, top_k=limit + 3)

        associations = []

        for mem_id, score in search_results:
            if score < 0.2:
                continue

            node = await self.get_memory(mem_id)
            if not node:
                continue

            associations.append({
                "id": f"assoc_{uuid.uuid4().hex[:8]}",
                "agent_id": agent_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_episode_ids": source_episode_ids,
                "related_concept_ids": [mem_id],
                "suggestion_text": node.content,
                "confidence": float(score),
                "tier": getattr(node, "tier", "unknown")
            })

            if len(associations) >= limit:
                break

        return associations

    # ==========================================================================
    # Conceptual Layer Proxy Methods
    # ==========================================================================

    async def define_concept(self, name: str, attributes: Dict[str, str]):
        await self._run_in_thread(self.soul.store_concept, name, attributes)

    async def reason_by_analogy(self, src: str, val: str, tgt: str):
        return await self._run_in_thread(self.soul.solve_analogy, src, val, tgt)

    async def cross_domain_inference(self, src: str, tgt: str, pat: str):
        return await self._run_in_thread(self.soul.solve_analogy, src, pat, tgt)

    async def inspect_concept(self, name: str, attr: str):
        return await self._run_in_thread(self.soul.extract_attribute, name, attr)

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    async def _run_in_thread(self, func, *args, **kwargs):
        """Run blocking function in thread pool."""
        import functools
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
