"""
Multi-Agent HAIM – Multi-agent system with shared HAIM memory
==============================================================
Demonstrates "collective consciousness" across multiple agents.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from ..core.engine import HAIMEngine
from ..core.exceptions import AgentNotFoundError


class MultiAgentHAIM:
    """
    Multi-agent system with shared HAIM memory
    Demonstrates "collective consciousness"
    """

    def __init__(self, num_agents: int = 3):
        self.agents = {}  # agent_id -> HAIMEngine
        self.shared_memory = HAIMEngine(dimension=16384)

        # Initialize agents with shared memory
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            self.agents[agent_id] = {
                "haim": self.shared_memory,  # All share same memory
                "role": self._get_agent_role(agent_id)
            }

    def _get_agent_role(self, agent_id: str) -> str:
        """Define agent roles"""
        roles = {
            "agent_0": "Research Agent",
            "agent_1": "Coding Agent",
            "agent_2": "Writing Agent"
        }
        return roles.get(agent_id, "General Agent")

    async def agent_learn(
        self,
        agent_id: str,
        content: str,
        metadata: dict = None
    ) -> str:
        """
        Agent stores memory in shared HAIM
        All agents can access this memory
        """
        if agent_id not in self.agents:
            raise AgentNotFoundError(agent_id)

        # Store in shared memory
        node_id = await self.shared_memory.store(content, metadata)

        # Update metadata with agent info
        node = await self.shared_memory.tier_manager.get_memory(node_id)
        if node:
            node.metadata = node.metadata or {}
            node.metadata["learned_by"] = agent_id
            node.metadata["agent_role"] = self.agents[agent_id]["role"]
            node.metadata["timestamp"] = datetime.now().isoformat()

        return node_id

    async def agent_recall(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Agent recalls memory from shared HAIM
        Can access memories learned by ANY agent
        """
        if agent_id not in self.agents:
            raise AgentNotFoundError(agent_id)

        # Query shared memory
        results = await self.shared_memory.query(query, top_k=top_k)

        # Enrich with agent context
        enriched = []
        for node_id, similarity in results:
            node = await self.shared_memory.tier_manager.get_memory(node_id)
            if node:
                enriched.append({
                    "node_id": node_id,
                    "content": node.content,
                    "similarity": similarity,
                    "metadata": node.metadata,
                    "learned_by": node.metadata.get("learned_by", "unknown"),
                    "agent_role": node.metadata.get("agent_role", "unknown")
                })

        return enriched

    async def cross_agent_learning(
        self,
        concept_a: str,
        concept_b: str,
        agent_id: str,
        success: bool = True
    ):
        """
        Strengthen connection between concepts across agents
        When ANY agent fires this connection, ALL agents benefit
        """
        if agent_id not in self.agents:
            raise AgentNotFoundError(agent_id)

        # Map concepts to memory IDs using holographic similarity
        mem_id_a = await self._concept_to_memory_id(concept_a)
        mem_id_b = await self._concept_to_memory_id(concept_b)

        if mem_id_a and mem_id_b:
            # Schedule binding in the background
            self._schedule_async_task(
                self.shared_memory.bind_memories(mem_id_a, mem_id_b, success=success)
            )

    async def _concept_to_memory_id(self, concept: str, min_similarity: float = 0.3) -> Optional[str]:
        """
        Map a concept string to the best matching memory ID.
        Uses the engine's vector search query() instead of manual O(N) scan.

        Args:
            concept: The concept string to search for
            min_similarity: Minimum similarity threshold for a match

        Returns:
            The memory ID if found with sufficient similarity, else None.
        """
        # Use the engine's query() method which uses optimized vector search
        # instead of manual linear scan through all HOT tier nodes
        results = await self.shared_memory.query(concept, top_k=1)

        if results:
            node_id, similarity = results[0]
            if similarity >= min_similarity:
                return node_id

        return None

    def _schedule_async_task(self, coro):
        """Schedule an async coroutine to run, handling the event loop appropriately."""
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, create a task
            loop.create_task(coro)
        except RuntimeError:
            # No running loop, run synchronously (for demo/testing purposes)
            try:
                asyncio.run(coro)
            except Exception:
                pass  # Silently fail in demo mode

    async def collective_orch_or(
        self,
        agent_id: str,
        query: str,
        max_collapse: int = 3
    ) -> List[Dict]:
        """
        Agent performs Orch OR on shared memories
        Collapses superposition based on collective free energy
        """
        if agent_id not in self.agents:
            raise AgentNotFoundError(agent_id)

        collapsed = await self.shared_memory.orchestrate_orch_or(max_collapse=max_collapse)

        # Enrich with agent context
        result = []
        for node in collapsed:
            result.append({
                "content": node.content,
                "free_energy_score": getattr(node, 'epistemic_value', 0.0),
                "metadata": node.metadata,
                "collapsed_by": agent_id,
                "agent_role": self.agents[agent_id]["role"]
            })

        return result

    async def demonstrate_collective_consciousness(self) -> Dict:
        """
        Demonstrate cross-agent learning
        Shows that when Agent A learns, Agent B knows
        """
        # Agent 0 (Research) learns something
        mem_0 = await self.agent_learn(
            agent_id="agent_0",
            content="MnemoCore Market Integrity Engine uses three signal groups: SURGE, FLOW, PATTERN",
            metadata={"category": "research", "importance": "high"}
        )

        # Agent 1 (Coding) learns something
        mem_1 = await self.agent_learn(
            agent_id="agent_1",
            content="HAIM uses hyperdimensional vectors with 10,000 dimensions",
            metadata={"category": "coding", "importance": "high"}
        )

        # Agent 2 (Writing) recalls BOTH memories
        recall_0 = await self.agent_recall(
            agent_id="agent_2",
            query="MnemoCore Engine",
            top_k=1
        )

        recall_1 = await self.agent_recall(
            agent_id="agent_2",
            query="HAIM dimensions",
            top_k=1
        )

        # Cross-agent learning: strengthen connection
        await self.cross_agent_learning(
            concept_a="MnemoCore Engine",
            concept_b="HAIM dimensions",
            agent_id="agent_2",
            success=True
        )

        return {
            "demonstration": "Collective Consciousness Demo",
            "agent_0_learned": mem_0,
            "agent_1_learned": mem_1,
            "agent_2_recalled_omega": recall_0,
            "agent_2_recalled_haim": recall_1,
            "cross_agent_connection": "Strengthened between Omega Engine and HAIM dimensions"
        }


__all__ = ["MultiAgentHAIM"]
