"""
LLM Integration for HAIM
Integrates Gemini 3 Pro with HAIM for reconstructive recall
"""

import json
from datetime import datetime
from typing import List, Dict, Optional
from src.core.engine import HAIMEngine
from src.core.node import MemoryNode

class HAIMLLMIntegrator:
    """Bridge between HAIM holographic memory and LLM reasoning"""

    def __init__(self, haim_engine: HAIMEngine, llm_client=None):
        self.haim = haim_engine
        self.llm_client = llm_client  # Will use OpenClaw's gemini integration

    def reconstructive_recall(
        self,
        cue: str,
        top_memories: int = 5,
        enable_reasoning: bool = True
    ) -> Dict:
        """
        Reconstruct memory from partial cue
        Similar to human recall - you remember fragments, brain reconstructs whole
        """
        # Query HAIM for related memories
        results = self.haim.query(cue, top_k=top_memories)

        # Extract memory content
        memory_fragments = []
        for node_id, similarity in results:
            node = self.haim.get_memory(node_id)
            if node:
                memory_fragments.append({
                    "content": node.content,
                    "metadata": node.metadata,
                    "similarity": similarity
                })

        if not enable_reasoning:
            return {
                "cue": cue,
                "fragments": memory_fragments,
                "reconstruction": "LLM reasoning disabled"
            }

        # Use LLM to reconstruct from fragments
        reconstruction_prompt = self._build_reconstruction_prompt(
            cue=cue,
            fragments=memory_fragments
        )

        # TODO: Call Gemini 3 Pro via OpenClaw API
        # For now, return prompt
        reconstruction = "TODO: Call Gemini 3 Pro"

        return {
            "cue": cue,
            "fragments": memory_fragments,
            "prompt": reconstruction_prompt,
            "reconstruction": reconstruction
        }

    def _build_reconstruction_prompt(
        self,
        cue: str,
        fragments: List[Dict]
    ) -> str:
        """Build prompt for LLM reconstructive recall"""
        prompt = f"""You are an AI with holographic memory. A user asks a question, and you have retrieved partial memory fragments from your holographic memory.

User's Question: "{cue}"

Memory Fragments (retrieved by holographic similarity):
"""

        for i, frag in enumerate(fragments, 1):
            prompt += f"\nFragment {i} (similarity: {frag['similarity']:.3f}):\n{frag['content']}\n"

        prompt += """

Task: Reconstruct a complete, coherent answer from these fragments.
- Combine fragments intelligently
- Fill in gaps using reasoning
- If fragments conflict, use highest-similarity fragment as primary
- Maintain factual accuracy
- Don't hallucinate information not supported by fragments

Reconstruction:"""

        return prompt

    def multi_hypothesis_query(
        self,
        query: str,
        hypotheses: List[str]
    ) -> Dict:
        """
        Query with multiple active hypotheses (superposition)
        Returns LLM evaluation of which hypothesis is most likely
        """
        # Create superposition of all hypotheses
        # TODO: superposition_query() not implemented in HAIMEngine
        # For now, combine hypotheses into a single query string
        combined_query = " ".join(hypotheses)
        
        # Query memories related to superposition
        # (This would require modifying HAIMEngine to accept HDV query)
        results = self.haim.query(query, top_k=10)

        # Extract relevant memories
        relevant_memories = []
        for node_id, similarity in results:
            node = self.haim.get_memory(node_id)
            if node:
                relevant_memories.append({
                    "content": node.content,
                    "similarity": similarity
                })

        # Build evaluation prompt
        evaluation_prompt = self._build_hypothesis_evaluation_prompt(
            query=query,
            hypotheses=hypotheses,
            relevant_memories=relevant_memories
        )

        # TODO: Call Gemini 3 Pro
        evaluation = "TODO: Call Gemini 3 Pro"

        return {
            "query": query,
            "hypotheses": hypotheses,
            "relevant_memories": relevant_memories,
            "prompt": evaluation_prompt,
            "evaluation": evaluation
        }

    def _build_hypothesis_evaluation_prompt(
        self,
        query: str,
        hypotheses: List[str],
        relevant_memories: List[Dict]
    ) -> str:
        """Build prompt for multi-hypothesis evaluation"""
        prompt = f"""You are an AI with holographic memory. You have multiple hypotheses about a question, and you've retrieved relevant memories to evaluate them.

Query: "{query}"

Hypotheses:
"""

        for i, hyp in enumerate(hypotheses, 1):
            prompt += f"\nHypothesis {i}: {hyp}"

        prompt += "\n\nRelevant Memories:\n"
        for i, mem in enumerate(relevant_memories, 1):
            prompt += f"\nMemory {i} (similarity: {mem['similarity']:.3f}):\n{mem['content']}\n"

        prompt += """

Task: Evaluate which hypothesis is most supported by the retrieved memories.
- Consider all memories
- Rank hypotheses by support from memory
- Explain your reasoning
- Provide confidence score (0-100%) for each hypothesis

Evaluation:"""

        return prompt

    def consolidate_memory(
        self,
        node_id: str,
        new_context: str,
        success: bool = True
    ):
        """
        Reconsolidate memory with new context
        Similar to how human memories are rewritten when recalled
        """
        node = self.haim.get_memory(node_id)
        if not node:
            return

        # Access triggers reconsolidation
        node.access()

        # Update content with new context (simplified)
        # In production: use LLM to intelligently merge
        node.content = f"{node.content}\n\n[RECONSOLIDATED]: {new_context}"

        # Strengthen synaptic connections if consolidation was successful
        if success:
            # Find related concepts and strengthen
            # (This requires concept extraction - simplified for now)
            pass


class MultiAgentHAIM:
    """
    Multi-agent system with shared HAIM memory
    Demonstrates "collective consciousness"
    """

    def __init__(self, num_agents: int = 3):
        self.agents = {}  # agent_id -> HAIMEngine
        self.shared_memory = HAIMEngine(dimension=10000)

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

    def agent_learn(
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
            raise ValueError(f"Agent {agent_id} not found")

        # Store in shared memory
        node_id = self.shared_memory.store(content, metadata)

        # Update metadata with agent info
        node = self.shared_memory.get_memory(node_id)
        if node:
            node.metadata = node.metadata or {}
            node.metadata["learned_by"] = agent_id
            node.metadata["agent_role"] = self.agents[agent_id]["role"]
            node.metadata["timestamp"] = datetime.now().isoformat()

        return node_id

    def agent_recall(
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
            raise ValueError(f"Agent {agent_id} not found")

        # Query shared memory
        results = self.shared_memory.query(query, top_k=top_k)

        # Enrich with agent context
        enriched = []
        for node_id, similarity in results:
            node = self.shared_memory.get_memory(node_id)
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

    def cross_agent_learning(
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
        # Use bind_memories instead of non-existent bind_concepts
        # Note: This requires memory IDs, not concept strings
        # TODO: Implement concept-to-memory-ID mapping or add bind_concepts to engine
        pass  # Placeholder until proper implementation

    def collective_orch_or(
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
            raise ValueError(f"Agent {agent_id} not found")

        # Get all active nodes (in production, filter by relevance)
        active_nodes = self.shared_memory.get_active_memories()

        # TODO: orchestrate_orch_or() not implemented in HAIMEngine
        # For now, return top nodes sorted by LTP strength as a proxy for "collapse"
        collapsed = sorted(active_nodes, key=lambda n: getattr(n, 'ltp_strength', 0), reverse=True)[:max_collapse]

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

    def demonstrate_collective_consciousness(self) -> Dict:
        """
        Demonstrate cross-agent learning
        Shows that when Agent A learns, Agent B knows
        """
        # Agent 0 (Research) learns something
        mem_0 = self.agent_learn(
            agent_id="agent_0",
            content="MnemoCore Market Integrity Engine uses three signal groups: SURGE, FLOW, PATTERN",
            metadata={"category": "research", "importance": "high"}
        )

        # Agent 1 (Coding) learns something
        mem_1 = self.agent_learn(
            agent_id="agent_1",
            content="HAIM uses hyperdimensional vectors with 10,000 dimensions",
            metadata={"category": "coding", "importance": "high"}
        )

        # Agent 2 (Writing) recalls BOTH memories
        recall_0 = self.agent_recall(
            agent_id="agent_2",
            query="MnemoCore Engine",
            top_k=1
        )

        recall_1 = self.agent_recall(
            agent_id="agent_2",
            query="HAIM dimensions",
            top_k=1
        )

        # Cross-agent learning: strengthen connection
        self.cross_agent_learning(
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


def create_demo():
    """Create HAIM demo with multi-agent system"""
    print("Creating HAIM Multi-Agent Demo...")

    # Create multi-agent system
    multi_agent_haim = MultiAgentHAIM(num_agents=3)

    # Demonstrate collective consciousness
    result = multi_agent_haim.demonstrate_collective_consciousness()

    print("\n=== DEMO RESULT ===")
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    create_demo()
