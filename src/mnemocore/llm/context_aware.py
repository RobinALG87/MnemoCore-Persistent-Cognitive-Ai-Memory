"""
Context-Aware LLM Integrator – Phase 6.0
=========================================
Context-Aware LLM Integration with Context Window Prioritization.

Enhances HAIMLLMIntegrator with intelligent memory ranking for optimal
LLM context window utilization.

Features:
- Token-aware memory ranking
- Semantic chunk splitting for long memories
- Model-specific token counting with tiktoken
- Budget-aware context selection
- RAG-ready context building
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from .config import LLMConfig
from .factory import LLMClientFactory
from .integrator import HAIMLLMIntegrator


class ContextAwareLLMIntegrator:
    """
    Phase 6.0: Context-Aware LLM Integration with Context Window Prioritization.

    Enhances HAIMLLMIntegrator with intelligent memory ranking for optimal
    LLM context window utilization.

    Features:
    - Token-aware memory ranking
    - Semantic chunk splitting for long memories
    - Model-specific token counting with tiktoken
    - Budget-aware context selection
    - RAG-ready context building

    Usage::

        integrator = ContextAwareLLMIntegrator(haim_engine, llm_config)
        response = integrator.query_with_optimized_context(
            "What are the key insights from recent discussions?",
            token_budget=8000
        )
    """

    def __init__(
        self,
        haim_engine,
        llm_client=None,
        llm_config: Optional[LLMConfig] = None,
        context_budget: Optional[int] = None,
        enable_chunking: bool = True,
    ):
        """
        Initialize the context-aware integrator.

        Args:
            haim_engine: HAIM engine instance
            llm_client: Optional LLM client
            llm_config: LLM configuration
            context_budget: Optional token budget override
            enable_chunking: Whether to enable semantic chunking
        """
        # Initialize base integrator
        self.base_integrator = HAIMLLMIntegrator(haim_engine, llm_client, llm_config)
        self.haim = haim_engine
        self.config = llm_config or LLMConfig.mock()

        # Import context optimizer
        from ..cognitive.context_optimizer import (
            ContextWindowPrioritizer,
            ContextBuilder,
            ChunkConfig,
        )

        # Setup prioritizer
        model_name = self.config.model or "gpt-4o"
        self.prioritizer = ContextWindowPrioritizer(
            model_name=model_name,
            token_budget=context_budget,
            chunk_config=ChunkConfig() if enable_chunking else None,
        )

        # Setup context builder
        self.context_builder = ContextBuilder(self.prioritizer)

    def query_with_optimized_context(
        self,
        query: str,
        memories: Optional[List] = None,
        token_budget: Optional[int] = None,
        top_k: int = 20,
    ) -> Dict:
        """
        Query with context-optimized memory selection.

        Args:
            query: The query string
            memories: Optional list of memories (auto-retrieved if None)
            token_budget: Optional token budget override
            top_k: Number of memories to retrieve if auto-retrieving

        Returns:
            Dict with selected memories, context string, and LLM response
        """
        # Retrieve memories if not provided
        if memories is None:
            from ..core.node import MemoryNode
            # Query HAIM for relevant memories
            results = self.haim.query(query, top_k=top_k)
            memories = []
            for node_id, similarity in results:
                node = self.haim.tier_manager.get_memory(node_id)
                if node:
                    memories.append(node)

        # Rank and select memories within budget
        budget = token_budget or self.prioritizer.token_budget
        optimization_result = self.prioritizer.select(
            memories=memories,
            query=query,
            promote_diversity=True,
        )

        # Build context string
        context_str = self.context_builder.build_rag_context(
            memories=memories,
            query=query,
            max_tokens=budget,
        )

        # Build final prompt
        prompt = f"{context_str}\n\nQuestion: {query}"

        # Call LLM
        response = self.base_integrator._call_llm(prompt)

        return {
            "query": query,
            "context": context_str,
            "response": response,
            "selected_memories": [
                {
                    "id": m.original_id,
                    "content": m.content[:200] + "..." if len(m.content) > 200 else m.content,
                    "score": m.score,
                    "tokens": m.token_count,
                }
                for m in optimization_result.ranked_memories
            ],
            "stats": {
                "total_tokens_used": optimization_result.total_tokens,
                "remaining_budget": optimization_result.remaining_budget,
                "memories_selected": len(optimization_result.ranked_memories),
                "coverage": optimization_result.coverage_stats,
            }
        }

    def get_ranked_memories(
        self,
        memories: List,
        query: Optional[str] = None,
        token_budget: Optional[int] = None,
    ) -> List[Dict]:
        """
        Get ranked memories without calling the LLM.

        Args:
            memories: List of memories to rank
            query: Optional query for relevance
            token_budget: Optional token budget

        Returns:
            List of ranked memory dictionaries
        """
        budget = token_budget or self.prioritizer.token_budget
        ranked = self.prioritizer.rank_with_budget(memories, budget, query)

        return [
            {
                "id": r.original_id,
                "content": r.content,
                "score": r.score,
                "tokens": r.token_count,
                "relevance": r.relevance,
                "importance": r.importance,
                "recency_weight": r.recency_weight,
            }
            for r in ranked
        ]

    @classmethod
    def from_config(cls, haim_engine, llm_config: LLMConfig, **kwargs) -> 'ContextAwareLLMIntegrator':
        """Create integrator from LLM configuration."""
        client = LLMClientFactory.create_client(llm_config)
        return cls(haim_engine=haim_engine, llm_client=client, llm_config=llm_config, **kwargs)


__all__ = ["ContextAwareLLMIntegrator"]
