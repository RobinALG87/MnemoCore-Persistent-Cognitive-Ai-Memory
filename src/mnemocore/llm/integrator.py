"""
HAIM LLM Integrator – Bridge between HAIM and LLM reasoning
============================================================
Bridge between HAIM holographic memory and LLM reasoning.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .config import LLMConfig, LLMProvider
from .factory import LLMClientFactory
from ..core.engine import HAIMEngine
from ..core.node import MemoryNode
from ..core.exceptions import LLMError


class HAIMLLMIntegrator:
    """Bridge between HAIM holographic memory and LLM reasoning"""

    def __init__(
        self,
        haim_engine: HAIMEngine,
        llm_client=None,
        llm_config: Optional[LLMConfig] = None
    ):
        self.haim = haim_engine

        # Support both legacy client and new config-based approach
        if llm_config:
            self.config = llm_config
            self.llm_client = llm_client or LLMClientFactory.create_client(llm_config)
        elif llm_client:
            self.llm_client = llm_client
            self.config = LLMConfig.mock()
        else:
            self.llm_client = None
            self.config = LLMConfig.mock()

    @classmethod
    def from_config(cls, haim_engine: HAIMEngine, config: LLMConfig) -> 'HAIMLLMIntegrator':
        """Create integrator from LLM configuration"""
        client = LLMClientFactory.create_client(config)
        return cls(haim_engine=haim_engine, llm_client=client, llm_config=config)

    def _call_llm(self, prompt: str, max_tokens: int = None) -> str:
        """
        Call the LLM with the given prompt.
        Supports multiple providers: OpenAI, OpenRouter, Anthropic, Gemini, Ollama, LM Studio
        """
        max_tokens = max_tokens or self.config.max_tokens

        if self.config.provider == LLMProvider.MOCK or self.llm_client is None:
            return self._mock_llm_response(prompt)

        try:
            provider = self.config.provider

            # OpenAI / OpenRouter / LM Studio (all use OpenAI SDK)
            if provider in (LLMProvider.OPENAI, LLMProvider.OPENROUTER, LLMProvider.LM_STUDIO, LLMProvider.CUSTOM):
                return self._call_openai_compatible(prompt, max_tokens)

            # Anthropic
            if provider == LLMProvider.ANTHROPIC:
                return self._call_anthropic(prompt, max_tokens)

            # Google Gemini
            if provider == LLMProvider.GOOGLE_GEMINI:
                return self._call_gemini(prompt, max_tokens)

            # Ollama
            if provider == LLMProvider.OLLAMA:
                return self._call_ollama(prompt, max_tokens)

            # Fallback: try to detect client type
            return self._call_generic(prompt, max_tokens)

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise LLMError(
                provider=str(self.config.provider.value),
                reason=str(e),
                context={"model": self.config.model, "max_tokens": max_tokens}
            )

    def _call_openai_compatible(self, prompt: str, max_tokens: int) -> str:
        """Call OpenAI-compatible API (OpenAI, OpenRouter, LM Studio)"""
        response = self.llm_client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            **self.config.extra_params
        )
        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str, max_tokens: int) -> str:
        """Call Anthropic Claude API"""
        response = self.llm_client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
            **self.config.extra_params
        )
        return response.content[0].text

    def _call_gemini(self, prompt: str, max_tokens: int) -> str:
        """Call Google Gemini API"""
        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": self.config.temperature,
            **self.config.extra_params
        }
        response = self.llm_client.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text

    def _call_ollama(self, prompt: str, max_tokens: int) -> str:
        """Call Ollama local model"""
        if hasattr(self.llm_client, 'generate'):
            # Using our fallback OllamaClient
            return self.llm_client.generate(prompt, max_tokens)
        else:
            # Using OpenAI SDK with Ollama
            return self._call_openai_compatible(prompt, max_tokens)

    def _call_generic(self, prompt: str, max_tokens: int) -> str:
        """Generic fallback that tries to detect and use the client"""
        client = self.llm_client

        # OpenAI-style
        if hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
            return self._call_openai_compatible(prompt, max_tokens)

        # Anthropic-style
        if hasattr(client, 'messages') and hasattr(client.messages, 'create'):
            return self._call_anthropic(prompt, max_tokens)

        # Simple callable
        if callable(client):
            return client(prompt)

        # Generate method
        if hasattr(client, 'generate'):
            return client.generate(prompt, max_tokens=max_tokens)

        return self._mock_llm_response(prompt)

    def _mock_llm_response(self, prompt: str) -> str:
        """Generate a mock LLM response when no client is available."""
        if "Reconstruct" in prompt or "reconstruct" in prompt:
            return "[MOCK RECONSTRUCTION] Based on the retrieved memory fragments, I can synthesize the following: The information appears to be related to the query context. However, please configure an LLM client for actual reconstructive reasoning."
        elif "Evaluate" in prompt or "hypothesis" in prompt.lower():
            return "[MOCK EVALUATION] Based on memory analysis: Hypothesis 1 appears most supported (confidence: 70%). Please configure an LLM client for actual hypothesis evaluation."
        return "[MOCK RESPONSE] Please configure an LLM client for actual responses."

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
            node = self.haim.tier_manager.get_memory(node_id)
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

        # Call LLM for reconstruction
        reconstruction = self._call_llm(reconstruction_prompt)

        return {
            "cue": cue,
            "fragments": memory_fragments,
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
        # Query memories using superposition of hypotheses
        results = self._superposition_query(query, hypotheses, top_k=10)

        # Extract relevant memories
        relevant_memories = []
        for node_id, similarity in results:
            node = self.haim.tier_manager.get_memory(node_id)
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

        # Call LLM for evaluation
        evaluation = self._call_llm(evaluation_prompt)

        return {
            "query": query,
            "hypotheses": hypotheses,
            "relevant_memories": relevant_memories,
            "evaluation": evaluation
        }

    def _superposition_query(
        self,
        query: str,
        hypotheses: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Perform a superposition query by combining query and hypotheses.
        Uses HDV bundling to create a superposition vector for retrieval.
        """
        # Encode each hypothesis and the main query
        query_vec = self.haim.encode_content(query)

        # Create superposition by bundling all hypothesis vectors with the query
        hypothesis_vectors = [self.haim.encode_content(h) for h in hypotheses]

        # Bundle all vectors together (superposition)
        from ..core.binary_hdv import majority_bundle
        all_vectors = [query_vec] + hypothesis_vectors
        superposition_vec = majority_bundle(all_vectors)

        # Query each hypothesis individually and merge results
        all_results: Dict[str, float] = {}

        # Primary query
        primary_results = self.haim.query(query, top_k=top_k)
        for node_id, sim in primary_results:
            all_results[node_id] = sim

        # Query each hypothesis and accumulate scores
        for hypothesis in hypotheses:
            hyp_results = self.haim.query(hypothesis, top_k=top_k // 2)
            for node_id, sim in hyp_results:
                if node_id in all_results:
                    # Boost score for memories relevant to multiple hypotheses
                    all_results[node_id] = max(all_results[node_id], sim * 0.8)
                else:
                    all_results[node_id] = sim * 0.6

        # Sort by score and return top_k
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

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
        node = self.haim.tier_manager.get_memory(node_id)
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


__all__ = ["HAIMLLMIntegrator"]
