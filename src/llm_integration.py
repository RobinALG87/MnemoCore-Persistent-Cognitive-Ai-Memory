"""
LLM Integration for HAIM
Multi-provider LLM support: OpenAI, OpenRouter, Anthropic, Google Gemini, and Local AI models
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from src.core.engine import HAIMEngine
from src.core.node import MemoryNode
from src.core.exceptions import (
    UnsupportedProviderError,
    AgentNotFoundError,
)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    GOOGLE_GEMINI = "google_gemini"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    CUSTOM = "custom"
    MOCK = "mock"


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: LLMProvider = LLMProvider.MOCK
    model: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # Provider-specific defaults
    @classmethod
    def openai(cls, model: str = "gpt-4", api_key: Optional[str] = None, **kwargs) -> 'LLMConfig':
        return cls(provider=LLMProvider.OPENAI, model=model, api_key=api_key, **kwargs)

    @classmethod
    def openrouter(cls, model: str = "anthropic/claude-3.5-sonnet", api_key: Optional[str] = None, **kwargs) -> 'LLMConfig':
        return cls(
            provider=LLMProvider.OPENROUTER,
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            extra_headers={"HTTP-Referer": "https://mnemocore.ai", "X-Title": "MnemoCore"},
            **kwargs
        )

    @classmethod
    def anthropic(cls, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None, **kwargs) -> 'LLMConfig':
        return cls(provider=LLMProvider.ANTHROPIC, model=model, api_key=api_key, **kwargs)

    @classmethod
    def google_gemini(cls, model: str = "gemini-1.5-pro", api_key: Optional[str] = None, **kwargs) -> 'LLMConfig':
        return cls(provider=LLMProvider.GOOGLE_GEMINI, model=model, api_key=api_key, **kwargs)

    @classmethod
    def ollama(cls, model: str = "llama3.1", base_url: str = "http://localhost:11434", **kwargs) -> 'LLMConfig':
        return cls(provider=LLMProvider.OLLAMA, model=model, base_url=base_url, **kwargs)

    @classmethod
    def lm_studio(cls, model: str = "local-model", base_url: str = "http://localhost:1234/v1", **kwargs) -> 'LLMConfig':
        return cls(provider=LLMProvider.LM_STUDIO, model=model, base_url=base_url, **kwargs)

    @classmethod
    def custom(cls, model: str, base_url: str, api_key: Optional[str] = None, **kwargs) -> 'LLMConfig':
        return cls(provider=LLMProvider.CUSTOM, model=model, base_url=base_url, api_key=api_key, **kwargs)

    @classmethod
    def mock(cls, **kwargs) -> 'LLMConfig':
        return cls(provider=LLMProvider.MOCK, **kwargs)


class LLMClientFactory:
    """Factory for creating LLM clients"""

    @staticmethod
    def create_client(config: LLMConfig) -> Any:
        """Create an LLM client based on configuration"""
        provider = config.provider

        if provider == LLMProvider.MOCK:
            return None

        if provider == LLMProvider.OPENAI:
            return LLMClientFactory._create_openai_client(config)

        if provider == LLMProvider.OPENROUTER:
            return LLMClientFactory._create_openrouter_client(config)

        if provider == LLMProvider.ANTHROPIC:
            return LLMClientFactory._create_anthropic_client(config)

        if provider == LLMProvider.GOOGLE_GEMINI:
            return LLMClientFactory._create_gemini_client(config)

        if provider == LLMProvider.OLLAMA:
            return LLMClientFactory._create_ollama_client(config)

        if provider == LLMProvider.LM_STUDIO:
            return LLMClientFactory._create_lm_studio_client(config)

        if provider == LLMProvider.CUSTOM:
            return LLMClientFactory._create_custom_client(config)

        supported = [p.value for p in LLMProvider]
        raise UnsupportedProviderError(str(provider.value), supported_providers=supported)

    @staticmethod
    def _create_openai_client(config: LLMConfig) -> Any:
        """Create OpenAI client"""
        try:
            from openai import OpenAI
            api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
            return OpenAI(api_key=api_key)
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            return None

    @staticmethod
    def _create_openrouter_client(config: LLMConfig) -> Any:
        """Create OpenRouter client (OpenAI-compatible)"""
        try:
            from openai import OpenAI
            api_key = config.api_key or os.environ.get("OPENROUTER_API_KEY")
            return OpenAI(
                base_url=config.base_url,
                api_key=api_key,
                default_headers=config.extra_headers
            )
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            return None

    @staticmethod
    def _create_anthropic_client(config: LLMConfig) -> Any:
        """Create Anthropic client"""
        try:
            import anthropic
            api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            return anthropic.Anthropic(api_key=api_key)
        except ImportError:
            logger.warning("anthropic package not installed. Install with: pip install anthropic")
            return None

    @staticmethod
    def _create_gemini_client(config: LLMConfig) -> Any:
        """Create Google Gemini client"""
        try:
            import google.generativeai as genai
            api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(config.model)
        except ImportError:
            logger.warning("google-generativeai package not installed. Install with: pip install google-generativeai")
            return None

    @staticmethod
    def _create_ollama_client(config: LLMConfig) -> Any:
        """Create Ollama client for local models"""
        try:
            from openai import OpenAI
            return OpenAI(base_url=config.base_url, api_key="ollama")
        except ImportError:
            # Fallback to direct HTTP calls
            return OllamaClient(base_url=config.base_url, model=config.model)

    @staticmethod
    def _create_lm_studio_client(config: LLMConfig) -> Any:
        """Create LM Studio client (OpenAI-compatible)"""
        try:
            from openai import OpenAI
            return OpenAI(base_url=config.base_url, api_key="lm-studio")
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            return None

    @staticmethod
    def _create_custom_client(config: LLMConfig) -> Any:
        """Create custom OpenAI-compatible client"""
        try:
            from openai import OpenAI
            return OpenAI(
                base_url=config.base_url,
                api_key=config.api_key or "custom"
            )
        except ImportError:
            logger.warning("openai package not installed. Install with: pip install openai")
            return None


class OllamaClient:
    """Fallback Ollama client using direct HTTP calls"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate response using Ollama API"""
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens}
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "")
        except urllib.error.URLError as e:
            return f"[Ollama Error: {str(e)}]"


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
            return f"[LLM Error: {str(e)}]"

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
        from src.core.binary_hdv import majority_bundle
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
            raise AgentNotFoundError(agent_id)

        # Store in shared memory
        node_id = self.shared_memory.store(content, metadata)

        # Update metadata with agent info
        node = self.shared_memory.tier_manager.get_memory(node_id)
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
            raise AgentNotFoundError(agent_id)

        # Query shared memory
        results = self.shared_memory.query(query, top_k=top_k)

        # Enrich with agent context
        enriched = []
        for node_id, similarity in results:
            node = self.shared_memory.tier_manager.get_memory(node_id)
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
        if agent_id not in self.agents:
            raise AgentNotFoundError(agent_id)

        # Map concepts to memory IDs using holographic similarity
        mem_id_a = self._concept_to_memory_id(concept_a)
        mem_id_b = self._concept_to_memory_id(concept_b)

        if mem_id_a and mem_id_b:
            # Schedule binding in the background
            self._schedule_async_task(
                self.shared_memory.bind_memories(mem_id_a, mem_id_b, success=success)
            )

    def _concept_to_memory_id(self, concept: str, min_similarity: float = 0.3) -> Optional[str]:
        """
        Map a concept string to the best matching memory ID.
        Uses holographic similarity to find the most relevant stored memory.
        Returns the memory ID if found with sufficient similarity, else None.
        """
        # Use synchronous encoding and search via tier manager for direct access
        query_vec = self.shared_memory.encode_content(concept)

        # Search in hot tier first (most recent/active memories)
        best_match_id = None
        best_similarity = 0.0

        # Check HOT tier
        for node_id, node in self.shared_memory.tier_manager.hot.items():
            sim = query_vec.similarity(node.hdv)
            if sim > best_similarity:
                best_similarity = sim
                best_match_id = node_id

        if best_similarity >= min_similarity:
            return best_match_id

        return None

    def _schedule_async_task(self, coro):
        """Schedule an async coroutine to run, handling the event loop appropriately."""
        import asyncio
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



class RLMIntegrator:
    """
    Phase 4.5: RLM (Recursive Language Models) Integrator.

    Bridges HAIMLLMIntegrator with the RecursiveSynthesizer to provide
    LLM-powered recursive memory queries.

    Usage::

        integrator = RLMIntegrator(llm_integrator)
        result = await integrator.rlm_query(
            "What do we know about X and how does it relate to Y?"
        )
        print(result["synthesis"])

    Without an LLM configured, falls back to heuristic decomposition
    and score-based synthesis.
    """

    def __init__(self, llm_integrator, config=None):
        from src.core.recursive_synthesizer import RecursiveSynthesizer, SynthesizerConfig
        self.llm_integrator = llm_integrator
        self.haim = llm_integrator.haim
        llm_call = None
        if llm_integrator.llm_client is not None:
            llm_call = llm_integrator._call_llm
        synth_config = config or SynthesizerConfig()
        self.synthesizer = RecursiveSynthesizer(
            engine=self.haim,
            config=synth_config,
            llm_call=llm_call,
        )

    async def rlm_query(self, query, context_text=None, project_id=None):
        """
        Execute a Phase 4.5 recursive memory query.

        Args:
            query:        The user question (can be complex/multi-topic).
            context_text: Optional large external text (Ripple environment).
            project_id:   Optional project scope for isolation masking.

        Returns:
            Dict: query, sub_queries, results, synthesis,
                  max_depth_hit, elapsed_ms, ripple_snippets, stats
        """
        from src.core.ripple_context import RippleContext
        ripple_ctx = None
        if context_text and context_text.strip():
            ripple_ctx = RippleContext(text=context_text, source_label="api_context")
        result = await self.synthesizer.synthesize(
            query=query,
            ripple_context=ripple_ctx,
            project_id=project_id,
        )
        return {
            "query": result.query,
            "sub_queries": result.sub_queries,
            "results": result.results,
            "synthesis": result.synthesis,
            "max_depth_hit": result.max_depth_hit,
            "elapsed_ms": result.total_elapsed_ms,
            "ripple_snippets": result.ripple_snippets,
            "stats": result.stats,
        }

    @classmethod
    def from_config(cls, haim_engine, llm_config, synth_config=None):
        """Create an RLMIntegrator directly from an LLMConfig."""
        llm_integrator = HAIMLLMIntegrator.from_config(haim_engine, llm_config)
        return cls(llm_integrator=llm_integrator, config=synth_config)


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
