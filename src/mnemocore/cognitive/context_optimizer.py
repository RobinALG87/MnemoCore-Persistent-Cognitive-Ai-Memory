"""
Context Window Prioritizer and Optimizer
=========================================
Phase 6: Intelligent memory ranking for LLM context window optimization.

This module provides:
- Token-aware memory ranking using multiple relevance signals
- Semantic chunk splitting for long memories
- Model-specific token counting with tiktoken
- Context window budget management
- Integration with LLM providers for optimal context utilization

Author: MnemoCore Infrastructure Team
Version: 1.0.0
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Tuple,
    Callable,
    Union,
    Iterator,
)
from enum import Enum
from collections import defaultdict

from loguru import logger

# Import memory models
from mnemocore.core.node import MemoryNode
from mnemocore.core.memory_model import (
    WorkingMemoryItem,
    Episode,
    SemanticConcept,
    Procedure,
)


class ModelProvider(Enum):
    """Supported LLM providers for token counting."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    META = "meta"
    MISTRAL = "mistral"
    GENERIC = "generic"


@dataclass
class ModelContextLimits:
    """Context window limits for common LLM models."""
    model_name: str
    max_tokens: int
    output_tokens: int = 2048
    provider: ModelProvider = ModelProvider.OPENAI

    @property
    def input_budget(self) -> int:
        """Tokens available for input (system + user context)."""
        return self.max_tokens - self.output_tokens


# Common model context limits
MODEL_LIMITS = {
    # OpenAI Models
    "gpt-4o": ModelContextLimits("gpt-4o", 128000, 4096, ModelProvider.OPENAI),
    "gpt-4o-mini": ModelContextLimits("gpt-4o-mini", 128000, 16384, ModelProvider.OPENAI),
    "gpt-4-turbo": ModelContextLimits("gpt-4-turbo", 128000, 4096, ModelProvider.OPENAI),
    "gpt-4": ModelContextLimits("gpt-4", 8192, 4096, ModelProvider.OPENAI),
    "gpt-3.5-turbo": ModelContextLimits("gpt-3.5-turbo", 16385, 4096, ModelProvider.OPENAI),
    "o1-preview": ModelContextLimits("o1-preview", 128000, 32768, ModelProvider.OPENAI),
    "o1-mini": ModelContextLimits("o1-mini", 128000, 65536, ModelProvider.OPENAI),

    # Anthropic Models
    "claude-3-5-sonnet-20241022": ModelContextLimits(
        "claude-3-5-sonnet-20241022", 200000, 8192, ModelProvider.ANTHROPIC
    ),
    "claude-3-5-sonnet-20240620": ModelContextLimits(
        "claude-3-5-sonnet-20240620", 200000, 8192, ModelProvider.ANTHROPIC
    ),
    "claude-3-opus-20240229": ModelContextLimits(
        "claude-3-opus-20240229", 200000, 4096, ModelProvider.ANTHROPIC
    ),
    "claude-3-sonnet-20240229": ModelContextLimits(
        "claude-3-sonnet-20240229", 200000, 4096, ModelProvider.ANTHROPIC
    ),
    "claude-3-haiku-20240307": ModelContextLimits(
        "claude-3-haiku-20240307", 200000, 4096, ModelProvider.ANTHROPIC
    ),

    # Google Models
    "gemini-1.5-pro": ModelContextLimits(
        "gemini-1.5-pro", 1000000, 8192, ModelProvider.GOOGLE
    ),
    "gemini-1.5-flash": ModelContextLimits(
        "gemini-1.5-flash", 1000000, 8192, ModelProvider.GOOGLE
    ),
    "gemini-pro": ModelContextLimits(
        "gemini-pro", 91728, 2048, ModelProvider.GOOGLE
    ),

    # Meta Models (via providers)
    "llama-3.1-405b": ModelContextLimits(
        "llama-3.1-405b", 131072, 4096, ModelProvider.META
    ),
    "llama-3.1-70b": ModelContextLimits(
        "llama-3.1-70b", 131072, 4096, ModelProvider.META
    ),
    "llama-3.1-8b": ModelContextLimits(
        "llama-3.1-8b", 131072, 4096, ModelProvider.META
    ),

    # Mistral Models
    "mistral-large": ModelContextLimits(
        "mistral-large", 128000, 4096, ModelProvider.MISTRAL
    ),
    "mixtral-8x7b": ModelContextLimits(
        "mixtral-8x7b", 32768, 4096, ModelProvider.MISTRAL
    ),
}


@dataclass
class RankedMemory:
    """A memory with its ranking score and metadata."""
    memory: Union[MemoryNode, WorkingMemoryItem, Episode, SemanticConcept, Procedure, Dict[str, Any]]
    score: float
    token_count: int
    relevance: float
    recency_weight: float
    importance: float
    chunk_index: int = 0
    total_chunks: int = 1
    original_id: str = ""

    @property
    def content(self) -> str:
        """Extract content from the memory."""
        if isinstance(self.memory, MemoryNode):
            return self.memory.content
        elif isinstance(self.memory, WorkingMemoryItem):
            return self.memory.content
        elif isinstance(self.memory, Episode):
            return f"{self.memory.goal or ''} {self.memory.context or ''}"
        elif isinstance(self.memory, SemanticConcept):
            return f"{self.memory.label}: {self.memory.description}"
        elif isinstance(self.memory, Procedure):
            return f"{self.memory.name}: {self.memory.description}"
        elif isinstance(self.memory, dict):
            return self.memory.get("content", str(self.memory))
        return str(self.memory)

    @property
    def memory_id(self) -> str:
        """Get unique ID for the memory."""
        if isinstance(self.memory, MemoryNode):
            return self.memory.id
        elif isinstance(self.memory, WorkingMemoryItem):
            return self.memory.id
        elif isinstance(self.memory, Episode):
            return self.memory.id
        elif isinstance(self.memory, SemanticConcept):
            return self.memory.id
        elif isinstance(self.memory, Procedure):
            return self.memory.id
        elif isinstance(self.memory, dict):
            return self.memory.get("id", str(id(self.memory)))
        return str(id(self.memory))


@dataclass
class ChunkConfig:
    """Configuration for semantic chunk splitting."""
    max_chunk_tokens: int = 512
    min_chunk_tokens: int = 128
    chunk_overlap_tokens: int = 64
    split_by_sentences: bool = True
    preserve_structure: bool = True
    semantic_threshold: float = 0.3  # Cosine similarity threshold for merging


@dataclass
class ScoringWeights:
    """Weights for the ranking formula."""
    relevance: float = 1.0
    recency: float = 0.5
    importance: float = 0.7
    token_efficiency: float = 0.3

    def normalize(self) -> "ScoringWeights":
        """Normalize weights to sum to 1.0."""
        total = self.relevance + self.recency + self.importance + self.token_efficiency
        if total == 0:
            return ScoringWeights()
        return ScoringWeights(
            relevance=self.relevance / total,
            recency=self.recency / total,
            importance=self.importance / total,
            token_efficiency=self.token_efficiency / total,
        )


@dataclass
class OptimizationResult:
    """Result of context window optimization."""
    ranked_memories: List[RankedMemory]
    total_tokens: int
    remaining_budget: int
    coverage_stats: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TokenCounter:
    """
    Token counting with tiktoken support.

    Supports multiple encoding schemes based on model provider.
    Falls back to heuristic estimation when tiktoken is unavailable.
    """

    # Heuristic token counts (chars per token approximation)
    HEURISTIC_RATIO = 4.0  # ~4 chars per token for most models

    # tiktoken encoding names by provider
    ENCODINGS = {
        ModelProvider.OPENAI: "cl100k_base",  # GPT-4, GPT-3.5-Turbo
        ModelProvider.ANTHROPIC: "cl100k_base",  # Approximation
        ModelProvider.GOOGLE: "cl100k_base",  # Approximation
        ModelProvider.COHERE: "cl100k_base",  # Approximation
        ModelProvider.META: "cl100k_base",  # Llama approximation
        ModelProvider.MISTRAL: "cl100k_base",  # Approximation
        ModelProvider.GENERIC: "cl100k_base",
    }

    _tiktoken_cache: Dict[str, Any] = {}

    def __init__(self, model_provider: ModelProvider = ModelProvider.OPENAI):
        self.provider = model_provider
        self.encoding_name = self.ENCODINGS.get(model_provider, "cl100k_base")
        self.encoding = self._get_encoding()

    def _get_encoding(self) -> Any:
        """Get tiktoken encoding, caching for performance."""
        if self.encoding_name in self._tiktoken_cache:
            return self._tiktoken_cache[self.encoding_name]

        try:
            import tiktoken
            encoding = tiktoken.get_encoding(self.encoding_name)
            self._tiktoken_cache[self.encoding_name] = encoding
            return encoding
        except ImportError:
            logger.warning(
                "tiktoken not available, using heuristic token counting. "
                "Install with: pip install tiktoken"
            )
            return None
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding: {e}, using heuristics")
            return None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Uses tiktoken if available, otherwise falls back to heuristic.
        """
        if not text:
            return 0

        if self.encoding is not None:
            try:
                return len(self.encoding.encode(text))
            except Exception as e:
                logger.debug(f"tiktoken encoding failed: {e}, falling back to heuristic")

        # Heuristic fallback
        return max(1, int(len(text) / self.HEURISTIC_RATIO))

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts efficiently."""
        if self.encoding is not None:
            try:
                # Batch encode for efficiency
                return [len(self.encoding.encode(text)) for text in texts]
            except Exception:
                pass

        return [self.count_tokens(text) for text in texts]

    @classmethod
    def for_model(cls, model_name: str) -> "TokenCounter":
        """Create TokenCounter for specific model."""
        # Detect provider from model name
        if model_name.startswith("gpt-") or model_name.startswith("o1-"):
            provider = ModelProvider.OPENAI
        elif model_name.startswith("claude-"):
            provider = ModelProvider.ANTHROPIC
        elif model_name.startswith("gemini-"):
            provider = ModelProvider.GOOGLE
        elif "llama" in model_name.lower():
            provider = ModelProvider.META
        elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
            provider = ModelProvider.MISTRAL
        else:
            provider = ModelProvider.GENERIC

        return cls(provider)


class SemanticChunker:
    """
    Split long memories into semantically coherent chunks.

    Strategies:
    1. Sentence boundary preservation
    2. Paragraph respect
    3. Token budget adherence
    4. Optional overlap for context continuity
    """

    def __init__(self, config: ChunkConfig, token_counter: TokenCounter):
        self.config = config
        self.counter = token_counter

        # Sentence patterns for splitting
        self.sentence_endings = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])(?=\n)|\n\n+'
        )
        self.paragraph_breaks = re.compile(r'\n\n+')

    def split(self, text: str, metadata: Optional[Dict] = None) -> List[str]:
        """
        Split text into semantically coherent chunks.

        Args:
            text: The text to split
            metadata: Optional metadata to include with each chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # If text is short enough, return as-is
        token_count = self.counter.count_tokens(text)
        if token_count <= self.config.max_chunk_tokens:
            return [text]

        # Strategy 1: Split by paragraphs first
        if self.config.preserve_structure:
            chunks = self._split_by_paragraphs(text)
            if self._within_limits(chunks):
                return chunks

        # Strategy 2: Split by sentences
        if self.config.split_by_sentences:
            chunks = self._split_by_sentences(text)
            if self._within_limits(chunks):
                return chunks

        # Strategy 3: Hard split by token count with overlap
        return self._split_by_tokens(text)

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraph boundaries."""
        paragraphs = self.paragraph_breaks.split(text.strip())
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            if not para.strip():
                continue

            para_tokens = self.counter.count_tokens(para)
            overlap_tokens = self.counter.count_tokens(current_chunk[-self.config.chunk_overlap_tokens:]
                                                       if len(current_chunk) > self.config.chunk_overlap_tokens
                                                       else current_chunk)

            # If paragraph fits, add it
            if current_tokens + para_tokens - overlap_tokens <= self.config.max_chunk_tokens:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
                current_tokens = self.counter.count_tokens(current_chunk)
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
                current_tokens = para_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentence boundaries."""
        sentences = self.sentence_endings.split(text.strip())
        return self._group_by_token_limit(sentences)

    def _split_by_tokens(self, text: str) -> List[str]:
        """Split text by hard token limit with overlap."""
        chunks = []
        chars_per_token = max(1, len(text) // self.counter.count_tokens(text))
        max_chars = int(self.config.max_chunk_tokens * chars_per_token)
        overlap_chars = int(self.config.chunk_overlap_tokens * chars_per_token)

        start = 0
        while start < len(text):
            end = start + max_chars
            # Try to end at a word boundary
            if end < len(text):
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap_chars

        return chunks

    def _group_by_token_limit(self, items: List[str]) -> List[str]:
        """Group items into chunks respecting token limits."""
        chunks = []
        current_chunk = ""

        for item in items:
            if not item.strip():
                continue

            test_chunk = current_chunk + " " + item if current_chunk else item
            test_tokens = self.counter.count_tokens(test_chunk)

            if test_tokens <= self.config.max_chunk_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If single item exceeds limit, it needs further splitting
                item_tokens = self.counter.count_tokens(item)
                if item_tokens > self.config.max_chunk_tokens:
                    # Recursively split the long item
                    sub_chunks = self._split_by_tokens(item)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = item

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _within_limits(self, chunks: List[str]) -> bool:
        """Check if all chunks are within token limits."""
        return all(
            self.config.min_chunk_tokens <= self.counter.count_tokens(c) <= self.config.max_chunk_tokens
            for c in chunks if c.strip()
        )


class ContextWindowPrioritizer:
    """
    Intelligent memory ranking for LLM context window optimization.

    Ranking Formula:
        score = relevance * recency_weight * importance * (1/token_cost)

    Where:
    - relevance: Semantic similarity to query/context
    - recency_weight: Temporal decay favoring recent memories
    - importance: LTP strength, access count, or explicit importance
    - token_cost: Normalized token count (efficiency penalty)

    Features:
    1. Multi-factor ranking with configurable weights
    2. Semantic chunk splitting for long memories
    3. Model-specific token counting
    4. Budget-aware selection
    5. Diversity promotion to avoid redundancy
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        token_budget: Optional[int] = None,
        weights: Optional[ScoringWeights] = None,
        chunk_config: Optional[ChunkConfig] = None,
        recency_half_life_hours: float = 24.0,
    ):
        """
        Initialize the context window prioritizer.

        Args:
            model_name: Name of the LLM model (for token limits/encoding)
            token_budget: Optional override for token budget
            weights: Scoring weights (uses defaults if None)
            chunk_config: Configuration for chunk splitting
            recency_half_life_hours: Half-life for recency decay
        """
        self.model_name = model_name
        self.model_limits = MODEL_LIMITS.get(
            model_name,
            ModelContextLimits(model_name, 128000, 4096, ModelProvider.GENERIC)
        )
        self.token_budget = token_budget or self.model_limits.input_budget
        self.counter = TokenCounter.for_model(model_name)
        self.weights = (weights or ScoringWeights()).normalize()
        self.chunk_config = chunk_config or ChunkConfig()
        self.recency_half_life_hours = recency_half_life_hours

        # Chunking enabled only for models with large context
        self.chunker = SemanticChunker(self.chunk_config, self.counter)

        # Diversity tracking
        self._diversity_threshold = 0.85  # Cosine similarity threshold

    def rank(
        self,
        memories: List[Union[MemoryNode, WorkingMemoryItem, Episode, Dict]],
        query: Optional[str] = None,
        context_embeddings: Optional[List[float]] = None,
    ) -> List[RankedMemory]:
        """
        Rank memories for context window inclusion.

        Args:
            memories: List of memories to rank
            query: Optional query string for relevance calculation
            context_embeddings: Optional pre-computed query embeddings

        Returns:
            List of RankedMemory objects, sorted by score descending
        """
        now = datetime.now(timezone.utc)
        ranked = []

        # Pre-compute query embedding if provided
        query_relevance_fn = self._make_relevance_function(query, context_embeddings)

        for memory in memories:
            # Extract content and metadata
            content = self._extract_content(memory)
            memory_id = self._extract_id(memory)
            importance = self._extract_importance(memory)
            created_at = self._extract_created_at(memory)
            access_count = self._extract_access_count(memory)

            # Calculate relevance score
            relevance = query_relevance_fn(content, memory)

            # Calculate recency weight (exponential decay)
            recency_weight = self._calculate_recency_weight(created_at, now)

            # Normalize importance to [0, 1]
            normalized_importance = min(1.0, importance)

            # Token counting
            token_count = self.counter.count_tokens(content)
            token_efficiency = 1.0 / (1.0 + token_count / 1000.0)  # Soft penalty

            # Calculate final score
            score = (
                (relevance ** self.weights.relevance) *
                (recency_weight ** self.weights.recency) *
                (normalized_importance ** self.weights.importance) *
                (token_efficiency ** self.weights.token_efficiency)
            )

            # Check if chunking is needed
            if token_count > self.chunk_config.max_chunk_tokens:
                chunks = self.chunker.split(content)
                for i, chunk in enumerate(chunks):
                    chunk_tokens = self.counter.count_tokens(chunk)
                    chunk_efficiency = 1.0 / (1.0 + chunk_tokens / 1000.0)
                    chunk_score = score * (chunk_efficiency / token_efficiency)

                    ranked.append(RankedMemory(
                        memory=memory,
                        score=chunk_score,
                        token_count=chunk_tokens,
                        relevance=relevance,
                        recency_weight=recency_weight,
                        importance=normalized_importance,
                        chunk_index=i,
                        total_chunks=len(chunks),
                        original_id=memory_id,
                    ))
            else:
                ranked.append(RankedMemory(
                    memory=memory,
                    score=score,
                    token_count=token_count,
                    relevance=relevance,
                    recency_weight=recency_weight,
                    importance=normalized_importance,
                    original_id=memory_id,
                ))

        # Sort by score descending
        ranked.sort(key=lambda x: x.score, reverse=True)

        return ranked

    def select(
        self,
        memories: List[Union[MemoryNode, WorkingMemoryItem, Episode, Dict]],
        query: Optional[str] = None,
        promote_diversity: bool = True,
        max_memories: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Select memories within token budget, optionally promoting diversity.

        Args:
            memories: List of memories to select from
            query: Optional query for relevance
            promote_diversity: Whether to avoid redundant similar memories
            max_memories: Optional cap on number of memories

        Returns:
            OptimizationResult with selected memories and statistics
        """
        # Rank all memories
        ranked = self.rank(memories, query)

        # Select within budget
        selected = []
        total_tokens = 0
        seen_embeddings = set()

        for ranked_mem in ranked:
            # Check budget
            if total_tokens + ranked_mem.token_count > self.token_budget:
                continue

            # Check diversity if enabled
            if promote_diversity and self._is_redundant(ranked_mem, seen_embeddings):
                continue

            # Check max memories
            if max_memories and len(selected) >= max_memories:
                break

            selected.append(ranked_mem)
            total_tokens += ranked_mem.token_count

            # Track for diversity
            if promote_diversity:
                seen_embeddings.add(self._get_content_signature(ranked_mem))

        # Build result
        return OptimizationResult(
            ranked_memories=selected,
            total_tokens=total_tokens,
            remaining_budget=self.token_budget - total_tokens,
            coverage_stats=self._compute_coverage_stats(selected, memories),
            metadata={
                "model_name": self.model_name,
                "token_budget": self.token_budget,
                "total_considered": len(memories),
                "total_selected": len(selected),
                "promoted_diversity": promote_diversity,
            }
        )

    def rank_with_budget(
        self,
        memories: List[Union[MemoryNode, WorkingMemoryItem, Episode, Dict]],
        token_budget: int,
        query: Optional[str] = None,
    ) -> List[RankedMemory]:
        """
        Rank and filter memories to fit within a specific token budget.

        This is the main entry point for context window optimization.

        Args:
            memories: List of memories to rank
            token_budget: Maximum tokens to include
            query: Optional query for relevance calculation

        Returns:
            List of RankedMemory objects that fit within budget
        """
        # Temporarily override budget
        original_budget = self.token_budget
        self.token_budget = token_budget

        try:
            result = self.select(memories, query, promote_diversity=False)
            return result.ranked_memories
        finally:
            self.token_budget = original_budget

    def _make_relevance_function(
        self,
        query: Optional[str],
        context_embeddings: Optional[List[float]],
    ) -> Callable[[str, Any], float]:
        """Create a relevance scoring function."""

        if context_embeddings:
            # Use provided embeddings
            return lambda content, memory: self._cosine_similarity(
                context_embeddings,
                getattr(memory, 'hdv', None)  # Will be handled in fallback
            )

        if query:
            # Simple keyword/semantic matching
            query_lower = query.lower()
            query_words = set(query_lower.split())

            def keyword_relevance(content: str, memory: Any) -> float:
                content_lower = content.lower()
                # Exact phrase match
                if query_lower in content_lower:
                    return 1.0

                # Word overlap
                content_words = set(content_lower.split())
                overlap = len(query_words & content_words)
                if overlap > 0:
                    return min(1.0, overlap / len(query_words) + 0.3)

                # Use HDV similarity if available
                if hasattr(memory, 'hdv') and memory.hdv is not None:
                    # Placeholder for HDV-based similarity
                    pass

                return 0.1  # Base relevance for all memories

            return keyword_relevance

        # Default: uniform relevance
        return lambda content, memory: 0.5

    def _calculate_recency_weight(
        self,
        created_at: datetime,
        now: datetime,
    ) -> float:
        """Calculate recency weight using exponential decay."""
        delta = now - created_at
        hours = delta.total_seconds() / 3600.0

        # Exponential decay with half-life
        decay_factor = math.exp(-math.log(2) * hours / self.recency_half_life_hours)

        # Ensure minimum weight
        return max(0.1, decay_factor)

    def _extract_content(self, memory: Any) -> str:
        """Extract content from various memory types."""
        if isinstance(memory, MemoryNode):
            return memory.content
        elif isinstance(memory, WorkingMemoryItem):
            return memory.content
        elif isinstance(memory, Episode):
            parts = [memory.goal or "", memory.context or ""]
            return " ".join(p for p in parts if p)
        elif isinstance(memory, dict):
            return memory.get("content", str(memory))
        return str(memory)

    def _extract_id(self, memory: Any) -> str:
        """Extract ID from memory."""
        if isinstance(memory, MemoryNode):
            return memory.id
        elif isinstance(memory, WorkingMemoryItem):
            return memory.id
        elif isinstance(memory, Episode):
            return memory.id
        elif isinstance(memory, dict):
            return memory.get("id", str(id(memory)))
        return str(id(memory))

    def _extract_importance(self, memory: Any) -> float:
        """Extract importance score from memory."""
        if isinstance(memory, MemoryNode):
            # Use LTP strength or access count
            return max(0.0, memory.ltp_strength)
        elif isinstance(memory, WorkingMemoryItem):
            return getattr(memory, 'importance', 0.5)
        elif isinstance(memory, dict):
            return memory.get('importance', memory.get('ltp_strength', 0.5))
        return 0.5

    def _extract_created_at(self, memory: Any) -> datetime:
        """Extract creation timestamp from memory."""
        if isinstance(memory, MemoryNode):
            return memory.created_at
        elif isinstance(memory, WorkingMemoryItem):
            return memory.created_at
        elif isinstance(memory, Episode):
            return memory.started_at
        elif isinstance(memory, dict):
            ts = memory.get('created_at', memory.get('timestamp'))
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except:
                    pass
            elif isinstance(ts, datetime):
                return ts
        return datetime.now(timezone.utc)

    def _extract_access_count(self, memory: Any) -> int:
        """Extract access count from memory."""
        if isinstance(memory, MemoryNode):
            return memory.access_count
        elif isinstance(memory, dict):
            return memory.get('access_count', 1)
        return 1

    def _is_redundant(
        self,
        ranked: RankedMemory,
        seen: set,
    ) -> bool:
        """Check if a memory is redundant with already-selected ones."""
        # Simple content-based deduplication
        signature = self._get_content_signature(ranked)
        return signature in seen

    def _get_content_signature(self, ranked: RankedMemory) -> str:
        """Get a signature for content comparison."""
        content = ranked.content.lower()
        # Use first 100 chars as signature
        return content[:100]

    def _cosine_similarity(self, a: Optional[Any], b: Optional[Any]) -> float:
        """Compute cosine similarity between vectors."""
        # Placeholder for actual vector similarity
        # Would use numpy or similar for real implementation
        return 0.0

    def _compute_coverage_stats(
        self,
        selected: List[RankedMemory],
        all_memories: List[Any],
    ) -> Dict[str, Any]:
        """Compute coverage statistics."""
        return {
            "selected_count": len(selected),
            "total_count": len(all_memories),
            "coverage_ratio": len(selected) / len(all_memories) if all_memories else 0.0,
            "avg_importance": sum(m.importance for m in selected) / len(selected) if selected else 0.0,
            "avg_relevance": sum(m.relevance for m in selected) / len(selected) if selected else 0.0,
        }


class ContextBuilder:
    """
    Build formatted context strings from ranked memories.

    Provides templates for different use cases:
    - Conversational context
    - RAG (Retrieval Augmented Generation)
    - Summarization
    - Analysis
    """

    def __init__(
        self,
        prioritizer: ContextWindowPrioritizer,
        template: Optional[str] = None,
    ):
        self.prioritizer = prioritizer
        self.template = template or self._default_template()

    def _default_template(self) -> str:
        """Default context template."""
        return """[Memory {index}]
ID: {id}
Relevance: {relevance:.2f}
{content}
"""

    def build_context(
        self,
        memories: List[Union[MemoryNode, Dict]],
        query: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Build a formatted context string from memories.

        Args:
            memories: List of memories to include
            query: Optional query for relevance ranking
            max_tokens: Optional token limit

        Returns:
            Formatted context string
        """
        budget = max_tokens or self.prioritizer.token_budget
        result = self.prioritizer.select(memories, query, promote_diversity=True)

        context_parts = []
        total_tokens = 0

        for i, ranked_mem in enumerate(result.ranked_memories, 1):
            formatted = self.template.format(
                index=i,
                id=ranked_mem.memory_id[:8],
                relevance=ranked_mem.relevance,
                content=ranked_mem.content,
            )

            token_count = self.prioritizer.counter.count_tokens(formatted)

            if total_tokens + token_count > budget:
                break

            context_parts.append(formatted)
            total_tokens += token_count

        return "\n".join(context_parts)

    def build_rag_context(
        self,
        memories: List[Union[MemoryNode, Dict]],
        query: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Build RAG-specific context with citations.

        Args:
            memories: List of memories to include
            query: Query for relevance
            max_tokens: Optional token limit

        Returns:
            Formatted RAG context with citations
        """
        template = """[Source {index}]
{content}

"""
        rag_template = """Use the following retrieved memories to answer the query.

Query: {query}

{context}

Citations: {citations}
"""

        budget = max_tokens or self.prioritizer.token_budget
        result = self.prioritizer.select(memories, query)

        context_parts = []
        citations = []
        total_tokens = 0

        for i, ranked_mem in enumerate(result.ranked_memories, 1):
            formatted = template.format(
                index=i,
                content=ranked_mem.content,
            )

            token_count = self.prioritizer.counter.count_tokens(formatted)

            # Reserve space for template wrapper
            if total_tokens + token_count > budget - 200:
                break

            context_parts.append(formatted)
            citations.append(f"[{i}]")
            total_tokens += token_count

        context = "\n".join(context_parts)
        citation_str = ", ".join(citations)

        return rag_template.format(
            query=query,
            context=context.strip(),
            citations=citation_str,
        )


def create_prioritizer(
    model_name: str = "gpt-4o",
    token_budget: Optional[int] = None,
    **kwargs
) -> ContextWindowPrioritizer:
    """
    Factory function to create a ContextWindowPrioritizer.

    Args:
        model_name: Name of the LLM model
        token_budget: Optional token budget override
        **kwargs: Additional arguments for ContextWindowPrioritizer

    Returns:
        Configured ContextWindowPrioritizer instance
    """
    return ContextWindowPrioritizer(
        model_name=model_name,
        token_budget=token_budget,
        **kwargs
    )


# Convenience function for quick ranking
def rank_memories(
    memories: List[Any],
    query: str,
    token_budget: int = 4096,
    model: str = "gpt-4o",
) -> List[str]:
    """
    Quick rank memories and return content list.

    Args:
        memories: List of memories to rank
        query: Query for relevance
        token_budget: Token budget
        model: Model name

    Returns:
        List of memory content strings, ranked by relevance
    """
    prioritizer = create_prioritizer(model_name=model, token_budget=token_budget)
    ranked = prioritizer.rank_with_budget(memories, token_budget, query)
    return [r.content for r in ranked]
