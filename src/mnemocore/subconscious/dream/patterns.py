"""
Pattern Extractor – Stage 2 of Dream Pipeline
==============================================
Extracts recurring patterns from memory content and metadata.

Patterns include:
- Recurring keywords/topics
- Temporal patterns (e.g., weekly activities)
- Semantic similarities across unrelated memories
- Temporal sequences (A often follows B)

Uses both heuristic analysis and optional LLM-based pattern detection.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from loguru import logger

if TYPE_CHECKING:
    from ...core.node import MemoryNode


class PatternExtractor:
    """
    Extracts recurring patterns from memory content and metadata.

    Patterns include:
    - Recurring keywords/topics
    - Temporal patterns (e.g., weekly activities)
    - Semantic similarities across unrelated memories
    - Temporal sequences (A often follows B)

    Uses both heuristic analysis and optional LLM-based pattern detection.
    """

    def __init__(
        self,
        min_frequency: int = 2,
        similarity_threshold: float = 0.75,
    ):
        self.min_frequency = min_frequency
        self.similarity_threshold = similarity_threshold

        # Common stopwords for filtering
        self._stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "as", "is", "was",
            "are", "were", "been", "be", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might",
            # Swedish stopwords
            "och", "eller", "men", "for", "av", "pa", "i", "med", "till",
            "fran", "som", "ar", "var", "varit", "blir", "blev", "ha",
            "har", "hade", "kommer", "skulle", "kunde", "maste",
        }

    async def extract(
        self,
        memories: List["MemoryNode"],
        llm_client: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract patterns from memories.

        Args:
            memories: List of memory nodes to analyze.
            llm_client: Optional LLM client for semantic pattern extraction.

        Returns:
            List of pattern dicts with metadata.
        """
        patterns = []

        # 1. Keyword frequency patterns
        keyword_patterns = self._extract_keyword_patterns(memories)
        patterns.extend(keyword_patterns)

        # 2. Temporal patterns
        temporal_patterns = self._extract_temporal_patterns(memories)
        patterns.extend(temporal_patterns)

        # 3. Metadata patterns
        metadata_patterns = self._extract_metadata_patterns(memories)
        patterns.extend(metadata_patterns)

        # 4. LLM-based semantic patterns (if available)
        if llm_client:
            semantic_patterns = await self._extract_semantic_patterns(
                memories, llm_client
            )
            patterns.extend(semantic_patterns)

        logger.info(f"[PatternExtractor] Extracted {len(patterns)} patterns")

        return patterns[:50]  # Limit to top patterns

    def _extract_keyword_patterns(self, memories: List["MemoryNode"]) -> List[Dict[str, Any]]:
        """Extract recurring keyword patterns."""
        word_counts = defaultdict(int)
        word_memories: Dict[str, Set[str]] = defaultdict(set)

        for memory in memories:
            # Tokenize and clean content
            words = self._tokenize(memory.content)
            for word in words:
                if word not in self._stopwords and len(word) > 3:
                    word_counts[word] += 1
                    word_memories[word].add(memory.id)

        # Filter by frequency
        patterns = []
        for word, count in word_counts.items():
            if count >= self.min_frequency:
                patterns.append({
                    "pattern_type": "keyword",
                    "pattern_value": word,
                    "frequency": count,
                    "memory_ids": list(word_memories[word])[:20],  # Limit
                })

        # Sort by frequency
        patterns.sort(key=lambda p: p["frequency"], reverse=True)
        return patterns[:20]

    def _extract_temporal_patterns(self, memories: List["MemoryNode"]) -> List[Dict[str, Any]]:
        """Extract temporal patterns (hour of day, day of week)."""
        if not memories:
            return []

        hour_counts = defaultdict(int)
        dow_counts = defaultdict(int)

        for memory in memories:
            hour_counts[memory.created_at.hour] += 1
            dow_counts[memory.created_at.weekday()] += 1

        patterns = []

        # Hour patterns
        peak_hour = max(hour_counts.items(), key=lambda x: x[1])
        if peak_hour[1] >= self.min_frequency:
            patterns.append({
                "pattern_type": "temporal_hour",
                "pattern_value": f"hour_{peak_hour[0]}",
                "frequency": peak_hour[1],
                "description": f"Most memories created around {peak_hour[0]}:00",
            })

        # Day of week patterns
        peak_dow = max(dow_counts.items(), key=lambda x: x[1])
        if peak_dow[1] >= self.min_frequency:
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            patterns.append({
                "pattern_type": "temporal_dow",
                "pattern_value": f"dow_{peak_dow[0]}",
                "frequency": peak_dow[1],
                "description": f"Most memories on {days[peak_dow[0]]}",
            })

        return patterns

    def _extract_metadata_patterns(self, memories: List["MemoryNode"]) -> List[Dict[str, Any]]:
        """Extract patterns from metadata fields."""
        category_counts = defaultdict(int)
        tag_counts: Dict[str, int] = defaultdict(int)

        for memory in memories:
            if cat := memory.metadata.get("category"):
                category_counts[cat] += 1

            tags = memory.metadata.get("tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    tag_counts[tag] += 1

        patterns = []

        # Category patterns
        for cat, count in category_counts.items():
            if count >= self.min_frequency:
                patterns.append({
                    "pattern_type": "category",
                    "pattern_value": cat,
                    "frequency": count,
                })

        # Tag patterns
        for tag, count in tag_counts.items():
            if count >= self.min_frequency:
                patterns.append({
                    "pattern_type": "tag",
                    "pattern_value": tag,
                    "frequency": count,
                })

        return patterns[:10]

    async def _extract_semantic_patterns(
        self,
        memories: List["MemoryNode"],
        llm_client: Any,
    ) -> List[Dict[str, Any]]:
        """Use LLM to extract semantic patterns."""
        if len(memories) < 5:
            return []

        # Sample memories for LLM analysis
        sample = memories[:20]
        contents = [f"{i+1}. {m.content[:150]}" for i, m in enumerate(sample)]

        prompt = f"""Analyze these memory fragments and identify 3-5 recurring themes or patterns.
Output ONLY a valid JSON array with this format:
[{{"theme": "pattern name", "description": "brief description", "evidence_count": N}}]

Memories:
{chr(10).join(contents)}
"""

        try:
            response = await llm_client.generate(prompt, max_tokens=300)

            # Parse JSON response
            if "[" in response:
                start = response.index("[")
                end = response.rindex("]") + 1
                from ...utils import json_compat as json
                parsed = json.loads(response[start:end])

                return [
                    {
                        "pattern_type": "semantic_theme",
                        "pattern_value": p.get("theme", "unknown"),
                        "description": p.get("description", ""),
                        "frequency": p.get("evidence_count", self.min_frequency),
                    }
                    for p in parsed
                ]
        except Exception as e:
            logger.debug(f"[PatternExtractor] LLM pattern extraction failed: {e}")

        return []

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for pattern extraction."""
        # Lowercase and extract words
        words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower())
        return words


__all__ = ["PatternExtractor"]
