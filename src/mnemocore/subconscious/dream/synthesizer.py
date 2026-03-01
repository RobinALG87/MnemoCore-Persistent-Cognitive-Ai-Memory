"""
Dream Synthesizer – Stage 3 of Dream Pipeline
==============================================
Wrapper around RecursiveSynthesizer for dream-time consolidation.

Uses the existing RecursiveSynthesizer to:
1. Synthesize patterns into higher-level abstractions
2. Create semantic bridges between related concepts
3. Generate "dream" memories from synthesis results

The synthesizer operates on pattern clusters rather than user queries.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from loguru import logger

if TYPE_CHECKING:
    from ...core.engine import HAIMEngine
    from ...core.recursive_synthesizer import RecursiveSynthesizer


@dataclass
class SynthesisResult:
    """Result from synthesizing a pattern."""
    pattern: Dict[str, Any]
    query: str
    results_count: int
    synthesis: str
    dream_memory_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "query": self.query,
            "results_count": self.results_count,
            "synthesis": self.synthesis,
            "dream_memory_id": self.dream_memory_id,
        }


class DreamSynthesizer:
    """
    Wrapper around RecursiveSynthesizer for dream-time consolidation.

    Uses the existing RecursiveSynthesizer to:
    1. Synthesize patterns into higher-level abstractions
    2. Create semantic bridges between related concepts
    3. Generate "dream" memories from synthesis results

    The synthesizer operates on pattern clusters rather than user queries.
    """

    def __init__(
        self,
        engine: "HAIMEngine",
        max_depth: int = 3,
        max_patterns: int = 10,
    ):
        self.engine = engine
        self.max_depth = max_depth
        self.max_patterns = max_patterns

        # Lazy load synthesizer
        self._synthesizer: Optional["RecursiveSynthesizer"] = None

    def _get_synthesizer(self) -> "RecursiveSynthesizer":
        """Get or create the synthesizer instance."""
        if self._synthesizer is None:
            from ...core.recursive_synthesizer import (
                RecursiveSynthesizer,
                SynthesizerConfig,
            )

            # Check if engine has LLM capability
            llm_call = getattr(self.engine, "subconscious_ai", None)
            if llm_call:
                llm_call = llm_call._model_client.generate

            self._synthesizer = RecursiveSynthesizer(
                engine=self.engine,
                config=SynthesizerConfig(
                    max_depth=self.max_depth,
                    max_sub_queries=5,
                    final_top_k=10,
                ),
                llm_call=llm_call,
            )
        return self._synthesizer

    async def synthesize_patterns(
        self,
        patterns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Synthesize patterns into higher-level abstractions.

        Args:
            patterns: List of extracted patterns.

        Returns:
            List of synthesis results.
        """
        if not patterns:
            return []

        synthesizer = self._get_synthesizer()
        results = []

        # Process top patterns
        top_patterns = patterns[:self.max_patterns]

        for pattern in top_patterns:
            # Build a query from the pattern
            query = self._pattern_to_query(pattern)

            try:
                synthesis = await synthesizer.synthesize(query)

                # Store synthesis as a "dream memory" if significant
                if synthesis.results and synthesis.synthesis:
                    dream_memory_id = await self._store_dream_memory(
                        pattern, synthesis
                    )

                    results.append({
                        "pattern": pattern,
                        "query": query,
                        "results_count": len(synthesis.results),
                        "synthesis": synthesis.synthesis[:500],  # Truncate
                        "dream_memory_id": dream_memory_id,
                    })

            except Exception as e:
                logger.warning(f"[DreamSynthesizer] Failed to synthesize pattern: {e}")

        return results

    def _pattern_to_query(self, pattern: Dict[str, Any]) -> str:
        """Convert a pattern into a search query."""
        ptype = pattern.get("pattern_type", "")
        value = pattern.get("pattern_value", "")

        if ptype == "keyword":
            return f"Memories related to {value}"
        elif ptype == "semantic_theme":
            desc = pattern.get("description", "")
            return f"{value}: {desc}"
        elif ptype == "category":
            return f"Memories in category: {value}"
        elif ptype == "temporal_hour":
            return f"Memories from around {value}"
        else:
            return str(value)

    async def _store_dream_memory(
        self,
        pattern: Dict[str, Any],
        synthesis: Any,
    ) -> Optional[str]:
        """Store synthesis result as a dream memory."""
        content = f"[DREAM SYNTHESIS] {pattern.get('pattern_value', 'unknown')}\n{synthesis.synthesis}"

        metadata = {
            "type": "dream_synthesis",
            "pattern_type": pattern.get("pattern_type"),
            "pattern_value": pattern.get("pattern_value"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "results_count": len(synthesis.results),
        }

        try:
            # Store using engine's async-friendly method
            mem_id = await asyncio.to_thread(
                self.engine.store,
                content,
                metadata=metadata,
            )
            return mem_id
        except Exception as e:
            logger.debug(f"[DreamSynthesizer] Failed to store dream memory: {e}")
            return None


__all__ = ["DreamSynthesizer", "SynthesisResult"]
