"""
Cognitive Router - Orchestrates System 1 (Fast) and System 2 (Slow) thinking.
"""

import logging
import time
from typing import Dict, Any, Tuple, Optional

from .engine import HAIMEngine
from .binary_hdv import majority_bundle, BinaryHDV
from .exceptions import MnemoCoreError

logger = logging.getLogger(__name__)


class CognitiveRouter:
    """
    Orchestrates System 1 (Fast) and System 2 (Slow) thinking.

    System 1: Intuitive, heuristic, fast memory retrieval.
    System 2: Analytical, epistemic search, heavy reasoning.
    """

    def __init__(self, engine: HAIMEngine):
        self.engine = engine
        self.complexity_threshold = 0.6  # Threshold for switching to Sys2

    async def route(self, impulse: str, context: dict = None) -> Tuple[str, Dict[str, Any]]:
        """
        Route the impulse to the appropriate system.
        Returns: (response, debug_info)
        """
        start_time = time.time()
        complexity = await self._assess_complexity(impulse)

        debug_info = {
            "impulse": impulse,
            "complexity_score": complexity,
            "timestamp": start_time
        }

        if complexity < self.complexity_threshold:
            # System 1: Fast Reflex
            debug_info["system"] = "Sys1 (Fast)"
            response = await self._system_1_reflex(impulse)
        else:
            # System 2: Heavy Reasoning
            debug_info["system"] = "Sys2 (Slow)"
            response = await self._system_2_reasoning(impulse, context)

        debug_info["duration"] = time.time() - start_time
        return response, debug_info

    async def _assess_complexity(self, text: str) -> float:
        """
        Heuristic to estimate cognitive load.
        """
        score = 0.0

        # Length heuristic
        if len(text.split()) > 20:
            score += 0.3

        # Complexity markers
        complex_markers = ["analyze", "compare", "why", "how", "plan", "design", "evaluate"]
        if any(marker in text.lower() for marker in complex_markers):
            score += 0.4

        # Uncertainty markers
        uncertainty = ["maybe", "unsure", "unknown", "complex"]
        if any(u in text.lower() for u in uncertainty):
            score += 0.2

        # Epistemic check (query engine for familiarity)
        # Low familiarity (high surprise) -> Higher complexity
        try:
            results = await self.engine.query(text, top_k=1, associative_jump=False)
            if results and results[0][1] > 0.8:
                # Strong memory match -> Familiar -> Lower complexity
                score -= 0.3
            elif not results or results[0][1] < 0.3:
                # No clue -> Novelty -> Higher complexity
                score += 0.4
        except MnemoCoreError as e:
            # Log domain errors but continue with heuristic-only assessment
            logger.debug(f"Complexity assessment query failed: {e}")
        except Exception as e:
            # Log unexpected errors but continue
            logger.warning(f"Unexpected error in complexity assessment: {e}")

        return min(1.0, max(0.0, score))

    async def _system_1_reflex(self, impulse: str) -> str:
        """
        Fast retrieval and simple association.
        """
        # 1. Quick memory lookup
        results = await self.engine.query(impulse, top_k=3)

        if not results:
            return "I don't have an immediate reflex for that."

        # 2. Synthesize simple answer from top memory (simulated)
        # Use engine.get_memory() instead of direct dict access
        top_mem_id, score = results[0]
        node = await self.engine.get_memory(top_mem_id)

        content = node.content if node else 'Unknown'
        return f"[Reflex] Based on memory ({score:.2f}): {content}"

    async def _system_2_reasoning(self, impulse: str, context: Optional[dict]) -> str:
        """Slow, deliberative process with Epistemic Drive."""
        eig: Optional[float] = None

        # 1. Epistemic Drive (Expected Information Gain)
        if self.engine.epistemic_drive_active:
            candidate_vec = self.engine.encode_content(impulse)

            # Build context vector from working memory or sample from engine
            ctx_vec: BinaryHDV

            if context and isinstance(context.get("working_memory"), list) and context["working_memory"]:
                vectors = []
                for item in context["working_memory"]:
                    vectors.append(self.engine.encode_content(str(item)))

                # Bundle all context vectors
                if vectors:
                    ctx_vec = majority_bundle(vectors)
                else:
                    ctx_vec = await self.engine._current_context_vector()
            else:
                ctx_vec = await self.engine._current_context_vector(sample_n=50)

            eig = self.engine.calculate_eig(candidate_vec, ctx_vec)

        # 2. Deep Search (Associative Jumps)
        results = await self.engine.query(impulse, top_k=10, associative_jump=True)

        # 3. Consolidation / Synthesis
        memories = []
        for mid, score in results:
            node = await self.engine.get_memory(mid)
            if node:
                memories.append(f"- {node.content} (conf: {score:.2f})")

        knowledge_block = "\n".join(memories)

        eig_line = f"\nEpistemic Drive (EIG): {eig:.2f}" if eig is not None else ""
        return f"[Reasoning] I have analyzed {len(memories)} data points.{eig_line}\nKey insights:\n{knowledge_block}"
