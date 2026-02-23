"""
Contradiction Detection Module (Phase 5.0)
==========================================
Detects contradicting memories in MnemoCore using a two-stage pipeline:

Stage 1: TextEncoder similarity search (fast, vector-based)
  - At /store time: compare new memory against top-5 existing memories
  - If similarity > SIMILARITY_THRESHOLD (0.80) → proceed to Stage 2

Stage 2: LLM-based semantic comparison (accurate, but heavier)
  - Uses SubconsciousAI connector to evaluate if two memories actually contradict
  - Avoids false positives from paraphrases (similarity doesn't mean contradiction)

On confirmed contradiction:
  - Both memories receive a 'contradiction_group_id' in their provenance lineage
  - Both are flagged in their metadata
  - The API returns an alert in the store response
  - Entries are added to a ContradictionRegistry for the /contradictions endpoint

Background scan:
  - ContradictionDetector.scan(nodes) can be called from ConsolidationWorker

Public API:
    detector = ContradictionDetector(engine)
    result = await detector.check_on_store(new_content, new_node, existing_nodes)
    all = detector.registry.list_all()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from loguru import logger

if TYPE_CHECKING:
    from .node import MemoryNode


# ------------------------------------------------------------------ #
#  Thresholds                                                         #
# ------------------------------------------------------------------ #

SIMILARITY_THRESHOLD: float = 0.80   # Above this → suspect contradiction
LLM_CONFIRM_MIN_SCORE: float = 0.70  # LLM contradiction confidence minimum


# ------------------------------------------------------------------ #
#  ContradictionRecord                                               #
# ------------------------------------------------------------------ #

@dataclass
class ContradictionRecord:
    """A detected contradiction between two memories."""
    group_id: str = field(default_factory=lambda: f"cg_{uuid.uuid4().hex[:12]}")
    memory_a_id: str = ""
    memory_b_id: str = ""
    similarity_score: float = 0.0
    llm_confirmed: bool = False
    detected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    resolved: bool = False
    resolution_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "memory_a_id": self.memory_a_id,
            "memory_b_id": self.memory_b_id,
            "similarity_score": round(self.similarity_score, 4),
            "llm_confirmed": self.llm_confirmed,
            "detected_at": self.detected_at,
            "resolved": self.resolved,
            "resolution_note": self.resolution_note,
        }


# ------------------------------------------------------------------ #
#  ContradictionRegistry                                             #
# ------------------------------------------------------------------ #

class ContradictionRegistry:
    """In-memory store of detected contradictions (survives until restart)."""

    def __init__(self) -> None:
        self._records: Dict[str, ContradictionRecord] = {}

    def register(self, record: ContradictionRecord) -> None:
        self._records[record.group_id] = record

    def resolve(self, group_id: str, note: Optional[str] = None) -> bool:
        if group_id in self._records:
            self._records[group_id].resolved = True
            self._records[group_id].resolution_note = note
            return True
        return False

    def list_all(self, unresolved_only: bool = True) -> List[ContradictionRecord]:
        recs = list(self._records.values())
        if unresolved_only:
            recs = [r for r in recs if not r.resolved]
        return sorted(recs, key=lambda r: r.detected_at, reverse=True)

    def list_for_memory(self, memory_id: str) -> List[ContradictionRecord]:
        return [
            r for r in self._records.values()
            if r.memory_a_id == memory_id or r.memory_b_id == memory_id
        ]

    def __len__(self) -> int:
        return len([r for r in self._records.values() if not r.resolved])


# ------------------------------------------------------------------ #
#  ContradictionDetector                                             #
# ------------------------------------------------------------------ #

class ContradictionDetector:
    """
    Two-stage contradiction detector.

    Stage 1: Vector similarity via the engine's binary HDV comparison.
    Stage 2: LLM semantic check via SubconsciousAI (optional).
    """

    def __init__(
        self,
        engine=None,  # HAIMEngine — optional; if None, similarity check uses fallback
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        top_k: int = 5,
        use_llm: bool = True,
    ) -> None:
        self.engine = engine
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.use_llm = use_llm
        self.registry = ContradictionRegistry()

    # ---- Similarity helpers -------------------------------------- #

    def _hamming_similarity(self, node_a: "MemoryNode", node_b: "MemoryNode") -> float:
        """
        Compute binary HDV similarity between two nodes.
        Similarity = 1 - normalized_hamming_distance.
        """
        try:
            import numpy as np
            a = node_a.hdv.data
            b = node_b.hdv.data
            xor = np.bitwise_xor(a, b)
            ham = float(bin(int.from_bytes(xor.tobytes(), "little")).count("1"))
            dim = len(a) * 8
            return 1.0 - ham / dim
        except Exception:
            return 0.0

    # ---- LLM contradiction check --------------------------------- #

    async def _llm_contradicts(
        self, content_a: str, content_b: str
    ) -> Tuple[bool, float]:
        """
        Ask SubconsciousAI if two contents contradict each other.
        Returns (is_contradiction, confidence_score).
        Falls back to False if LLM is unavailable.
        """
        if not self.engine or not self.use_llm:
            return False, 0.0

        try:
            subcon = getattr(self.engine, "subconscious_ai", None)
            if subcon is None:
                return False, 0.0

            prompt = (
                "Do the following two statements contradict each other? "
                "Answer with a JSON object: {\"contradiction\": true/false, \"confidence\": 0.0-1.0}.\n\n"
                f"Statement A: {content_a[:500]}\n"
                f"Statement B: {content_b[:500]}"
            )
            raw = await subcon.generate(prompt, max_tokens=64)
            
            # Robust JSON extraction
            import json as _json
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                logger.debug(f"No JSON found in LLM response: {raw}")
                return False, 0.0
                
            parsed = _json.loads(match.group())
            return bool(parsed.get("contradiction", False)), float(parsed.get("confidence", 0.0))
        except Exception as exc:
            logger.debug(f"LLM contradiction check failed: {exc}")
            return False, 0.0

    # ---- Flag helpers ------------------------------------------- #

    def _flag_node(self, node: "MemoryNode", group_id: str) -> None:
        """Attach contradiction metadata to a node's provenance and metadata fields."""
        node.metadata["contradiction_group_id"] = group_id
        node.metadata["contradicted_at"] = datetime.now(timezone.utc).isoformat()

        prov = getattr(node, "provenance", None)
        if prov is not None:
            prov.mark_contradicted(group_id)

    # ---- Main API ------------------------------------------------ #

    async def check_on_store(
        self,
        new_node: "MemoryNode",
        candidates: Optional[List["MemoryNode"]] = None,
    ) -> Optional[ContradictionRecord]:
        """
        Check a newly stored node against existing memories.

        Args:
            new_node:   The node just stored.
            candidates: Optional pre-fetched list of nodes to compare against.
                        If None and engine is available, fetches via HDV search.

        Returns:
            ContradictionRecord if a contradiction was detected, else None.
        """
        # Fetch candidates if not provided
        if candidates is None and self.engine is not None:
            try:
                results = await self.engine.query(
                    new_node.content, top_k=self.top_k
                )
                nodes = []
                for mem_id, _score in results:
                    n = await self.engine.get_memory(mem_id)
                    if n and n.id != new_node.id:
                        nodes.append(n)
                candidates = nodes
            except Exception as e:
                logger.debug(f"ContradictionDetector: candidate fetch failed: {e}")
                candidates = []

        if not candidates:
            return None

        # Stage 1: similarity filter
        high_sim_candidates = []
        for cand in candidates:
            sim = self._hamming_similarity(new_node, cand)
            if sim >= self.similarity_threshold:
                high_sim_candidates.append((cand, sim))

        if not high_sim_candidates:
            return None

        # Stage 2: LLM confirmation for high-similarity candidates
        high_sim_candidates.sort(key=lambda x: x[1], reverse=True)
        
        for cand, sim in high_sim_candidates:
            is_contradiction = False
            llm_confirmed = False

            if self.use_llm:
                is_contradiction, conf = await self._llm_contradicts(
                    new_node.content, cand.content
                )
                llm_confirmed = is_contradiction and conf >= LLM_CONFIRM_MIN_SCORE
            else:
                # Without LLM, use very high similarity as a soft contradiction signal
                is_contradiction = sim >= 0.92
                llm_confirmed = False

            if is_contradiction:
                # Register the contradiction
                record = ContradictionRecord(
                    memory_a_id=new_node.id,
                    memory_b_id=cand.id,
                    similarity_score=sim,
                    llm_confirmed=llm_confirmed,
                )
                self.registry.register(record)
                self._flag_node(new_node, record.group_id)
                self._flag_node(cand, record.group_id)

                logger.warning(
                    f"⚠️  Contradiction detected: {new_node.id[:8]} ↔ {cand.id[:8]} "
                    f"(sim={sim:.3f}, llm_confirmed={llm_confirmed}, group={record.group_id})"
                )
                return record

        return None

    async def scan(self, nodes: "List[MemoryNode]") -> List[ContradictionRecord]:
        """
        Background scan: compare each node against its peers in the provided list.
        Called periodically from ConsolidationWorker.

        Returns all newly detected contradiction records.
        """
        found: List[ContradictionRecord] = []
        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._hamming_similarity(nodes[i], nodes[j])
                if sim < self.similarity_threshold:
                    continue
                is_contradiction, _ = await self._llm_contradicts(
                    nodes[i].content, nodes[j].content
                )
                if not is_contradiction:
                    continue
                record = ContradictionRecord(
                    memory_a_id=nodes[i].id,
                    memory_b_id=nodes[j].id,
                    similarity_score=sim,
                    llm_confirmed=True,
                )
                self.registry.register(record)
                self._flag_node(nodes[i], record.group_id)
                self._flag_node(nodes[j], record.group_id)
                found.append(record)

        if found:
            logger.info(f"ContradictionDetector background scan: {len(found)} contradictions found in {n} nodes")
        return found


# ------------------------------------------------------------------ #
#  Module singleton                                                  #
# ------------------------------------------------------------------ #

_DETECTOR: ContradictionDetector | None = None


def get_contradiction_detector(engine=None) -> ContradictionDetector:
    """Return the shared ContradictionDetector singleton."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = ContradictionDetector(engine=engine)
    elif engine is not None and _DETECTOR.engine is None:
        _DETECTOR.engine = engine
    return _DETECTOR
