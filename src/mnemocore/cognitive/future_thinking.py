"""
Episodic Future Thinking (EFT) Module
======================================
Phase 7.0 - Cognitive simulation of probable future scenarios.

Episodic Future Thinking (EFT) is the cognitive capacity to project oneself
into the future and pre-experience potential events. This module implements:

1. Scenario Generation: Probable futures based on historical patterns
2. Temporal Decay: Future scenarios decay over time (forgetting curve)
3. Attention Integration: Scenarios influence attention prioritization
4. Prediction Storage: Integration with PredictionStore for verification

Architecture:
  - EpisodeFutureSimulator: Main entry point for scenario generation
  - ScenarioNode: Represents a simulated future event
  - ScenarioStore: Manages scenario lifecycle with decay
  - AttentionIntegration: Connects scenarios to XOR attention system

References:
  - Schacter et al. (2017): "The Future of Memory: An Integrative
    Perspective on the Future of episodic thought"
  - Buckner & Carroll (2007): "Self-projection and the brain"
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import math

import numpy as np
from loguru import logger

from mnemocore.core.binary_hdv import BinaryHDV, majority_bundle
from mnemocore.core.config import get_config

if TYPE_CHECKING:
    from mnemocore.core.prediction_store import PredictionStore
    from mnemocore.core.anticipatory import AnticipatoryEngine
    from mnemocore.core.tier_manager import TierManager
    from mnemocore.core.synapse_index import SynapseIndex
    from mnemocore.core.attention import XORAttentionMasker, AttentionResult


# ==============================================================================
# Configuration
# ==============================================================================


@dataclass
class EFTConfig:
    """Configuration for Episodic Future Thinking module."""
    enabled: bool = True

    # Scenario generation
    max_scenarios_per_simulation: int = 5
    min_similarity_threshold: float = 0.55  # Min pattern match to generate scenario
    temporal_horizon_hours: float = 24.0  # How far into the future to simulate
    branching_factor: int = 3  # Alternative futures per context

    # Decay parameters
    scenario_decay_lambda: float = 0.05  # Decay rate per hour
    scenario_half_life_hours: float = 12.0  # Half-life of scenario relevance
    min_scenario_confidence: float = 0.1  # Prune scenarios below this

    # Attention integration
    attention_boost_factor: float = 0.2  # How much scenarios influence attention
    scenario_attention_weight: float = 0.15  # Weight in composite attention score

    # Storage
    max_stored_scenarios: int = 100
    persist_scenarios: bool = True

    def validate(self) -> None:
        assert 0.0 <= self.min_similarity_threshold <= 1.0
        assert 0.0 <= self.scenario_decay_lambda <= 1.0
        assert 0.0 <= self.attention_boost_factor <= 1.0


# ==============================================================================
# Scenario Models
# ==============================================================================


class ScenarioType(Enum):
    """Types of future scenarios."""
    CONTINUATION = "continuation"  # Direct continuation of current pattern
    BRANCHING = "branching"  # Alternative based on similar historical patterns
    ANOMALY = "anomaly"  # Unexpected/unlikely but high-impact event
    GOAL_DIRECTED = "goal_directed"  # Scenario based on explicit goals
    ANTICIPATED = "anticipated"  # Based on anticipatory engine predictions


@dataclass
class ScenarioNode:
    """
    Represents a simulated future event/episode.

    A scenario is a hypothetical future state that:
      - Has a confidence score (probability)
      - Decays over time
      - Links to related memories (pattern sources)
      - Can be verified against actual future events
    """
    id: str = field(default_factory=lambda: f"scenario_{uuid.uuid4().hex[:12]}")
    type: ScenarioType = ScenarioType.CONTINUATION

    # Content
    content: str = ""
    context_summary: str = ""  # What context triggered this scenario
    projected_timestamp: Optional[str] = None  # When this scenario is projected to occur

    # Confidence and probability
    confidence: float = 0.5  # Initial confidence [0, 1]
    probability: float = 0.5  # Estimated probability [0, 1]

    # Temporal dynamics
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_accessed: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    decay_factor: float = 1.0  # Current decay multiplier

    # Pattern sources (which memories informed this scenario)
    source_memory_ids: List[str] = field(default_factory=list)
    pattern_similarity: float = 0.0  # Average similarity to source patterns

    # Vector representation (for attention integration)
    hdv: Optional[BinaryHDV] = field(default=None, repr=False)

    # Verification
    verified: bool = False
    verification_outcome: Optional[bool] = None  # True=occurred, False=did_not_occur
    verified_at: Optional[str] = None

    # Tags and metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------------
    # Decay and lifecycle
    # --------------------------------------------------------------

    def age_hours(self) -> float:
        """Hours since scenario creation."""
        created = datetime.fromisoformat(self.created_at)
        now = datetime.now(timezone.utc)
        return (now - created).total_seconds() / 3600.0

    def apply_decay(self, decay_lambda: float, half_life_hours: float) -> float:
        """
        Apply exponential decay to scenario confidence.

        Formula: confidence = initial * exp(-lambda * t)
        Where t is time in hours.
        """
        age = self.age_hours()
        self.decay_factor = math.exp(-decay_lambda * age)
        return self.current_confidence()

    def current_confidence(self) -> float:
        """Get confidence after decay is applied."""
        return max(self.confidence * self.decay_factor, 0.0)

    def is_expired(self, threshold: float = 0.1) -> bool:
        """Check if scenario has decayed below relevance threshold."""
        return self.current_confidence() < threshold

    # --------------------------------------------------------------
    # Verification
    # --------------------------------------------------------------

    def verify(self, occurred: bool, notes: Optional[str] = None) -> None:
        """Mark scenario as verified against reality."""
        self.verified = True
        self.verification_outcome = occurred
        self.verified_at = datetime.now(timezone.utc).isoformat()
        if notes:
            self.metadata["verification_notes"] = notes

    # --------------------------------------------------------------
    # Serialization
    # --------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "context_summary": self.context_summary,
            "projected_timestamp": self.projected_timestamp,
            "confidence": round(self.confidence, 4),
            "probability": round(self.probability, 4),
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "decay_factor": round(self.decay_factor, 4),
            "source_memory_ids": self.source_memory_ids,
            "pattern_similarity": round(self.pattern_similarity, 4),
            "verified": self.verified,
            "verification_outcome": self.verification_outcome,
            "verified_at": self.verified_at,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScenarioNode":
        """Deserialize from dictionary."""
        scenario_type = ScenarioType(d.get("type", ScenarioType.CONTINUATION.value))
        return cls(
            id=d.get("id", f"scenario_{uuid.uuid4().hex[:12]}"),
            type=scenario_type,
            content=d.get("content", ""),
            context_summary=d.get("context_summary", ""),
            projected_timestamp=d.get("projected_timestamp"),
            confidence=d.get("confidence", 0.5),
            probability=d.get("probability", 0.5),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            last_accessed=d.get("last_accessed", datetime.now(timezone.utc).isoformat()),
            decay_factor=d.get("decay_factor", 1.0),
            source_memory_ids=d.get("source_memory_ids", []),
            pattern_similarity=d.get("pattern_similarity", 0.0),
            verified=d.get("verified", False),
            verification_outcome=d.get("verification_outcome"),
            verified_at=d.get("verified_at"),
            tags=d.get("tags", []),
            metadata=d.get("metadata", {}),
        )


# ==============================================================================
# Scenario Store with Decay
# ==============================================================================


class ScenarioStore:
    """
    Manages future scenario storage with automatic decay.

    Scenarios are stored in-memory and optionally persisted.
    Decay is applied on access and during periodic cleanup.
    """

    def __init__(
        self,
        config: Optional[EFTConfig] = None,
        storage_path: Optional[str] = None
    ):
        self.config = config or EFTConfig()
        self._scenarios: Dict[str, ScenarioNode] = {}
        self._storage_path = storage_path or "./data/scenarios.jsonl"
        self._lock = asyncio.Lock()

    # --------------------------------------------------------------
    # CRUD operations
    # --------------------------------------------------------------

    async def store(self, scenario: ScenarioNode) -> str:
        """Store a new scenario."""
        async with self._lock:
            # Add to local index
            self._scenarios[scenario.id] = scenario

            # HARD LIMIT: strictly enforce max_stored_scenarios
            # First try gentle cleanup of low confidence
            if len(self._scenarios) > self.config.max_stored_scenarios:
                await self._cleanup_low_confidence()
            
            # If still over limit, remove oldest scenarios (FIFO logic)
            if len(self._scenarios) > self.config.max_stored_scenarios:
                sorted_scenarios = sorted(
                    self._scenarios.items(),
                    key=lambda x: (x[1].last_accessed or "", x[1].confidence)
                )
                
                overflow = len(self._scenarios) - self.config.max_stored_scenarios
                for i in range(overflow):
                    sid, _ = sorted_scenarios[i]
                    del self._scenarios[sid]

            if self.config.persist_scenarios:
                await self._persist_scenario(scenario)

            logger.debug(
                f"Stored scenario {scenario.id} | "
                f"type={scenario.type.value} | confidence={scenario.confidence:.2f}"
            )
            return scenario.id

    async def get(self, scenario_id: str) -> Optional[ScenarioNode]:
        """Retrieve a scenario and update its access time."""
        async with self._lock:
            scenario = self._scenarios.get(scenario_id)
            if scenario:
                scenario.last_accessed = datetime.now(timezone.utc).isoformat()
                # Apply decay on access
                scenario.apply_decay(
                    self.config.scenario_decay_lambda,
                    self.config.scenario_half_life_hours
                )
            return scenario

    async def list_active(
        self,
        min_confidence: float = 0.1,
        scenario_type: Optional[ScenarioType] = None
    ) -> List[ScenarioNode]:
        """List all active scenarios above confidence threshold."""
        async with self._lock:
            active = []
            for scenario in self._scenarios.values():
                # Skip verified scenarios
                if scenario.verified:
                    continue

                # Apply decay
                current_conf = scenario.apply_decay(
                    self.config.scenario_decay_lambda,
                    self.config.scenario_half_life_hours
                )

                if current_conf >= min_confidence:
                    if scenario_type is None or scenario.type == scenario_type:
                        active.append(scenario)

            # Sort by confidence descending
            active.sort(key=lambda s: s.current_confidence(), reverse=True)
            return active

    async def get_by_source_memory(self, memory_id: str) -> List[ScenarioNode]:
        """Get all scenarios derived from a specific memory."""
        async with self._lock:
            return [
                s for s in self._scenarios.values()
                if memory_id in s.source_memory_ids and not s.verified
            ]

    # --------------------------------------------------------------
    # Verification and lifecycle
    # --------------------------------------------------------------

    async def verify(
        self,
        scenario_id: str,
        occurred: bool,
        notes: Optional[str] = None
    ) -> Optional[ScenarioNode]:
        """Verify a scenario against reality."""
        async with self._lock:
            scenario = self._scenarios.get(scenario_id)
            if scenario:
                scenario.verify(occurred, notes)
                logger.info(
                    f"Scenario {scenario_id} verified: "
                    f"{'OCCURRED' if occurred else 'DID NOT OCCUR'}"
                )
            return scenario

    async def cleanup_expired(self) -> int:
        """Remove scenarios that have decayed below threshold."""
        async with self._lock:
            return await self._cleanup_low_confidence()

    async def _cleanup_low_confidence(self) -> int:
        """Internal cleanup of low-confidence scenarios."""
        to_remove = []
        for sid, scenario in self._scenarios.items():
            if scenario.verified:
                # Keep verified scenarios for learning
                continue

            conf = scenario.apply_decay(
                self.config.scenario_decay_lambda,
                self.config.scenario_half_life_hours
            )
            if conf < self.config.min_scenario_confidence:
                to_remove.append(sid)

        for sid in to_remove:
            del self._scenarios[sid]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} expired scenarios")

        return len(to_remove)

    # --------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------

    async def _list_active_unlocked(
        self,
        min_confidence: float = 0.1,
        scenario_type: Optional[ScenarioType] = None
    ) -> List[ScenarioNode]:
        """
        Internal unlocked version of list_active.

        Must be called with self._lock already held.
        """
        active = []
        for scenario in self._scenarios.values():
            # Skip verified scenarios
            if scenario.verified:
                continue

            # Apply decay
            current_conf = scenario.apply_decay(
                self.config.scenario_decay_lambda,
                self.config.scenario_half_life_hours
            )

            if current_conf >= min_confidence:
                if scenario_type is None or scenario.type == scenario_type:
                    active.append(scenario)

        # Sort by confidence descending
        active.sort(key=lambda s: s.current_confidence(), reverse=True)
        return active

    async def list_active(
        self,
        min_confidence: float = 0.1,
        scenario_type: Optional[ScenarioType] = None
    ) -> List[ScenarioNode]:
        """List all active scenarios above confidence threshold."""
        async with self._lock:
            return await self._list_active_unlocked(min_confidence, scenario_type)

    async def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        async with self._lock:
            active = await self._list_active_unlocked()
            verified = [s for s in self._scenarios.values() if s.verified]

            by_type = {}
            for s in active:
                by_type[s.type.value] = by_type.get(s.type.value, 0) + 1

            return {
                "total_scenarios": len(self._scenarios),
                "active_scenarios": len(active),
                "verified_scenarios": len(verified),
                "by_type": by_type,
                "avg_confidence": (
                    sum(s.current_confidence() for s in active) / len(active)
                    if active else 0.0
                ),
            }

    # --------------------------------------------------------------
    # Persistence
    # --------------------------------------------------------------

    async def _persist_scenario(self, scenario: ScenarioNode) -> None:
        """Append scenario to storage file."""
        try:
            import json
            from pathlib import Path

            Path(self._storage_path).parent.mkdir(parents=True, exist_ok=True)

            with open(self._storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(scenario.to_dict()) + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist scenario {scenario.id}: {e}")

    async def load_from_storage(self) -> int:
        """Load scenarios from persistent storage."""
        try:
            import json
            from pathlib import Path

            path = Path(self._storage_path)
            if not path.exists():
                return 0

            count = 0
            async with self._lock:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            scenario = ScenarioNode.from_dict(data)
                            # Only load unverified scenarios (verified are for audit)
                            if not scenario.verified:
                                self._scenarios[scenario.id] = scenario
                                count += 1
                        except Exception as e:
                            logger.warning(f"Failed to load scenario: {e}")

            logger.info(f"Loaded {count} scenarios from {self._storage_path}")
            return count

        except Exception as e:
            logger.error(f"Failed to load scenarios from storage: {e}")
            return 0


# ==============================================================================
# Main Simulator
# ==============================================================================


class EpisodeFutureSimulator:
    """
    Main entry point for Episodic Future Thinking.

    Generates probable future scenarios based on:
      1. Historical pattern matching (synaptic graph traversal)
      2. Temporal sequences (episodic chains)
      3. Contextual similarity (HDV vector space)
      4. Anticipatory engine predictions

    Scenarios influence attention prioritization and can be verified
    against actual future events.
    """

    def __init__(
        self,
        config: Optional[EFTConfig] = None,
        prediction_store: Optional["PredictionStore"] = None,
        anticipatory_engine: Optional["AnticipatoryEngine"] = None,
        tier_manager: Optional["TierManager"] = None,
        synapse_index: Optional["SynapseIndex"] = None,
        dimension: int = 16384,
    ):
        self.config = config or EFTConfig()
        self.prediction_store = prediction_store
        self.anticipatory_engine = anticipatory_engine
        self.tier_manager = tier_manager
        self.synapse_index = synapse_index
        self.dimension = dimension

        self.scenario_store = ScenarioStore(self.config)
        self._encoder = None  # Lazy-loaded TextEncoder
        self._lock = asyncio.Lock()

        # Attention integration (lazy initialization)
        self._attention_masker: Optional["XORAttentionMasker"] = None

    # --------------------------------------------------------------
    # Public API: Scenario Simulation
    # --------------------------------------------------------------

    async def simulate(
        self,
        context: str,
        context_hdv: Optional[BinaryHDV] = None,
        horizon_hours: Optional[float] = None,
        max_scenarios: Optional[int] = None,
        scenario_types: Optional[List[ScenarioType]] = None,
    ) -> List[ScenarioNode]:
        """
        Generate probable future scenarios based on current context.

        Args:
            context: Text description of current situation
            context_hdv: Optional pre-computed HDV for context
            horizon_hours: How far into the future to simulate
            max_scenarios: Maximum number of scenarios to generate
            scenario_types: Which types of scenarios to generate

        Returns:
            List of generated scenarios, sorted by confidence
        """
        if not self.config.enabled:
            return []

        horizon = horizon_hours or self.config.temporal_horizon_hours
        max_gen = max_scenarios or self.config.max_scenarios_per_simulation

        # Generate context HDV if not provided
        if context_hdv is None:
            context_hdv = self._encode_text(context)

        logger.info(
            f"Simulating future scenarios | context_length={len(context)} | "
            f"horizon={horizon}h | max={max_gen}"
        )

        scenarios = []

        # 1. Continuation scenarios (pattern extension)
        if self._should_generate_type(ScenarioType.CONTINUATION, scenario_types):
            continuation = await self._generate_continuation_scenarios(
                context, context_hdv, horizon, max_gen
            )
            scenarios.extend(continuation)

        # 2. Branching scenarios (alternative patterns)
        if self._should_generate_type(ScenarioType.BRANCHING, scenario_types):
            branching = await self._generate_branching_scenarios(
                context, context_hdv, horizon, max_gen
            )
            scenarios.extend(branching)

        # 3. Anticipated scenarios (from anticipatory engine)
        if self._should_generate_type(ScenarioType.ANTICIPATED, scenario_types):
            anticipated = await self._generate_anticipated_scenarios(
                context, context_hdv, horizon, max_gen
            )
            scenarios.extend(anticipated)

        # 4. Goal-directed scenarios (if goal metadata present)
        if self._should_generate_type(ScenarioType.GOAL_DIRECTED, scenario_types):
            goal_directed = await self._generate_goal_scenarios(
                context, context_hdv, horizon, max_gen
            )
            scenarios.extend(goal_directed)

        # Store scenarios
        for scenario in scenarios:
            await self.scenario_store.store(scenario)

        # Sort by confidence and limit
        scenarios.sort(key=lambda s: s.current_confidence(), reverse=True)
        result = scenarios[:max_gen]

        logger.info(
            f"Generated {len(result)} scenarios (total: {len(scenarios)})"
        )

        return result

    async def simulate_from_memory_ids(
        self,
        memory_ids: List[str],
        horizon_hours: Optional[float] = None,
        max_scenarios: Optional[int] = None,
    ) -> List[ScenarioNode]:
        """
        Generate scenarios based on specific memories as pattern sources.

        Useful for explicit "what if" exploration of past patterns.
        """
        if not memory_ids:
            return []

        # Retrieve memories
        if self.tier_manager:
            nodes = await self.tier_manager.get_memories_batch(memory_ids)
        else:
            return []

        valid_nodes = [n for n in nodes if n is not None]
        if not valid_nodes:
            return []

        # Build context from memories
        context_texts = [n.content for n in valid_nodes if n.content]
        context = " | ".join(context_texts[:3])  # Use first 3

        # Bundle their HDVs for pattern matching
        hdvs = [n.hdv for n in valid_nodes if hasattr(n, "hdv") and n.hdv]
        context_hdv = majority_bundle(hdvs) if hdvs else self._encode_text(context)

        scenarios = await self.simulate(
            context=context,
            context_hdv=context_hdv,
            horizon_hours=horizon_hours,
            max_scenarios=max_scenarios,
        )

        # Update source tracking
        for scenario in scenarios:
            scenario.source_memory_ids = memory_ids[:5]  # Limit sources
            scenario.pattern_similarity = sum(
                context_hdv.similarity(n.hdv) for n in valid_nodes if hasattr(n, "hdv") and n.hdv
            ) / max(len(valid_nodes), 1)

        return scenarios

    # --------------------------------------------------------------
    # Scenario Type Generators
    # --------------------------------------------------------------

    async def _generate_continuation_scenarios(
        self,
        context: str,
        context_hdv: BinaryHDV,
        horizon_hours: float,
        max_scenarios: int,
    ) -> List[ScenarioNode]:
        """
        Generate scenarios that directly continue the current pattern.

        Method: Look for similar historical patterns and project their
        continuation forward in time.
        """
        scenarios = []

        # Find similar memories in HOT tier
        similar_memories = await self._find_similar_memories(
            context_hdv, limit=max_scenarios * 2
        )

        for memory_id, similarity in similar_memories[:max_scenarios]:
            if similarity < self.config.min_similarity_threshold:
                continue

            # Get the memory node
            memory = None
            if self.tier_manager:
                memory = await self.tier_manager.get_memory(memory_id)

            if not memory:
                continue

            # Look at what followed this memory in episodic chain
            next_memory = None
            if self.tier_manager:
                next_memory = await self.tier_manager.get_next_in_chain(memory_id)

            # Generate scenario content
            if next_memory:
                content = f"Similar to '{memory.content[:100]}...', " \
                         f"likely followed by: {next_memory.content[:100]}..."
                confidence = similarity * 0.7  # Discount for projection
            else:
                content = f"Pattern similar to '{memory.content[:100]}...' " \
                         f"continues in a similar direction"
                confidence = similarity * 0.5

            # Project timestamp
            projected_time = datetime.now(timezone.utc) + timedelta(hours=horizon_hours)

            scenario = ScenarioNode(
                type=ScenarioType.CONTINUATION,
                content=content,
                context_summary=context[:200],
                projected_timestamp=projected_time.isoformat(),
                confidence=confidence,
                probability=confidence,
                source_memory_ids=[memory_id],
                pattern_similarity=similarity,
                hdv=self._encode_text(content),
            )
            scenarios.append(scenario)

        return scenarios

    async def _generate_branching_scenarios(
        self,
        context: str,
        context_hdv: BinaryHDV,
        horizon_hours: float,
        max_scenarios: int,
    ) -> List[ScenarioNode]:
        """
        Generate alternative scenarios based on divergent patterns.

        Method: Use synapse graph to find branching points and generate
        alternative continuations.
        """
        scenarios = []

        if not self.synapse_index:
            return scenarios

        # Find neighbors through synapse graph
        similar_memories = await self._find_similar_memories(
            context_hdv, limit=10
        )

        for memory_id, base_similarity in similar_memories[:3]:
            # Get multi-hop neighbors
            neighbors = await self.synapse_index.get_multi_hop_neighbors(
                memory_id, depth=self.config.branching_factor
            )

            # Generate scenarios from alternative paths
            for neighbor_id, path_strength in list(neighbors.items())[:2]:
                neighbor = None
                if self.tier_manager:
                    neighbor = await self.tier_manager.get_memory(neighbor_id)

                if not neighbor:
                    continue

                # Confidence based on path strength
                confidence = base_similarity * path_strength * 0.6

                if confidence < self.config.min_scenario_confidence:
                    continue

                projected_time = datetime.now(timezone.utc) + timedelta(
                    hours=horizon_hours * (1.0 - path_strength * 0.5)
                )

                scenario = ScenarioNode(
                    type=ScenarioType.BRANCHING,
                    content=f"Alternative path: {neighbor.content[:150]}...",
                    context_summary=f"Branching from {memory_id[:8]}",
                    projected_timestamp=projected_time.isoformat(),
                    confidence=confidence,
                    probability=confidence * 0.5,  # Alternatives are less probable
                    source_memory_ids=[memory_id, neighbor_id],
                    pattern_similarity=base_similarity,
                    hdv=self._encode_text(neighbor.content),
                )
                scenarios.append(scenario)

        return scenarios

    async def _generate_anticipated_scenarios(
        self,
        context: str,
        context_hdv: BinaryHDV,
        horizon_hours: float,
        max_scenarios: int,
    ) -> List[ScenarioNode]:
        """
        Generate scenarios based on anticipatory engine predictions.

        Integrates with the anticipatory system to create scenarios
        from pre-loaded memories.
        """
        scenarios = []

        if not self.anticipatory_engine:
            return scenarios

        # Get anticipatory predictions
        # We use a representative memory ID from our context
        similar = await self._find_similar_memories(context_hdv, limit=1)
        if not similar:
            return scenarios

        representative_id = similar[0][0]
        predicted_ids = await self.anticipatory_engine.predict_and_preload(
            representative_id
        )

        for pred_id in predicted_ids[:max_scenarios]:
            memory = None
            if self.tier_manager:
                memory = await self.tier_manager.get_memory(pred_id)

            if not memory:
                continue

            projected_time = datetime.now(timezone.utc) + timedelta(hours=horizon_hours)

            scenario = ScenarioNode(
                type=ScenarioType.ANTICIPATED,
                content=f"Anticipated: {memory.content[:150]}...",
                context_summary="From anticipatory engine",
                projected_timestamp=projected_time.isoformat(),
                confidence=0.6,  # Anticipatory predictions are moderately confident
                probability=0.5,
                source_memory_ids=[pred_id],
                pattern_similarity=0.5,
                hdv=memory.hdv if hasattr(memory, "hdv") else self._encode_text(memory.content),
            )
            scenarios.append(scenario)

        return scenarios

    async def _generate_goal_scenarios(
        self,
        context: str,
        context_hdv: BinaryHDV,
        horizon_hours: float,
        max_scenarios: int,
    ) -> List[ScenarioNode]:
        """
        Generate goal-directed scenarios.

        These scenarios explore paths toward explicit goals or desired
        outcomes. If goal metadata is present in context, use it to
        project forward.
        """
        scenarios = []

        # Check for goal-related patterns in similar memories
        similar_memories = await self._find_similar_memories(
            context_hdv, limit=max_scenarios
        )

        for memory_id, similarity in similar_memories:
            if similarity < self.config.min_similarity_threshold:
                continue

            memory = None
            if self.tier_manager:
                memory = await self.tier_manager.get_memory(memory_id)

            if not memory or not memory.metadata:
                continue

            # Check for goal-related metadata
            goal = memory.metadata.get("goal") or memory.metadata.get("objective")
            outcome = memory.metadata.get("outcome") or memory.metadata.get("result")

            if not goal:
                continue

            # Generate scenario projecting toward this goal
            if outcome:
                content = f"Working toward '{goal}', potential outcome: {outcome}"
            else:
                content = f"Continuing progress toward '{goal}'"

            projected_time = datetime.now(timezone.utc) + timedelta(hours=horizon_hours)

            scenario = ScenarioNode(
                type=ScenarioType.GOAL_DIRECTED,
                content=content,
                context_summary=f"Goal: {goal}",
                projected_timestamp=projected_time.isoformat(),
                confidence=similarity * 0.65,
                probability=similarity * 0.5,
                source_memory_ids=[memory_id],
                pattern_similarity=similarity,
                hdv=self._encode_text(content),
                tags=["goal", "directed"],
            )
            scenarios.append(scenario)

        return scenarios

    # --------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------

    def _should_generate_type(
        self,
        scenario_type: ScenarioType,
        allowed_types: Optional[List[ScenarioType]] = None
    ) -> bool:
        """Check if a scenario type should be generated."""
        if allowed_types is None:
            return True
        return scenario_type in allowed_types

    async def _find_similar_memories(
        self,
        query_hdv: BinaryHDV,
        limit: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Find memories similar to the query HDV.

        Returns list of (memory_id, similarity) tuples.
        """
        if not self.tier_manager:
            return []

        # Search HOT tier first
        hot_results = self.tier_manager.search_hot(query_hdv, top_k=limit)

        # Filter by similarity threshold
        filtered = [
            (mid, sim) for mid, sim in hot_results
            if sim >= self.config.min_similarity_threshold
        ]

        return filtered[:limit]

    def _encode_text(self, text: str) -> BinaryHDV:
        """Encode text to BinaryHDV, with lazy encoder initialization."""
        if self._encoder is None:
            from mnemocore.core.binary_hdv import TextEncoder
            self._encoder = TextEncoder(dimension=self.dimension)
        return self._encoder.encode(text)

    # --------------------------------------------------------------
    # Integration with PredictionStore
    # --------------------------------------------------------------

    async def promote_to_prediction(
        self,
        scenario_id: str,
        confidence_threshold: float = 0.7,
    ) -> Optional[str]:
        """
        Promote a high-confidence scenario to a formal prediction.

        Useful for scenarios that have been validated and should be
        tracked for verification.
        """
        scenario = await self.scenario_store.get(scenario_id)
        if not scenario:
            return None

        if scenario.current_confidence() < confidence_threshold:
            logger.debug(
                f"Scenario {scenario_id} confidence {scenario.current_confidence():.2f} "
                f"below threshold {confidence_threshold}"
            )
            return None

        if not self.prediction_store:
            logger.warning("No PredictionStore available for promotion")
            return None

        # Create prediction from scenario
        prediction_id = self.prediction_store.create(
            content=scenario.content,
            confidence=scenario.probability,
            deadline_days=scenario.age_hours() / 24.0,
            related_memory_ids=scenario.source_memory_ids,
            tags=scenario.tags + ["eft_promoted"],
        )

        logger.info(f"Promoted scenario {scenario_id} to prediction {prediction_id}")
        return prediction_id

    async def verify_scenario_as_outcome(
        self,
        scenario_id: str,
        occurred: bool,
        notes: Optional[str] = None,
    ) -> Optional[ScenarioNode]:
        """
        Verify a scenario and optionally create/update a prediction.

        Bridges the gap between future scenarios and verified predictions.
        """
        scenario = await self.scenario_store.verify(scenario_id, occurred, notes)

        if scenario and self.prediction_store:
            # Create a corresponding prediction record for learning
            self.prediction_store.create(
                content=scenario.content,
                confidence=scenario.probability,
                tags=scenario.tags + ["eft_verified"],
            )

            # If scenario didn't occur, mark prediction as falsified
            if not occurred:
                # The prediction store doesn't return the ID from create,
                # but we can use the scenario metadata
                pass

        return scenario


# ==============================================================================
# Attention Integration
# ==============================================================================


class AttentionIntegration:
    """
    Integrates future scenarios with the XOR attention system.

    Scenarios influence attention by:
      1. Boosting attention to memories related to likely futures
      2. Providing novelty signals for unexpected potential outcomes
      3. Weighting retrieval based on temporal proximity to scenarios
    """

    def __init__(
        self,
        simulator: EpisodeFutureSimulator,
        attention_masker: Optional["XORAttentionMasker"] = None,
        config: Optional[EFTConfig] = None,
    ):
        self.simulator = simulator
        self.attention_masker = attention_masker
        self.config = config or EFTConfig()

    async def rerank_with_scenarios(
        self,
        query_results: List[Tuple[str, float]],
        context_hdv: BinaryHDV,
        query: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Re-rank query results based on active scenario relevance.

        Memories that align with likely future scenarios get a boost.
        """
        if not self.config.enabled:
            return query_results

        # Get active scenarios
        scenarios = await self.simulator.scenario_store.list_active()

        if not scenarios:
            return query_results

        # Calculate scenario boost for each result
        reranked = []
        for node_id, base_score in query_results:
            boost = await self._calculate_scenario_boost(
                node_id, scenarios, context_hdv
            )
            # Apply boost with configured factor
            adjusted_score = base_score * (1.0 + boost * self.config.attention_boost_factor)
            reranked.append((node_id, adjusted_score))

        # Re-sort
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    async def _calculate_scenario_boost(
        self,
        node_id: str,
        scenarios: List[ScenarioNode],
        context_hdv: BinaryHDV,
    ) -> float:
        """Calculate attention boost based on scenario relevance."""
        total_boost = 0.0

        for scenario in scenarios:
            # Check if node is a source of this scenario
            if node_id in scenario.source_memory_ids:
                # Direct source memories get higher boost
                total_boost += scenario.current_confidence() * 0.5
            else:
                # Check vector similarity
                if scenario.hdv:
                    similarity = scenario.hdv.similarity(context_hdv)
                    if similarity > 0.6:
                        total_boost += similarity * scenario.current_confidence() * 0.2

        return min(total_boost, 1.0)  # Cap at 1.0

    async def get_scenario_aware_attention_mask(
        self,
        query_hdv: BinaryHDV,
        context_hdv: BinaryHDV,
    ) -> Optional[BinaryHDV]:
        """
        Generate an attention mask that incorporates scenario information.

        The mask highlights features that are relevant to likely futures.
        """
        scenarios = await self.simulator.scenario_store.list_active()

        if not scenarios:
            return None

        # Bundle scenario HDVs
        scenario_hdvs = [s.hdv for s in scenarios if s.hdv]
        if not scenario_hdvs:
            return None

        # Create scenario bundle
        scenario_bundle = majority_bundle(scenario_hdvs)

        # XOR with context to find novel future-relevant features
        if self.attention_masker:
            base_mask = self.attention_masker.build_attention_mask(
                query_hdv, context_hdv
            )
            # Combine with scenario information
            combined = majority_bundle([base_mask, scenario_bundle])
            return combined

        return scenario_bundle


# ==============================================================================
# Utility functions
# ==============================================================================


async def create_future_thinking_pipeline(
    prediction_store: Optional["PredictionStore"] = None,
    anticipatory_engine: Optional["AnticipatoryEngine"] = None,
    tier_manager: Optional["TierManager"] = None,
    synapse_index: Optional["SynapseIndex"] = None,
    attention_masker: Optional["XORAttentionMasker"] = None,
    config: Optional[EFTConfig] = None,
) -> Tuple[EpisodeFutureSimulator, AttentionIntegration]:
    """
    Factory function to create a complete EFT pipeline.

    Returns:
        (simulator, attention_integration) tuple
    """
    cfg = config or EFTConfig()
    cfg.validate()

    dimension = get_config().dimensionality

    simulator = EpisodeFutureSimulator(
        config=cfg,
        prediction_store=prediction_store,
        anticipatory_engine=anticipatory_engine,
        tier_manager=tier_manager,
        synapse_index=synapse_index,
        dimension=dimension,
    )

    attention_integration = AttentionIntegration(
        simulator=simulator,
        attention_masker=attention_masker,
        config=cfg,
    )

    # Load persisted scenarios
    await simulator.scenario_store.load_from_storage()

    logger.info("Episodic Future Thinking pipeline initialized")

    return simulator, attention_integration


# ==============================================================================
# Exports
# ==============================================================================

__all__ = [
    "EFTConfig",
    "ScenarioType",
    "ScenarioNode",
    "ScenarioStore",
    "EpisodeFutureSimulator",
    "AttentionIntegration",
    "create_future_thinking_pipeline",
]
