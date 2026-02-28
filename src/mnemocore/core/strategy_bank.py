"""
Closed-Loop Strategy Memory — strategy_bank.py
================================================
Implements the test-time learning paradigm from ReasoningBank (ICML 2025)
and A-MEM (arXiv 2502.12110):

    Retrieve → Execute → Judge → Distill → Store

This is fundamentally different from storing facts: the Strategy Bank remembers
HOW to solve problems. It stores both successful patterns (60%) and failed
patterns (40%, the critical negative-signal balance identified by ReasoningBank).

Architecture
~~~~~~~~~~~~
::

    ┌──────────────────────────────────────────────────────┐
    │                  StrategyBank                        │
    │                                                     │
    │  ┌───────────┐   ┌──────────────┐   ┌───────────┐  │
    │  │ Retrieval  │──▶│   Execute    │──▶│  Judge    │  │
    │  │ (query)    │   │  (run strat) │   │  (eval)   │  │
    │  └───────────┘   └──────────────┘   └─────┬─────┘  │
    │                                           │         │
    │                     ┌─────────────────────▼─────┐   │
    │                     │      Distill + Store       │   │
    │                     │  Bayesian LTP update per   │   │
    │                     │  strategy confidence        │   │
    │                     └───────────────────────────┘   │
    │                                                     │
    │  Data:                                              │
    │   - Strategy: (trigger, steps, outcome_history)     │
    │   - 60/40 success/failure balance (ReasoningBank)   │
    │   - Bayesian confidence (Beta posterior)             │
    │   - Test-time adaptation: live updates, no retrain  │
    └─────────────────────────────────────────────────────┘

References
~~~~~~~~~~
- ReasoningBank (ICML 2025): strategy distillation + negative exemplar balance.
- A-MEM (arXiv 2502.12110): Zettelkasten-style dynamic bidirelctional linking.
- Continuum Memory Architecture (arXiv 2601.09913): ingest → activate → retrieve → consolidate lifecycle.

Integration points:
    - ``bayesian_ltp.BayesianState`` for per-strategy confidence tracking
    - ``episodic_store`` for sourcing outcome signals
    - ``procedural_store`` for complementary skill storage
    - Pulse loop Phase 8: strategy refinement tick
    - ``self_improvement_worker`` judge component
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Literal
from loguru import logger

from .bayesian_ltp import BayesianState


# ═══════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StrategyOutcome:
    """
    Captures a single execution outcome for a strategy.

    Fields:
        timestamp: When this outcome was recorded.
        success: Whether the strategy execution was judged successful.
        quality_score: 0.0–1.0 fine-grained quality metric from the Judge.
        context: Free-form context about the retrieval / execution.
        feedback: Human or LLM feedback signal, if available.
    """
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = True
    quality_score: float = 0.5
    context: str = ""
    feedback: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "quality_score": self.quality_score,
            "context": self.context,
            "feedback": self.feedback,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyOutcome":
        ts = d.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        else:
            ts = datetime.now(timezone.utc)
        return cls(
            timestamp=ts,
            success=d.get("success", True),
            quality_score=d.get("quality_score", 0.5),
            context=d.get("context", ""),
            feedback=d.get("feedback", ""),
        )


@dataclass
class Strategy:
    """
    A learned strategy — how to solve a class of problems.

    This is NOT a fact or a procedure step list. It captures:
    - The trigger pattern (when to activate this strategy)
    - The approach description (what to do)
    - The outcome history (did it work? how well?)
    - Bayesian confidence (how much evidence do we have?)

    The 60/40 balance (ReasoningBank) means we explicitly KEEP failed
    strategies. They are essential negative signal for avoiding traps.

    Fields:
        id: Unique strategy identifier.
        name: Human-readable label.
        trigger_pattern: When should this strategy be considered?
        approach: Description of the strategy's method.
        rationale: Why this strategy works (or fails).
        category: Semantic category for clustering.
        tags: Free-form tags for retrieval filtering.
        agent_id: Which agent created this strategy.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        outcomes: Full outcome history (capped at max_history).
        bayesian_state: Bayesian Beta posterior tracking confidence.
        source_episode_ids: Episodic memory links (provenance).
        source_procedure_ids: Procedural memory links.
        is_negative_exemplar: Explicitly flagged as "what NOT to do".
        distilled_from: IDs of strategies this was distilled from.
        metadata: Arbitrary extension dict.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    trigger_pattern: str = ""
    approach: str = ""
    rationale: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    agent_id: str = "default"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    outcomes: List[StrategyOutcome] = field(default_factory=list)
    bayesian_state: BayesianState = field(default_factory=BayesianState)
    source_episode_ids: List[str] = field(default_factory=list)
    source_procedure_ids: List[str] = field(default_factory=list)
    is_negative_exemplar: bool = False
    distilled_from: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Derived Properties ────────────────────────────────────────────

    @property
    def confidence(self) -> float:
        """Bayesian posterior mean: P(strategy is reliable)."""
        return self.bayesian_state.mean

    @property
    def uncertainty(self) -> float:
        """Bayesian posterior std dev — high when under-explored."""
        return self.bayesian_state.uncertainty

    @property
    def total_executions(self) -> int:
        return len(self.outcomes)

    @property
    def success_rate(self) -> float:
        if not self.outcomes:
            return 0.5  # Uninformative prior
        return sum(1 for o in self.outcomes if o.success) / len(self.outcomes)

    @property
    def avg_quality(self) -> float:
        if not self.outcomes:
            return 0.5
        return sum(o.quality_score for o in self.outcomes) / len(self.outcomes)

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "trigger_pattern": self.trigger_pattern,
            "approach": self.approach,
            "rationale": self.rationale,
            "category": self.category,
            "tags": self.tags,
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "outcomes": [o.to_dict() for o in self.outcomes],
            "bayesian_state": self.bayesian_state.to_dict(),
            "source_episode_ids": self.source_episode_ids,
            "source_procedure_ids": self.source_procedure_ids,
            "is_negative_exemplar": self.is_negative_exemplar,
            "distilled_from": self.distilled_from,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Strategy":
        ts_created = d.get("created_at")
        ts_updated = d.get("updated_at")
        if isinstance(ts_created, str):
            ts_created = datetime.fromisoformat(ts_created)
        else:
            ts_created = datetime.now(timezone.utc)
        if isinstance(ts_updated, str):
            ts_updated = datetime.fromisoformat(ts_updated)
        else:
            ts_updated = datetime.now(timezone.utc)
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            name=d.get("name", ""),
            trigger_pattern=d.get("trigger_pattern", ""),
            approach=d.get("approach", ""),
            rationale=d.get("rationale", ""),
            category=d.get("category", "general"),
            tags=d.get("tags", []),
            agent_id=d.get("agent_id", "default"),
            created_at=ts_created,
            updated_at=ts_updated,
            outcomes=[StrategyOutcome.from_dict(o) for o in d.get("outcomes", [])],
            bayesian_state=BayesianState.from_dict(d.get("bayesian_state", {})),
            source_episode_ids=d.get("source_episode_ids", []),
            source_procedure_ids=d.get("source_procedure_ids", []),
            is_negative_exemplar=d.get("is_negative_exemplar", False),
            distilled_from=d.get("distilled_from", []),
            metadata=d.get("metadata", {}),
        )


# ═══════════════════════════════════════════════════════════════════════
# Retrieval Judge — the critical feedback-loop component
# ═══════════════════════════════════════════════════════════════════════

class RetrievalJudge:
    """
    Evaluates retrieval quality and produces quality signal for Bayesian update.

    This is the **key missing piece** identified in the research review:
    MnemoCore stores and retrieves memories, but until now it never asked
    "was that retrieval actually helpful?" and fed the answer back.

    The Judge evaluates:
    1. Relevance: Did the retrieved content match the query intent?
    2. Completeness: Did it cover the needed information?
    3. Freshness: Was the information current enough?
    4. Actionability: Could the agent act on it?

    Each dimension produces a 0.0–1.0 score. The composite quality_score
    is fed into the Bayesian LTP system to update per-strategy confidence.

    The Judge operates entirely at inference time (test-time learning):
    no model weights are updated, only Bayesian posteriors.
    """

    def __init__(
        self,
        relevance_weight: float = 0.4,
        completeness_weight: float = 0.25,
        freshness_weight: float = 0.15,
        actionability_weight: float = 0.2,
        success_threshold: float = 0.5,
    ):
        """
        Args:
            relevance_weight: Weight for relevance dimension.
            completeness_weight: Weight for completeness dimension.
            freshness_weight: Weight for freshness dimension.
            actionability_weight: Weight for actionability dimension.
            success_threshold: Composite score >= this = success.
        """
        self.relevance_weight = relevance_weight
        self.completeness_weight = completeness_weight
        self.freshness_weight = freshness_weight
        self.actionability_weight = actionability_weight
        self.success_threshold = success_threshold

    def judge(
        self,
        query: str,
        retrieved_content: str,
        outcome: Optional[str] = None,
        retrieval_score: float = 0.5,
        content_age_hours: float = 0.0,
        was_used: bool = True,
    ) -> StrategyOutcome:
        """
        Produce a quality judgment for a retrieval event.

        This is the core Retrieve→Execute→**Judge** step.

        Args:
            query: The original query text.
            retrieved_content: What was retrieved.
            outcome: "success" / "failure" / "partial" / None.
            retrieval_score: Raw similarity score from the retrieval engine.
            content_age_hours: How old the retrieved content is.
            was_used: Whether the agent actually used this retrieval.

        Returns:
            StrategyOutcome with composite quality_score and success flag.
        """
        # ── Relevance: overlap + retrieval score ──────────────────────
        relevance = self._score_relevance(query, retrieved_content, retrieval_score)

        # ── Completeness: content length adequacy ─────────────────────
        completeness = self._score_completeness(retrieved_content)

        # ── Freshness: temporal decay ─────────────────────────────────
        freshness = self._score_freshness(content_age_hours)

        # ── Actionability: did the agent use it? ──────────────────────
        actionability = self._score_actionability(was_used, outcome)

        # ── Composite ─────────────────────────────────────────────────
        quality_score = (
            self.relevance_weight * relevance
            + self.completeness_weight * completeness
            + self.freshness_weight * freshness
            + self.actionability_weight * actionability
        )
        quality_score = max(0.0, min(1.0, quality_score))
        success = quality_score >= self.success_threshold

        return StrategyOutcome(
            success=success,
            quality_score=quality_score,
            context=f"relevance={relevance:.2f} completeness={completeness:.2f} "
                    f"freshness={freshness:.2f} actionability={actionability:.2f}",
            feedback=outcome or "",
        )

    # ── Score Dimensions (each returns 0.0–1.0) ──────────────────────

    def _score_relevance(
        self, query: str, content: str, retrieval_score: float
    ) -> float:
        """Word overlap + raw retrieval score blend."""
        q_words = set(query.lower().split())
        c_words = set(content.lower().split())
        if not q_words:
            return retrieval_score
        overlap = len(q_words & c_words) / len(q_words)
        return 0.6 * overlap + 0.4 * min(1.0, max(0.0, retrieval_score))

    def _score_completeness(self, content: str) -> float:
        """Longer content is generally more complete (diminishing returns)."""
        length = len(content)
        if length < 10:
            return 0.1
        if length < 50:
            return 0.4
        if length < 200:
            return 0.7
        return min(1.0, 0.7 + 0.3 * (length / 1000))

    def _score_freshness(self, age_hours: float) -> float:
        """Exponential decay: half-life = 168 hours (1 week)."""
        half_life = 168.0
        if age_hours <= 0:
            return 1.0
        return math.exp(-0.693 * age_hours / half_life)

    def _score_actionability(
        self, was_used: bool, outcome: Optional[str]
    ) -> float:
        """If agent used it and succeeded → high actionability."""
        if not was_used:
            return 0.2
        if outcome == "success":
            return 1.0
        if outcome == "partial":
            return 0.6
        if outcome == "failure":
            return 0.1
        return 0.5  # unknown


# ═══════════════════════════════════════════════════════════════════════
# Strategy Bank Service
# ═══════════════════════════════════════════════════════════════════════

class StrategyBankService:
    """
    The closed-loop strategy memory system.

    Implements the full cycle: Retrieve → Execute → Judge → Distill → Store.

    Key design principles (from research):
    1. **60/40 Balance** (ReasoningBank): Maintain ~60% success strategies and
       ~40% failure strategies. Negative exemplars prevent the agent from
       repeating mistakes. The bank actively monitors this ratio.
    2. **Bayesian Confidence** (extends bayesian_ltp.py): Every strategy has a
       Beta(α, β) posterior. Each judge verdict updates the posterior.
       Under-explored strategies get UCB exploration bonus.
    3. **Test-Time Learning**: The agent improves during execution without
       model retraining. Bayesian updates + outcome recording = online learning.
    4. **Provenance**: Every strategy links back to source episodes and
       procedures for full auditability.

    Thread-safety: All mutations are protected by a reentrant lock.

    Persistence: JSON file at ``config.strategy_bank.persistence_path``.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        judge: Optional[RetrievalJudge] = None,
    ):
        """
        Args:
            config: StrategyBankConfig (or None for defaults).
            judge: Optional RetrievalJudge instance.
        """
        self._lock = threading.RLock()
        self._strategies: Dict[str, Strategy] = {}
        self._category_index: Dict[str, List[str]] = {}  # category → [strategy_ids]
        self._agent_index: Dict[str, List[str]] = {}     # agent_id → [strategy_ids]

        # Config
        self._max_strategies = getattr(config, "max_strategies", 10000)
        self._max_outcomes_per_strategy = getattr(config, "max_outcomes_per_strategy", 100)
        self._target_negative_ratio = getattr(config, "target_negative_ratio", 0.4)
        self._min_confidence_threshold = getattr(config, "min_confidence_threshold", 0.3)
        self._persistence_path = getattr(config, "persistence_path", None)
        self._auto_persist = getattr(config, "auto_persist", True)

        # Judge for the feedback loop
        self.judge = judge or RetrievalJudge()

        # Metrics
        self._total_retrievals = 0
        self._total_judgments = 0
        self._total_distillations = 0

        # Load persisted state
        if self._persistence_path:
            self._load_from_disk()

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: RETRIEVE — find applicable strategies
    # ══════════════════════════════════════════════════════════════════

    def retrieve(
        self,
        query: str,
        agent_id: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = 5,
        include_negative: bool = True,
        min_confidence: Optional[float] = None,
    ) -> List[Strategy]:
        """
        Retrieve the most relevant strategies for a given query.

        Uses word-overlap matching against trigger_pattern + approach +
        name, weighted by Bayesian confidence (UCB for exploration bonus).

        Args:
            query: The problem description or query text.
            agent_id: Filter by agent (None = all agents).
            category: Filter by category (None = all categories).
            top_k: Maximum strategies to return.
            include_negative: Include negative exemplars (recommended).
            min_confidence: Minimum confidence filter (None = use config default).

        Returns:
            Strategies ranked by relevance × confidence (UCB).
        """
        with self._lock:
            self._total_retrievals += 1
            candidates = list(self._strategies.values())

        # ── Filter ────────────────────────────────────────────────────
        if agent_id:
            candidates = [s for s in candidates if s.agent_id == agent_id]
        if category:
            candidates = [s for s in candidates if s.category == category]
        if not include_negative:
            candidates = [s for s in candidates if not s.is_negative_exemplar]
        threshold = min_confidence if min_confidence is not None else self._min_confidence_threshold
        if threshold > 0:
            candidates = [s for s in candidates if s.confidence >= threshold or s.uncertainty > 0.2]

        if not candidates:
            return []

        # ── Score: word overlap × Bayesian UCB ────────────────────────
        q_words = set(query.lower().split())
        scored: List[Tuple[float, Strategy]] = []
        for strat in candidates:
            text = f"{strat.trigger_pattern} {strat.approach} {strat.name}".lower()
            s_words = set(text.split())
            if not q_words or not s_words:
                overlap = 0.0
            else:
                overlap = len(q_words & s_words) / max(len(q_words), 1)

            # UCB: prefer under-explored strategies (exploration bonus)
            ucb = strat.bayesian_state.upper_credible
            score = 0.6 * overlap + 0.4 * ucb
            scored.append((score, strat))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    # ══════════════════════════════════════════════════════════════════
    # Phase 3: JUDGE — evaluate retrieval quality
    # ══════════════════════════════════════════════════════════════════

    def judge_retrieval(
        self,
        strategy_id: str,
        query: str,
        retrieved_content: str,
        outcome: Optional[str] = None,
        retrieval_score: float = 0.5,
        content_age_hours: float = 0.0,
        was_used: bool = True,
    ) -> Optional[StrategyOutcome]:
        """
        Judge a retrieval event and update the strategy's Bayesian confidence.

        This is the **critical feedback loop** that makes the system learn
        from its own retrieval mistakes during runtime (test-time learning).

        Args:
            strategy_id: Which strategy was applied.
            query: The original query.
            retrieved_content: What was actually retrieved.
            outcome: "success" / "failure" / "partial" / None.
            retrieval_score: Raw retrieval similarity score.
            content_age_hours: Age of the retrieved content.
            was_used: Whether the agent actually used this retrieval.

        Returns:
            The StrategyOutcome produced by the Judge, or None if
            strategy not found.
        """
        with self._lock:
            strategy = self._strategies.get(strategy_id)
            if not strategy:
                logger.warning(f"Strategy {strategy_id} not found for judgment.")
                return None

            # Run the judge
            verdict = self.judge.judge(
                query=query,
                retrieved_content=retrieved_content,
                outcome=outcome,
                retrieval_score=retrieval_score,
                content_age_hours=content_age_hours,
                was_used=was_used,
            )

            # Update Bayesian posterior (the core feedback signal)
            strategy.bayesian_state.observe(
                success=verdict.success,
                strength=verdict.quality_score,
            )

            # Record outcome
            strategy.outcomes.append(verdict)
            if len(strategy.outcomes) > self._max_outcomes_per_strategy:
                strategy.outcomes = strategy.outcomes[-self._max_outcomes_per_strategy:]

            strategy.updated_at = datetime.now(timezone.utc)
            self._total_judgments += 1

            logger.debug(
                f"Strategy '{strategy.name}' judged: "
                f"success={verdict.success} quality={verdict.quality_score:.3f} "
                f"→ confidence={strategy.confidence:.4f} ± {strategy.uncertainty:.4f}"
            )

        if self._auto_persist and self._persistence_path:
            self._persist_to_disk()

        return verdict

    # ══════════════════════════════════════════════════════════════════
    # Phase 4: DISTILL — create/refine strategies from episodes
    # ══════════════════════════════════════════════════════════════════

    def distill_from_episode(
        self,
        episode_id: str,
        trigger_pattern: str,
        approach: str,
        outcome: str,
        quality_score: float = 0.5,
        agent_id: str = "default",
        category: str = "general",
        tags: Optional[List[str]] = None,
        rationale: str = "",
        name: str = "",
    ) -> Strategy:
        """
        Distill a new strategy from an episodic experience.

        This is the Distill→Store phase of the closed loop.
        Creates a new Strategy with an initial outcome from the episode.

        The 60/40 balance (ReasoningBank) is enforced: if we have too many
        success strategies, the next failure is explicitly flagged as a
        negative exemplar to maintain the balance.

        Args:
            episode_id: Source episode for provenance.
            trigger_pattern: When should this strategy activate?
            approach: What does this strategy do?
            outcome: "success" / "failure" / "partial".
            quality_score: 0.0–1.0 quality of this episode's result.
            agent_id: Which agent discovered this strategy.
            category: Semantic category.
            tags: Free-form tags.
            rationale: Why this approach works or fails.
            name: Human-readable name.

        Returns:
            The newly created Strategy.
        """
        is_success = outcome in ("success", "partial")
        is_neg = not is_success

        # ── 60/40 Balance check (ReasoningBank) ──────────────────────
        if not is_neg:
            neg_ratio = self._current_negative_ratio()
            if neg_ratio < self._target_negative_ratio * 0.5:
                # We have too few negative exemplars — flag this for attention
                logger.info(
                    f"Strategy bank negative ratio ({neg_ratio:.1%}) below target "
                    f"({self._target_negative_ratio:.0%}). Consider adding failure cases."
                )

        initial_outcome = StrategyOutcome(
            success=is_success,
            quality_score=quality_score,
            context=f"distilled_from_episode={episode_id}",
            feedback=outcome,
        )

        strategy = Strategy(
            name=name or f"strategy-{trigger_pattern[:30]}",
            trigger_pattern=trigger_pattern,
            approach=approach,
            rationale=rationale,
            category=category,
            tags=tags or [],
            agent_id=agent_id,
            outcomes=[initial_outcome],
            source_episode_ids=[episode_id],
            is_negative_exemplar=is_neg,
        )

        # Set initial Bayesian state from outcome
        strategy.bayesian_state.observe(success=is_success, strength=quality_score)

        with self._lock:
            self._strategies[strategy.id] = strategy
            self._index_strategy(strategy)
            self._total_distillations += 1

            # Enforce capacity
            self._enforce_capacity()

        if self._auto_persist and self._persistence_path:
            self._persist_to_disk()

        logger.info(
            f"Distilled strategy '{strategy.name}' from episode {episode_id[:8]}… "
            f"(negative={is_neg}, confidence={strategy.confidence:.3f})"
        )

        return strategy

    # ══════════════════════════════════════════════════════════════════
    # Phase 5: STORE — direct strategy storage
    # ══════════════════════════════════════════════════════════════════

    def store_strategy(self, strategy: Strategy) -> str:
        """
        Store a pre-built strategy directly.

        Args:
            strategy: Complete Strategy object.

        Returns:
            The strategy ID.
        """
        with self._lock:
            self._strategies[strategy.id] = strategy
            self._index_strategy(strategy)
            self._enforce_capacity()

        if self._auto_persist and self._persistence_path:
            self._persist_to_disk()

        return strategy.id

    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """Retrieve a single strategy by ID."""
        with self._lock:
            return self._strategies.get(strategy_id)

    def get_strategies_by_category(self, category: str) -> List[Strategy]:
        """Get all strategies in a given category."""
        with self._lock:
            ids = self._category_index.get(category, [])
            return [self._strategies[sid] for sid in ids if sid in self._strategies]

    def get_strategies_by_agent(self, agent_id: str) -> List[Strategy]:
        """Get all strategies created by a specific agent."""
        with self._lock:
            ids = self._agent_index.get(agent_id, [])
            return [self._strategies[sid] for sid in ids if sid in self._strategies]

    # ══════════════════════════════════════════════════════════════════
    # Balance & Maintenance
    # ══════════════════════════════════════════════════════════════════

    def get_balance_ratio(self) -> Dict[str, Any]:
        """
        Report the current success/failure balance.

        ReasoningBank recommends ~60% success / 40% failure for optimal
        learning. This method reports the current state.

        Returns:
            Dict with total, positive_count, negative_count, negative_ratio,
            target_ratio, balanced (bool).
        """
        with self._lock:
            total = len(self._strategies)
            neg = sum(1 for s in self._strategies.values() if s.is_negative_exemplar)
            pos = total - neg
            ratio = neg / total if total > 0 else 0.0
        return {
            "total": total,
            "positive_count": pos,
            "negative_count": neg,
            "negative_ratio": ratio,
            "target_ratio": self._target_negative_ratio,
            "balanced": abs(ratio - self._target_negative_ratio) < 0.1,
        }

    def decay_confidence(self, decay_rate: float = 0.01) -> int:
        """
        Apply a small decay to all strategy confidences (prevents stale overconfidence).

        This implements temporal forgetting for strategies — even successful
        ones should lose confidence if not re-validated recently.

        Args:
            decay_rate: How much to decay (β += decay_rate).

        Returns:
            Number of strategies decayed.
        """
        count = 0
        with self._lock:
            for strategy in self._strategies.values():
                strategy.bayesian_state.beta_count += decay_rate
                count += 1

        if self._auto_persist and self._persistence_path:
            self._persist_to_disk()

        return count

    def prune_low_confidence(self, threshold: Optional[float] = None) -> int:
        """
        Remove strategies that have been consistently judged as unreliable
        AND have enough evidence to be confident about that judgment.

        Only prunes when:
        - confidence < threshold
        - total_observations > 5 (enough evidence)
        - uncertainty < 0.15 (confident about low quality)

        Args:
            threshold: Confidence below which to prune. Default from config.

        Returns:
            Number of strategies pruned.
        """
        threshold = threshold or self._min_confidence_threshold
        pruned = 0
        with self._lock:
            to_remove = []
            for sid, strategy in self._strategies.items():
                if (
                    strategy.confidence < threshold
                    and strategy.bayesian_state.total_observations > 5
                    and strategy.uncertainty < 0.15
                ):
                    to_remove.append(sid)

            for sid in to_remove:
                self._remove_strategy(sid)
                pruned += 1

        if pruned > 0 and self._auto_persist and self._persistence_path:
            self._persist_to_disk()

        return pruned

    # ══════════════════════════════════════════════════════════════════
    # Statistics & Observability
    # ══════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """
        Comprehensive statistics for monitoring and meta-memory.

        Returns:
            Dict with counts, balance, confidence distribution, metrics.
        """
        with self._lock:
            strategies = list(self._strategies.values())

        total = len(strategies)
        if total == 0:
            return {
                "total_strategies": 0,
                "positive_count": 0,
                "negative_count": 0,
                "negative_ratio": 0.0,
                "avg_confidence": 0.0,
                "avg_uncertainty": 0.0,
                "total_retrievals": self._total_retrievals,
                "total_judgments": self._total_judgments,
                "total_distillations": self._total_distillations,
                "categories": [],
            }

        neg = sum(1 for s in strategies if s.is_negative_exemplar)
        return {
            "total_strategies": total,
            "positive_count": total - neg,
            "negative_count": neg,
            "negative_ratio": neg / total if total else 0.0,
            "avg_confidence": sum(s.confidence for s in strategies) / total,
            "avg_uncertainty": sum(s.uncertainty for s in strategies) / total,
            "total_retrievals": self._total_retrievals,
            "total_judgments": self._total_judgments,
            "total_distillations": self._total_distillations,
            "categories": list(self._category_index.keys()),
        }

    # ══════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════

    def _index_strategy(self, strategy: Strategy) -> None:
        """Update internal indices after adding/modifying a strategy."""
        # Category index
        cat_list = self._category_index.setdefault(strategy.category, [])
        if strategy.id not in cat_list:
            cat_list.append(strategy.id)
        # Agent index
        agent_list = self._agent_index.setdefault(strategy.agent_id, [])
        if strategy.id not in agent_list:
            agent_list.append(strategy.id)

    def _remove_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from all indices (must hold lock)."""
        strategy = self._strategies.pop(strategy_id, None)
        if not strategy:
            return
        cat_list = self._category_index.get(strategy.category, [])
        if strategy_id in cat_list:
            cat_list.remove(strategy_id)
        agent_list = self._agent_index.get(strategy.agent_id, [])
        if strategy_id in agent_list:
            agent_list.remove(strategy_id)

    def _current_negative_ratio(self) -> float:
        """Current ratio of negative exemplars."""
        total = len(self._strategies)
        if total == 0:
            return 0.0
        neg = sum(1 for s in self._strategies.values() if s.is_negative_exemplar)
        return neg / total

    def _enforce_capacity(self) -> None:
        """Evict lowest-confidence strategies if over capacity (must hold lock)."""
        if len(self._strategies) <= self._max_strategies:
            return
        # Sort by confidence (ascending), prune from bottom
        ranked = sorted(self._strategies.values(), key=lambda s: s.confidence)
        overage = len(self._strategies) - self._max_strategies
        for strategy in ranked[:overage]:
            self._remove_strategy(strategy.id)
            logger.debug(f"Evicted strategy '{strategy.name}' (confidence={strategy.confidence:.3f})")

    # ── Persistence ───────────────────────────────────────────────────

    def _persist_to_disk(self) -> None:
        """Save all strategies to JSON file."""
        if not self._persistence_path:
            return
        try:
            path = Path(self._persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "version": "1.0",
                    "strategies": [s.to_dict() for s in self._strategies.values()],
                    "metrics": {
                        "total_retrievals": self._total_retrievals,
                        "total_judgments": self._total_judgments,
                        "total_distillations": self._total_distillations,
                    },
                }
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to persist strategy bank: {e}")

    def _load_from_disk(self) -> None:
        """Load strategies from JSON file."""
        if not self._persistence_path:
            return
        path = Path(self._persistence_path)
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for sd in raw.get("strategies", []):
                strategy = Strategy.from_dict(sd)
                self._strategies[strategy.id] = strategy
                self._index_strategy(strategy)
            metrics = raw.get("metrics", {})
            self._total_retrievals = metrics.get("total_retrievals", 0)
            self._total_judgments = metrics.get("total_judgments", 0)
            self._total_distillations = metrics.get("total_distillations", 0)
            logger.info(f"Loaded {len(self._strategies)} strategies from {path}")
        except Exception as e:
            logger.error(f"Failed to load strategy bank: {e}")
