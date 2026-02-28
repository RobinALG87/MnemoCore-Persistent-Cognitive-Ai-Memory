"""
Tests for Closed-Loop Strategy Memory (strategy_bank.py)
=========================================================
Covers StrategyBankService, RetrievalJudge, Strategy, StrategyOutcome.

Research basis: ReasoningBank + A-MEM (arXiv 2502.12110)
Retrieve → Execute → Judge → Distill → Store cycle.
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mnemocore.core.strategy_bank import (
    Strategy,
    StrategyBankService,
    StrategyOutcome,
    RetrievalJudge,
)
from mnemocore.core.bayesian_ltp import BayesianState


# ═══════════════════════════════════════════════════════════════════════
# Strategy dataclass
# ═══════════════════════════════════════════════════════════════════════

class TestStrategy:

    def test_default_strategy(self):
        s = Strategy(trigger_pattern="test", approach="do thing", rationale="because")
        assert s.trigger_pattern == "test"
        assert s.approach == "do thing"
        assert s.rationale == "because"
        assert s.confidence > 0
        assert s.uncertainty > 0
        assert s.is_negative_exemplar is False
        assert s.category == "general"

    def test_success_rate_no_outcomes(self):
        s = Strategy(trigger_pattern="t", approach="a", rationale="r")
        assert s.success_rate == 0.5  # Uninformative Bayesian prior

    def test_success_rate_with_outcomes(self):
        s = Strategy(trigger_pattern="t", approach="a", rationale="r")
        s.outcomes.append(StrategyOutcome(success=True, quality_score=0.8))
        s.outcomes.append(StrategyOutcome(success=False, quality_score=0.2))
        assert s.success_rate == 0.5

    def test_avg_quality_no_outcomes(self):
        s = Strategy(trigger_pattern="t", approach="a", rationale="r")
        assert s.avg_quality == 0.5  # Uninformative prior

    def test_avg_quality(self):
        s = Strategy(trigger_pattern="t", approach="a", rationale="r")
        s.outcomes.append(StrategyOutcome(success=True, quality_score=0.8))
        s.outcomes.append(StrategyOutcome(success=True, quality_score=0.6))
        assert abs(s.avg_quality - 0.7) < 0.01

    def test_to_dict_roundtrip(self):
        s = Strategy(
            trigger_pattern="pattern",
            approach="do this",
            rationale="explanation",
            category="debugging",
            agent_id="agent-1",
        )
        s.outcomes.append(StrategyOutcome(success=True, quality_score=0.9))
        d = s.to_dict()
        s2 = Strategy.from_dict(d)
        assert s2.trigger_pattern == s.trigger_pattern
        assert s2.approach == s.approach
        assert s2.category == s.category
        assert s2.agent_id == s.agent_id
        assert len(s2.outcomes) == 1

    def test_negative_exemplar(self):
        s = Strategy(
            trigger_pattern="bad pattern",
            approach="wrong approach",
            rationale="learn from mistake",
            is_negative_exemplar=True,
        )
        assert s.is_negative_exemplar is True


# ═══════════════════════════════════════════════════════════════════════
# RetrievalJudge
# ═══════════════════════════════════════════════════════════════════════

class TestRetrievalJudge:

    def test_default_weights(self):
        judge = RetrievalJudge()
        assert abs(judge.relevance_weight + judge.completeness_weight +
                    judge.freshness_weight + judge.actionability_weight - 1.0) < 0.01

    def test_judge_perfect_match(self):
        judge = RetrievalJudge()
        outcome = judge.judge(
            query="how to debug python code",
            retrieved_content="how to debug python code effectively using pdb and breakpoints",
            outcome="success",
            retrieval_score=0.95,
        )
        assert outcome.success is True
        assert outcome.quality_score > 0.5

    def test_judge_no_content(self):
        judge = RetrievalJudge()
        outcome = judge.judge(
            query="anything",
            retrieved_content="",
            outcome=None,
            retrieval_score=0.0,
        )
        assert outcome.quality_score < 0.5

    def test_judge_irrelevant(self):
        judge = RetrievalJudge()
        outcome = judge.judge(
            query="quantum physics theory",
            retrieved_content="cooking pasta requires boiling water first",
            outcome="failure",
            retrieval_score=0.1,
        )
        assert outcome.quality_score < 0.5


# ═══════════════════════════════════════════════════════════════════════
# StrategyBankService
# ═══════════════════════════════════════════════════════════════════════

class TestStrategyBankService:

    def _make_strategy(self, **kwargs):
        defaults = dict(trigger_pattern="test", approach="do it", rationale="why")
        defaults.update(kwargs)
        return Strategy(**defaults)

    def test_store_and_retrieve(self):
        svc = StrategyBankService()
        s = self._make_strategy(
            trigger_pattern="deploy to production",
            approach="run CI/CD pipeline",
            rationale="ensures quality",
            category="devops",
        )
        svc.store_strategy(s)
        results = svc.retrieve("deploy production", top_k=5)
        assert len(results) >= 1
        assert results[0].trigger_pattern == "deploy to production"

    def test_retrieve_empty(self):
        svc = StrategyBankService()
        results = svc.retrieve("nonexistent query")
        assert results == []

    def test_judge_retrieval_updates_bayesian(self):
        svc = StrategyBankService()
        s = self._make_strategy(
            trigger_pattern="fix bug in module",
            approach="add unit test first",
            rationale="TDD approach",
        )
        svc.store_strategy(s)
        initial_alpha = s.bayesian_state.alpha
        verdict = svc.judge_retrieval(
            strategy_id=s.id,
            query="fix bug in module",
            retrieved_content="add unit test first to reproduce the bug",
            outcome="success",
            retrieval_score=0.8,
        )
        assert verdict is not None
        assert s.bayesian_state.alpha != initial_alpha or len(s.outcomes) > 0

    def test_distill_from_episode(self):
        svc = StrategyBankService()
        result = svc.distill_from_episode(
            episode_id="ep-1",
            trigger_pattern="memory leak detected",
            approach="use memory profiler, find allocation source",
            outcome="success",
            quality_score=0.8,
            agent_id="agent-1",
            rationale="profiling reveals root cause",
        )
        assert result is not None
        assert result.agent_id == "agent-1"
        assert len(result.source_episode_ids) == 1
        assert result.source_episode_ids[0] == "ep-1"

    def test_balance_ratio(self):
        svc = StrategyBankService()
        s1 = self._make_strategy(trigger_pattern="good", is_negative_exemplar=False)
        s2 = self._make_strategy(trigger_pattern="bad", is_negative_exemplar=True)
        svc.store_strategy(s1)
        svc.store_strategy(s2)
        ratio = svc.get_balance_ratio()
        assert "negative_ratio" in ratio

    def test_decay_confidence(self):
        svc = StrategyBankService()
        s = self._make_strategy()
        svc.store_strategy(s)
        initial_conf = s.confidence
        svc.decay_confidence(0.1)
        assert s.confidence <= initial_conf

    def test_prune_low_confidence(self):
        svc = StrategyBankService()
        s = self._make_strategy()
        svc.store_strategy(s)
        s.bayesian_state = BayesianState(alpha=0.01, beta_count=10.0)
        for _ in range(10):
            s.outcomes.append(StrategyOutcome(success=False, quality_score=0.01))
        pruned = svc.prune_low_confidence(threshold=0.05)
        assert isinstance(pruned, int)

    def test_get_stats(self):
        svc = StrategyBankService()
        svc.store_strategy(self._make_strategy(category="cat1"))
        svc.store_strategy(self._make_strategy(category="cat2"))
        stats = svc.get_stats()
        assert stats["total_strategies"] == 2
        assert "cat1" in stats["categories"]
        assert "cat2" in stats["categories"]

    def test_persistence(self, tmp_path):
        path = tmp_path / "strategies.json"
        svc = StrategyBankService(config=MagicMock(
            persistence_path=str(path),
            auto_persist=True,
            max_strategies=10000,
        ))
        svc.store_strategy(self._make_strategy(trigger_pattern="persistent"))
        assert path.exists()

        svc2 = StrategyBankService(config=MagicMock(
            persistence_path=str(path),
            auto_persist=True,
            max_strategies=10000,
        ))
        assert len(svc2._strategies) == 1

    def test_category_index(self):
        svc = StrategyBankService()
        svc.store_strategy(self._make_strategy(trigger_pattern="debug a", category="debugging"))
        svc.store_strategy(self._make_strategy(trigger_pattern="debug b", category="debugging"))
        svc.store_strategy(self._make_strategy(trigger_pattern="deploy c", category="deployment"))
        results = svc.retrieve("debug", top_k=10, category="debugging")
        assert all(s.category == "debugging" for s in results)
