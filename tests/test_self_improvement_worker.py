"""
Dedicated tests for SelfImprovementWorker (Phase 5.4).
=======================================================
Covers:
  - Candidate selection heuristics (short content, missing metadata, near-duplicates)
  - Proposal generation (normalize, metadata_repair, deduplicate)
  - Validation gate edge cases and combinations
  - Decision logging + log cap enforcement
  - Cooldown mechanism
  - Backpressure detection
  - Error resilience (tier_manager failures)
  - Full cycle with mixed candidates
  - Metrics accuracy after multiple cycles
  - Factory function with real config
  - Background loop start / stop / cancel
  - Resource gate timeout mid-cycle
  - Dry-run vs non-dry-run behaviour
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from mnemocore.core.config import SelfImprovementConfig
from mnemocore.subconscious.self_improvement_worker import (
    SelfImprovementWorker,
    SelfImprovementMetrics,
    ValidationGates,
    ImprovementCandidate,
    ImprovementProposal,
    create_self_improvement_worker,
)


# ─── Helpers ─────────────────────────────────────────────────────────

def _mock_engine(hot_memories=None):
    """Create a mock engine with tier_manager and meta_memory."""
    engine = MagicMock()
    engine.tier_manager = MagicMock()
    engine.tier_manager.get_hot_recent = MagicMock(return_value=hot_memories or [])
    engine.meta_memory = MagicMock()
    engine.meta_memory.record_metric = MagicMock()
    # Remove subconscious queue so backpressure never fires by default
    if hasattr(engine, "_subconscious_queue"):
        del engine._subconscious_queue
    return engine


def _make_worker(config=None, engine=None, hot_memories=None):
    """Convenience factory."""
    eng = engine or _mock_engine(hot_memories)
    cfg = config or SelfImprovementConfig(enabled=True, dry_run=True)
    return SelfImprovementWorker(eng, config=cfg)


def _candidate(node_id="n1", content="test content here for testing",
               metadata=None, tier="hot", reason="test"):
    return ImprovementCandidate(
        node_id=node_id,
        content=content,
        metadata=metadata or {},
        tier=tier,
        selection_reason=reason,
    )


def _proposal(candidate=None, proposed=None, score=0.2, imp_type="normalize"):
    c = candidate or _candidate()
    return ImprovementProposal(
        candidate=c,
        improvement_type=imp_type,
        proposed_content=proposed or c.content,
        rationale="test proposal",
        expected_improvement_score=score,
    )


# ═══════════════════════════════════════════════════════════════════════
# Candidate Selection
# ═══════════════════════════════════════════════════════════════════════

class TestCandidateSelection:
    """Tests for _select_candidates heuristics."""

    async def test_short_content_detected(self):
        """Content < 20 chars triggers 'short_content' heuristic."""
        w = _make_worker(hot_memories=[
            {"id": "n1", "content": "hi", "metadata": {}},
        ])
        candidates = await w._select_candidates()
        assert len(candidates) == 1
        assert candidates[0].selection_reason == "short_content"
        assert candidates[0].score == 0.3

    async def test_missing_metadata_detected(self):
        """Missing agent_id or importance triggers 'missing_metadata'."""
        w = _make_worker(hot_memories=[
            {"id": "n1", "content": "A proper length content string for testing here",
             "metadata": {"importance": 0.8}},  # missing agent_id
        ])
        candidates = await w._select_candidates()
        assert len(candidates) == 1
        assert "missing_metadata" in candidates[0].selection_reason
        assert "agent_id" in candidates[0].selection_reason

    async def test_complete_metadata_not_selected(self):
        """Content with all expected fields and enough length is NOT selected."""
        w = _make_worker(hot_memories=[
            {"id": "n1",
             "content": "A proper length content string for testing here",
             "metadata": {"agent_id": "ag1", "importance": 0.5}},
        ])
        candidates = await w._select_candidates()
        assert len(candidates) == 0

    async def test_near_duplicate_detection(self):
        """Two memories with identical content (case-insensitive) trigger dedup."""
        w = _make_worker(hot_memories=[
            {"id": "n1", "content": "A proper length content string for testing here",
             "metadata": {"agent_id": "a", "importance": 0.5}},
            {"id": "n2", "content": "A proper length content string for testing here",
             "metadata": {"agent_id": "a", "importance": 0.5}},
        ])
        candidates = await w._select_candidates()
        dedup = [c for c in candidates if "near_duplicate" in c.selection_reason]
        assert len(dedup) >= 1

    async def test_batch_size_limit(self):
        """Never returns more candidates than batch_size."""
        cfg = SelfImprovementConfig(enabled=True, batch_size=2)
        mems = [
            {"id": f"n{i}", "content": "x", "metadata": {}}
            for i in range(20)
        ]
        w = _make_worker(config=cfg, hot_memories=mems)
        candidates = await w._select_candidates()
        assert len(candidates) <= 2

    async def test_cooldown_prevents_re_selection(self):
        """A recently-evaluated node should not be selected again."""
        w = _make_worker(hot_memories=[
            {"id": "n1", "content": "x", "metadata": {}},
        ])
        # First run: should pick it
        c1 = await w._select_candidates()
        assert len(c1) == 1
        # Second run: cooldown blocks it
        c2 = await w._select_candidates()
        assert len(c2) == 0

    async def test_cooldown_expires(self):
        """After cooldown_minutes expire, node is re-eligible."""
        cfg = SelfImprovementConfig(enabled=True, cooldown_minutes=0)  # 0 min = immediate
        w = _make_worker(config=cfg, hot_memories=[
            {"id": "n1", "content": "x", "metadata": {}},
        ])
        c1 = await w._select_candidates()
        assert len(c1) == 1
        # Force cooldown to the past
        w._cooldown["n1"] = time.time() - 1
        c2 = await w._select_candidates()
        assert len(c2) == 1

    async def test_empty_node_id_skipped(self):
        """Entries without an id are silently skipped."""
        w = _make_worker(hot_memories=[
            {"content": "something", "metadata": {}},
        ])
        candidates = await w._select_candidates()
        assert len(candidates) == 0

    async def test_empty_content_skipped(self):
        """Entries with empty content are silently skipped."""
        w = _make_worker(hot_memories=[
            {"id": "n1", "content": "", "metadata": {}},
        ])
        candidates = await w._select_candidates()
        assert len(candidates) == 0

    async def test_tier_manager_error_returns_empty(self):
        """If tier_manager.get_hot_recent raises, return empty list gracefully."""
        w = _make_worker()
        w.engine.tier_manager.get_hot_recent = MagicMock(
            side_effect=RuntimeError("connection error")
        )
        candidates = await w._select_candidates()
        assert candidates == []


# ═══════════════════════════════════════════════════════════════════════
# Proposal Generation
# ═══════════════════════════════════════════════════════════════════════

class TestProposalGeneration:
    """Tests for the rule-based proposal generators."""

    def test_normalize_collapses_whitespace(self):
        w = _make_worker()
        c = _candidate(content="  hello   world  \n  test  ", reason="other")
        proposals = w._generate_proposals(c)
        # Default reason triggers normalize
        assert len(proposals) == 1
        assert proposals[0].improvement_type == "normalize"
        assert proposals[0].proposed_content == "hello world test"

    def test_normalize_score_proportional_to_change(self):
        w = _make_worker()
        c = _candidate(content="no   extra   spaces   here   ok", reason="other")
        proposals = w._generate_proposals(c)
        assert proposals[0].expected_improvement_score > 0.1

    def test_short_content_generates_metadata_repair(self):
        w = _make_worker()
        c = _candidate(content="hi", reason="short_content")
        proposals = w._generate_proposals(c)
        assert proposals[0].improvement_type == "metadata_repair"

    def test_missing_metadata_generates_metadata_repair(self):
        w = _make_worker()
        c = _candidate(reason="missing_metadata:agent_id")
        proposals = w._generate_proposals(c)
        assert proposals[0].improvement_type == "metadata_repair"
        assert proposals[0].expected_improvement_score == 0.2

    def test_near_duplicate_generates_deduplicate(self):
        w = _make_worker()
        c = _candidate(reason="near_duplicate_of:n0")
        proposals = w._generate_proposals(c)
        assert proposals[0].improvement_type == "deduplicate"
        assert proposals[0].expected_improvement_score == 0.4


# ═══════════════════════════════════════════════════════════════════════
# Validation Gates — Edge Cases
# ═══════════════════════════════════════════════════════════════════════

class TestValidationGatesEdgeCases:
    """Extended edge-case tests for the 5 gates."""

    def _gates(self, **kwargs):
        return ValidationGates(SelfImprovementConfig(**kwargs))

    def test_semantic_drift_empty_strings(self):
        """Both original and proposed empty should pass."""
        g = self._gates()
        p = _proposal(
            candidate=_candidate(content=""),
            proposed="",
        )
        assert g.semantic_drift_gate(p) is True

    def test_semantic_drift_one_word_match(self):
        g = self._gates(min_semantic_similarity=0.5)
        p = _proposal(
            candidate=_candidate(content="hello"),
            proposed="hello",
        )
        assert g.semantic_drift_gate(p) is True

    def test_semantic_drift_partial_overlap(self):
        """50% overlap with threshold 0.82 should fail."""
        g = self._gates(min_semantic_similarity=0.82)
        p = _proposal(
            candidate=_candidate(content="alpha beta gamma delta"),
            proposed="alpha beta epsilon zeta",
        )
        # Jaccard: intersection={alpha,beta}/union={alpha,beta,gamma,delta,epsilon,zeta} = 2/6 ≈ 0.33
        assert g.semantic_drift_gate(p) is False

    def test_fact_safety_strict_allows_small_additions(self):
        """Up to 10% new tokens allowed in strict mode."""
        g = self._gates(safety_mode="strict")
        # 10 tokens original, add 1 new ≤ max(3, 1) = 3
        p = _proposal(
            candidate=_candidate(content="a b c d e f g h i j"),
            proposed="a b c d e f g h i j extra",
        )
        assert g.fact_safety_gate(p) is True

    def test_fact_safety_strict_rejects_many_additions(self):
        g = self._gates(safety_mode="strict")
        p = _proposal(
            candidate=_candidate(content="hello"),
            proposed="alpha beta gamma delta epsilon zeta eta theta hello",
        )
        assert g.fact_safety_gate(p) is False

    def test_resource_gate_exact_boundary(self):
        """Elapsed == max should fail (not <)."""
        g = self._gates()
        assert g.resource_gate(20.0, 20.0) is False

    def test_resource_gate_just_under(self):
        g = self._gates()
        assert g.resource_gate(19.999, 20.0) is True

    def test_validate_records_all_gate_results(self):
        g = self._gates()
        p = _proposal(score=0.2)
        g.validate(p, 0.5, 20.0)
        assert set(p.validator_results.keys()) == {
            "semantic_drift", "fact_safety", "structure", "policy", "resource"
        }

    def test_validate_rejection_reasons_populated(self):
        g = self._gates(min_improvement_score=0.99)  # score=0.2 will fail structure
        p = _proposal(score=0.2)
        result = g.validate(p, 0.5, 20.0)
        assert result is False
        assert "structure" in p.rejection_reasons
        assert p.accepted is False

    def test_validate_multiple_gate_failures(self):
        """Multiple gates can fail simultaneously."""
        g = self._gates(min_improvement_score=0.99, min_semantic_similarity=0.99)
        p = _proposal(
            candidate=_candidate(content="hello world"),
            proposed="completely different text here now",
            score=0.01,
        )
        result = g.validate(p, 0.5, 20.0)
        assert result is False
        assert len(p.rejection_reasons) >= 2


# ═══════════════════════════════════════════════════════════════════════
# Decision Logging
# ═══════════════════════════════════════════════════════════════════════

class TestDecisionLogging:
    """Tests for _log_decision and log capping."""

    def test_log_entry_fields(self):
        w = _make_worker()
        p = _proposal()
        p.accepted = True
        p.validator_results = {"semantic_drift": True, "fact_safety": True,
                                "structure": True, "policy": True, "resource": True}
        w._log_decision(p)
        assert len(w._decision_log) == 1
        entry = w._decision_log[0]
        assert entry["accepted"] is True
        assert entry["node_id"] == p.candidate.node_id
        assert entry["improvement_type"] == p.improvement_type
        assert "timestamp" in entry

    def test_log_rejected_entry(self):
        w = _make_worker()
        p = _proposal()
        p.accepted = False
        p.rejection_reasons = ["structure"]
        w._log_decision(p)
        assert w._decision_log[0]["accepted"] is False
        assert w._decision_log[0]["rejection_reasons"] == ["structure"]

    def test_log_capping(self):
        """Log should not exceed _max_log_entries."""
        w = _make_worker()
        w._max_log_entries = 5
        for i in range(10):
            p = _proposal(candidate=_candidate(node_id=f"n{i}"))
            p.accepted = True
            p.validator_results = {}
            w._log_decision(p)
        assert len(w._decision_log) == 5
        # Oldest entries should be gone
        assert w._decision_log[0]["node_id"] == "n5"

    def test_get_decision_log_default_limit(self):
        w = _make_worker()
        for i in range(100):
            p = _proposal(candidate=_candidate(node_id=f"n{i}"))
            p.accepted = True
            p.validator_results = {}
            w._log_decision(p)
        log = w.get_decision_log()  # default last_n=50
        assert len(log) == 50

    def test_get_decision_log_custom_limit(self):
        w = _make_worker()
        for i in range(10):
            p = _proposal(candidate=_candidate(node_id=f"n{i}"))
            p.accepted = True
            p.validator_results = {}
            w._log_decision(p)
        log = w.get_decision_log(last_n=3)
        assert len(log) == 3


# ═══════════════════════════════════════════════════════════════════════
# Backpressure
# ═══════════════════════════════════════════════════════════════════════

class TestBackpressure:
    """Tests for _check_backpressure."""

    def test_no_queue_no_backpressure(self):
        w = _make_worker()
        assert w._check_backpressure() is False

    def test_small_queue_no_backpressure(self):
        w = _make_worker()
        w.engine._subconscious_queue = MagicMock()
        w.engine._subconscious_queue.qsize = MagicMock(return_value=10)
        assert w._check_backpressure() is False

    def test_large_queue_triggers_backpressure(self):
        w = _make_worker()
        w.engine._subconscious_queue = MagicMock()
        w.engine._subconscious_queue.qsize = MagicMock(return_value=100)
        assert w._check_backpressure() is True

    def test_boundary_50_no_backpressure(self):
        """qsize=50 should NOT trigger (> 50 required)."""
        w = _make_worker()
        w.engine._subconscious_queue = MagicMock()
        w.engine._subconscious_queue.qsize = MagicMock(return_value=50)
        assert w._check_backpressure() is False

    def test_boundary_51_triggers_backpressure(self):
        w = _make_worker()
        w.engine._subconscious_queue = MagicMock()
        w.engine._subconscious_queue.qsize = MagicMock(return_value=51)
        assert w._check_backpressure() is True

    def test_broken_queue_no_crash(self):
        """If qsize raises, backpressure returns False gracefully."""
        w = _make_worker()
        w.engine._subconscious_queue = MagicMock()
        w.engine._subconscious_queue.qsize = MagicMock(side_effect=RuntimeError)
        assert w._check_backpressure() is False


# ═══════════════════════════════════════════════════════════════════════
# Full Cycle — run_once
# ═══════════════════════════════════════════════════════════════════════

class TestRunOnce:
    """Tests for the full run_once cycle."""

    async def test_cycle_with_mixed_candidates(self):
        """A mix of short, missing-metadata, and normal memories."""
        w = _make_worker(hot_memories=[
            {"id": "short1", "content": "hi", "metadata": {}},
            {"id": "missing1", "content": "A longer piece of content for importance",
             "metadata": {"importance": 0.5}},  # missing agent_id
            {"id": "ok1", "content": "Normal well-formed memory content here",
             "metadata": {"agent_id": "a", "importance": 0.5}},
        ])
        result = await w.run_once()
        assert result["candidates_found"] >= 2  # short1 + missing1
        assert result["proposals_generated"] >= 2
        assert result["dry_run"] is True
        assert result["elapsed_seconds"] >= 0
        assert result["timestamp"] is not None

    async def test_cycle_metrics_updated(self):
        w = _make_worker(hot_memories=[
            {"id": "n1", "content": "x", "metadata": {}},
        ])
        result = await w.run_once()
        assert w.metrics.attempts_total >= 1
        assert len(w.metrics.cycle_durations) == 1
        assert len(w.metrics.candidates_per_cycle) == 1

    async def test_cycle_records_meta_memory(self):
        w = _make_worker(hot_memories=[
            {"id": "n1", "content": "x", "metadata": {}},
        ])
        await w.run_once()
        w.engine.meta_memory.record_metric.assert_called()

    async def test_cycle_no_candidates_still_records(self):
        w = _make_worker(hot_memories=[])
        result = await w.run_once()
        assert result["candidates_found"] == 0
        assert w.last_run is not None
        assert len(w.metrics.cycle_durations) == 1

    async def test_cycle_backpressure_skip(self):
        w = _make_worker()
        w.engine._subconscious_queue = MagicMock()
        w.engine._subconscious_queue.qsize = MagicMock(return_value=200)
        result = await w.run_once()
        assert result["backpressure_skip"] is True
        assert result["candidates_found"] == 0

    async def test_cycle_dry_run_no_commits(self):
        w = _make_worker(hot_memories=[
            {"id": "n1", "content": "x", "metadata": {}},
        ])
        result = await w.run_once()
        assert w.metrics.commits_total == 0

    async def test_cycle_non_dry_run_still_no_commits_phase0(self):
        """Even with dry_run=False, Phase 0 does not commit."""
        cfg = SelfImprovementConfig(enabled=True, dry_run=False)
        w = _make_worker(config=cfg, hot_memories=[
            {"id": "n1", "content": "x", "metadata": {}},
        ])
        result = await w.run_once()
        assert w.metrics.commits_total == 0

    async def test_cycle_resource_gate_budget_exhaustion(self):
        """When max_cycle_seconds is very low, the cycle should cut short."""
        cfg = SelfImprovementConfig(enabled=True, max_cycle_seconds=0)
        mems = [{"id": f"n{i}", "content": "x", "metadata": {}} for i in range(10)]
        w = _make_worker(config=cfg, hot_memories=mems)
        result = await w.run_once()
        # Some candidates found, but may not all be processed
        assert result["candidates_found"] >= 1

    async def test_multiple_cycles_accumulate_metrics(self):
        w = _make_worker(hot_memories=[
            {"id": "n1", "content": "x", "metadata": {}},
        ])
        await w.run_once()
        # Reset cooldown so n1 is eligible again
        w._cooldown.clear()
        await w.run_once()
        assert len(w.metrics.cycle_durations) == 2
        assert w.metrics.attempts_total >= 2

    async def test_meta_memory_error_does_not_crash(self):
        """If meta_memory.record_metric raises, cycle still completes."""
        w = _make_worker(hot_memories=[
            {"id": "n1", "content": "x", "metadata": {}},
        ])
        w.engine.meta_memory.record_metric = MagicMock(
            side_effect=RuntimeError("db error")
        )
        result = await w.run_once()
        # Should still complete
        assert result["elapsed_seconds"] >= 0


# ═══════════════════════════════════════════════════════════════════════
# Lifecycle — start / stop
# ═══════════════════════════════════════════════════════════════════════

class TestLifecycle:
    """Tests for start/stop background loop."""

    async def test_start_creates_task(self):
        w = _make_worker()
        await w.start()
        assert w._running is True
        assert w._task is not None
        await w.stop()

    async def test_stop_cancels_task(self):
        w = _make_worker()
        await w.start()
        await w.stop()
        assert w._running is False

    async def test_start_disabled_noop(self):
        cfg = SelfImprovementConfig(enabled=False)
        w = _make_worker(config=cfg)
        await w.start()
        assert w._running is False
        assert w._task is None

    async def test_stop_without_start_safe(self):
        w = _make_worker()
        await w.stop()  # should not raise
        assert w._running is False

    async def test_double_stop_safe(self):
        w = _make_worker()
        await w.start()
        await w.stop()
        await w.stop()  # second stop should not raise
        assert w._running is False


# ═══════════════════════════════════════════════════════════════════════
# Metrics Snapshot
# ═══════════════════════════════════════════════════════════════════════

class TestMetricsSnapshot:
    """Tests for SelfImprovementMetrics snapshot accuracy."""

    def test_initial_snapshot(self):
        m = SelfImprovementMetrics()
        snap = m.snapshot()
        assert snap["mnemocore_self_improve_attempts_total"] == 0
        assert snap["mnemocore_self_improve_commits_total"] == 0
        assert snap["mnemocore_self_improve_rejects_total"] == 0
        assert snap["mnemocore_self_improve_cycle_duration_seconds"] == 0.0
        assert snap["mnemocore_self_improve_quality_delta"] == 0.0
        assert snap["mnemocore_self_improve_backpressure_skips_total"] == 0
        assert snap["total_cycles"] == 0

    def test_snapshot_with_data(self):
        m = SelfImprovementMetrics()
        m.attempts_total = 42
        m.rejects_total = 7
        m.cycle_durations.append(2.5)
        m.cycle_durations.append(1.8)
        m.candidates_per_cycle.append(5)
        m.quality_deltas.append(0.35)
        snap = m.snapshot()
        assert snap["mnemocore_self_improve_attempts_total"] == 42
        assert snap["mnemocore_self_improve_rejects_total"] == 7
        assert snap["mnemocore_self_improve_cycle_duration_seconds"] == 1.8  # last
        assert snap["total_cycles"] == 2

    def test_snapshot_candidates_shows_latest(self):
        m = SelfImprovementMetrics()
        m.candidates_per_cycle.extend([3, 7, 2])
        snap = m.snapshot()
        assert snap["mnemocore_self_improve_candidates_in_cycle"] == 2


# ═══════════════════════════════════════════════════════════════════════
# get_stats
# ═══════════════════════════════════════════════════════════════════════

class TestGetStats:
    """Tests for the public get_stats API."""

    def test_stats_structure(self):
        w = _make_worker()
        stats = w.get_stats()
        assert "last_run" in stats
        assert "running" in stats
        assert "config" in stats
        assert "metrics" in stats
        assert "decision_log_size" in stats
        assert "cooldown_entries" in stats
        assert "last_cycle" in stats

    def test_stats_config_values(self):
        cfg = SelfImprovementConfig(enabled=True, dry_run=True, batch_size=16)
        w = _make_worker(config=cfg)
        stats = w.get_stats()
        assert stats["config"]["enabled"] is True
        assert stats["config"]["dry_run"] is True
        assert stats["config"]["batch_size"] == 16

    async def test_stats_after_cycle(self):
        w = _make_worker(hot_memories=[
            {"id": "n1", "content": "x", "metadata": {}},
        ])
        await w.run_once()
        stats = w.get_stats()
        assert stats["last_run"] is not None
        assert stats["last_cycle"]["candidates_found"] >= 1


# ═══════════════════════════════════════════════════════════════════════
# Factory Function
# ═══════════════════════════════════════════════════════════════════════

class TestFactory:
    """Tests for create_self_improvement_worker."""

    def test_factory_creates_worker(self):
        engine = _mock_engine()
        worker = create_self_improvement_worker(engine)
        assert isinstance(worker, SelfImprovementWorker)

    def test_factory_uses_global_config(self):
        engine = _mock_engine()
        worker = create_self_improvement_worker(engine)
        # Should use global config's self_improvement section
        assert worker.cfg is not None

    def test_factory_worker_has_gates(self):
        engine = _mock_engine()
        worker = create_self_improvement_worker(engine)
        assert isinstance(worker.gates, ValidationGates)
