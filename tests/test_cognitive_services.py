"""
Comprehensive tests for Phase 5.0–5.4 cognitive services.
==========================================================
Covers:
  - SemanticStoreService (consolidation, decay, stats)
  - EpisodicStoreService (chain verification, LTP, repair)
  - ProceduralStoreService (persistence, matching, creation)
  - SelfImprovementWorker (lifecycle, candidate selection, validation gates)
  - PulseLoop (phase execution, stats)
  - Cognitive config dataclasses
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mnemocore.core.config import (
    EpisodicConfig,
    ProceduralConfig,
    SemanticConfig,
    SelfImprovementConfig,
    MetaMemoryConfig,
)


# =====================================================================
# SemanticStoreService Tests
# =====================================================================

class TestSemanticStoreService:
    """Tests for the upgraded SemanticStoreService."""

    def _make_store(self, qdrant_store=None, config=None):
        from mnemocore.core.semantic_store import SemanticStoreService
        return SemanticStoreService(
            qdrant_store=qdrant_store,
            config=config or SemanticConfig(),
        )

    def _make_hdv(self):
        from mnemocore.core.binary_hdv import BinaryHDV
        return BinaryHDV.random()

    def _make_concept(self, label="test_concept", hdv=None):
        from mnemocore.core.semantic_store import SemanticConcept
        import uuid
        return SemanticConcept(
            id=f"sc_{uuid.uuid4().hex[:12]}",
            label=label,
            description=f"A test concept: {label}",
            tags=["test"],
            prototype_hdv=hdv or self._make_hdv(),
            support_episode_ids=[],
            reliability=0.5,
            last_updated_at=datetime.now(timezone.utc),
            metadata={},
        )

    def test_init_defaults(self):
        store = self._make_store()
        assert store.concept_count == 0

    def test_upsert_concept_local(self):
        store = self._make_store()
        concept = self._make_concept("test_concept")
        store.upsert_concept(concept)
        assert store.concept_count == 1

    def test_find_nearby_concepts_empty(self):
        store = self._make_store()
        results = store.find_nearby_concepts(self._make_hdv())
        assert results == []

    def test_find_nearby_concepts_finds_similar(self):
        store = self._make_store()
        hdv = self._make_hdv()
        concept = self._make_concept("similar", hdv=hdv)
        store.upsert_concept(concept)
        results = store.find_nearby_concepts(hdv, top_k=1, min_similarity=0.5)
        # Same HDV should match itself perfectly
        assert len(results) >= 1
        assert results[0].label == "similar"

    def test_consolidate_from_content_creates_new(self):
        store = self._make_store()
        hdv = self._make_hdv()
        result = store.consolidate_from_content(
            "brand_new_concept", hdv, episode_ids=["ep1"]
        )
        assert result is not None
        assert store.concept_count == 1

    def test_consolidate_from_content_merges_existing(self):
        store = self._make_store()
        hdv = self._make_hdv()
        # First create
        store.consolidate_from_content("existing", hdv, episode_ids=["ep1"])
        initial_reliability = store.get_all_concepts()[0].reliability

        # Consolidate again with same HDV — should merge
        result = store.consolidate_from_content("existing", hdv, episode_ids=["ep2"])
        if result:
            assert result.reliability >= initial_reliability

    def test_consolidate_empty_content_returns_none(self):
        store = self._make_store()
        result = store.consolidate_from_content("", self._make_hdv(), [])
        assert result is None

    def test_decay_all_reliability(self):
        store = self._make_store()
        concept = self._make_concept("decay_test")
        concept.reliability = 0.8
        store.upsert_concept(concept)
        decayed = store.decay_all_reliability(decay_rate=0.1)
        assert decayed >= 1
        all_c = store.get_all_concepts()
        assert all_c[0].reliability < 0.8

    def test_decay_doesnt_go_below_zero(self):
        store = self._make_store()
        concept = self._make_concept("floor_test")
        concept.reliability = 0.01
        store.upsert_concept(concept)
        store.decay_all_reliability(decay_rate=0.5)
        assert store.get_all_concepts()[0].reliability >= 0.0

    def test_adjust_concept_reliability(self):
        store = self._make_store()
        concept = self._make_concept("adjust_test")
        concept.reliability = 0.5
        store.upsert_concept(concept)
        store.adjust_concept_reliability(concept.id, 0.2)
        assert store.get_all_concepts()[0].reliability == pytest.approx(0.7, abs=0.01)

    def test_get_stats(self):
        store = self._make_store()
        concept = self._make_concept("a")
        store.upsert_concept(concept)
        stats = store.get_stats()
        assert stats["concept_count"] == 1
        assert stats["upsert_count"] == 1

    async def test_upsert_concept_persistent_without_qdrant(self):
        store = self._make_store(qdrant_store=None)
        concept = self._make_concept("no_qdrant")
        await store.upsert_concept_persistent(concept)
        assert store.concept_count == 1


# =====================================================================
# EpisodicStoreService Tests
# =====================================================================

class TestEpisodicStoreService:
    """Tests for the upgraded EpisodicStoreService."""

    def _make_store(self, config=None):
        from mnemocore.core.episodic_store import EpisodicStoreService
        return EpisodicStoreService(config=config or EpisodicConfig())

    def test_start_episode(self):
        store = self._make_store()
        ep_id = store.start_episode("agent1", goal="test")
        assert ep_id is not None
        assert ep_id.startswith("ep_")
        active = store.get_active_episodes("agent1")
        assert len(active) == 1

    def test_start_episode_enforces_max_active(self):
        config = EpisodicConfig(max_active_episodes_per_agent=2)
        store = self._make_store(config)
        store.start_episode("agent1", goal="g1")
        store.start_episode("agent1", goal="g2")
        # Third should auto-end the oldest
        ep3 = store.start_episode("agent1", goal="g3")
        assert ep3 is not None
        active = store.get_active_episodes("agent1")
        assert len(active) <= 2

    def test_append_event(self):
        store = self._make_store()
        ep_id = store.start_episode("agent1")
        store.append_event(ep_id, kind="action", content="clicked button")
        ep = store.get_episode(ep_id)
        assert len(ep.events) == 1
        assert ep.events[0].content == "clicked button"

    def test_append_event_to_nonexistent_episode(self):
        store = self._make_store()
        # Should not raise
        store.append_event("fake_id", kind="test", content="noop")

    def test_end_episode(self):
        store = self._make_store()
        ep_id = store.start_episode("agent1")
        store.append_event(ep_id, kind="action", content="test")
        store.end_episode(ep_id, outcome="success")
        assert store.get_episode(ep_id) is not None
        # Should be in history, not active
        assert len(store.get_active_episodes("agent1")) == 0

    def test_end_episode_calculates_ltp(self):
        store = self._make_store()
        ep_id = store.start_episode("agent1")
        for i in range(5):
            store.append_event(ep_id, kind="action", content=f"event_{i}")
        store.end_episode(ep_id, outcome="success", reward=0.8)
        ep = store.get_episode(ep_id)
        assert ep.ltp_strength > 0

    def test_episode_chaining(self):
        store = self._make_store()
        ep1 = store.start_episode("agent1")
        store.end_episode(ep1, outcome="success")
        ep2 = store.start_episode("agent1")
        store.end_episode(ep2, outcome="success")

        ep2_obj = store.get_episode(ep2)
        assert ep1 in ep2_obj.links_prev

    def test_verify_chain_integrity(self):
        store = self._make_store()
        ep1 = store.start_episode("agent1")
        store.end_episode(ep1, outcome="success")
        ep2 = store.start_episode("agent1")
        store.end_episode(ep2, outcome="success")

        report = store.verify_chain_integrity("agent1")
        assert "total_episodes" in report
        assert report["total_episodes"] == 2
        assert "chain_healthy" in report

    def test_repair_chain(self):
        store = self._make_store()
        ep1 = store.start_episode("agent1")
        store.end_episode(ep1, outcome="success")
        ep2 = store.start_episode("agent1")
        store.end_episode(ep2, outcome="success")

        # Break the chain manually
        ep2_obj = store.get_episode(ep2)
        ep2_obj.links_prev = ["broken_id"]

        repairs = store.repair_chain("agent1")
        assert repairs >= 1

    def test_get_episodes_by_outcome(self):
        store = self._make_store()
        ep1 = store.start_episode("agent1")
        store.end_episode(ep1, outcome="success")
        ep2 = store.start_episode("agent1")
        store.end_episode(ep2, outcome="failure")

        successes = store.get_episodes_by_outcome("agent1", "success")
        failures = store.get_episodes_by_outcome("agent1", "failure")
        assert len(successes) == 1
        assert len(failures) == 1

    def test_max_history_enforcement(self):
        config = EpisodicConfig(max_history_per_agent=3)
        store = self._make_store(config)
        for i in range(5):
            ep = store.start_episode("agent1")
            store.end_episode(ep, outcome="success")

        stats = store.get_stats()
        assert stats["total_history_episodes"] <= 3

    def test_get_stats(self):
        store = self._make_store()
        ep = store.start_episode("agent1")
        stats = store.get_stats()
        assert stats["active_episodes"] == 1
        assert "total_history_episodes" in stats
        assert "agents_tracked" in stats

    def test_get_all_agent_ids(self):
        store = self._make_store()
        store.start_episode("agent_a")
        store.start_episode("agent_b")
        agents = store.get_all_agent_ids()
        assert "agent_a" in agents
        assert "agent_b" in agents

    def test_get_recent(self):
        store = self._make_store()
        ep1 = store.start_episode("agent1", goal="first")
        store.end_episode(ep1, outcome="success")
        ep2 = store.start_episode("agent1", goal="second")
        store.end_episode(ep2, outcome="success")
        recent = store.get_recent("agent1", limit=1)
        assert len(recent) == 1


# =====================================================================
# ProceduralStoreService Tests
# =====================================================================

class TestProceduralStoreService:
    """Tests for the upgraded ProceduralStoreService."""

    def _make_store(self, config=None, tmp_dir=None):
        from mnemocore.core.procedural_store import ProceduralStoreService
        cfg = config or ProceduralConfig(
            persistence_path=(
                os.path.join(tmp_dir, "procedures.json")
                if tmp_dir else None
            )
        )
        return ProceduralStoreService(config=cfg)

    def test_create_procedure_from_episode(self):
        store = self._make_store()
        proc = store.create_procedure_from_episode(
            name="greet_user",
            description="Greet the user with hello",
            steps=[
                {"instruction": "detect_greeting"},
                {"instruction": "respond_with_hello"},
            ],
            trigger_pattern="hello hi hey",
            tags=["greeting"],
        )
        assert proc is not None
        assert proc.name == "greet_user"
        assert len(proc.steps) == 2

    def test_find_applicable_procedures(self):
        store = self._make_store()
        store.create_procedure_from_episode(
            name="greet_user",
            description="Greet",
            steps=[{"instruction": "greet"}],
            trigger_pattern="hello",
            tags=["greeting", "hello"],
        )
        found = store.find_applicable_procedures("hello world")
        assert len(found) >= 1
        assert found[0].name == "greet_user"

    def test_find_returns_empty_for_no_match(self):
        store = self._make_store()
        store.create_procedure_from_episode(
            name="cook_pasta",
            description="Cook",
            steps=[{"instruction": "boil"}],
            trigger_pattern="pasta cook",
            tags=["cooking"],
        )
        found = store.find_applicable_procedures("quantum physics")
        assert len(found) == 0

    def test_record_procedure_outcome_success(self):
        store = self._make_store()
        proc = store.create_procedure_from_episode(
            name="test_proc",
            description="Test",
            steps=[{"instruction": "step1"}],
            trigger_pattern="test",
            tags=["test"],
        )
        initial_rel = proc.reliability
        store.record_procedure_outcome(proc.id, success=True)
        updated = store.get_procedure(proc.id)
        assert updated.reliability > initial_rel
        assert updated.success_count == 1

    def test_record_procedure_outcome_failure(self):
        store = self._make_store()
        proc = store.create_procedure_from_episode(
            name="fail_proc",
            description="Fail",
            steps=[{"instruction": "step1"}],
            trigger_pattern="fail",
            tags=["test"],
        )
        initial_rel = proc.reliability
        store.record_procedure_outcome(proc.id, success=False)
        updated = store.get_procedure(proc.id)
        assert updated.reliability < initial_rel
        assert updated.failure_count == 1

    def test_decay_all_reliability(self):
        store = self._make_store()
        proc = store.create_procedure_from_episode(
            name="decay_proc",
            description="Decay",
            steps=[{"instruction": "s"}],
            trigger_pattern="test",
            tags=["test"],
        )
        proc.reliability = 0.9
        decayed = store.decay_all_reliability(decay_rate=0.1)
        assert decayed >= 1
        assert store.get_procedure(proc.id).reliability < 0.9

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp_dir=tmp)
            store.create_procedure_from_episode(
                name="persistent",
                description="Persist test",
                steps=[{"instruction": "step1"}],
                trigger_pattern="persist",
                tags=["test"],
            )

            # Create new store from same path — should load
            store2 = self._make_store(tmp_dir=tmp)
            all_procs = list(store2._procedures.values())
            assert len(all_procs) == 1
            assert all_procs[0].name == "persistent"

    def test_get_stats(self):
        store = self._make_store()
        store.create_procedure_from_episode(
            name="p1",
            description="d",
            steps=[{"instruction": "s"}],
            trigger_pattern="t1",
            tags=[],
        )
        store.create_procedure_from_episode(
            name="p2",
            description="d",
            steps=[{"instruction": "s"}],
            trigger_pattern="t2",
            tags=[],
        )
        stats = store.get_stats()
        assert stats["total_procedures"] == 2


# =====================================================================
# SelfImprovementWorker Tests
# =====================================================================

class TestSelfImprovementWorker:
    """Tests for the Phase 0 SelfImprovementWorker."""

    def _make_worker(self, config=None, engine=None):
        from mnemocore.subconscious.self_improvement_worker import (
            SelfImprovementWorker,  # direct module import, no __init__.py
        )
        mock_engine = engine or MagicMock()
        mock_engine.tier_manager = MagicMock()
        mock_engine.tier_manager.get_hot_recent = MagicMock(return_value=[])
        mock_engine.meta_memory = MagicMock()
        mock_engine.meta_memory.record_metric = MagicMock()

        cfg = config or SelfImprovementConfig(enabled=True, dry_run=True)
        return SelfImprovementWorker(mock_engine, config=cfg)

    def test_init(self):
        worker = self._make_worker()
        assert worker._running is False
        assert worker.cfg.dry_run is True
        assert worker.cfg.enabled is True

    async def test_start_stop_lifecycle(self):
        worker = self._make_worker()
        await worker.start()
        assert worker._running is True
        assert worker._task is not None
        await worker.stop()
        assert worker._running is False

    async def test_start_disabled(self):
        cfg = SelfImprovementConfig(enabled=False)
        worker = self._make_worker(config=cfg)
        await worker.start()
        assert worker._running is False
        assert worker._task is None

    async def test_run_once_no_candidates(self):
        worker = self._make_worker()
        result = await worker.run_once()
        assert result["candidates_found"] == 0
        assert result["dry_run"] is True
        assert result["elapsed_seconds"] >= 0

    async def test_run_once_with_candidates(self):
        worker = self._make_worker()
        worker.engine.tier_manager.get_hot_recent = MagicMock(
            return_value=[
                {
                    "id": "node_1",
                    "content": "x",  # Very short = short_content heuristic
                    "metadata": {},
                },
                {
                    "id": "node_2",
                    "content": "A longer piece of content for testing purposes here",
                    "metadata": {},  # Missing agent_id, importance
                },
            ]
        )
        result = await worker.run_once()
        assert result["candidates_found"] >= 1
        assert result["proposals_generated"] >= 1

    async def test_run_once_dry_run_no_commits(self):
        worker = self._make_worker()
        worker.engine.tier_manager.get_hot_recent = MagicMock(
            return_value=[
                {"id": "n1", "content": "x", "metadata": {}},
            ]
        )
        result = await worker.run_once()
        assert worker.metrics.commits_total == 0

    async def test_backpressure_skip(self):
        worker = self._make_worker()
        # Simulate a loaded queue
        worker.engine._subconscious_queue = MagicMock()
        worker.engine._subconscious_queue.qsize = MagicMock(return_value=100)
        result = await worker.run_once()
        assert result["backpressure_skip"] is True
        assert worker.metrics.backpressure_skips_total >= 1

    def test_get_stats(self):
        worker = self._make_worker()
        stats = worker.get_stats()
        assert "config" in stats
        assert "metrics" in stats
        assert "running" in stats
        assert stats["config"]["dry_run"] is True

    def test_get_decision_log(self):
        worker = self._make_worker()
        log = worker.get_decision_log()
        assert isinstance(log, list)


# =====================================================================
# Validation Gates Tests
# =====================================================================

class TestValidationGates:
    """Tests for the 5 SelfImprovement validation gates."""

    def _make_gates(self, config=None):
        from mnemocore.subconscious.self_improvement_worker import (
            ValidationGates,  # direct module import
        )
        return ValidationGates(config or SelfImprovementConfig())

    def _make_proposal(self, original="hello world test", proposed="hello world test", score=0.2):
        from mnemocore.subconscious.self_improvement_worker import (
            ImprovementCandidate,  # direct module import
            ImprovementProposal,
        )
        candidate = ImprovementCandidate(
            node_id="test_node",
            content=original,
            metadata={},
            tier="hot",
            selection_reason="test",
        )
        return ImprovementProposal(
            candidate=candidate,
            improvement_type="normalize",
            proposed_content=proposed,
            rationale="test",
            expected_improvement_score=score,
        )

    def test_semantic_drift_gate_passes_identical(self):
        gates = self._make_gates()
        proposal = self._make_proposal("hello world", "hello world")
        assert gates.semantic_drift_gate(proposal) is True

    def test_semantic_drift_gate_fails_completely_different(self):
        gates = self._make_gates()
        proposal = self._make_proposal(
            "hello world",
            "completely unrelated quantum mechanics discussion about particles"
        )
        assert gates.semantic_drift_gate(proposal) is False

    def test_fact_safety_gate_strict_passes_subset(self):
        gates = self._make_gates(SelfImprovementConfig(safety_mode="strict"))
        proposal = self._make_proposal(
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps"
        )
        assert gates.fact_safety_gate(proposal) is True

    def test_fact_safety_gate_strict_fails_new_content(self):
        gates = self._make_gates(SelfImprovementConfig(safety_mode="strict"))
        proposal = self._make_proposal(
            "hello",
            "hello plus many completely new words that were never in original text at all whoa"
        )
        assert gates.fact_safety_gate(proposal) is False

    def test_fact_safety_gate_balanced_always_passes(self):
        gates = self._make_gates(SelfImprovementConfig(safety_mode="balanced"))
        proposal = self._make_proposal(
            "hello",
            "completely new content that is entirely different"
        )
        assert gates.fact_safety_gate(proposal) is True

    def test_structure_gate_passes_above_threshold(self):
        gates = self._make_gates(SelfImprovementConfig(min_improvement_score=0.1))
        proposal = self._make_proposal(score=0.2)
        assert gates.structure_gate(proposal) is True

    def test_structure_gate_fails_below_threshold(self):
        gates = self._make_gates(SelfImprovementConfig(min_improvement_score=0.5))
        proposal = self._make_proposal(score=0.1)
        assert gates.structure_gate(proposal) is False

    def test_policy_gate_always_passes_phase0(self):
        gates = self._make_gates()
        proposal = self._make_proposal()
        assert gates.policy_gate(proposal) is True

    def test_resource_gate_passes_under_budget(self):
        gates = self._make_gates()
        assert gates.resource_gate(1.0, 20.0) is True

    def test_resource_gate_fails_over_budget(self):
        gates = self._make_gates()
        assert gates.resource_gate(25.0, 20.0) is False

    def test_validate_all_gates_pass(self):
        gates = self._make_gates()
        proposal = self._make_proposal(
            "hello world test", "hello world test", score=0.2
        )
        result = gates.validate(proposal, 0.1, 20.0)
        assert result is True
        assert proposal.accepted is True
        assert all(proposal.validator_results.values())


# =====================================================================
# Config Tests
# =====================================================================

class TestCognitiveConfig:
    """Tests for the new cognitive config dataclasses."""

    def test_self_improvement_config_defaults(self):
        cfg = SelfImprovementConfig()
        assert cfg.enabled is False
        assert cfg.dry_run is True
        assert cfg.safety_mode == "strict"
        assert cfg.interval_seconds == 300
        assert cfg.batch_size == 8

    def test_episodic_config_defaults(self):
        cfg = EpisodicConfig()
        assert cfg.max_active_episodes_per_agent >= 1
        assert cfg.max_history_per_agent >= 1

    def test_semantic_config_defaults(self):
        cfg = SemanticConfig()
        assert 0 < cfg.min_similarity_threshold < 1

    def test_procedural_config_defaults(self):
        cfg = ProceduralConfig()
        assert cfg.enable_semantic_matching is True

    def test_meta_memory_config_defaults(self):
        cfg = MetaMemoryConfig()
        assert cfg.max_metrics_history > 0

    def test_configs_are_frozen(self):
        cfg = SelfImprovementConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.enabled = True  # type: ignore


# =====================================================================
# Metrics Tests
# =====================================================================

class TestSelfImprovementMetrics:
    """Tests for the in-process metrics tracker."""

    def test_snapshot_initial(self):
        from mnemocore.subconscious.self_improvement_worker import (
            SelfImprovementMetrics,  # direct module import
        )
        m = SelfImprovementMetrics()
        snap = m.snapshot()
        assert snap["mnemocore_self_improve_attempts_total"] == 0
        assert snap["mnemocore_self_improve_commits_total"] == 0
        assert snap["total_cycles"] == 0

    def test_snapshot_after_updates(self):
        from mnemocore.subconscious.self_improvement_worker import (
            SelfImprovementMetrics,  # direct module import
        )
        m = SelfImprovementMetrics()
        m.attempts_total = 10
        m.rejects_total = 3
        m.cycle_durations.append(1.5)
        snap = m.snapshot()
        assert snap["mnemocore_self_improve_attempts_total"] == 10
        assert snap["mnemocore_self_improve_rejects_total"] == 3
        assert snap["total_cycles"] == 1


# =====================================================================
# PulseLoop Tests
# =====================================================================

class TestPulseLoop:
    """Tests for pulse.py phase implementations."""

    def _make_pulse(self):
        from mnemocore.core.pulse import PulseLoop
        mock_engine = MagicMock()
        mock_engine.episodic_store = MagicMock()
        mock_engine.episodic_store.get_all_agent_ids = MagicMock(
            return_value=["agent1"]
        )
        mock_engine.episodic_store.verify_chain_integrity = MagicMock(
            return_value={"broken_links": 0}
        )
        mock_engine.episodic_store.repair_chain = MagicMock(return_value=0)
        mock_engine.semantic_store = MagicMock()
        mock_engine.semantic_store.consolidate_from_content = MagicMock()
        mock_engine.semantic_store.decay_all_reliability = MagicMock()
        mock_engine.gap_detector = None
        mock_engine.meta_memory = MagicMock()
        mock_engine.meta_memory.record_metric = MagicMock()
        mock_engine.meta_memory.generate_proposals_from_metrics = AsyncMock(
            return_value=None
        )
        mock_engine.procedural_store = MagicMock()
        mock_engine.procedural_store.decay_all_reliability = MagicMock()
        mock_engine.association_network = None
        mock_engine.working_memory = MagicMock()
        mock_engine.working_memory.get_context = MagicMock(return_value=[])

        config = MagicMock()
        config.interval_seconds = 30
        config.enabled = True

        pulse = PulseLoop(mock_engine, config)
        return pulse, mock_engine

    def test_init(self):
        pulse, _ = self._make_pulse()
        assert pulse._tick_count == 0
        assert pulse._running is False

    def test_get_stats(self):
        pulse, _ = self._make_pulse()
        stats = pulse.get_stats()
        assert "tick_count" in stats
        assert "running" in stats
        assert "phase_errors" in stats


# =====================================================================
# Factory function tests
# =====================================================================

class TestFactoryFunctions:
    """Test factory functions for creating workers."""

    def test_create_self_improvement_worker(self):
        from mnemocore.subconscious.self_improvement_worker import (
            SelfImprovementWorker,  # direct module import
        )
        mock_engine = MagicMock()
        mock_engine.tier_manager = MagicMock()
        cfg = SelfImprovementConfig(enabled=False, dry_run=True)
        worker = SelfImprovementWorker(mock_engine, config=cfg)
        assert isinstance(worker, SelfImprovementWorker)
        assert worker.cfg.enabled is False
