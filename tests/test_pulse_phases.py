"""
Comprehensive tests for PulseLoop (src/mnemocore/core/pulse.py).
================================================================
Covers:
  - All 7 cognitive phases individually
  - Phase skip logic (insight every 5th, procedure every 3rd, meta every 10th)
  - Error isolation between phases (one crash ≠ full tick failure)
  - Phase timing and error tracking
  - Start / stop lifecycle
  - Disabled config
  - WM maintenance (prune_all)
  - Episodic chaining (verify + repair)
  - Semantic refresh (episodic → semantic consolidation + decay)
  - Gap detection (knowledge gap metrics)
  - Insight generation (cross-domain associations)
  - Procedure refinement (outcome tracking + reliability decay)
  - Meta self-reflection (anomaly proposals + phase metrics)
  - get_stats accuracy
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from mnemocore.core.pulse import PulseLoop, PulseTick
from mnemocore.core.config import PulseConfig


# ─── Helpers ─────────────────────────────────────────────────────────

def _make_config(**overrides):
    defaults = {"enabled": True, "interval_seconds": 1, "max_agents_per_tick": 50,
                "max_episodes_per_tick": 200}
    defaults.update(overrides)
    return PulseConfig(**defaults)


def _mock_episode(ep_id="ep1", agent_id="a1", outcome="success", events=None,
                  is_active=False, goal=None):
    """Create a mock Episode with the right interface."""
    ep = MagicMock()
    ep.id = ep_id
    ep.agent_id = agent_id
    ep.outcome = outcome
    ep.is_active = is_active
    ep.goal = goal
    ep.events = events or []
    return ep


def _mock_event(content="test event", metadata=None):
    ev = MagicMock()
    ev.content = content
    ev.metadata = metadata or {}
    return ev


def _make_container(**components):
    """Create a container-like object with optional components."""
    container = MagicMock()
    # Clear all attributes then set only what's provided
    container.working_memory = components.get("working_memory", None)
    container.episodic_store = components.get("episodic_store", None)
    container.semantic_store = components.get("semantic_store", None)
    container.procedural_store = components.get("procedural_store", None)
    container.meta_memory = components.get("meta_memory", None)
    container.engine = components.get("engine", None)
    return container


def _make_pulse(container=None, config=None, **container_kw):
    c = container or _make_container(**container_kw)
    cfg = config or _make_config()
    return PulseLoop(c, cfg)


# ═══════════════════════════════════════════════════════════════════════
# PulseTick Enum
# ═══════════════════════════════════════════════════════════════════════

class TestPulseTickEnum:
    def test_all_seven_phases(self):
        assert len(PulseTick) == 11

    def test_phase_values(self):
        expected = {
            "wm_maintenance", "episodic_chaining", "semantic_refresh",
            "gap_detection", "insight_generation", "procedure_refinement",
            "meta_self_reflection",
            "strategy_refinement", "graph_maintenance",
            "scheduler_tick", "exchange_discovery",
        }
        assert {t.value for t in PulseTick} == expected


# ═══════════════════════════════════════════════════════════════════════
# Initialization
# ═══════════════════════════════════════════════════════════════════════

class TestPulseInit:
    def test_initial_state(self):
        pulse = _make_pulse()
        assert pulse._running is False
        assert pulse._tick_count == 0
        assert pulse._meta_tick_count == 0
        assert pulse._phase_durations == {}
        assert pulse._phase_errors == {}
        assert pulse._last_tick_at is None


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: WM Maintenance
# ═══════════════════════════════════════════════════════════════════════

class TestWMMaintenance:
    async def test_prune_called(self):
        wm = MagicMock()
        pulse = _make_pulse(working_memory=wm)
        await pulse._wm_maintenance()
        wm.prune_all.assert_called_once()

    async def test_no_wm_noop(self):
        """Should not crash if working_memory is None."""
        pulse = _make_pulse()
        await pulse._wm_maintenance()  # no error

    async def test_prune_error_propagates_to_run_phase(self):
        """Errors in prune_all are caught by _run_phase."""
        wm = MagicMock()
        wm.prune_all.side_effect = RuntimeError("pruning failed")
        pulse = _make_pulse(working_memory=wm)
        await pulse._run_phase("wm_maintenance", pulse._wm_maintenance)
        assert pulse._phase_errors.get("wm_maintenance", 0) == 1


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Episodic Chaining
# ═══════════════════════════════════════════════════════════════════════

class TestEpisodicChaining:
    async def test_skips_if_no_episodic_store(self):
        pulse = _make_pulse()
        await pulse._episodic_chaining()  # no crash

    async def test_healthy_chains_no_repair(self):
        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(return_value=["a1"])
        episodic.verify_chain_integrity = MagicMock(
            return_value={"chain_healthy": True, "broken_links": 0}
        )
        pulse = _make_pulse(episodic_store=episodic)
        await pulse._episodic_chaining()
        episodic.repair_chain.assert_not_called()

    async def test_broken_chain_triggers_repair(self):
        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(return_value=["a1"])
        episodic.verify_chain_integrity = MagicMock(
            return_value={"chain_healthy": False, "broken_links": 2}
        )
        episodic.repair_chain = MagicMock(return_value=2)
        pulse = _make_pulse(episodic_store=episodic)
        await pulse._episodic_chaining()
        episodic.repair_chain.assert_called_once_with("a1")

    async def test_max_agents_per_tick_limit(self):
        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(return_value=[f"a{i}" for i in range(100)])
        episodic.verify_chain_integrity = MagicMock(
            return_value={"chain_healthy": True, "broken_links": 0}
        )
        config = _make_config(max_agents_per_tick=3)
        pulse = _make_pulse(episodic_store=episodic, config=config)
        await pulse._episodic_chaining()
        assert episodic.verify_chain_integrity.call_count == 3

    async def test_error_in_chaining_caught(self):
        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(side_effect=RuntimeError("db error"))
        pulse = _make_pulse(episodic_store=episodic)
        # Should not raise
        await pulse._episodic_chaining()


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Semantic Refresh
# ═══════════════════════════════════════════════════════════════════════

class TestSemanticRefresh:
    async def test_skips_if_no_stores(self):
        pulse = _make_pulse()
        await pulse._semantic_refresh()  # no crash

    async def test_skips_if_only_episodic(self):
        pulse = _make_pulse(episodic_store=MagicMock())
        await pulse._semantic_refresh()  # no crash — semantic is None

    async def test_decay_called_even_without_episodes(self):
        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(return_value=[])
        semantic = MagicMock()
        semantic.decay_all_reliability = MagicMock(return_value=5)
        pulse = _make_pulse(episodic_store=episodic, semantic_store=semantic)
        await pulse._semantic_refresh()
        semantic.decay_all_reliability.assert_called_once_with(decay_rate=0.005)

    async def test_consolidation_triggered_for_success_episodes(self):
        ev = _mock_event(content="learned something important")
        ep = _mock_episode(outcome="success", events=[ev])

        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(return_value=["a1"])
        episodic.get_episodes_by_outcome = MagicMock(return_value=[ep])

        semantic = MagicMock()
        semantic.consolidate_from_content = MagicMock(return_value=MagicMock())
        semantic.decay_all_reliability = MagicMock(return_value=0)

        engine = MagicMock()
        engine.binary_encoder = MagicMock()
        engine.binary_encoder.encode = MagicMock(return_value=MagicMock())

        pulse = _make_pulse(
            episodic_store=episodic,
            semantic_store=semantic,
            engine=engine,
        )
        await pulse._semantic_refresh()
        semantic.consolidate_from_content.assert_called()

    async def test_skips_empty_events(self):
        ep = _mock_episode(outcome="success", events=[])
        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(return_value=["a1"])
        episodic.get_episodes_by_outcome = MagicMock(return_value=[ep])
        semantic = MagicMock()
        semantic.decay_all_reliability = MagicMock(return_value=0)

        pulse = _make_pulse(episodic_store=episodic, semantic_store=semantic)
        await pulse._semantic_refresh()
        semantic.consolidate_from_content.assert_not_called()

    async def test_skips_short_combined_content(self):
        """Combined content < 10 chars should be skipped."""
        ev = _mock_event(content="hi")
        ep = _mock_episode(outcome="success", events=[ev])
        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(return_value=["a1"])
        episodic.get_episodes_by_outcome = MagicMock(return_value=[ep])
        semantic = MagicMock()
        semantic.decay_all_reliability = MagicMock(return_value=0)

        pulse = _make_pulse(episodic_store=episodic, semantic_store=semantic)
        await pulse._semantic_refresh()
        semantic.consolidate_from_content.assert_not_called()

    async def test_encoder_failure_does_not_stop_cycle(self):
        ev = _mock_event(content="a long enough content to pass the threshold")
        ep = _mock_episode(outcome="success", events=[ev])
        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(return_value=["a1"])
        episodic.get_episodes_by_outcome = MagicMock(return_value=[ep])
        semantic = MagicMock()
        semantic.decay_all_reliability = MagicMock(return_value=0)
        engine = MagicMock()
        engine.binary_encoder = MagicMock()
        engine.binary_encoder.encode = MagicMock(side_effect=RuntimeError("encode error"))

        pulse = _make_pulse(
            episodic_store=episodic, semantic_store=semantic, engine=engine
        )
        await pulse._semantic_refresh()
        # Should still call decay even though encoding failed
        semantic.decay_all_reliability.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Gap Detection
# ═══════════════════════════════════════════════════════════════════════

class TestGapDetection:
    async def test_skips_if_no_engine(self):
        pulse = _make_pulse()
        await pulse._gap_detection()  # no crash

    async def test_skips_if_no_gap_detector(self):
        engine = MagicMock()
        engine.gap_detector = None
        pulse = _make_pulse(engine=engine)
        await pulse._gap_detection()

    async def test_records_gap_metrics(self):
        gap = MagicMock()
        gap.priority = 0.9
        engine = MagicMock()
        engine.gap_detector = MagicMock()
        engine.gap_detector.get_gaps = MagicMock(return_value=[gap])
        meta = MagicMock()
        meta.record_metric = MagicMock()
        container = _make_container(engine=engine, meta_memory=meta)
        pulse = _make_pulse(container=container)
        await pulse._gap_detection()
        # Should record 2 metrics
        assert meta.record_metric.call_count == 2

    async def test_no_gaps_records_zeros(self):
        engine = MagicMock()
        engine.gap_detector = MagicMock()
        engine.gap_detector.get_gaps = MagicMock(return_value=[])
        meta = MagicMock()
        container = _make_container(engine=engine, meta_memory=meta)
        pulse = _make_pulse(container=container)
        await pulse._gap_detection()
        meta.record_metric.assert_any_call("knowledge_gap_count", 0.0, "pulse_tick")


# ═══════════════════════════════════════════════════════════════════════
# Phase 5: Insight Generation
# ═══════════════════════════════════════════════════════════════════════

class TestInsightGeneration:
    async def test_skips_on_wrong_tick(self):
        """Only runs every 5th tick."""
        pulse = _make_pulse()
        pulse._tick_count = 3  # Not divisible by 5
        await pulse._insight_generation()  # should skip silently

    async def test_runs_on_5th_tick(self):
        engine = MagicMock()
        soul = MagicMock()
        soul.get_all_concepts = MagicMock(return_value=[MagicMock(), MagicMock()])
        engine.soul = soul
        assoc = MagicMock()
        assoc.get_strongest_edges = MagicMock(return_value=[MagicMock()])
        engine.associations = assoc
        pulse = _make_pulse(engine=engine)
        pulse._tick_count = 5
        await pulse._insight_generation()  # should not raise

    async def test_fallback_to_edges_attribute(self):
        """If get_strongest_edges missing, uses _edges."""
        engine = MagicMock()
        soul = MagicMock()
        soul.get_all_concepts = MagicMock(return_value=[MagicMock(), MagicMock()])
        engine.soul = soul
        assoc = MagicMock(spec=[])  # no get_strongest_edges
        edge = MagicMock()
        edge.weight = 0.9
        assoc._edges = {"e1": edge}
        engine.associations = assoc
        pulse = _make_pulse(engine=engine)
        pulse._tick_count = 10
        await pulse._insight_generation()

    async def test_no_soul_skips(self):
        engine = MagicMock()
        engine.soul = None
        engine.associations = MagicMock()
        pulse = _make_pulse(engine=engine)
        pulse._tick_count = 5
        await pulse._insight_generation()  # no crash


# ═══════════════════════════════════════════════════════════════════════
# Phase 6: Procedure Refinement
# ═══════════════════════════════════════════════════════════════════════

class TestProcedureRefinement:
    async def test_skips_on_wrong_tick(self):
        """Only runs every 3rd tick."""
        pulse = _make_pulse()
        pulse._tick_count = 2
        await pulse._procedure_refinement()  # skip

    async def test_runs_on_3rd_tick(self):
        ep_event = _mock_event(metadata={"procedure_id": "proc1"})
        ep = _mock_episode(outcome="success", events=[ep_event], is_active=False)
        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(return_value=["a1"])
        episodic.get_recent = MagicMock(return_value=[ep])
        procedural = MagicMock()
        procedural.record_procedure_outcome = MagicMock()
        procedural.decay_all_reliability = MagicMock(return_value=3)
        pulse = _make_pulse(episodic_store=episodic, procedural_store=procedural)
        pulse._tick_count = 3
        await pulse._procedure_refinement()
        procedural.record_procedure_outcome.assert_called_once_with("proc1", True)

    async def test_failure_outcome_records_false(self):
        ep_event = _mock_event(metadata={"procedure_id": "proc1"})
        ep = _mock_episode(outcome="failure", events=[ep_event], is_active=False)
        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(return_value=["a1"])
        episodic.get_recent = MagicMock(return_value=[ep])
        procedural = MagicMock()
        procedural.decay_all_reliability = MagicMock(return_value=0)
        pulse = _make_pulse(episodic_store=episodic, procedural_store=procedural)
        pulse._tick_count = 6
        await pulse._procedure_refinement()
        procedural.record_procedure_outcome.assert_called_once_with("proc1", False)

    async def test_skips_active_episodes(self):
        ep = _mock_episode(is_active=True, events=[_mock_event()])
        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(return_value=["a1"])
        episodic.get_recent = MagicMock(return_value=[ep])
        procedural = MagicMock()
        procedural.decay_all_reliability = MagicMock(return_value=0)
        pulse = _make_pulse(episodic_store=episodic, procedural_store=procedural)
        pulse._tick_count = 9
        await pulse._procedure_refinement()
        procedural.record_procedure_outcome.assert_not_called()

    async def test_events_without_procedure_id_skipped(self):
        ep_event = _mock_event(metadata={"other_key": "val"})
        ep = _mock_episode(outcome="success", events=[ep_event], is_active=False)
        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(return_value=["a1"])
        episodic.get_recent = MagicMock(return_value=[ep])
        procedural = MagicMock()
        procedural.decay_all_reliability = MagicMock(return_value=0)
        pulse = _make_pulse(episodic_store=episodic, procedural_store=procedural)
        pulse._tick_count = 3
        await pulse._procedure_refinement()
        procedural.record_procedure_outcome.assert_not_called()

    async def test_decay_called(self):
        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(return_value=[])
        procedural = MagicMock()
        procedural.decay_all_reliability = MagicMock(return_value=5)
        pulse = _make_pulse(episodic_store=episodic, procedural_store=procedural)
        pulse._tick_count = 3
        await pulse._procedure_refinement()
        procedural.decay_all_reliability.assert_called_once_with(decay_rate=0.002)


# ═══════════════════════════════════════════════════════════════════════
# Phase 7: Meta Self-Reflection
# ═══════════════════════════════════════════════════════════════════════

class TestMetaSelfReflection:
    async def test_skips_on_wrong_meta_tick(self):
        """Reflection only fires every N meta ticks."""
        pulse = _make_pulse()
        pulse._meta_tick_count = 0  # Will become 1 in the method
        await pulse._meta_self_reflection()
        # tick 1 (not divisible by 10) — should skip

    async def test_fires_on_10th_meta_tick(self):
        meta = MagicMock()
        meta._config = MagicMock()
        meta._config.reflection_interval_ticks = 10
        meta.record_metric = MagicMock()
        meta.generate_proposals_from_metrics = AsyncMock(return_value=None)
        engine = MagicMock()
        semantic = MagicMock()
        semantic.get_stats = MagicMock(return_value={"concept_count": 42})
        episodic = MagicMock()
        episodic.get_stats = MagicMock(return_value={
            "active_episodes": 3, "total_history_episodes": 100
        })
        container = _make_container(
            meta_memory=meta, engine=engine,
            semantic_store=semantic, episodic_store=episodic,
        )
        pulse = _make_pulse(container=container)
        pulse._meta_tick_count = 9  # Will become 10 inside
        pulse._tick_count = 50
        await pulse._meta_self_reflection()
        meta.record_metric.assert_called()
        meta.generate_proposals_from_metrics.assert_awaited_once()

    async def test_records_phase_duration_metrics(self):
        meta = MagicMock()
        meta._config = None  # No config → default interval 10
        meta.record_metric = MagicMock()
        meta.generate_proposals_from_metrics = AsyncMock()
        engine = MagicMock()
        container = _make_container(meta_memory=meta, engine=engine)
        pulse = _make_pulse(container=container)
        pulse._meta_tick_count = 9
        pulse._phase_durations = {"wm_maintenance": 0.01, "episodic_chaining": 0.02}
        await pulse._meta_self_reflection()
        # Check phase duration metrics recorded
        calls = [str(c) for c in meta.record_metric.call_args_list]
        assert any("pulse_phase_wm_maintenance" in c for c in calls)

    async def test_records_phase_error_metrics(self):
        meta = MagicMock()
        meta._config = None
        meta.record_metric = MagicMock()
        meta.generate_proposals_from_metrics = AsyncMock()
        engine = MagicMock()
        container = _make_container(meta_memory=meta, engine=engine)
        pulse = _make_pulse(container=container)
        pulse._meta_tick_count = 9
        pulse._phase_errors = {"gap_detection": 3}
        await pulse._meta_self_reflection()
        calls = [str(c) for c in meta.record_metric.call_args_list]
        assert any("gap_detection_errors" in c for c in calls)

    async def test_meta_memory_error_caught(self):
        meta = MagicMock()
        meta._config = None
        meta.record_metric = MagicMock(side_effect=RuntimeError("db error"))
        engine = MagicMock()
        container = _make_container(meta_memory=meta, engine=engine)
        pulse = _make_pulse(container=container)
        pulse._meta_tick_count = 9
        # Should not raise
        await pulse._meta_self_reflection()


# ═══════════════════════════════════════════════════════════════════════
# Full Tick
# ═══════════════════════════════════════════════════════════════════════

class TestFullTick:
    async def test_tick_increments_counter(self):
        pulse = _make_pulse()
        await pulse.tick()
        assert pulse._tick_count == 1

    async def test_tick_sets_last_tick_at(self):
        pulse = _make_pulse()
        assert pulse._last_tick_at is None
        await pulse.tick()
        assert pulse._last_tick_at is not None

    async def test_multiple_ticks_accumulate(self):
        pulse = _make_pulse()
        await pulse.tick()
        await pulse.tick()
        await pulse.tick()
        assert pulse._tick_count == 3

    async def test_tick_records_all_phase_durations(self):
        pulse = _make_pulse()
        await pulse.tick()
        # All 7 phases should have durations (even if 0.0)
        assert "wm_maintenance" in pulse._phase_durations
        assert "episodic_chaining" in pulse._phase_durations
        assert "semantic_refresh" in pulse._phase_durations

    async def test_phase_error_does_not_stop_subsequent_phases(self):
        """A failure in one phase should not prevent the next from running."""
        wm = MagicMock()
        wm.prune_all = MagicMock(side_effect=RuntimeError("crash"))
        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(return_value=[])
        pulse = _make_pulse(working_memory=wm, episodic_store=episodic)
        await pulse.tick()
        # WM crashed but episodic still ran
        assert pulse._phase_errors.get("wm_maintenance", 0) == 1
        assert "episodic_chaining" in pulse._phase_durations

    async def test_multiple_phase_errors_accumulated(self):
        wm = MagicMock()
        wm.prune_all = MagicMock(side_effect=RuntimeError("crash"))
        pulse = _make_pulse(working_memory=wm)
        await pulse.tick()
        await pulse.tick()
        assert pulse._phase_errors.get("wm_maintenance", 0) == 2


# ═══════════════════════════════════════════════════════════════════════
# Start / Stop Lifecycle
# ═══════════════════════════════════════════════════════════════════════

class TestPulseLifecycle:
    async def test_start_sets_running(self):
        pulse = _make_pulse(config=_make_config(interval_seconds=1))
        task = asyncio.create_task(pulse.start())
        await asyncio.sleep(0.05)
        assert pulse._running is True
        pulse.stop()
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_start_disabled_returns_immediately(self):
        pulse = _make_pulse(config=_make_config(enabled=False))
        await pulse.start()
        assert pulse._running is False

    def test_stop_without_start(self):
        pulse = _make_pulse()
        pulse.stop()  # should not raise
        assert pulse._running is False

    async def test_start_runs_tick(self):
        pulse = _make_pulse(config=_make_config(interval_seconds=0.05))
        task = asyncio.create_task(pulse.start())
        await asyncio.sleep(0.15)
        assert pulse._tick_count >= 1
        pulse.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ═══════════════════════════════════════════════════════════════════════
# get_stats
# ═══════════════════════════════════════════════════════════════════════

class TestPulseStats:
    def test_stats_fields(self):
        pulse = _make_pulse()
        stats = pulse.get_stats()
        assert "running" in stats
        assert "tick_count" in stats
        assert "meta_tick_count" in stats
        assert "last_tick_at" in stats
        assert "phase_durations" in stats
        assert "phase_errors" in stats
        assert "interval_seconds" in stats

    def test_stats_initial_values(self):
        pulse = _make_pulse()
        stats = pulse.get_stats()
        assert stats["running"] is False
        assert stats["tick_count"] == 0
        assert stats["last_tick_at"] is None

    async def test_stats_after_tick(self):
        pulse = _make_pulse()
        await pulse.tick()
        stats = pulse.get_stats()
        assert stats["tick_count"] == 1
        assert stats["last_tick_at"] is not None

    def test_stats_interval_from_config(self):
        pulse = _make_pulse(config=_make_config(interval_seconds=42))
        stats = pulse.get_stats()
        assert stats["interval_seconds"] == 42


# ═══════════════════════════════════════════════════════════════════════
# Phase Skip Logic — Integration
# ═══════════════════════════════════════════════════════════════════════

class TestPhaseSkipLogic:
    """Verify that tick-modulus gates work correctly across multiple ticks."""

    async def test_insight_generation_only_every_5_ticks(self):
        engine = MagicMock()
        soul = MagicMock()
        soul.get_all_concepts = MagicMock(return_value=[MagicMock(), MagicMock()])
        engine.soul = soul
        assoc = MagicMock()
        assoc.get_strongest_edges = MagicMock(return_value=[])
        engine.associations = assoc

        pulse = _make_pulse(engine=engine)
        call_counts = []
        for i in range(10):
            assoc.get_strongest_edges.reset_mock()
            await pulse.tick()
            call_counts.append(assoc.get_strongest_edges.call_count)

        # Should fire on tick 5 and 10 (indexes 4 and 9)
        assert call_counts[4] == 1  # tick 5
        assert call_counts[9] == 1  # tick 10
        assert call_counts[0] == 0  # tick 1 — skipped
        assert call_counts[2] == 0  # tick 3 — skipped

    async def test_procedure_refinement_only_every_3_ticks(self):
        episodic = MagicMock()
        episodic.get_all_agent_ids = MagicMock(return_value=[])
        procedural = MagicMock()
        procedural.decay_all_reliability = MagicMock(return_value=0)

        pulse = _make_pulse(episodic_store=episodic, procedural_store=procedural)
        decay_calls = []
        for i in range(9):
            procedural.decay_all_reliability.reset_mock()
            await pulse.tick()
            decay_calls.append(procedural.decay_all_reliability.call_count)

        # Should fire on ticks 3, 6, 9
        assert decay_calls[2] == 1  # tick 3
        assert decay_calls[5] == 1  # tick 6
        assert decay_calls[8] == 1  # tick 9
        assert decay_calls[0] == 0  # tick 1
        assert decay_calls[1] == 0  # tick 2


# ═══════════════════════════════════════════════════════════════════════
# _run_phase — Isolated
# ═══════════════════════════════════════════════════════════════════════

class TestRunPhase:
    async def test_success_records_duration(self):
        pulse = _make_pulse()

        async def noop():
            pass

        await pulse._run_phase("test_phase", noop)
        assert "test_phase" in pulse._phase_durations
        assert pulse._phase_durations["test_phase"] >= 0

    async def test_error_records_duration_and_error_count(self):
        pulse = _make_pulse()

        async def fail():
            raise ValueError("boom")

        await pulse._run_phase("test_phase", fail)
        assert pulse._phase_errors["test_phase"] == 1
        assert "test_phase" in pulse._phase_durations

    async def test_error_count_accumulates(self):
        pulse = _make_pulse()

        async def fail():
            raise ValueError("boom")

        await pulse._run_phase("test_phase", fail)
        await pulse._run_phase("test_phase", fail)
        assert pulse._phase_errors["test_phase"] == 2
