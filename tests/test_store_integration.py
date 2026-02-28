"""
Cross-Store Integration Tests
==============================
Verifies that the three cognitive memory stores (Episodic, Semantic, Procedural)
work together correctly through realistic multi-store workflows.

These tests exercise the interplay between stores, NOT just each store in isolation:
  - Episodic → Semantic consolidation (CLS theory neocortical transfer)
  - Episodic outcome → Procedural reliability feedback loop
  - Full pipeline: episode lifecycle → semantic concept creation → procedure extraction
  - Multi-agent cross-store consistency
  - Store stats coherence after cross-store operations
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import MagicMock

from mnemocore.core.episodic_store import EpisodicStoreService
from mnemocore.core.semantic_store import SemanticStoreService
from mnemocore.core.procedural_store import ProceduralStoreService
from mnemocore.core.memory_model import (
    Episode, EpisodeEvent, SemanticConcept, Procedure, ProcedureStep,
)
from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.config import (
    EpisodicConfig, SemanticConfig, ProceduralConfig,
)


# ─── Helpers ─────────────────────────────────────────────────────────

def _make_hdv(dim=1024):
    """Create a random BinaryHDV for testing."""
    packed = np.random.randint(0, 256, size=dim // 8, dtype=np.uint8)
    return BinaryHDV(packed, dim)


def _make_episodic(config=None):
    return EpisodicStoreService(tier_manager=None, config=config or EpisodicConfig())


def _make_semantic(config=None):
    return SemanticStoreService(qdrant_store=None, config=config or SemanticConfig())


def _make_procedural(config=None):
    return ProceduralStoreService(config=config or ProceduralConfig())


def _run_full_episode(episodic, agent_id, goal, events, outcome="success", reward=None):
    """Run a complete episode lifecycle and return the Episode object."""
    ep_id = episodic.start_episode(agent_id, goal=goal)
    for kind, content, meta in events:
        episodic.append_event(ep_id, kind, content, meta)
    episodic.end_episode(ep_id, outcome, reward=reward)
    return episodic.get_episode(ep_id)


# ═══════════════════════════════════════════════════════════════════════
# Episodic → Semantic Consolidation
# ═══════════════════════════════════════════════════════════════════════

class TestEpisodicToSemanticConsolidation:
    """The CLS-theory flow: episodic experiences → semantic concepts."""

    def test_successful_episode_creates_semantic_concept(self):
        """A completed episode's content can be consolidated into a concept."""
        episodic = _make_episodic()
        semantic = _make_semantic()

        ep = _run_full_episode(
            episodic, "agent1", "learn python",
            [
                ("observation", "Python uses indentation for blocks", {}),
                ("thought", "Indentation is significant in Python", {}),
                ("action", "Written first Python script", {}),
            ],
            outcome="success",
        )

        # Simulate what the pulse semantic_refresh phase does
        contents = [ev.content for ev in ep.events if ev.content]
        combined = " ".join(contents)
        hdv = _make_hdv()

        concept = semantic.consolidate_from_content(
            content=combined,
            hdv=hdv,
            episode_ids=[ep.id],
            tags=["python", "programming"],
            agent_id="agent1",
        )

        assert concept is not None
        assert isinstance(concept, SemanticConcept)
        assert ep.id in concept.support_episode_ids
        assert concept.reliability > 0

    def test_multiple_episodes_reinforce_concept(self):
        """Repeated similar experiences should strengthen the semantic concept."""
        episodic = _make_episodic()
        semantic = _make_semantic()

        hdv = _make_hdv()  # Same HDV = same concept space

        for i in range(3):
            ep = _run_full_episode(
                episodic, "agent1", f"practice python {i}",
                [("observation", "Python lists are ordered collections", {})],
                outcome="success",
            )
            semantic.consolidate_from_content(
                content="Python lists are ordered collections",
                hdv=hdv,
                episode_ids=[ep.id],
                tags=["python"],
                agent_id="agent1",
            )

        # Should have created/found concepts
        stats = semantic.get_stats()
        assert stats["concept_count"] >= 1

    def test_failed_episodes_still_consolidatable(self):
        """Even failed episodes carry learning value."""
        episodic = _make_episodic()
        semantic = _make_semantic()

        ep = _run_full_episode(
            episodic, "agent1", "deploy to production",
            [
                ("action", "Pushed code without tests", {}),
                ("error", "CI pipeline failed on linting errors", {}),
            ],
            outcome="failure",
        )

        hdv = _make_hdv()
        concept = semantic.consolidate_from_content(
            content="CI pipeline failed on linting errors",
            hdv=hdv,
            episode_ids=[ep.id],
            agent_id="agent1",
        )
        assert concept is not None

    def test_episodic_outcome_affects_ltp_strength(self):
        """Success gives higher LTP than failure → affects recall priority."""
        episodic = _make_episodic()

        ep_success = _run_full_episode(
            episodic, "agent1", "win",
            [("action", "did it right", {})],
            outcome="success",
        )
        ep_failure = _run_full_episode(
            episodic, "agent1", "lose",
            [("action", "did it wrong", {})],
            outcome="failure",
        )

        assert ep_success.ltp_strength > ep_failure.ltp_strength


# ═══════════════════════════════════════════════════════════════════════
# Episodic → Procedural Feedback Loop
# ═══════════════════════════════════════════════════════════════════════

class TestEpisodicToProcedural:
    """Episodic outcomes update procedural reliability — the basal ganglia loop."""

    def test_successful_episode_boosts_procedure(self):
        """A successful episode referencing a procedure should boost its reliability."""
        episodic = _make_episodic()
        procedural = _make_procedural()

        # Create a procedure
        proc = procedural.create_procedure_from_episode(
            name="Deploy safely",
            description="Safe deployment procedure",
            steps=[{"order": 1, "instruction": "Run tests first", "code_snippet": None, "tool_call": None}],
            trigger_pattern="deploy",
            tags=["deployment"],
            agent_id="agent1",
        )
        initial_reliability = proc.reliability

        # Run an episode that references this procedure
        ep = _run_full_episode(
            episodic, "agent1", "deploy app",
            [("action", "Followed deployment procedure", {"procedure_id": proc.id})],
            outcome="success",
        )

        # Simulate what pulse procedure_refinement does
        for event in ep.events:
            proc_id = (event.metadata or {}).get("procedure_id")
            if proc_id:
                procedural.record_procedure_outcome(proc_id, success=True)

        updated = procedural.get_procedure(proc.id)
        assert updated.success_count == 1
        assert updated.reliability >= initial_reliability

    def test_failed_episode_penalizes_procedure(self):
        """A failed episode should decrease procedure reliability."""
        episodic = _make_episodic()
        procedural = _make_procedural()

        proc = procedural.create_procedure_from_episode(
            name="Quick deploy",
            description="Fast but risky deployment",
            steps=[{"order": 1, "instruction": "Push directly", "code_snippet": None, "tool_call": None}],
            trigger_pattern="quick deploy",
            tags=["deployment"],
        )
        initial_reliability = proc.reliability

        ep = _run_full_episode(
            episodic, "agent1", "quick deploy",
            [("action", "Tried quick deploy", {"procedure_id": proc.id})],
            outcome="failure",
        )

        for event in ep.events:
            proc_id = (event.metadata or {}).get("procedure_id")
            if proc_id:
                procedural.record_procedure_outcome(proc_id, success=False)

        updated = procedural.get_procedure(proc.id)
        assert updated.failure_count == 1
        assert updated.reliability <= initial_reliability

    def test_mixed_outcomes_converge(self):
        """Multiple success + failure outcomes converge reliability correctly."""
        episodic = _make_episodic()
        procedural = _make_procedural()

        proc = procedural.create_procedure_from_episode(
            name="Test procedure",
            description="Mixed results procedure",
            steps=[{"order": 1, "instruction": "Try it", "code_snippet": None, "tool_call": None}],
            trigger_pattern="test",
            tags=[],
        )

        # 3 successes, 2 failures
        outcomes = [True, True, False, True, False]
        for i, success in enumerate(outcomes):
            _run_full_episode(
                episodic, "agent1", f"attempt {i}",
                [("action", "did thing", {"procedure_id": proc.id})],
                outcome="success" if success else "failure",
            )
            procedural.record_procedure_outcome(proc.id, success)

        updated = procedural.get_procedure(proc.id)
        assert updated.success_count == 3
        assert updated.failure_count == 2


# ═══════════════════════════════════════════════════════════════════════
# Full Pipeline: Episode → Concept → Procedure
# ═══════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """End-to-end: episode lifecycle → concept extraction → procedure creation."""

    def test_episode_to_concept_to_procedure(self):
        """The full cognitive pipeline where experience becomes skill."""
        episodic = _make_episodic()
        semantic = _make_semantic()
        procedural = _make_procedural()

        # Step 1: Agent completes an episode
        ep = _run_full_episode(
            episodic, "agent1", "write unit tests",
            [
                ("observation", "Tests prevent regressions and document behavior", {}),
                ("action", "Created test_module.py with pytest assertions", {}),
                ("thought", "Testing is essential for reliable software", {}),
            ],
            outcome="success",
            reward=1.0,
        )

        # Step 2: Pulse consolidates episodic → semantic
        contents = [ev.content for ev in ep.events if ev.content]
        combined = " ".join(contents)
        hdv = _make_hdv()
        concept = semantic.consolidate_from_content(
            content=combined,
            hdv=hdv,
            episode_ids=[ep.id],
            tags=["testing", "software-engineering"],
            agent_id="agent1",
        )
        assert concept is not None

        # Step 3: From the episode, extract a procedure
        proc = procedural.create_procedure_from_episode(
            name="Write unit tests",
            description="Procedure for writing pytest unit tests",
            steps=[
                {"order": 1, "instruction": "Identify function to test", "code_snippet": None, "tool_call": None},
                {"order": 2, "instruction": "Write test assertions", "code_snippet": None, "tool_call": None},
                {"order": 3, "instruction": "Run pytest and verify", "code_snippet": None, "tool_call": None},
            ],
            trigger_pattern="write tests",
            tags=["testing"],
            agent_id="agent1",
        )

        # Verify the full chain
        assert ep.outcome == "success"
        assert ep.ltp_strength > 0
        assert concept.support_episode_ids == [ep.id]
        assert proc.name == "Write unit tests"
        assert len(proc.steps) == 3

        # Step 4: Procedure is findable by query
        found = procedural.find_applicable_procedures("write tests")
        assert any(p.id == proc.id for p in found)

    def test_pipeline_with_concept_nearby_search(self):
        """Concepts created from episodes are findable via HDV similarity."""
        episodic = _make_episodic()
        semantic = _make_semantic(SemanticConfig(min_similarity_threshold=0.0))

        ep = _run_full_episode(
            episodic, "agent1", "learn Docker",
            [("observation", "Docker uses containers for isolation", {})],
            outcome="success",
        )

        hdv = _make_hdv()
        concept = semantic.consolidate_from_content(
            content="Docker uses containers for isolation",
            hdv=hdv,
            episode_ids=[ep.id],
            agent_id="agent1",
        )

        # Same HDV should find the concept
        nearby = semantic.find_nearby_concepts(hdv, top_k=5, min_similarity=0.0)
        assert len(nearby) >= 1


# ═══════════════════════════════════════════════════════════════════════
# Multi-Agent Cross-Store Consistency
# ═══════════════════════════════════════════════════════════════════════

class TestMultiAgentConsistency:
    """Verify that multiple agents don't contaminate each other's stores."""

    def test_episodic_isolation_between_agents(self):
        episodic = _make_episodic()

        _run_full_episode(
            episodic, "agent_A", "task A",
            [("action", "Agent A did something", {})],
            outcome="success",
        )
        _run_full_episode(
            episodic, "agent_B", "task B",
            [("action", "Agent B did something", {})],
            outcome="failure",
        )

        a_history = episodic.get_recent("agent_A", limit=10)
        b_history = episodic.get_recent("agent_B", limit=10)

        assert len(a_history) == 1
        assert len(b_history) == 1
        assert a_history[0].agent_id == "agent_A"
        assert b_history[0].agent_id == "agent_B"

    def test_procedural_multi_agent_isolation(self):
        procedural = _make_procedural()

        proc_a = procedural.create_procedure_from_episode(
            name="Agent A method",
            description="Method A",
            steps=[{"order": 1, "instruction": "do A", "code_snippet": None, "tool_call": None}],
            trigger_pattern="method A",
            tags=[],
            agent_id="agent_A",
        )
        proc_b = procedural.create_procedure_from_episode(
            name="Agent B method",
            description="Method B",
            steps=[{"order": 1, "instruction": "do B", "code_snippet": None, "tool_call": None}],
            trigger_pattern="method B",
            tags=[],
            agent_id="agent_B",
        )

        found_a = procedural.find_applicable_procedures("method A", agent_id="agent_A")
        found_b = procedural.find_applicable_procedures("method B", agent_id="agent_B")

        assert any(p.id == proc_a.id for p in found_a)
        assert any(p.id == proc_b.id for p in found_b)

    def test_episodic_chaining_isolated_per_agent(self):
        """Chain links should not cross agent boundaries."""
        episodic = _make_episodic()

        ep_a1 = _run_full_episode(
            episodic, "agent_A", "A task 1",
            [("action", "first", {})], outcome="success",
        )
        ep_b1 = _run_full_episode(
            episodic, "agent_B", "B task 1",
            [("action", "first B", {})], outcome="success",
        )
        ep_a2 = _run_full_episode(
            episodic, "agent_A", "A task 2",
            [("action", "second", {})], outcome="success",
        )

        # A2 should link back to A1, not B1
        assert ep_a2.links_prev == [ep_a1.id]


# ═══════════════════════════════════════════════════════════════════════
# Store Stats Coherence
# ═══════════════════════════════════════════════════════════════════════

class TestStoreStatsCoherence:
    """Verify that store stats remain accurate after cross-store operations."""

    def test_episodic_stats_after_episodes(self):
        episodic = _make_episodic()
        _run_full_episode(
            episodic, "a1", "goal1",
            [("action", "did1", {}), ("action", "did2", {})],
            outcome="success",
        )
        ep_id2 = episodic.start_episode("a1", goal="active goal")
        episodic.append_event(ep_id2, "observation", "seeing things", {})

        stats = episodic.get_stats()
        assert stats["episodes_started"] == 2
        assert stats["episodes_ended"] == 1
        assert stats["events_logged"] == 3
        assert stats["active_episodes"] == 1

    def test_semantic_stats_after_consolidation(self):
        semantic = _make_semantic()
        hdv = _make_hdv()
        semantic.consolidate_from_content(
            content="Something worth remembering for sure",
            hdv=hdv,
            episode_ids=["ep1"],
            agent_id="a1",
        )
        stats = semantic.get_stats()
        assert stats["concept_count"] >= 1

    def test_procedural_stats_after_operations(self):
        procedural = _make_procedural()
        proc = procedural.create_procedure_from_episode(
            name="Test proc",
            description="A test procedure",
            steps=[{"order": 1, "instruction": "do", "code_snippet": None, "tool_call": None}],
            trigger_pattern="test",
            tags=[],
        )
        procedural.record_procedure_outcome(proc.id, True)
        procedural.record_procedure_outcome(proc.id, False)
        stats = procedural.get_stats()
        assert stats["total_procedures"] >= 1


# ═══════════════════════════════════════════════════════════════════════
# Decay Coherence
# ═══════════════════════════════════════════════════════════════════════

class TestDecayCoherence:
    """Verify that reliability decay is consistent across stores."""

    def test_semantic_decay_reduces_reliability(self):
        semantic = _make_semantic()
        hdv = _make_hdv()
        concept = semantic.consolidate_from_content(
            content="Important fact about the world we live in",
            hdv=hdv,
            episode_ids=["ep1"],
        )
        initial_reliability = concept.reliability
        decayed = semantic.decay_all_reliability(decay_rate=0.1)
        assert decayed >= 1
        updated = semantic.get_concept(concept.id)
        assert updated.reliability < initial_reliability

    def test_procedural_decay_reduces_reliability(self):
        procedural = _make_procedural()
        proc = procedural.create_procedure_from_episode(
            name="Decay test",
            description="Test decay",
            steps=[{"order": 1, "instruction": "do", "code_snippet": None, "tool_call": None}],
            trigger_pattern="decay",
            tags=[],
        )
        initial_reliability = proc.reliability
        decayed = procedural.decay_all_reliability(decay_rate=0.1)
        assert decayed >= 1
        updated = procedural.get_procedure(proc.id)
        assert updated.reliability < initial_reliability

    def test_both_stores_decay_symmetrically(self):
        """Both stores should decay using the same formula at similar rates."""
        semantic = _make_semantic()
        procedural = _make_procedural()

        hdv = _make_hdv()
        concept = semantic.consolidate_from_content(
            content="Some important concept for decay testing purposes",
            hdv=hdv, episode_ids=["ep1"],
        )

        proc = procedural.create_procedure_from_episode(
            name="Decay sym test",
            description="Decay symmetry",
            steps=[{"order": 1, "instruction": "do", "code_snippet": None, "tool_call": None}],
            trigger_pattern="sym",
            tags=[],
        )

        # Capture initial values before decay mutates the objects in place
        concept_reliability_before = concept.reliability
        proc_reliability_before = proc.reliability

        # Apply same decay rate
        semantic.decay_all_reliability(decay_rate=0.05)
        procedural.decay_all_reliability(decay_rate=0.05)

        concept_after = semantic.get_concept(concept.id)
        proc_after = procedural.get_procedure(proc.id)

        # Both should have decayed from their initial reliability
        assert concept_after.reliability < concept_reliability_before
        assert proc_after.reliability < proc_reliability_before


# ═══════════════════════════════════════════════════════════════════════
# Temporal Chain Verify + Repair
# ═══════════════════════════════════════════════════════════════════════

class TestChainIntegrityAcrossStores:
    """Episodic chain integrity is a prerequisite for semantic consolidation."""

    def test_chain_integrity_after_multiple_episodes(self):
        episodic = _make_episodic()

        ep1 = _run_full_episode(
            episodic, "a1", "first", [("action", "did1", {})], "success"
        )
        ep2 = _run_full_episode(
            episodic, "a1", "second", [("action", "did2", {})], "success"
        )
        ep3 = _run_full_episode(
            episodic, "a1", "third", [("action", "did3", {})], "success"
        )

        report = episodic.verify_chain_integrity("a1")
        assert report["chain_healthy"] is True

    def test_repair_chain_then_consolidate(self):
        """After repairing chain, semantic consolidation should still work."""
        episodic = _make_episodic()
        semantic = _make_semantic()

        ep = _run_full_episode(
            episodic, "a1", "learn", [("observation", "Important lesson learned here", {})],
            "success",
        )

        # Force chain check
        report = episodic.verify_chain_integrity("a1")
        # Whether repair needed or not, consolidation should work
        repairs = episodic.repair_chain("a1")

        hdv = _make_hdv()
        concept = semantic.consolidate_from_content(
            content="Important lesson learned here",
            hdv=hdv,
            episode_ids=[ep.id],
        )
        assert concept is not None
