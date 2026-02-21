"""
Tests for Phase 5.0 Contradiction Detection Module.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timezone

from mnemocore.core.contradiction import (
    ContradictionRecord,
    ContradictionRegistry,
    ContradictionDetector,
    get_contradiction_detector,
)
from mnemocore.core.provenance import ProvenanceRecord


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _make_node(memory_id: str, content: str = "test content", hdv_data: bytes = None):
    import numpy as np
    node = MagicMock()
    node.id = memory_id
    node.content = content
    # Create a fake HDV with random binary data
    data = np.frombuffer(hdv_data or bytes(2048), dtype=np.uint8) if hdv_data else np.zeros(2048, dtype=np.uint8)
    node.hdv = MagicMock()
    node.hdv.data = data
    node.metadata = {}
    node.provenance = None
    return node


# ------------------------------------------------------------------ #
#  ContradictionRecord                                               #
# ------------------------------------------------------------------ #

class TestContradictionRecord:
    def test_auto_group_id(self):
        r = ContradictionRecord(memory_a_id="a", memory_b_id="b")
        assert r.group_id.startswith("cg_")

    def test_to_dict(self):
        r = ContradictionRecord(memory_a_id="mem_a", memory_b_id="mem_b", similarity_score=0.87)
        d = r.to_dict()
        assert d["memory_a_id"] == "mem_a"
        assert d["similarity_score"] == pytest.approx(0.87, abs=0.001)
        assert d["resolved"] is False


# ------------------------------------------------------------------ #
#  ContradictionRegistry                                             #
# ------------------------------------------------------------------ #

class TestContradictionRegistry:
    def test_register_and_list(self):
        reg = ContradictionRegistry()
        rec = ContradictionRecord(memory_a_id="a", memory_b_id="b")
        reg.register(rec)
        all_recs = reg.list_all(unresolved_only=False)
        assert len(all_recs) == 1

    def test_unresolved_only(self):
        reg = ContradictionRegistry()
        r1 = ContradictionRecord(memory_a_id="a", memory_b_id="b")
        r2 = ContradictionRecord(memory_a_id="c", memory_b_id="d")
        reg.register(r1)
        reg.register(r2)
        reg.resolve(r1.group_id, note="fixed")
        unresolved = reg.list_all(unresolved_only=True)
        assert len(unresolved) == 1
        assert unresolved[0].group_id == r2.group_id

    def test_resolve_unknown_returns_false(self):
        reg = ContradictionRegistry()
        assert reg.resolve("cg_nonexistent") is False

    def test_resolve_sets_note(self):
        reg = ContradictionRegistry()
        r = ContradictionRecord(memory_a_id="a", memory_b_id="b")
        reg.register(r)
        reg.resolve(r.group_id, note="duplicate info")
        assert reg._records[r.group_id].resolution_note == "duplicate info"

    def test_list_for_memory(self):
        reg = ContradictionRegistry()
        r = ContradictionRecord(memory_a_id="mem_x", memory_b_id="mem_y")
        reg.register(r)
        found = reg.list_for_memory("mem_x")
        assert len(found) == 1
        not_found = reg.list_for_memory("mem_z")
        assert len(not_found) == 0

    def test_len_counts_unresolved(self):
        reg = ContradictionRegistry()
        r1 = ContradictionRecord(memory_a_id="a", memory_b_id="b")
        r2 = ContradictionRecord(memory_a_id="c", memory_b_id="d")
        reg.register(r1)
        reg.register(r2)
        assert len(reg) == 2
        reg.resolve(r1.group_id)
        assert len(reg) == 1


# ------------------------------------------------------------------ #
#  ContradictionDetector.hamming_similarity                          #
# ------------------------------------------------------------------ #

class TestHammingSimilarity:
    def test_identical_nodes_similarity_one(self):
        import numpy as np
        data = np.random.randint(0, 255, 2048, dtype=np.uint8).tobytes()
        a = _make_node("a", hdv_data=data)
        b = _make_node("b", hdv_data=data)
        detector = ContradictionDetector(engine=None, use_llm=False)
        sim = detector._hamming_similarity(a, b)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_random_nodes_similarity_near_half(self):
        """Random binary vectors should have ~50% similarity (Hamming)."""
        import numpy as np
        rng = np.random.default_rng(42)
        a = _make_node("a", hdv_data=rng.integers(0, 255, 2048, dtype=np.uint8).tobytes())
        b = _make_node("b", hdv_data=rng.integers(0, 255, 2048, dtype=np.uint8).tobytes())
        detector = ContradictionDetector(engine=None, use_llm=False)
        sim = detector._hamming_similarity(a, b)
        # Should be around 0.5 ± 0.1
        assert 0.4 < sim < 0.6


# ------------------------------------------------------------------ #
#  check_on_store                                                    #
# ------------------------------------------------------------------ #

class TestCheckOnStore:
    @pytest.mark.asyncio
    async def test_no_contradiction_when_candidates_empty(self):
        detector = ContradictionDetector(engine=None, use_llm=False)
        node = _make_node("new_mem")
        result = await detector.check_on_store(node, candidates=[])
        assert result is None

    @pytest.mark.asyncio
    async def test_high_similarity_without_llm_flags_very_high(self):
        """Without LLM, only similarity >= 0.90 counts as contradiction."""
        import numpy as np
        identical_data = bytes(2048)
        new_node = _make_node("new", hdv_data=identical_data)
        existing = _make_node("old", hdv_data=identical_data)
        new_node.provenance = ProvenanceRecord.new(origin_type="observation")
        existing.provenance = ProvenanceRecord.new(origin_type="observation")

        detector = ContradictionDetector(
            engine=None,
            use_llm=False,
            similarity_threshold=0.80,
        )
        result = await detector.check_on_store(new_node, candidates=[existing])
        # Identical nodes have similarity=1.0 → contradiction without LLM
        assert result is not None
        assert result.memory_a_id == "new"
        assert result.memory_b_id == "old"

    @pytest.mark.asyncio
    async def test_contradiction_flags_provenance(self):
        import numpy as np
        identical_data = bytes(2048)
        new_node = _make_node("new2", hdv_data=identical_data)
        existing = _make_node("old2", hdv_data=identical_data)
        new_node.provenance = ProvenanceRecord.new(origin_type="observation")
        existing.provenance = ProvenanceRecord.new(origin_type="observation")

        detector = ContradictionDetector(engine=None, use_llm=False, similarity_threshold=0.80)
        result = await detector.check_on_store(new_node, candidates=[existing])
        if result:
            # Both nodes should be flagged
            assert "contradiction_group_id" in new_node.metadata
            assert new_node.provenance.is_contradicted()
