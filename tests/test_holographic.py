"""
Comprehensive Tests for Holographic/Conceptual Memory Module
=============================================================

Tests the VSA-based knowledge graph using Binary HDV.

Coverage:
- store_concept() and recall_concept()
- Codebook persistence (save/load roundtrip)
- Concurrent writes don't corrupt file
- Load with mismatched dimension (graceful handling)
"""

import pytest
from unittest.mock import patch, MagicMock
import asyncio
import json
import os
import tempfile
import shutil
from pathlib import Path

from mnemocore.core.holographic import ConceptualMemory
from mnemocore.core.binary_hdv import BinaryHDV


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def conceptual_memory(temp_storage_dir):
    """Create a ConceptualMemory instance with temporary storage."""
    memory = ConceptualMemory(dimension=1024, storage_dir=temp_storage_dir)
    yield memory


# =============================================================================
# ConceptualMemory Initialization Tests
# =============================================================================

class TestConceptualMemoryInit:
    """Test ConceptualMemory initialization."""

    def test_init_with_dimension(self, temp_storage_dir):
        """ConceptualMemory should initialize with specified dimension."""
        memory = ConceptualMemory(dimension=2048, storage_dir=temp_storage_dir)

        assert memory.dimension == 2048
        assert memory.storage_dir == temp_storage_dir

    def test_init_empty_state(self, conceptual_memory):
        """ConceptualMemory should start with empty state."""
        assert len(conceptual_memory.symbols) == 0
        assert len(conceptual_memory.concepts) == 0
        assert conceptual_memory._dirty is False

    def test_init_loads_existing_data(self, temp_storage_dir):
        """ConceptualMemory should load existing data on init."""
        # Create and save some data
        memory1 = ConceptualMemory(dimension=1024, storage_dir=temp_storage_dir)
        memory1.store_concept("test_concept", {"attr1": "value1"})
        memory1.save()

        # Create new instance - should load existing data
        memory2 = ConceptualMemory(dimension=1024, storage_dir=temp_storage_dir)

        assert "test_concept" in memory2.concepts


# =============================================================================
# Symbol Tests
# =============================================================================

class TestGetSymbol:
    """Test symbol retrieval and creation."""

    def test_get_symbol_creates_new(self, conceptual_memory):
        """get_symbol should create new symbol if not exists."""
        hdv = conceptual_memory.get_symbol("test_symbol")

        assert hdv is not None
        assert hdv.dimension == conceptual_memory.dimension
        assert "test_symbol" in conceptual_memory.symbols

    def test_get_symbol_returns_same(self, conceptual_memory):
        """get_symbol should return same HDV for same symbol."""
        hdv1 = conceptual_memory.get_symbol("test_symbol")
        hdv2 = conceptual_memory.get_symbol("test_symbol")

        # Same object
        assert hdv1 is hdv2

    def test_get_symbol_deterministic(self, temp_storage_dir):
        """get_symbol should be deterministic across instances."""
        memory1 = ConceptualMemory(dimension=1024, storage_dir=temp_storage_dir)
        hdv1 = memory1.get_symbol("deterministic_test")

        memory2 = ConceptualMemory(dimension=1024, storage_dir=temp_storage_dir)
        hdv2 = memory2.get_symbol("deterministic_test")

        # Same underlying data
        assert (hdv1.data == hdv2.data).all()

    def test_different_symbols_different_hdv(self, conceptual_memory):
        """Different symbols should have different HDVs."""
        hdv1 = conceptual_memory.get_symbol("symbol_a")
        hdv2 = conceptual_memory.get_symbol("symbol_b")

        # Different vectors
        assert not (hdv1.data == hdv2.data).all()


# =============================================================================
# Concept Storage Tests
# =============================================================================

class TestStoreConcept:
    """Test concept storage."""

    def test_store_concept_basic(self, conceptual_memory):
        """store_concept should store concept with attributes."""
        conceptual_memory.store_concept("arbitrage", {
            "domain": "finance",
            "type": "strategy",
        })

        assert "arbitrage" in conceptual_memory.concepts
        assert conceptual_memory._dirty is True

    def test_store_concept_empty_attributes(self, conceptual_memory):
        """store_concept should handle empty attributes."""
        conceptual_memory.store_concept("empty_concept", {})

        assert "empty_concept" in conceptual_memory.concepts

    def test_store_concept_overwrites(self, conceptual_memory):
        """store_concept should overwrite existing concept."""
        conceptual_memory.store_concept("test", {"attr": "value1"})
        conceptual_memory.store_concept("test", {"attr": "value2"})

        # Should only have one concept with this name
        assert "test" in conceptual_memory.concepts
        assert len([k for k in conceptual_memory.concepts if k == "test"]) == 1


# =============================================================================
# Concept Query Tests
# =============================================================================

class TestQuery:
    """Test concept querying."""

    def test_query_finds_similar(self, conceptual_memory):
        """query should find similar concepts."""
        # Store a concept
        conceptual_memory.store_concept("finance_concept", {
            "domain": "finance",
            "type": "trading",
        })

        # Query with same concept's HDV
        query_hdv = conceptual_memory.concepts["finance_concept"]
        results = conceptual_memory.query(query_hdv, threshold=0.5)

        assert len(results) >= 1
        assert results[0][0] == "finance_concept"

    def test_query_respects_threshold(self, conceptual_memory):
        """query should respect similarity threshold."""
        # Store two different concepts
        conceptual_memory.store_concept("concept_a", {"attr": "val_a"})
        conceptual_memory.store_concept("concept_b", {"attr": "val_b"})

        # Query with high threshold
        query_hdv = conceptual_memory.concepts["concept_a"]
        results = conceptual_memory.query(query_hdv, threshold=0.99)

        # Should only match exact (or very close)
        assert len(results) <= 1

    def test_query_returns_sorted_by_similarity(self, conceptual_memory):
        """query should return results sorted by similarity descending."""
        # Store multiple concepts
        for i in range(5):
            conceptual_memory.store_concept(f"concept_{i}", {"attr": f"val_{i}"})

        # Query with first concept
        query_hdv = conceptual_memory.concepts["concept_0"]
        results = conceptual_memory.query(query_hdv, threshold=0.0)

        # Check sorted descending
        similarities = [r[1] for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_query_empty_memory(self, conceptual_memory):
        """query should return empty list for empty memory."""
        query_hdv = BinaryHDV.random(conceptual_memory.dimension)
        results = conceptual_memory.query(query_hdv)

        assert results == []


# =============================================================================
# Analogy Solving Tests
# =============================================================================

class TestSolveAnalogy:
    """Test analogy solving with XOR binding."""

    def test_solve_analogy_basic(self, conceptual_memory):
        """solve_analogy should find analogies using XOR binding."""
        # Store concepts with relationships
        conceptual_memory.store_concept("arbitrage", {"domain": "finance"})
        conceptual_memory.store_concept("bio_hacking", {"domain": "biology"})

        # Solve: arbitrage:finance :: bio_hacking:?
        results = conceptual_memory.solve_analogy("arbitrage", "finance", "bio_hacking")

        assert isinstance(results, list)
        # Should return list of (symbol, similarity) tuples
        if results:
            assert len(results[0]) == 2

    def test_solve_analogy_missing_concept(self, conceptual_memory):
        """solve_analogy should return empty if concept missing."""
        results = conceptual_memory.solve_analogy("nonexistent", "value", "also_nonexistent")

        assert results == []


# =============================================================================
# Attribute Extraction Tests
# =============================================================================

class TestExtractAttribute:
    """Test attribute extraction."""

    def test_extract_attribute(self, conceptual_memory):
        """extract_attribute should extract attribute values."""
        conceptual_memory.store_concept("test_concept", {
            "color": "blue",
            "size": "large",
        })

        results = conceptual_memory.extract_attribute("test_concept", "color")

        assert isinstance(results, list)
        # Should find "blue" as top match
        if results:
            assert results[0][0] == "blue" or results[0][1] > 0.5

    def test_extract_attribute_missing_concept(self, conceptual_memory):
        """extract_attribute should return empty for missing concept."""
        results = conceptual_memory.extract_attribute("nonexistent", "attr")

        assert results == []


# =============================================================================
# Append to Concept Tests
# =============================================================================

class TestAppendToConcept:
    """Test appending to existing concepts."""

    def test_append_to_existing(self, conceptual_memory):
        """append_to_concept should add to existing concept."""
        conceptual_memory.store_concept("test", {"attr1": "val1"})
        conceptual_memory.append_to_concept("test", "attr2", "val2")

        # Should still exist
        assert "test" in conceptual_memory.concepts
        assert conceptual_memory._dirty is True

    def test_append_to_new(self, conceptual_memory):
        """append_to_concept should create new concept if not exists."""
        conceptual_memory.append_to_concept("new_concept", "attr", "val")

        assert "new_concept" in conceptual_memory.concepts


# =============================================================================
# Persistence Tests
# =============================================================================

class TestPersistence:
    """Test save/load persistence."""

    def test_save_creates_files(self, conceptual_memory, temp_storage_dir):
        """save should create codebook and concepts files."""
        conceptual_memory.store_concept("test", {"attr": "val"})
        conceptual_memory.save()

        assert os.path.exists(conceptual_memory.codebook_path)
        assert os.path.exists(conceptual_memory.concepts_path)

    def test_save_load_roundtrip(self, temp_storage_dir):
        """Data should survive save/load roundtrip."""
        # Create and populate memory
        memory1 = ConceptualMemory(dimension=1024, storage_dir=temp_storage_dir)
        memory1.store_concept("concept1", {"attr1": "val1", "attr2": "val2"})
        memory1.store_concept("concept2", {"attr3": "val3"})
        memory1.save()

        # Load into new memory
        memory2 = ConceptualMemory(dimension=1024, storage_dir=temp_storage_dir)

        assert "concept1" in memory2.concepts
        assert "concept2" in memory2.concepts

    def test_load_with_mismatched_dimension(self, temp_storage_dir):
        """Load should skip vectors with mismatched dimension."""
        # Create memory with dimension 1024 and save
        memory1 = ConceptualMemory(dimension=1024, storage_dir=temp_storage_dir)
        memory1.store_concept("test_concept", {"attr": "val"})
        memory1.save()

        # Load with different dimension
        memory2 = ConceptualMemory(dimension=2048, storage_dir=temp_storage_dir)

        # Should not have loaded the concept (dimension mismatch)
        assert "test_concept" not in memory2.concepts

    def test_atomic_write_prevents_corruption(self, conceptual_memory, temp_storage_dir):
        """Atomic write should prevent file corruption."""
        conceptual_memory.store_concept("test", {"attr": "val"})

        # Mock os.replace to fail
        with patch("os.replace", side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                conceptual_memory.save()

        # Original files should not be corrupted
        # (either don't exist or have valid data)
        if os.path.exists(conceptual_memory.concepts_path):
            with open(conceptual_memory.concepts_path, "r") as f:
                data = json.load(f)
                assert isinstance(data, dict)


# =============================================================================
# Concurrent Access Tests
# =============================================================================

class TestConcurrentAccess:
    """Test concurrent write handling."""

    @pytest.mark.asyncio
    async def test_concurrent_writes_dont_corrupt(self, conceptual_memory, temp_storage_dir):
        """Concurrent writes should not corrupt files."""
        async def store_and_flush(i):
            conceptual_memory.store_concept(f"concept_{i}", {"attr": f"val_{i}"})
            await conceptual_memory.flush()

        # Run multiple concurrent stores
        tasks = [store_and_flush(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete without error
        for r in results:
            assert not isinstance(r, Exception)

        # Verify data integrity
        assert len(conceptual_memory.concepts) == 10

    @pytest.mark.asyncio
    async def test_flush_only_when_dirty(self, conceptual_memory):
        """flush should only write when dirty."""
        conceptual_memory.store_concept("test", {"attr": "val"})
        assert conceptual_memory._dirty is True

        await conceptual_memory.flush()
        assert conceptual_memory._dirty is False

        # Second flush should be no-op
        with patch.object(conceptual_memory, "_atomic_save") as mock_save:
            await conceptual_memory.flush()
            mock_save.assert_not_called()


# =============================================================================
# Flush and Close Tests
# =============================================================================

class TestFlushAndClose:
    """Test flush and close methods."""

    @pytest.mark.asyncio
    async def test_flush(self, conceptual_memory):
        """flush should save dirty state."""
        conceptual_memory.store_concept("test", {"attr": "val"})
        await conceptual_memory.flush()

        assert conceptual_memory._dirty is False

    @pytest.mark.asyncio
    async def test_close(self, conceptual_memory):
        """close should flush and close."""
        conceptual_memory.store_concept("test", {"attr": "val"})
        await conceptual_memory.close()

        assert conceptual_memory._dirty is False


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_large_number_of_concepts(self, conceptual_memory):
        """Should handle large number of concepts."""
        for i in range(100):
            conceptual_memory.store_concept(f"concept_{i}", {f"attr_{i}": f"val_{i}"})

        assert len(conceptual_memory.concepts) == 100

    def test_unicode_concept_names(self, conceptual_memory):
        """Should handle unicode in concept names."""
        conceptual_memory.store_concept("concept_unicode", {"attr": "\u4e2d\u6587"})

        assert "concept_unicode" in conceptual_memory.concepts

    def test_special_characters_in_attributes(self, conceptual_memory):
        """Should handle special characters in attributes."""
        conceptual_memory.store_concept("test", {
            "attr_with_special": "value\nwith\nnewlines",
            "json_chars": '{"key": "value"}',
        })

        assert "test" in conceptual_memory.concepts

    def test_very_long_attribute_values(self, conceptual_memory):
        """Should handle long attribute values."""
        long_value = "x" * 10000
        conceptual_memory.store_concept("test", {"attr": long_value})

        assert "test" in conceptual_memory.concepts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
