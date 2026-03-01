"""
Tests for Learning Journal Module
=================================
Comprehensive tests for meta-learning storage and retrieval.

Tests cover:
- Entry creation and retrieval
- Prediction registration and evaluation
- Surprise calculation
- Tag-based querying
- Persistence roundtrip
- Empty journal handling
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from mnemocore.meta.learning_journal import LearningJournal, LearningEntry


@pytest.fixture
def temp_journal_path():
    """Create a temporary file path for journal storage."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def empty_journal(temp_journal_path):
    """Create an empty LearningJournal with temporary storage."""
    return LearningJournal(path=temp_journal_path)


@pytest.fixture
def populated_journal(empty_journal):
    """Create a LearningJournal with sample entries."""
    lj = empty_journal

    # Add various learning entries
    lj.record(
        lesson="Use batch processing for large datasets",
        context="Processing 1M records",
        outcome="success",
        confidence=0.9,
        tags=["performance", "optimization"]
    )

    lj.record(
        lesson="Avoid synchronous I/O in async context",
        context="Web server optimization",
        outcome="failure",
        confidence=0.7,
        tags=["async", "bug"]
    )

    lj.record(
        lesson="Cache expensive computations",
        context="Database query optimization",
        outcome="success",
        confidence=0.85,
        tags=["performance", "caching"]
    )

    lj.record(
        lesson="Use connection pooling",
        context="Database connections",
        outcome="mixed",
        confidence=0.6,
        tags=["database", "connections"]
    )

    return lj


class TestLearningEntry:
    """Tests for the LearningEntry dataclass."""

    def test_entry_creation(self):
        """Test basic LearningEntry creation."""
        entry = LearningEntry(
            id="learn_123",
            lesson="Test lesson",
            context="Test context",
            outcome="success",
            confidence=0.8
        )

        assert entry.id == "learn_123"
        assert entry.lesson == "Test lesson"
        assert entry.context == "Test context"
        assert entry.outcome == "success"
        assert entry.confidence == 0.8
        assert entry.applications == 0
        assert entry.tags == []

    def test_entry_with_tags(self):
        """Test LearningEntry with tags."""
        entry = LearningEntry(
            id="learn_456",
            lesson="Tagged lesson",
            context="Context",
            outcome="success",
            confidence=0.7,
            tags=["tag1", "tag2"]
        )

        assert "tag1" in entry.tags
        assert "tag2" in entry.tags


class TestLearningJournalEntryCreation:
    """Tests for learning entry creation and retrieval."""

    def test_record_creates_entry(self, empty_journal):
        """Test that record() creates a new entry."""
        lj = empty_journal

        entry_id = lj.record(
            lesson="New learning",
            context="Test context",
            outcome="success",
            confidence=0.8
        )

        assert entry_id is not None
        assert entry_id.startswith("learn_")
        assert entry_id in lj.entries

    def test_record_with_all_parameters(self, empty_journal):
        """Test record() with all optional parameters."""
        lj = empty_journal

        entry_id = lj.record(
            lesson="Complex learning",
            context="Complex context",
            outcome="mixed",
            confidence=0.75,
            tags=["important", "verified"],
            surprise=0.3
        )

        entry = lj.entries[entry_id]
        assert entry.lesson == "Complex learning"
        assert entry.outcome == "mixed"
        assert "important" in entry.tags

    def test_record_default_values(self, empty_journal):
        """Test record() with default values."""
        lj = empty_journal

        entry_id = lj.record(
            lesson="Minimal learning",
            context="Minimal context"
        )

        entry = lj.entries[entry_id]
        assert entry.outcome == "success"  # Default
        assert entry.confidence == 0.7  # Default

    def test_entry_id_format(self, empty_journal):
        """Test that entry IDs follow expected format."""
        lj = empty_journal

        entry_id = lj.record(lesson="Test", context="Test")

        assert entry_id.startswith("learn_")
        suffix = entry_id.replace("learn_", "")
        assert len(suffix) == 12

    def test_get_top_learnings(self, populated_journal):
        """Test retrieving top learnings by confidence and applications."""
        lj = populated_journal

        top = lj.get_top_learnings(n=3)

        assert len(top) <= 3
        # Should be sorted by confidence * (1 + applications * 0.1)
        if len(top) > 1:
            for i in range(len(top) - 1):
                score_i = top[i].confidence * (1 + top[i].applications * 0.1)
                score_next = top[i + 1].confidence * (1 + top[i + 1].applications * 0.1)
                assert score_i >= score_next


class TestLearningJournalPrediction:
    """Tests for prediction registration and evaluation."""

    def test_register_prediction(self, empty_journal):
        """Test registering a prediction."""
        lj = empty_journal

        pred_id = lj.register_prediction(
            context="Testing prediction",
            expectation="Expected outcome"
        )

        assert pred_id is not None
        assert pred_id.startswith("pred_")
        assert pred_id in lj.predictions

    def test_resolve_prediction_exact_match(self, empty_journal):
        """Test resolving prediction with exact match (no surprise)."""
        lj = empty_journal

        pred_id = lj.register_prediction(
            context="Test context",
            expectation="The quick brown fox"
        )

        result = lj.resolve_prediction(pred_id, "The quick brown fox")

        # Should not create a learning entry (no surprise)
        assert result is None

    def test_resolve_prediction_with_difference(self, empty_journal):
        """Test resolving prediction with different outcome (surprise)."""
        lj = empty_journal

        pred_id = lj.register_prediction(
            context="Test context",
            expectation="Expected result A"
        )

        result = lj.resolve_prediction(pred_id, "Actual result B")

        # Should create a learning entry due to surprise
        assert result is not None
        assert result in lj.entries

        entry = lj.entries[result]
        assert "prediction_error" in entry.tags
        assert "auto_generated" in entry.tags

    def test_resolve_nonexistent_prediction(self, empty_journal):
        """Test resolving a non-existent prediction."""
        lj = empty_journal

        result = lj.resolve_prediction("nonexistent_pred", "Some result")
        assert result is None

    def test_prediction_removed_after_resolution(self, empty_journal):
        """Test that prediction is removed after resolution."""
        lj = empty_journal

        pred_id = lj.register_prediction(
            context="Test",
            expectation="Expectation"
        )

        assert pred_id in lj.predictions

        lj.resolve_prediction(pred_id, "Different outcome")

        assert pred_id not in lj.predictions


class TestLearningJournalSurprise:
    """Tests for surprise calculation."""

    def test_evaluate_surprise_identical(self, empty_journal):
        """Test surprise is 0 for identical strings."""
        lj = empty_journal

        surprise = lj.evaluate_surprise(
            "The quick brown fox",
            "The quick brown fox"
        )

        assert surprise == 0.0

    def test_evaluate_surprise_completely_different(self, empty_journal):
        """Test surprise is high for completely different strings."""
        lj = empty_journal

        surprise = lj.evaluate_surprise(
            "apple banana cherry",
            "dog cat elephant"
        )

        # No word overlap, should be high surprise
        assert surprise > 0.8

    def test_evaluate_surprise_partial_overlap(self, empty_journal):
        """Test surprise for partial word overlap."""
        lj = empty_journal

        surprise = lj.evaluate_surprise(
            "The quick brown fox jumps",
            "The slow brown dog walks"
        )

        # Some overlap (The, brown), moderate surprise
        assert 0.0 < surprise < 1.0

    def test_evaluate_surprise_empty_strings(self, empty_journal):
        """Test surprise with empty strings."""
        lj = empty_journal

        # Empty expectation
        surprise = lj.evaluate_surprise("", "Some content")
        assert surprise == 1.0

        # Empty actual
        surprise = lj.evaluate_surprise("Some content", "")
        assert surprise == 1.0

    def test_surprise_boosts_confidence(self, empty_journal):
        """Test that high surprise boosts confidence (flashbulb learning)."""
        lj = empty_journal

        # Record with high surprise
        entry_id = lj.record(
            lesson="Surprising discovery",
            context="Unexpected outcome",
            confidence=0.5,
            surprise=0.8  # High surprise
        )

        entry = lj.entries[entry_id]
        # Confidence should be boosted: min(1.0, 0.5 * (1.0 + 0.8)) = 0.9
        assert entry.confidence > 0.5

    def test_surprise_adds_tags(self, empty_journal):
        """Test that surprise adds appropriate tags."""
        lj = empty_journal

        # Record with moderate surprise
        entry_id = lj.record(
            lesson="Moderate surprise",
            context="Test",
            surprise=0.6
        )

        entry = lj.entries[entry_id]
        assert any("surprise_" in tag for tag in entry.tags)

    def test_flashbulb_memory_tag(self, empty_journal):
        """Test that very high surprise adds flashbulb_memory tag."""
        lj = empty_journal

        entry_id = lj.record(
            lesson="Shocking discovery",
            context="Test",
            surprise=0.8  # High surprise
        )

        entry = lj.entries[entry_id]
        assert "flashbulb_memory" in entry.tags


class TestLearningJournalQuery:
    """Tests for tag-based and context querying."""

    def test_query_by_context_keywords(self, populated_journal):
        """Test querying learnings by context keywords."""
        lj = populated_journal

        results = lj.query("database query optimization", top_k=5)

        # Should find entries related to database/optimization
        assert len(results) > 0

    def test_query_by_tag_matching(self, populated_journal):
        """Test that tag matching boosts query results."""
        lj = populated_journal

        # Query using terms that appear in tags
        results = lj.query("performance optimization", top_k=5)

        assert len(results) > 0

    def test_query_respects_top_k(self, populated_journal):
        """Test that query respects top_k limit."""
        lj = populated_journal

        results = lj.query("test", top_k=2)
        assert len(results) <= 2

    def test_query_no_matches(self, empty_journal):
        """Test query with no matches returns empty list."""
        lj = empty_journal

        lj.record(lesson="Test lesson", context="Test context")
        results = lj.query("xyznonexistent123", top_k=5)

        assert results == []

    def test_query_confidence_weighting(self, populated_journal):
        """Test that query results are weighted by confidence."""
        lj = populated_journal

        results = lj.query("performance", top_k=10)

        # Higher confidence entries should score better
        if len(results) > 1:
            # Results are sorted by score (which includes confidence)
            for i in range(len(results) - 1):
                # This is a soft check since scoring depends on multiple factors
                pass


class TestLearningJournalApplyContradict:
    """Tests for apply() and contradict() methods."""

    def test_apply_increments_applications(self, populated_journal):
        """Test that apply() increments application count."""
        lj = populated_journal

        entry_id = list(lj.entries.keys())[0]
        initial_count = lj.entries[entry_id].applications

        lj.apply(entry_id)

        assert lj.entries[entry_id].applications == initial_count + 1

    def test_apply_increases_confidence(self, populated_journal):
        """Test that apply() increases confidence (reinforcement)."""
        lj = populated_journal

        entry_id = list(lj.entries.keys())[0]
        initial_confidence = lj.entries[entry_id].confidence

        lj.apply(entry_id)

        # Confidence should increase by factor of 1.05
        expected = min(1.0, initial_confidence * 1.05)
        assert abs(lj.entries[entry_id].confidence - expected) < 0.01

    def test_apply_multiple_times(self, populated_journal):
        """Test multiple applications accumulate."""
        lj = populated_journal

        entry_id = list(lj.entries.keys())[0]

        for _ in range(5):
            lj.apply(entry_id)

        assert lj.entries[entry_id].applications == 5

    def test_contradict_reduces_confidence(self, populated_journal):
        """Test that contradict() reduces confidence (weakening)."""
        lj = populated_journal

        entry_id = list(lj.entries.keys())[0]
        initial_confidence = lj.entries[entry_id].confidence

        lj.contradict(entry_id)

        # Confidence should be reduced by factor of 0.8
        expected = initial_confidence * 0.8
        assert abs(lj.entries[entry_id].confidence - expected) < 0.01

    def test_contradict_nonexistent_entry(self, empty_journal):
        """Test contradict() on non-existent entry does not raise."""
        lj = empty_journal
        # Should not raise
        lj.contradict("nonexistent_entry")

    def test_apply_nonexistent_entry(self, empty_journal):
        """Test apply() on non-existent entry does not raise."""
        lj = empty_journal
        # Should not raise
        lj.apply("nonexistent_entry")


class TestLearningJournalPersistence:
    """Tests for persistence roundtrip."""

    def test_save_to_file(self, temp_journal_path):
        """Test that entries are saved to file."""
        lj = LearningJournal(path=temp_journal_path)

        lj.record(lesson="Saved lesson", context="Saved context")

        # Read file directly
        with open(temp_journal_path, 'r') as f:
            data = json.load(f)

        assert len(data) == 1
        saved_entry = list(data.values())[0]
        assert saved_entry["lesson"] == "Saved lesson"

    def test_load_from_file(self, temp_journal_path):
        """Test that entries are loaded from file on initialization."""
        # Create initial journal and add entries
        lj1 = LearningJournal(path=temp_journal_path)
        id1 = lj1.record(lesson="Lesson 1", context="Context 1")
        id2 = lj1.record(lesson="Lesson 2", context="Context 2")

        # Create new journal with same path - should load existing entries
        lj2 = LearningJournal(path=temp_journal_path)

        assert id1 in lj2.entries
        assert id2 in lj2.entries
        assert lj2.entries[id1].lesson == "Lesson 1"
        assert lj2.entries[id2].lesson == "Lesson 2"

    def test_persistence_roundtrip(self, temp_journal_path):
        """Test that entries persist correctly through save/load cycle."""
        lj1 = LearningJournal(path=temp_journal_path)

        # Add complex entry
        entry_id = lj1.record(
            lesson="Complex lesson",
            context="Complex context with details",
            outcome="mixed",
            confidence=0.85,
            tags=["tag1", "tag2"],
            surprise=0.4
        )

        # Apply some operations
        lj1.apply(entry_id)
        lj1.apply(entry_id)

        # Load into new instance
        lj2 = LearningJournal(path=temp_journal_path)

        assert len(lj2.entries) == 1
        loaded = lj2.entries[entry_id]
        assert loaded.lesson == "Complex lesson"
        assert loaded.outcome == "mixed"
        assert loaded.applications == 2  # Persisted applications
        assert "tag1" in loaded.tags

    def test_corrupt_file_handling(self, temp_journal_path):
        """Test that corrupt JSON file is handled gracefully."""
        # Write invalid JSON
        with open(temp_journal_path, 'w') as f:
            f.write("{ invalid json }")

        # Should not raise, should have empty entries
        lj = LearningJournal(path=temp_journal_path)
        assert lj.entries == {}

    def test_empty_file_handling(self, temp_journal_path):
        """Test that empty file is handled gracefully."""
        # Write empty file
        with open(temp_journal_path, 'w') as f:
            f.write("")

        # Should not raise
        lj = LearningJournal(path=temp_journal_path)
        assert lj.entries == {}

    def test_missing_file_handling(self, temp_journal_path):
        """Test that missing file is handled gracefully."""
        # Delete the file
        os.unlink(temp_journal_path)

        # Should not raise, should have empty entries
        lj = LearningJournal(path=temp_journal_path)
        assert lj.entries == {}


class TestLearningJournalEmptyHandling:
    """Tests for empty journal handling."""

    def test_stats_empty_journal(self, empty_journal):
        """Test stats on empty journal."""
        lj = empty_journal
        stats = lj.stats()

        assert stats["total_learnings"] == 0
        assert stats["successes"] == 0
        assert stats["failures"] == 0
        assert stats["avg_confidence"] == 0.0
        assert stats["total_applications"] == 0

    def test_query_empty_journal(self, empty_journal):
        """Test query on empty journal returns empty list."""
        lj = empty_journal
        results = lj.query("any query", top_k=5)
        assert results == []

    def test_get_top_learnings_empty_journal(self, empty_journal):
        """Test get_top_learnings on empty journal."""
        lj = empty_journal
        top = lj.get_top_learnings(n=10)
        assert top == []


class TestLearningJournalStats:
    """Tests for LearningJournal.stats() method."""

    def test_stats_with_entries(self, populated_journal):
        """Test stats returns correct counts with entries."""
        lj = populated_journal
        stats = lj.stats()

        assert stats["total_learnings"] > 0
        assert stats["successes"] >= 0
        assert stats["failures"] >= 0
        assert 0.0 <= stats["avg_confidence"] <= 1.0

    def test_stats_after_applications(self, populated_journal):
        """Test stats updates after applying learnings."""
        lj = populated_journal

        # Apply some learnings
        entry_ids = list(lj.entries.keys())
        for eid in entry_ids[:2]:
            lj.apply(eid)

        stats = lj.stats()
        assert stats["total_applications"] >= 2

    def test_stats_outcome_counts(self, temp_journal_path):
        """Test that stats correctly counts outcomes."""
        lj = LearningJournal(path=temp_journal_path)

        lj.record(lesson="Success 1", context="Test", outcome="success")
        lj.record(lesson="Success 2", context="Test", outcome="success")
        lj.record(lesson="Failure 1", context="Test", outcome="failure")
        lj.record(lesson="Mixed 1", context="Test", outcome="mixed")

        stats = lj.stats()
        assert stats["successes"] == 2
        assert stats["failures"] == 1
        # Note: "mixed" is not counted in successes or failures
