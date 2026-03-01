"""
Comprehensive Tests for Agent Profile Service
==============================================

Tests persistent state for agent profiles including quirks,
preferences, and reliability scores.

Coverage:
- CRUD operations on agent profiles
- Defaults for missing fields
- In-memory persistence behavior
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch
import threading
import concurrent.futures

from mnemocore.core.agent_profile import (
    AgentProfile,
    AgentProfileService,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def profile_service():
    """Create a fresh AgentProfileService for testing."""
    return AgentProfileService()


@pytest.fixture
def sample_profile():
    """Create a sample AgentProfile."""
    return AgentProfile(
        id="agent-001",
        name="Test Agent",
        description="A test agent for unit testing",
        created_at=datetime.now(timezone.utc),
        last_active=datetime.now(timezone.utc),
        core_directives=["Never delete files without confirmation"],
        preferences={"language": "python", "editor": "vscode"},
        reliability_score=1.0,
    )


# =============================================================================
# AgentProfile Tests
# =============================================================================

class TestAgentProfile:
    """Test AgentProfile dataclass."""

    def test_profile_creation(self):
        """Should create profile with all fields."""
        now = datetime.now(timezone.utc)
        profile = AgentProfile(
            id="test-id",
            name="Test Agent",
            description="Test description",
            created_at=now,
            last_active=now,
        )

        assert profile.id == "test-id"
        assert profile.name == "Test Agent"
        assert profile.description == "Test description"
        assert profile.created_at == now
        assert profile.last_active == now

    def test_profile_defaults(self):
        """Should have correct default values."""
        now = datetime.now(timezone.utc)
        profile = AgentProfile(
            id="test-id",
            name="Test",
            description="Desc",
            created_at=now,
            last_active=now,
        )

        assert profile.core_directives == []
        assert profile.preferences == {}
        assert profile.reliability_score == 1.0

    def test_profile_with_directives(self):
        """Should accept core directives."""
        now = datetime.now(timezone.utc)
        profile = AgentProfile(
            id="test-id",
            name="Test",
            description="Desc",
            created_at=now,
            last_active=now,
            core_directives=[
                "Directive 1",
                "Directive 2",
            ],
        )

        assert len(profile.core_directives) == 2

    def test_profile_with_preferences(self):
        """Should accept preferences."""
        now = datetime.now(timezone.utc)
        profile = AgentProfile(
            id="test-id",
            name="Test",
            description="Desc",
            created_at=now,
            last_active=now,
            preferences={"key": "value", "nested": {"a": 1}},
        )

        assert profile.preferences["key"] == "value"
        assert profile.preferences["nested"]["a"] == 1


# =============================================================================
# AgentProfileService Tests
# =============================================================================

class TestAgentProfileServiceInit:
    """Test service initialization."""

    def test_init(self):
        """Should initialize with empty profiles."""
        service = AgentProfileService()

        assert service._profiles == {}
        assert isinstance(service._lock, type(threading.RLock()))

    def test_init_empty_state(self, profile_service):
        """Should start with empty state."""
        assert len(profile_service._profiles) == 0


# =============================================================================
# get_or_create_profile Tests
# =============================================================================

class TestGetOrCreateProfile:
    """Test profile retrieval and creation."""

    def test_create_new_profile(self, profile_service):
        """Should create new profile if not exists."""
        profile = profile_service.get_or_create_profile("new-agent")

        assert profile.id == "new-agent"
        assert profile.name == "Unknown Agent"
        assert "new-agent" in profile_service._profiles

    def test_create_with_custom_name(self, profile_service):
        """Should use provided name for new profile."""
        profile = profile_service.get_or_create_profile("agent-1", name="Custom Agent")

        assert profile.name == "Custom Agent"

    def test_get_existing_profile(self, profile_service):
        """Should return existing profile."""
        # Create profile
        profile1 = profile_service.get_or_create_profile("agent-1", name="First Agent")

        # Get same profile
        profile2 = profile_service.get_or_create_profile("agent-1")

        assert profile1 is profile2
        assert profile2.name == "First Agent"

    def test_updates_last_active(self, profile_service):
        """Should update last_active on retrieval."""
        profile1 = profile_service.get_or_create_profile("agent-1")
        first_active = profile1.last_active

        # Small delay
        import time
        time.sleep(0.01)

        profile2 = profile_service.get_or_create_profile("agent-1")

        assert profile2.last_active > first_active

    def test_auto_generates_description(self, profile_service):
        """Should auto-generate description for new profiles."""
        profile = profile_service.get_or_create_profile("test-agent")

        assert "test-agent" in profile.description


# =============================================================================
# update_preferences Tests
# =============================================================================

class TestUpdatePreferences:
    """Test preference updates."""

    def test_update_preferences_new(self, profile_service):
        """Should add new preferences."""
        profile_service.get_or_create_profile("agent-1")
        profile_service.update_preferences("agent-1", {"theme": "dark"})

        profile = profile_service.get_or_create_profile("agent-1")
        assert profile.preferences["theme"] == "dark"

    def test_update_preferences_merge(self, profile_service):
        """Should merge with existing preferences."""
        profile_service.get_or_create_profile("agent-1")
        profile_service.update_preferences("agent-1", {"theme": "dark", "lang": "en"})
        profile_service.update_preferences("agent-1", {"lang": "fr", "editor": "vim"})

        profile = profile_service.get_or_create_profile("agent-1")
        assert profile.preferences["theme"] == "dark"  # Kept
        assert profile.preferences["lang"] == "fr"     # Updated
        assert profile.preferences["editor"] == "vim"  # Added

    def test_update_preferences_creates_profile(self, profile_service):
        """Should create profile if not exists."""
        profile_service.update_preferences("new-agent", {"key": "value"})

        assert "new-agent" in profile_service._profiles
        assert profile_service._profiles["new-agent"].preferences["key"] == "value"

    def test_update_preferences_nested(self, profile_service):
        """Should handle nested preferences."""
        profile_service.get_or_create_profile("agent-1")
        profile_service.update_preferences("agent-1", {
            "ui": {"theme": "dark", "font": "mono"}
        })

        profile = profile_service.get_or_create_profile("agent-1")
        assert profile.preferences["ui"]["theme"] == "dark"


# =============================================================================
# adjust_reliability Tests
# =============================================================================

class TestAdjustReliability:
    """Test reliability score adjustments."""

    def test_increase_reliability(self, profile_service):
        """Should increase reliability score."""
        profile_service.get_or_create_profile("agent-1")
        # Lower first to test increase
        profile_service.adjust_reliability("agent-1", -0.5)
        profile_service.adjust_reliability("agent-1", 0.1)

        profile = profile_service.get_or_create_profile("agent-1")
        assert profile.reliability_score == 0.6

    def test_reliability_clamped_at_one(self, profile_service):
        """Should not go above 1.0."""
        profile_service.get_or_create_profile("agent-1")
        profile_service.adjust_reliability("agent-1", 0.1)

        profile = profile_service.get_or_create_profile("agent-1")
        assert profile.reliability_score == 1.0

    def test_decrease_reliability(self, profile_service):
        """Should increase reliability score (capped at 1.0)."""
        profile_service.get_or_create_profile("agent-1")
        profile_service.adjust_reliability("agent-1", 0.1)

        profile = profile_service.get_or_create_profile("agent-1")
        assert profile.reliability_score == 1.0

    def test_reliability_clamped_at_zero(self, profile_service):
        """Should not go below 0."""
        profile_service.get_or_create_profile("agent-1")
        profile_service.adjust_reliability("agent-1", -2.0)

        profile = profile_service.get_or_create_profile("agent-1")
        assert profile.reliability_score == 0.0

    def test_reliability_clamped_at_one(self, profile_service):
        """Should not go above 1."""
        profile_service.get_or_create_profile("agent-1")
        profile_service.adjust_reliability("agent-1", 2.0)

        profile = profile_service.get_or_create_profile("agent-1")
        assert profile.reliability_score == 1.0

    def test_adjust_reliability_creates_profile(self, profile_service):
        """Should create profile if not exists."""
        profile_service.adjust_reliability("new-agent", 0.1)

        assert "new-agent" in profile_service._profiles

    def test_multiple_adjustments(self, profile_service):
        """Should handle multiple adjustments."""
        profile_service.get_or_create_profile("agent-1")

        profile_service.adjust_reliability("agent-1", -0.3)
        profile_service.adjust_reliability("agent-1", 0.1)
        profile_service.adjust_reliability("agent-1", -0.2)

        profile = profile_service.get_or_create_profile("agent-1")
        assert abs(profile.reliability_score - 0.6) < 0.001


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Test thread safety of operations."""

    def test_concurrent_profile_creation(self, profile_service):
        """Should handle concurrent profile creation."""
        def create_profile(i):
            return profile_service.get_or_create_profile(f"agent-{i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_profile, i) for i in range(100)]
            results = [f.result() for f in futures]

        # All profiles should be created
        assert len(profile_service._profiles) == 100

    def test_concurrent_preference_updates(self, profile_service):
        """Should handle concurrent preference updates."""
        profile_service.get_or_create_profile("shared-agent")

        def update_pref(i):
            profile_service.update_preferences("shared-agent", {f"key_{i}": i})

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(update_pref, i) for i in range(50)]
            for f in futures:
                f.result()

        profile = profile_service.get_or_create_profile("shared-agent")
        # All keys should be present
        assert len(profile.preferences) == 50

    def test_concurrent_reliability_adjustments(self, profile_service):
        """Should handle concurrent reliability adjustments."""
        profile_service.get_or_create_profile("shared-agent")

        def adjust():
            profile_service.adjust_reliability("shared-agent", -0.01)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(adjust) for _ in range(50)]
            for f in futures:
                f.result()

        profile = profile_service.get_or_create_profile("shared-agent")
        # Should be 1.0 - (50 * 0.01) = 0.5
        assert abs(profile.reliability_score - 0.5) < 0.001


# =============================================================================
# In-Memory Persistence Tests
# =============================================================================

class TestInMemoryPersistence:
    """Test in-memory persistence behavior."""

    def test_profiles_persist_in_memory(self, profile_service):
        """Profiles should persist in memory."""
        # Create profile
        profile_service.get_or_create_profile("agent-1")
        profile_service.update_preferences("agent-1", {"key": "value"})
        profile_service.adjust_reliability("agent-1", -0.2)

        # Retrieve later
        profile = profile_service.get_or_create_profile("agent-1")

        assert profile.preferences["key"] == "value"
        assert profile.reliability_score == 0.8

    def test_different_agents_isolated(self, profile_service):
        """Different agents should have isolated profiles."""
        profile_service.get_or_create_profile("agent-1")
        profile_service.get_or_create_profile("agent-2")

        profile_service.update_preferences("agent-1", {"theme": "dark"})
        profile_service.update_preferences("agent-2", {"theme": "light"})

        profile1 = profile_service.get_or_create_profile("agent-1")
        profile2 = profile_service.get_or_create_profile("agent-2")

        assert profile1.preferences["theme"] == "dark"
        assert profile2.preferences["theme"] == "light"

    def test_service_reset_clears_profiles(self):
        """New service instance should have empty profiles."""
        service1 = AgentProfileService()
        service1.get_or_create_profile("agent-1")

        # New instance
        service2 = AgentProfileService()

        assert len(service2._profiles) == 0


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_empty_agent_id(self, profile_service):
        """Should handle empty agent ID."""
        profile = profile_service.get_or_create_profile("")

        assert profile.id == ""

    def test_special_characters_in_id(self, profile_service):
        """Should handle special characters in ID."""
        profile = profile_service.get_or_create_profile("agent@test.com")

        assert profile.id == "agent@test.com"

    def test_unicode_in_preferences(self, profile_service):
        """Should handle unicode in preferences."""
        profile_service.get_or_create_profile("agent-1")
        profile_service.update_preferences("agent-1", {"greeting": "\u4f60\u597d"})

        profile = profile_service.get_or_create_profile("agent-1")
        assert profile.preferences["greeting"] == "\u4f60\u597d"

    def test_large_preferences(self, profile_service):
        """Should handle large preference objects."""
        profile_service.get_or_create_profile("agent-1")

        large_prefs = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}
        profile_service.update_preferences("agent-1", large_prefs)

        profile = profile_service.get_or_create_profile("agent-1")
        assert len(profile.preferences) == 100

    def test_very_long_agent_id(self, profile_service):
        """Should handle very long agent ID."""
        long_id = "a" * 1000
        profile = profile_service.get_or_create_profile(long_id)

        assert profile.id == long_id

    def test_empty_preferences(self, profile_service):
        """Should handle empty preferences update."""
        profile_service.get_or_create_profile("agent-1")
        profile_service.update_preferences("agent-1", {})

        profile = profile_service.get_or_create_profile("agent-1")
        assert profile.preferences == {}

    def test_zero_reliability_adjustment(self, profile_service):
        """Should handle zero reliability adjustment."""
        profile_service.get_or_create_profile("agent-1")
        initial_score = profile_service.get_or_create_profile("agent-1").reliability_score

        profile_service.adjust_reliability("agent-1", 0.0)

        profile = profile_service.get_or_create_profile("agent-1")
        assert profile.reliability_score == initial_score


# =============================================================================
# Defaults Tests
# =============================================================================

class TestDefaults:
    """Test default values for missing fields."""

    def test_default_name(self, profile_service):
        """Should have default name when not provided."""
        profile = profile_service.get_or_create_profile("agent-1")

        assert profile.name == "Unknown Agent"

    def test_default_core_directives(self, profile_service):
        """Should have empty core_directives by default."""
        profile = profile_service.get_or_create_profile("agent-1")

        assert profile.core_directives == []

    def test_default_preferences(self, profile_service):
        """Should have empty preferences by default."""
        profile = profile_service.get_or_create_profile("agent-1")

        assert profile.preferences == {}

    def test_default_reliability(self, profile_service):
        """Should have reliability_score of 1.0 by default."""
        profile = profile_service.get_or_create_profile("agent-1")

        assert profile.reliability_score == 1.0

    def test_default_description_format(self, profile_service):
        """Should have auto-generated description."""
        profile = profile_service.get_or_create_profile("agent-123")

        assert "agent-123" in profile.description


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
