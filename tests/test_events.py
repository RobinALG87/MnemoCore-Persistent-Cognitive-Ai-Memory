"""
Tests for MnemoCore Events & Webhook System
============================================

Tests for EventBus, WebhookManager, event schemas, and integration.
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mnemocore.events import (
    EventBus,
    WebhookManager,
    WebhookConfig,
    WebhookRetryConfig,
    WebhookDelivery,
    WebhookSignature,
    Event,
    validate_event,
    get_schema_for_event_type,
    emit_event,
    emit_memory_created,
    emit_consolidation_completed,
    emit_contradiction_detected,
    emit_dream_started,
    emit_dream_completed,
    emit_dream_failed,
)


# =============================================================================
# EventBus Tests
# =============================================================================

class TestEventBus:
    """Tests for EventBus functionality."""

    @pytest.fixture
    async def event_bus(self):
        """Create a fresh EventBus for each test."""
        bus = EventBus()
        await bus.start()
        yield bus
        await bus.stop()

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, event_bus):
        """Test basic subscribe and publish."""
        received = []

        async def handler(event):
            received.append(event)

        sub_id = event_bus.subscribe("test.event", handler)

        event = await event_bus.publish("test.event", {"key": "value"})

        # Wait for delivery
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].type == "test.event"
        assert received[0].data["key"] == "value"

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, event_bus):
        """Test wildcard pattern matching."""
        received = []

        async def handler(event):
            received.append(event.type)

        event_bus.subscribe("memory.*", handler)

        await event_bus.publish("memory.created", {"id": "1"})
        await event_bus.publish("memory.deleted", {"id": "2"})
        await event_bus.publish("other.event", {"id": "3"})

        await asyncio.sleep(0.1)

        assert len(received) == 2
        assert "memory.created" in received
        assert "memory.deleted" in received

    @pytest.mark.asyncio
    async def test_event_filter(self, event_bus):
        """Test event filtering in subscriptions."""
        received = []

        async def handler(event):
            received.append(event.data["id"])

        # Only receive events with id > 5
        event_bus.subscribe(
            "test.event",
            handler,
            event_filter=lambda e: e.data.get("id", 0) > 5
        )

        await event_bus.publish("test.event", {"id": 3})
        await event_bus.publish("test.event", {"id": 7})
        await event_bus.publish("test.event", {"id": 10})

        await asyncio.sleep(0.1)

        assert len(received) == 2
        assert 7 in received
        assert 10 in received

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""
        received = []

        async def handler(event):
            received.append(event.type)

        sub_id = event_bus.subscribe("test.event", handler)
        event_bus.unsubscribe(sub_id)

        await event_bus.publish("test.event", {})

        await asyncio.sleep(0.1)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_enable_disable_subscription(self, event_bus):
        """Test enabling/disabling subscriptions."""
        received = []

        async def handler(event):
            received.append(event.type)

        sub_id = event_bus.subscribe("test.event", handler)

        await event_bus.publish("test.event", {})
        await asyncio.sleep(0.1)
        assert len(received) == 1

        event_bus.disable_subscription(sub_id)
        await event_bus.publish("test.event", {})
        await asyncio.sleep(0.1)
        assert len(received) == 1  # No new events

        event_bus.enable_subscription(sub_id)
        await event_bus.publish("test.event", {})
        await asyncio.sleep(0.1)
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_list_subscriptions(self, event_bus):
        """Test listing subscriptions."""
        async def handler1(event): pass
        async def handler2(event): pass

        event_bus.subscribe("test.event", handler1)
        event_bus.subscribe("test.event", handler2)
        event_bus.subscribe("other.*", handler1, enabled=False)

        subs = event_bus.list_subscriptions()

        assert len(subs) == 3
        enabled_subs = event_bus.list_subscriptions(enabled_only=True)
        assert len(enabled_subs) == 2

    @pytest.mark.asyncio
    async def test_event_history(self, event_bus):
        """Test event history tracking."""
        await event_bus.publish("event1", {})
        await event_bus.publish("event2", {})
        await event_bus.publish("event1", {})

        await asyncio.sleep(0.1)

        history = await event_bus.get_history(limit=10)
        assert len(history) == 3

        event1_history = await event_bus.get_history(event_type="event1")
        assert len(event1_history) == 2

    @pytest.mark.asyncio
    async def test_metrics(self, event_bus):
        """Test EventBus metrics."""
        metrics = event_bus.get_metrics()

        assert "events_published" in metrics
        assert "events_delivered" in metrics
        assert "subscription_count" in metrics


# =============================================================================
# Event Schema Tests
# =============================================================================

class TestEventSchemas:
    """Tests for event schema validation."""

    def test_memory_created_schema_valid(self):
        """Validate valid memory.created event."""
        is_valid, errors = validate_event("memory.created", {
            "memory_id": "abc123",
            "content": "Hello world",
            "tier": "hot",
            "ltp_strength": 0.8,
        })

        assert is_valid
        assert len(errors) == 0

    def test_memory_created_schema_missing_required(self):
        """Detect missing required fields."""
        is_valid, errors = validate_event("memory.created", {
            "content": "Hello world",
            # Missing memory_id and tier
        })

        assert not is_valid
        assert any("memory_id" in e for e in errors)
        assert any("tier" in e for e in errors)

    def test_memory_created_schema_wrong_type(self):
        """Detect incorrect field types."""
        is_valid, errors = validate_event("memory.created", {
            "memory_id": "abc123",
            "content": "Hello world",
            "tier": "invalid_tier",  # Not in enum
            "ltp_strength": "not_a_number",
        })

        assert not is_valid
        assert any("tier" in e for e in errors)

    def test_consolidation_completed_schema(self):
        """Validate consolidation.completed event."""
        is_valid, errors = validate_event("consolidation.completed", {
            "cycle_id": "cons_123",
            "duration_seconds": 30.5,
            "memories_processed": 100,
            "memories_consolidated": 50,
        })

        assert is_valid
        assert len(errors) == 0

    def test_contradiction_detected_schema(self):
        """Validate contradiction.detected event."""
        is_valid, errors = validate_event("contradiction.detected", {
            "group_id": "cg_123",
            "memory_a_id": "mem_a",
            "memory_b_id": "mem_b",
            "similarity_score": 0.85,
            "llm_confirmed": True,
        })

        assert is_valid
        assert len(errors) == 0

    def test_get_schema_for_event_type(self):
        """Test retrieving schema by event type."""
        schema = get_schema_for_event_type("memory.created")
        assert schema is not None
        assert schema.event_type == "memory.created"

        unknown = get_schema_for_event_type("unknown.event")
        assert unknown is None


# =============================================================================
# Webhook Signature Tests
# =============================================================================

class TestWebhookSignature:
    """Tests for HMAC signature generation and verification."""

    def test_sign_and_verify(self):
        """Test signing and verifying webhooks."""
        secret = "test_secret"
        payload = '{"test": "data"}'

        signature = WebhookSignature.sign(payload, secret)
        is_valid = WebhookSignature.verify(payload, secret, signature)

        assert is_valid

    def test_wrong_secret_fails(self):
        """Test that wrong secret fails verification."""
        payload = '{"test": "data"}'

        signature = WebhookSignature.sign(payload, "correct_secret")
        is_valid = WebhookSignature.verify(payload, "wrong_secret", signature)

        assert not is_valid

    def test_tampered_payload_fails(self):
        """Test that tampered payload fails verification."""
        secret = "test_secret"
        original = '{"test": "data"}'
        tampered = '{"test": "modified"}'

        signature = WebhookSignature.sign(original, secret)
        is_valid = WebhookSignature.verify(tampered, secret, signature)

        assert not is_valid

    def test_expired_timestamp_fails(self):
        """Test that old timestamps are rejected."""
        import time

        secret = "test_secret"
        payload = '{"test": "data"}'

        # Create signature with old timestamp
        old_timestamp = int(time.time()) - 400  # More than 5 minutes ago
        signature = WebhookSignature.sign(payload, secret, timestamp=old_timestamp)

        is_valid = WebhookSignature.verify(payload, secret, signature)

        assert not is_valid


# =============================================================================
# Integration Helper Tests
# =============================================================================

class TestEventIntegration:
    """Tests for event integration helper functions."""

    @pytest.mark.asyncio
    async def test_emit_memory_created(self):
        """Test emit_memory_created helper."""
        bus = EventBus()
        await bus.start()

        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe("memory.created", handler)

        await emit_memory_created(
            event_bus=bus,
            memory_id="mem_123",
            content="Test content",
            tier="hot",
            ltp_strength=0.8,
        )

        await asyncio.sleep(0.1)
        await bus.stop()

        assert len(received) == 1
        assert received[0].data["memory_id"] == "mem_123"
        assert received[0].data["tier"] == "hot"

    @pytest.mark.asyncio
    async def test_emit_consolidation_completed(self):
        """Test emit_consolidation_completed helper."""
        bus = EventBus()
        await bus.start()

        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe("consolidation.completed", handler)

        await emit_consolidation_completed(
            event_bus=bus,
            cycle_id="cons_123",
            duration_seconds=45.0,
            memories_processed=100,
            memories_consolidated=50,
        )

        await asyncio.sleep(0.1)
        await bus.stop()

        assert len(received) == 1
        assert received[0].data["cycle_id"] == "cons_123"

    @pytest.mark.asyncio
    async def test_emit_contradiction_detected(self):
        """Test emit_contradiction_detected helper."""
        bus = EventBus()
        await bus.start()

        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe("contradiction.detected", handler)

        await emit_contradiction_detected(
            event_bus=bus,
            group_id="cg_123",
            memory_a_id="mem_a",
            memory_b_id="mem_b",
            similarity_score=0.85,
            llm_confirmed=True,
        )

        await asyncio.sleep(0.1)
        await bus.stop()

        assert len(received) == 1
        assert received[0].data["group_id"] == "cg_123"

    @pytest.mark.asyncio
    async def test_emit_dream_events(self):
        """Test dream event helpers."""
        bus = EventBus()
        await bus.start()

        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe("dream.*", handler)

        await emit_dream_started(
            event_bus=bus,
            session_id="dream_123",
            trigger="idle",
        )

        await emit_dream_completed(
            event_bus=bus,
            session_id="dream_123",
            duration_seconds=60.0,
            memories_processed=100,
        )

        await asyncio.sleep(0.1)
        await bus.stop()

        assert len(received) == 2
        assert received[0].type == "dream.started"
        assert received[1].type == "dream.completed"

    @pytest.mark.asyncio
    async def test_emit_dream_failed(self):
        """Test emit_dream_failed helper."""
        bus = EventBus()
        await bus.start()

        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe("dream.failed", handler)

        await emit_dream_failed(
            event_bus=bus,
            session_id="dream_123",
            error="Test error",
            stage="clustering",
        )

        await asyncio.sleep(0.1)
        await bus.stop()

        assert len(received) == 1
        assert received[0].data["error"] == "Test error"


# =============================================================================
# Event To/From Dict Tests
# =============================================================================

class TestEventDataclass:
    """Tests for Event dataclass serialization."""

    def test_to_dict(self):
        """Test converting Event to dictionary."""
        event = Event(
            id="evt_123",
            type="test.event",
            data={"key": "value"},
            metadata={"source": "test"},
        )

        result = event.to_dict()

        assert result["id"] == "evt_123"
        assert result["type"] == "test.event"
        assert result["data"]["key"] == "value"
        assert "published_at" in result

    def test_from_dict(self):
        """Test creating Event from dictionary."""
        data = {
            "id": "evt_123",
            "type": "test.event",
            "data": {"key": "value"},
            "metadata": {"source": "test"},
            "published_at": "2024-01-01T00:00:00+00:00",
        }

        event = Event.from_dict(data)

        assert event.id == "evt_123"
        assert event.type == "test.event"
        assert event.data["key"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
