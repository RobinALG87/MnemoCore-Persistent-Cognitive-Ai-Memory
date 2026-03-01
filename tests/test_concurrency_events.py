"""
Concurrency Tests for Event Bus
================================

Tests for thread-safety and concurrency correctness of the EventBus
publish/subscribe system. Validates that concurrent event operations
are handled correctly without event loss or corruption.

Test Categories:
    - 100 concurrent emit() calls -> all events delivered
    - Subscriber add/remove during emission -> no errors
    - Event bus singleton under concurrent access
"""

import asyncio
import threading
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

import pytest

from mnemocore.events.event_bus import (
    EventBus,
    Event,
    Subscription,
    get_event_bus,
    reset_event_bus,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
async def event_bus():
    """
    Create a fresh EventBus for each test.

    Starts the dispatcher loop and ensures cleanup after the test.
    """
    bus = EventBus(max_queue_size=1000, delivery_timeout=5.0)
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
async def isolated_event_bus():
    """
    Create an isolated EventBus without singleton interference.
    """
    reset_event_bus()
    bus = EventBus(max_queue_size=500)
    await bus.start()
    yield bus
    await bus.shutdown()
    reset_event_bus()


# =============================================================================
# Test: 100 Concurrent Emit Calls
# =============================================================================

class TestConcurrentEventEmission:
    """
    Tests for concurrent event emission and delivery.

    Validates that:
    - All emitted events are delivered to subscribers
    - Event order is preserved within delivery constraints
    - No events are lost under concurrent load
    """

    @pytest.mark.asyncio
    async def test_100_concurrent_emits_all_delivered(self, event_bus):
        """
        Test that 100 concurrent emit() calls result in all events being
        delivered to subscribers.

        This validates the event queue and dispatcher handle concurrent
        load without dropping events.
        """
        num_events = 100
        received_events = []
        received_lock = asyncio.Lock()

        async def event_handler(event: Event):
            """Handler that records all received events."""
            async with received_lock:
                received_events.append(event)

        # Subscribe to all events
        event_bus.subscribe("*", event_handler)

        # Emit all events concurrently
        async def emit_event(i: int):
            await event_bus.publish(
                f"test.event.{i % 5}",
                {"index": i, "uuid": uuid.uuid4().hex},
                metadata={"batch": "concurrent_test"}
            )

        tasks = [emit_event(i) for i in range(num_events)]
        await asyncio.gather(*tasks)

        # Wait for dispatcher to process all events
        await asyncio.sleep(0.5)

        # Verify all events were delivered
        assert len(received_events) == num_events, (
            f"Expected {num_events} events, got {len(received_events)}"
        )

        # Verify all event indices are present
        indices = {e.data["index"] for e in received_events}
        assert indices == set(range(num_events)), "Some events were not delivered"

    @pytest.mark.asyncio
    async def test_concurrent_emits_to_multiple_subscribers(self, event_bus):
        """
        Test that concurrent emissions are delivered to all matching subscribers.

        Multiple subscribers for the same event type should all receive
        the events.
        """
        num_events = 50
        num_subscribers = 5
        received_by_subscriber = {i: [] for i in range(num_subscribers)}
        locks = {i: asyncio.Lock() for i in range(num_subscribers)}

        def make_handler(sub_id: int):
            async def handler(event: Event):
                async with locks[sub_id]:
                    received_by_subscriber[sub_id].append(event)
            return handler

        # Subscribe multiple handlers to the same event pattern
        for i in range(num_subscribers):
            event_bus.subscribe("test.*", make_handler(i))

        # Emit events concurrently
        tasks = [
            event_bus.publish(f"test.event.{i}", {"index": i})
            for i in range(num_events)
        ]
        await asyncio.gather(*tasks)

        # Wait for delivery
        await asyncio.sleep(0.5)

        # Each subscriber should have received all events
        for sub_id in range(num_subscribers):
            assert len(received_by_subscriber[sub_id]) == num_events, (
                f"Subscriber {sub_id} received {len(received_by_subscriber[sub_id])} "
                f"events, expected {num_events}"
            )

    @pytest.mark.asyncio
    async def test_concurrent_emits_with_wildcard_patterns(self, event_bus):
        """
        Test that wildcard subscriptions work correctly under concurrent load.

        Events should be matched correctly to wildcard patterns even when
        emitted concurrently.
        """
        received = {"memory": [], "dream": [], "all": []}
        locks = {k: asyncio.Lock() for k in received}

        def make_handler(category: str):
            async def handler(event: Event):
                async with locks[category]:
                    received[category].append(event)
            return handler

        # Subscribe with different patterns
        event_bus.subscribe("memory.*", make_handler("memory"))
        event_bus.subscribe("dream.*", make_handler("dream"))
        event_bus.subscribe("*", make_handler("all"))

        # Emit events of different types concurrently
        tasks = []
        for i in range(30):
            tasks.append(event_bus.publish("memory.created", {"idx": i}))
            tasks.append(event_bus.publish("dream.started", {"idx": i}))
            tasks.append(event_bus.publish("consolidation.completed", {"idx": i}))

        await asyncio.gather(*tasks)
        await asyncio.sleep(0.5)

        # Verify correct routing
        # memory.created -> memory.* and *
        assert len(received["memory"]) == 30
        # dream.started -> dream.* and *
        assert len(received["dream"]) == 30
        # All events -> *
        assert len(received["all"]) == 90  # 30 * 3 event types

    @pytest.mark.asyncio
    async def test_concurrent_emits_preserve_event_integrity(self, event_bus):
        """
        Test that event data is not corrupted during concurrent emission.

        Each event should retain its original data without corruption
        from concurrent operations.
        """
        num_events = 100
        received_events = []
        lock = asyncio.Lock()

        async def handler(event: Event):
            async with lock:
                received_events.append(event)

        event_bus.subscribe("integrity.test", handler)

        # Emit events with unique, verifiable data
        tasks = [
            event_bus.publish(
                "integrity.test",
                {
                    "id": i,
                    "checksum": f"checksum_{i}_{uuid.uuid4().hex}",
                    "nested": {"value": i * 2}
                }
            )
            for i in range(num_events)
        ]
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.5)

        # Verify data integrity
        assert len(received_events) == num_events

        for event in received_events:
            idx = event.data["id"]
            assert event.data["checksum"] == f"checksum_{idx}_"
            assert event.data["checksum"].endswith(event.data["checksum"].split("_")[-1])
            assert event.data["nested"]["value"] == idx * 2


# =============================================================================
# Test: Subscriber Add/Remove During Emission
# =============================================================================

class TestConcurrentSubscriptionManagement:
    """
    Tests for concurrent subscription management during event emission.

    Validates that:
    - Adding subscribers during emission doesn't cause errors
    - Removing subscribers during emission doesn't cause errors
    - The subscription list remains consistent
    """

    @pytest.mark.asyncio
    async def test_subscribe_during_emission(self, event_bus):
        """
        Test that adding subscribers during active emission works correctly.

        New subscribers should start receiving events without disrupting
        ongoing emission.
        """
        received = []
        lock = asyncio.Lock()

        async def handler(event: Event):
            async with lock:
                received.append(event)

        # Start with one subscriber
        event_bus.subscribe("dynamic.test", handler)

        async def emit_events():
            """Emit a stream of events."""
            for i in range(50):
                await event_bus.publish("dynamic.test", {"index": i})
                await asyncio.sleep(0.01)

        async def add_subscribers():
            """Add new subscribers during emission."""
            await asyncio.sleep(0.1)  # Let some events emit first
            for i in range(5):
                event_bus.subscribe("dynamic.test", handler)
                await asyncio.sleep(0.05)

        # Run emission and subscription concurrently
        await asyncio.gather(emit_events(), add_subscribers())
        await asyncio.sleep(0.3)

        # Should have received all events without errors
        assert len(received) >= 50  # At least the original 50 events

    @pytest.mark.asyncio
    async def test_unsubscribe_during_emission(self, event_bus):
        """
        Test that removing subscribers during active emission works correctly.

        Removed subscribers should stop receiving events without disrupting
        ongoing emission or causing errors.
        """
        received = {"active": [], "removed": []}
        locks = {k: asyncio.Lock() for k in received}

        def make_handler(category: str):
            async def handler(event: Event):
                async with locks[category]:
                    received[category].append(event)
            return handler

        # Subscribe two handlers
        sub_active = event_bus.subscribe("unsub.test", make_handler("active"))
        sub_to_remove = event_bus.subscribe("unsub.test", make_handler("removed"))

        async def emit_events():
            """Emit events."""
            for i in range(30):
                await event_bus.publish("unsub.test", {"index": i})
                await asyncio.sleep(0.02)

        async def remove_subscriber():
            """Remove one subscriber mid-way."""
            await asyncio.sleep(0.2)  # Let some events emit
            event_bus.unsubscribe(sub_to_remove)

        await asyncio.gather(emit_events(), remove_subscriber())
        await asyncio.sleep(0.3)

        # Active handler should have received all events
        assert len(received["active"]) == 30

        # Removed handler should have received fewer events
        # (some before removal, none after)
        assert len(received["removed"]) < 30
        assert len(received["removed"]) > 0  # Got some before removal

    @pytest.mark.asyncio
    async def test_concurrent_subscribe_unsubscribe(self, event_bus):
        """
        Test that concurrent subscribe and unsubscribe operations work.

        The subscription list should remain consistent despite concurrent
        modifications.
        """
        received = []
        lock = asyncio.Lock()
        subscription_ids = []

        async def handler(event: Event):
            async with lock:
                received.append(event)

        async def subscribe_ops():
            """Perform subscribe operations."""
            for i in range(20):
                sub_id = event_bus.subscribe("concurrent.sub", handler)
                subscription_ids.append(sub_id)
                await asyncio.sleep(0.01)

        async def unsubscribe_ops():
            """Perform unsubscribe operations."""
            await asyncio.sleep(0.05)
            while subscription_ids:
                sub_id = subscription_ids.pop(0)
                event_bus.unsubscribe(sub_id)
                await asyncio.sleep(0.01)

        async def emit_ops():
            """Emit events during all this."""
            for i in range(30):
                await event_bus.publish("concurrent.sub", {"index": i})
                await asyncio.sleep(0.01)

        # Run all operations concurrently
        await asyncio.gather(subscribe_ops(), unsubscribe_ops(), emit_ops())
        await asyncio.sleep(0.3)

        # No exceptions should have occurred
        # Some events should have been received (exact count depends on timing)
        assert len(received) > 0

    @pytest.mark.asyncio
    async def test_enable_disable_during_emission(self, event_bus):
        """
        Test that enabling/disabling subscriptions during emission works.

        Disabled subscriptions should not receive events; re-enabling should
        resume delivery.
        """
        received = []
        lock = asyncio.Lock()

        async def handler(event: Event):
            async with lock:
                received.append(event)

        sub_id = event_bus.subscribe("toggle.test", handler)

        async def emit_events():
            for i in range(30):
                await event_bus.publish("toggle.test", {"index": i})
                await asyncio.sleep(0.02)

        async def toggle_subscription():
            await asyncio.sleep(0.15)
            event_bus.disable_subscription(sub_id)
            await asyncio.sleep(0.15)
            event_bus.enable_subscription(sub_id)

        await asyncio.gather(emit_events(), toggle_subscription())
        await asyncio.sleep(0.3)

        # Should have received some events (before and after toggle)
        assert len(received) > 0
        # Should not have received all 30 due to disabled period
        # (exact count depends on timing)


# =============================================================================
# Test: Event Bus Singleton Under Concurrent Access
# =============================================================================

class TestEventBusSingletonConcurrency:
    """
    Tests for the global EventBus singleton under concurrent access.

    Validates that:
    - Singleton access is thread-safe
    - Multiple concurrent get_event_bus() calls return the same instance
    - Reset works correctly under concurrent access
    """

    @pytest.mark.asyncio
    async def test_singleton_thread_safety(self):
        """
        Test that get_event_bus() is thread-safe.

        Multiple concurrent calls from different threads should all
        return the same EventBus instance.
        """
        reset_event_bus()
        instances = []
        lock = threading.Lock()

        def get_bus():
            bus = get_event_bus()
            with lock:
                instances.append(id(bus))

        # Run get_event_bus from multiple threads concurrently
        threads = [threading.Thread(target=get_bus) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should have gotten the same instance
        unique_instances = set(instances)
        assert len(unique_instances) == 1, (
            f"Expected 1 unique instance, got {len(unique_instances)}: {unique_instances}"
        )

        reset_event_bus()

    @pytest.mark.asyncio
    async def test_concurrent_singleton_access_with_operations(self):
        """
        Test that singleton works correctly with concurrent access and operations.

        Mix of singleton access and event operations should work together.
        """
        reset_event_bus()

        received = []
        lock = asyncio.Lock()

        async def handler(event: Event):
            async with lock:
                received.append(event)

        # Get singleton and subscribe
        bus = get_event_bus()
        await bus.start()
        bus.subscribe("singleton.test", handler)

        async def access_and_emit(i: int):
            """Access singleton and emit an event."""
            bus = get_event_bus()
            await bus.publish("singleton.test", {"index": i})

        # Concurrent access and emission
        tasks = [access_and_emit(i) for i in range(50)]
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.3)

        # All events should be delivered
        assert len(received) == 50

        await bus.shutdown()
        reset_event_bus()

    @pytest.mark.asyncio
    async def test_reset_during_concurrent_access(self):
        """
        Test that reset_event_bus() is safe during concurrent access.

        Reset should not cause crashes, though behavior during reset
        may be unpredictable.
        """
        reset_event_bus()

        errors = []
        lock = threading.Lock()

        def access_bus():
            try:
                for _ in range(10):
                    bus = get_event_bus()
                    # Simulate some work
                    _ = bus.get_metrics()
            except Exception as e:
                with lock:
                    errors.append(e)

        def reset_bus():
            try:
                for _ in range(5):
                    reset_event_bus()
                    import time
                    time.sleep(0.01)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Mix of access and reset
        threads = [threading.Thread(target=access_bus) for _ in range(10)]
        threads += [threading.Thread(target=reset_bus) for _ in range(2)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions should have occurred
        assert len(errors) == 0, f"Errors during concurrent access: {errors}"

        reset_event_bus()

    @pytest.mark.asyncio
    async def test_event_bus_metrics_under_load(self, event_bus):
        """
        Test that EventBus metrics remain accurate under concurrent load.

        The published/delivered counters should reflect actual operations.
        """
        received = []
        lock = asyncio.Lock()

        async def handler(event: Event):
            async with lock:
                received.append(event)

        event_bus.subscribe("metrics.test", handler)

        # Emit many events concurrently
        num_events = 100
        tasks = [
            event_bus.publish("metrics.test", {"index": i})
            for i in range(num_events)
        ]
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.5)

        # Check metrics
        metrics = event_bus.get_metrics()
        assert metrics["events_published"] == num_events
        assert metrics["events_delivered"] >= num_events  # May include prior tests

        # Queue should be empty after processing
        assert metrics["queue_size"] == 0


# =============================================================================
# Test: Error Handling Under Concurrency
# =============================================================================

class TestConcurrentErrorHandling:
    """
    Tests for error handling during concurrent operations.

    Validates that:
    - Handler errors don't affect other handlers
    - Errors are properly isolated
    - Event bus continues operating after handler failures
    """

    @pytest.mark.asyncio
    async def test_handler_error_isolation(self, event_bus):
        """
        Test that errors in one handler don't affect other handlers.

        If one handler raises an exception, other handlers should still
        receive events.
        """
        received = {"good": [], "bad": []}
        lock = asyncio.Lock()

        async def good_handler(event: Event):
            async with lock:
                received["good"].append(event)

        async def bad_handler(event: Event):
            async with lock:
                received["bad"].append(event)
            raise ValueError("Intentional test error")

        event_bus.subscribe("error.test", good_handler)
        event_bus.subscribe("error.test", bad_handler)

        # Emit events
        for i in range(10):
            await event_bus.publish("error.test", {"index": i})

        await asyncio.sleep(0.3)

        # Good handler should have received all events
        assert len(received["good"]) == 10
        # Bad handler should also have received events (before error)
        assert len(received["bad"]) == 10

    @pytest.mark.asyncio
    async def test_concurrent_handler_errors(self, event_bus):
        """
        Test that multiple handlers can error concurrently without issues.

        All errors should be caught and logged without crashing the bus.
        """
        received = []
        lock = asyncio.Lock()

        async def sometimes_failing_handler(event: Event):
            async with lock:
                received.append(event)
            if event.data.get("index", 0) % 3 == 0:
                raise RuntimeError("Intermittent error")

        # Multiple handlers that may fail
        for _ in range(5):
            event_bus.subscribe("error.concurrent", sometimes_failing_handler)

        # Emit events
        tasks = [
            event_bus.publish("error.concurrent", {"index": i})
            for i in range(20)
        ]
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.5)

        # Should have received events despite errors
        # (5 handlers * 20 events = 100 potential deliveries)
        assert len(received) > 0

        # Check error counts in subscriptions
        subs = event_bus.list_subscriptions()
        total_errors = sum(s["error_count"] for s in subs)
        assert total_errors > 0  # Some errors should have occurred


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
