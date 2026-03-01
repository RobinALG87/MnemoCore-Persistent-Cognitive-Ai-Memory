"""
EventBus - Internal Publish-Subscribe for Memory Events
========================================================

A high-performance async event bus for internal MnemoCore communication.

Features:
    - Type-safe event publishing and subscription
    - Wildcard pattern matching for event types
    - Event filtering with custom predicates
    - Async handler execution with error isolation
    - Delivery tracking and monitoring
    - Graceful shutdown with pending event flushing

Event Type Hierarchy:
    memory.*
        - memory.created
        - memory.accessed
        - memory.deleted
        - memory.consolidated
    consolidation.*
        - consolidation.started
        - consolidation.completed
    contradiction.*
        - contradiction.detected
        - contradiction.resolved
    dream.*
        - dream.started
        - dream.completed
        - dream.failed
    synapse.*
        - synapse.formed
        - synapse.fired
    gap.*
        - gap.detected
        - gap.filled

Example:
    ```python
    bus = EventBus()

    # Subscribe to specific event
    async def on_memory_created(event):
        print(f"New memory: {event.data['memory_id']}")

    bus.subscribe("memory.created", on_memory_created)

    # Subscribe with wildcard
    bus.subscribe("dream.*", on_dream_event)

    # Subscribe with filter
    bus.subscribe(
        "memory.created",
        on_important_memory,
        filter=lambda e: e.data.get("importance", 0) > 0.8
    )

    # Publish event
    await bus.publish("memory.created", {
        "memory_id": "abc123",
        "content": "...",
        "importance": 0.9
    })
    ```
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatch
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)
from loguru import logger

if TYPE_CHECKING:
    from .webhook_manager import WebhookManager


# =============================================================================
# Event Data Structure
# =============================================================================

@dataclass
class Event:
    """
    Represents a single event in the system.

    Attributes:
        id: Unique event identifier
        type: Event type (e.g., "memory.created")
        data: Event payload (event-specific data)
        metadata: Optional metadata (timestamp, source, correlation_id)
        published_at: When the event was published
    """
    id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex}")
    type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    published_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data,
            "metadata": self.metadata,
            "published_at": self.published_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary representation."""
        published_at = data.get("published_at")
        if isinstance(published_at, str):
            published_at = datetime.fromisoformat(published_at)
        elif published_at is None:
            published_at = datetime.now(timezone.utc)

        return cls(
            id=data.get("id", f"evt_{uuid.uuid4().hex}"),
            type=data.get("type", ""),
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
            published_at=published_at,
        )


# =============================================================================
# Event Handler Types
# =============================================================================

EventHandler = Callable[[Event], Awaitable[None]]
EventFilter = Callable[[Event], bool]


# =============================================================================
# Subscription Information
# =============================================================================

@dataclass
class Subscription:
    """Represents a single event subscription."""
    id: str
    event_pattern: str
    handler: EventHandler
    filter: Optional[EventFilter] = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    delivery_count: int = 0
    error_count: int = 0
    last_delivery_at: Optional[datetime] = None
    enabled: bool = True


# =============================================================================
# EventBus Implementation
# =============================================================================

class EventBus:
    """
    Async publish-subscribe event bus for MnemoCore.

    The EventBus manages event subscriptions and delivery, supporting:
    - Pattern-based subscriptions (e.g., "memory.*")
    - Per-event filtering
    - Isolated error handling per handler
    - Delivery metrics and monitoring

    Thread Safety:
        The EventBus is designed for use within a single event loop.
        For multi-process scenarios, use Redis pub/sub or similar.

    Example:
        ```python
        bus = EventBus()

        # Simple subscription
        bus.subscribe("memory.created", my_handler)

        # Wildcard subscription
        bus.subscribe("dream.*", dream_handler)

        # Filtered subscription
        bus.subscribe(
            "memory.created",
            important_handler,
            filter=lambda e: e.data.get("importance", 0) > 0.8
        )

        # Publish
        await bus.publish("memory.created", {"memory_id": "..."})

        # Shutdown
        await bus.shutdown()
        ```
    """

    def __init__(
        self,
        webhook_manager: Optional["WebhookManager"] = None,
        max_queue_size: int = 10000,
        delivery_timeout: float = 30.0,
    ):
        """
        Initialize the EventBus.

        Args:
            webhook_manager: Optional webhook manager for external delivery
            max_queue_size: Maximum pending events before backpressure
            delivery_timeout: Max seconds to wait for handler completion
        """
        self._webhook_manager = webhook_manager

        # Subscription storage: {event_pattern: {subscription_id: Subscription}}
        self._subscriptions: Dict[str, Dict[str, Subscription]] = defaultdict(dict)

        # Handler queue for async processing
        self._queue: asyncio.Queue[Optional[Event]] = asyncio.Queue(maxsize=max_queue_size)
        self._delivery_timeout = delivery_timeout

        # Background task management
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()

        # Event history for debugging (circular buffer using deque for O(1) operations)
        self._history_size = 1000
        self._history: deque = deque(maxlen=self._history_size)
        self._history_lock = asyncio.Lock()

        # Metrics
        self._events_published = 0
        self._events_delivered = 0
        self._events_failed = 0

    # ======================================================================
    # Subscription Management
    # ======================================================================

    def subscribe(
        self,
        event_pattern: str,
        handler: EventHandler,
        event_filter: Optional[EventFilter] = None,
        enabled: bool = True,
    ) -> str:
        """
        Subscribe to events matching a pattern.

        Args:
            event_pattern: Event type pattern (e.g., "memory.created" or "memory.*")
            handler: Async function to call when matching events are published
            event_filter: Optional predicate to filter events before handler
            enabled: Whether subscription is initially enabled

        Returns:
            Subscription ID for later management

        Example:
            ```python
            sub_id = bus.subscribe(
                "memory.created",
                my_handler,
                filter=lambda e: e.data.get("important", False)
            )
            ```
        """
        subscription_id = f"sub_{uuid.uuid4().hex[:12]}"

        subscription = Subscription(
            id=subscription_id,
            event_pattern=event_pattern,
            handler=handler,
            filter=event_filter,
            enabled=enabled,
        )

        self._subscriptions[event_pattern][subscription_id] = subscription

        logger.debug(
            f"[EventBus] Subscribed {subscription_id} to '{event_pattern}' "
            f"(filter={event_filter is not None}, enabled={enabled})"
        )

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe by subscription ID.

        Args:
            subscription_id: ID returned from subscribe()

        Returns:
            True if subscription was found and removed
        """
        for pattern, subs in self._subscriptions.items():
            if subscription_id in subs:
                del subs[subscription_id]
                logger.debug(f"[EventBus] Unsubscribed {subscription_id} from '{pattern}'")
                return True
        return False

    def unsubscribe_pattern(self, event_pattern: str) -> int:
        """
        Remove all subscriptions for a pattern.

        Args:
            event_pattern: Event pattern to unsubscribe

        Returns:
            Number of subscriptions removed
        """
        if event_pattern in self._subscriptions:
            count = len(self._subscriptions[event_pattern])
            del self._subscriptions[event_pattern]
            logger.debug(f"[EventBus] Unsubscribed {count} handlers from '{event_pattern}'")
            return count
        return 0

    def enable_subscription(self, subscription_id: str) -> bool:
        """Enable a subscription by ID."""
        for pattern, subs in self._subscriptions.items():
            if subscription_id in subs:
                subs[subscription_id].enabled = True
                return True
        return False

    def disable_subscription(self, subscription_id: str) -> bool:
        """Disable a subscription by ID."""
        for pattern, subs in self._subscriptions.items():
            if subscription_id in subs:
                subs[subscription_id].enabled = False
                return True
        return False

    def list_subscriptions(
        self,
        event_pattern: Optional[str] = None,
        enabled_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List subscriptions with optional filtering.

        Args:
            event_pattern: Filter by event pattern (None = all)
            enabled_only: Only return enabled subscriptions

        Returns:
            List of subscription info dictionaries
        """
        results = []

        patterns = [event_pattern] if event_pattern else self._subscriptions.keys()

        for pattern in patterns:
            for sub in self._subscriptions.get(pattern, {}).values():
                if enabled_only and not sub.enabled:
                    continue

                results.append({
                    "id": sub.id,
                    "event_pattern": sub.event_pattern,
                    "created_at": sub.created_at.isoformat(),
                    "delivery_count": sub.delivery_count,
                    "error_count": sub.error_count,
                    "last_delivery_at": sub.last_delivery_at.isoformat() if sub.last_delivery_at else None,
                    "enabled": sub.enabled,
                    "has_filter": sub.filter is not None,
                })

        return results

    # ======================================================================
    # Event Publishing
    # ======================================================================

    async def publish(
        self,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """
        Publish an event to all matching subscribers.

        Args:
            event_type: Event type (e.g., "memory.created")
            data: Event payload
            metadata: Optional metadata (source, correlation_id, etc.)

        Returns:
            The published Event object

        Example:
            ```python
            await bus.publish("memory.created", {
                "memory_id": "abc123",
                "content": "Hello world",
                "tier": "hot"
            }, metadata={"source": "api"})
            ```
        """
        event = Event(
            type=event_type,
            data=data,
            metadata=metadata or {},
        )

        await self._publish_event(event)
        return event

    async def publish_batch(
        self,
        events: List[Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]],
    ) -> List[Event]:
        """
        Publish multiple events efficiently.

        Args:
            events: List of (event_type, data, metadata) tuples

        Returns:
            List of published Event objects
        """
        published = []

        for event_type, data, metadata in events:
            event = Event(
                type=event_type,
                data=data,
                metadata=metadata or {},
            )
            published.append(event)

        # Batch publish
        for event in published:
            await self._publish_event(event)

        return published

    async def _publish_event(self, event: Event) -> None:
        """Internal method to queue event for delivery."""
        self._events_published += 1

        # Add to history
        await self._add_to_history(event)

        # Queue for dispatcher
        try:
            await asyncio.wait_for(
                self._queue.put(event),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"[EventBus] Event queue full, dropping event {event.id}")
            self._events_failed += 1

    # ======================================================================
    # Dispatcher Loop
    # ======================================================================

    async def start(self) -> None:
        """Start the event dispatcher background task."""
        if self._running:
            return

        self._running = True
        self._dispatcher_task = asyncio.create_task(self._dispatcher_loop())
        logger.info("[EventBus] Started event dispatcher")

    async def stop(self) -> None:
        """Stop the event dispatcher and flush pending events."""
        if not self._running:
            return

        self._running = False

        # Signal dispatcher to stop
        await self._queue.put(None)

        # Wait for dispatcher to finish
        if self._dispatcher_task:
            await asyncio.wait_for(self._dispatcher_task, timeout=30.0)
            self._dispatcher_task = None

        logger.info("[EventBus] Stopped event dispatcher")

    async def shutdown(self) -> None:
        """Shutdown the event bus completely."""
        await self.stop()

        # Clear all subscriptions
        self._subscriptions.clear()

        logger.info("[EventBus] Shutdown complete")

    async def _dispatcher_loop(self) -> None:
        """
        Main dispatcher loop that processes queued events.

        Events are delivered to matching subscribers in parallel.
        Errors in handlers are isolated and logged.
        """
        logger.debug("[EventBus] Dispatcher loop started")

        while self._running:
            try:
                # Get next event (with timeout for periodic checks)
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )

                # None is the shutdown signal
                if event is None:
                    break

                # Find matching subscriptions
                subscriptions = self._find_matching_subscriptions(event)

                if subscriptions:
                    # Deliver to all handlers concurrently
                    await self._deliver_event(event, subscriptions)

                # Also deliver to webhook manager if configured
                if self._webhook_manager:
                    asyncio.create_task(
                        self._webhook_manager.send_webhooks_for_event(event)
                    )

                self._events_delivered += 1

            except asyncio.TimeoutError:
                # Periodic timeout, continue loop
                continue
            except Exception as e:
                logger.error(f"[EventBus] Dispatcher error: {e}", exc_info=True)

        logger.debug("[EventBus] Dispatcher loop stopped")

    def _find_matching_subscriptions(self, event: Event) -> List[Subscription]:
        """Find all subscriptions matching the event type."""
        matches = []

        for pattern, subs in self._subscriptions.items():
            # Use fnmatch for wildcard matching
            if fnmatch(event.type, pattern):
                for sub in subs.values():
                    if not sub.enabled:
                        continue
                    # Apply event filter if present
                    if sub.filter and not sub.filter(event):
                        continue
                    matches.append(sub)

        return matches

    async def _deliver_event(
        self,
        event: Event,
        subscriptions: List[Subscription],
    ) -> None:
        """
        Deliver event to all matching subscriptions.

        Each handler is executed concurrently with timeout protection.
        Errors are isolated per handler.
        """
        tasks = []

        for sub in subscriptions:
            task = asyncio.create_task(
                self._deliver_to_handler(event, sub)
            )
            tasks.append(task)

        # Wait for all handlers (with individual timeouts)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update subscription metrics
        for sub, result in zip(subscriptions, results):
            if isinstance(result, Exception):
                sub.error_count += 1
            else:
                sub.delivery_count += 1
                sub.last_delivery_at = datetime.now(timezone.utc)

    async def _deliver_to_handler(
        self,
        event: Event,
        subscription: Subscription,
    ) -> None:
        """
        Deliver event to a single handler with timeout protection.

        Exceptions are caught and logged but don't affect other handlers.
        """
        try:
            await asyncio.wait_for(
                subscription.handler(event),
                timeout=self._delivery_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[EventBus] Handler {subscription.id} timeout "
                f"for event {event.id}"
            )
            subscription.error_count += 1
        except Exception as e:
            logger.error(
                f"[EventBus] Handler {subscription.id} error "
                f"for event {event.id}: {e}",
                exc_info=True
            )
            subscription.error_count += 1
            raise

    # ======================================================================
    # Event History
    # ======================================================================

    async def _add_to_history(self, event: Event) -> None:
        """Add event to circular history buffer (deque with maxlen handles overflow automatically)."""
        async with self._history_lock:
            self._history.append(event)

    async def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get event history with optional filtering.

        Args:
            event_type: Filter by event type (None = all)
            limit: Maximum events to return

        Returns:
            List of event dictionaries
        """
        async with self._history_lock:
            history = list(self._history)

        if event_type:
            history = [e for e in history if fnmatch(e.type, event_type)]

        return [
            e.to_dict()
            for e in history[-limit:]
        ]

    # ======================================================================
    # Metrics and Monitoring
    # ======================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get current EventBus metrics."""
        subscription_count = sum(
            len(subs)
            for subs in self._subscriptions.values()
        )

        return {
            "events_published": self._events_published,
            "events_delivered": self._events_delivered,
            "events_failed": self._events_failed,
            "queue_size": self._queue.qsize(),
            "subscription_count": subscription_count,
            "running": self._running,
            "history_size": len(self._history),
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics including per-subscription metrics."""
        stats = {
            "metrics": self.get_metrics(),
            "subscriptions": self.list_subscriptions(),
        }

        return stats


# =============================================================================
# Global Singleton
# =============================================================================

_EVENT_BUS: Optional[EventBus] = None
_EVENT_BUS_LOCK = threading.Lock()


def get_event_bus(webhook_manager: Optional["WebhookManager"] = None) -> EventBus:
    """
    Get or create the global EventBus singleton.

    Thread-safe: Uses threading.Lock to prevent race conditions
    during concurrent first access.

    Args:
        webhook_manager: Optional webhook manager for external delivery

    Returns:
        The shared EventBus instance
    """
    global _EVENT_BUS

    with _EVENT_BUS_LOCK:
        if _EVENT_BUS is None:
            _EVENT_BUS = EventBus(webhook_manager=webhook_manager)
        elif webhook_manager is not None and _EVENT_BUS._webhook_manager is None:
            _EVENT_BUS._webhook_manager = webhook_manager

    return _EVENT_BUS


def reset_event_bus() -> None:
    """Reset the global EventBus singleton (useful for testing)."""
    global _EVENT_BUS
    with _EVENT_BUS_LOCK:
        _EVENT_BUS = None
