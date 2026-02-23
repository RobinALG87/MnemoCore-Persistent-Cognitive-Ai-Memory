"""
WebhookManager - External Webhook Delivery with Retry Logic
===========================================================

Manages external webhook subscriptions for MnemoCore events.

Features:
    - Webhook registration with HMAC signature verification
    - Exponential backoff retry logic
    - Per-event-type subscription filtering
    - Delivery status tracking and monitoring
    - Graceful handling of unavailable endpoints
    - Webhook delivery history for debugging

Webhook Configuration:
    Each webhook can be configured to receive specific event types:
    - on_consolidation: consolidation.completed events
    - on_contradiction: contradiction.detected events
    - on_dream_complete: dream.completed events
    - Any event type from the EventBus

Security:
    Webhooks are signed using HMAC-SHA256 with a shared secret.
    Recipients should verify signatures to ensure authenticity.

Example:
    ```python
    manager = WebhookManager()

    # Register webhook
    config = await manager.register_webhook(
        url="https://example.com/mnemocore-webhook",
        events=["memory.created", "consolidation.completed"],
        secret="my_shared_secret",
        headers={"X-Custom-Header": "value"}
    )

    # Webhooks are automatically triggered when events are published
    # via the EventBus integration

    # Check delivery status
    status = await manager.get_webhook_status(config.id)
    ```

Webhook Payload Format:
    ```json
    {
        "id": "evt_abc123",
        "type": "memory.created",
        "data": {
            "memory_id": "...",
            "content": "...",
            "tier": "hot"
        },
        "metadata": {
            "timestamp": "2024-01-01T00:00:00Z"
        },
        "webhook_delivery": {
            "webhook_id": "wh_xyz",
            "attempt": 1,
            "sent_at": "2024-01-01T00:00:00Z"
        }
    }
    ```
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from loguru import logger

import aiohttp

from .event_bus import Event


# =============================================================================
# Webhook Configuration
# =============================================================================

@dataclass
class WebhookConfig:
    """
    Configuration for a single webhook endpoint.

    Attributes:
        id: Unique webhook identifier
        url: Target URL for webhook delivery
        events: List of event types to deliver (empty = all events)
        secret: Shared secret for HMAC signature
        headers: Additional HTTP headers to include
        enabled: Whether webhook is active
        retry_config: Retry behavior configuration
        created_at: When webhook was registered
        updated_at: Last configuration update
    """
    id: str
    url: str
    events: List[str] = field(default_factory=list)
    secret: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    retry_config: "RetryConfig" = field(default_factory=lambda: RetryConfig())
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def matches_event(self, event_type: str) -> bool:
        """Check if this webhook should receive the given event type."""
        if not self.events:
            return True  # Empty list = subscribe to all
        return event_type in self.events

    def to_dict(self, scrub_secret: bool = True) -> Dict[str, Any]:
        """Convert to dictionary, optionally hiding the secret."""
        return {
            "id": self.id,
            "url": self.url,
            "events": self.events,
            "secret": "***" if scrub_secret else self.secret,
            "headers": self.headers,
            "enabled": self.enabled,
            "retry_config": self.retry_config.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebhookConfig":
        """Create from dictionary representation."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        retry_data = data.get("retry_config", {})
        retry_config = RetryConfig(**retry_data) if retry_data else RetryConfig()

        return cls(
            id=data["id"],
            url=data["url"],
            events=data.get("events", []),
            secret=data.get("secret", ""),
            headers=data.get("headers", {}),
            enabled=data.get("enabled", True),
            retry_config=retry_config,
            created_at=created_at or datetime.now(timezone.utc),
            updated_at=updated_at or datetime.now(timezone.utc),
        )


# =============================================================================
# Retry Configuration with Exponential Backoff
# =============================================================================

@dataclass
class RetryConfig:
    """
    Retry behavior for failed webhook deliveries.

    Uses exponential backoff: delay = base_delay * (2 ^ (attempt - 1))

    Attributes:
        max_attempts: Maximum delivery attempts (including first)
        base_delay_seconds: Initial delay before first retry
        max_delay_seconds: Maximum delay between attempts
        timeout_seconds: HTTP request timeout
    """
    max_attempts: int = 5
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 300.0  # 5 minutes
    timeout_seconds: float = 30.0

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given retry attempt.

        Args:
            attempt: Attempt number (1 = first retry)

        Returns:
            Delay in seconds with exponential backoff
        """
        delay = self.base_delay_seconds * (2 ** (attempt - 1))
        return min(delay, self.max_delay_seconds)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_attempts": self.max_attempts,
            "base_delay_seconds": self.base_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "timeout_seconds": self.timeout_seconds,
        }


# =============================================================================
# Webhook Delivery Record
# =============================================================================

@dataclass
class WebhookDelivery:
    """
    Record of a webhook delivery attempt.

    Attributes:
        id: Unique delivery identifier
        webhook_id: ID of the webhook being delivered
        event_id: ID of the event being delivered
        status: Delivery status (pending, success, failed, retrying)
        http_status: HTTP status code if response received
        attempt: Attempt number (1-indexed)
        created_at: When delivery was attempted
        completed_at: When delivery completed (if finished)
        error_message: Error message if failed
        response_body: Response body (truncated if large)
    """
    id: str
    webhook_id: str
    event_id: str
    status: str = "pending"  # pending, success, failed, retrying
    http_status: Optional[int] = None
    attempt: int = 1
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    response_body: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "webhook_id": self.webhook_id,
            "event_id": self.event_id,
            "status": self.status,
            "http_status": self.http_status,
            "attempt": self.attempt,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "response_body": self.response_body,
        }


# =============================================================================
# Webhook Signature (HMAC)
# =============================================================================

class WebhookSignature:
    """
    HMAC-SHA256 signature generation and verification for webhooks.

    The signature is computed as:
        hmac_sha256(secret, timestamp + "." + payload)

    The signature is included in the X-MnemoCore-Signature header:
        X-MnemoCore-Signature: t=<timestamp>,v1=<signature>
    """

    SIGNATURE_HEADER = "X-MnemoCore-Signature"
    TIMESTAMP_HEADER = "X-MnemoCore-Timestamp"
    ALGORITHM = "sha256"
    VERSION_PREFIX = "v1"
    TOLERANCE_SECONDS = 300  # 5 minutes

    @classmethod
    def sign(
        cls,
        payload: str,
        secret: str,
        timestamp: Optional[int] = None,
    ) -> str:
        """
        Generate HMAC signature for a payload.

        Args:
            payload: JSON string payload to sign
            secret: Shared secret key
            timestamp: Unix timestamp (default: current time)

        Returns:
            Signature header value: "t=<timestamp>,v1=<signature>"
        """
        if timestamp is None:
            timestamp = int(time.time())

        # Create message: timestamp.payload
        message = f"{timestamp}.{payload}"

        # Compute HMAC
        signature = hmac.new(
            secret.encode(),
            message.encode(),
            getattr(hashlib, cls.ALGORITHM)
        ).hexdigest()

        return f"t={timestamp},{cls.VERSION_PREFIX}={signature}"

    @classmethod
    def verify(
        cls,
        payload: str,
        secret: str,
        signature_header: str,
        timestamp_header: Optional[str] = None,
    ) -> bool:
        """
        Verify HMAC signature from a webhook.

        Args:
            payload: Received JSON payload
            secret: Shared secret key
            signature_header: Value from X-MnemoCore-Signature header
            timestamp_header: Value from X-MnemoCore-Timestamp header (legacy)

        Returns:
            True if signature is valid and timestamp is fresh
        """
        try:
            # Parse signature header
            timestamp = None
            signature = None

            for part in signature_header.split(","):
                key, value = part.split("=", 1)
                if key == "t":
                    timestamp = int(value)
                elif key.startswith("v"):
                    signature = value

            # Legacy: separate timestamp header
            if timestamp is None and timestamp_header:
                timestamp = int(timestamp_header)

            if timestamp is None or signature is None:
                return False

            # Check timestamp freshness
            now = int(time.time())
            if abs(now - timestamp) > cls.TOLERANCE_SECONDS:
                logger.warning(
                    f"[WebhookSignature] Timestamp too old: {timestamp} vs {now}"
                )
                return False

            # Recompute signature
            expected = cls.sign(payload, secret, timestamp)

            # Constant-time comparison
            return hmac.compare_digest(signature_header, expected)

        except (ValueError, AttributeError) as e:
            logger.warning(f"[WebhookSignature] Verification error: {e}")
            return False


# =============================================================================
# Webhook Manager
# =============================================================================

class WebhookManager:
    """
    Manages webhook registrations and delivery for MnemoCore events.

    Integration with EventBus:
        WebhookManager can be integrated with EventBus to automatically
        deliver published events to registered webhooks.

    Persistence:
        Webhook configurations are persisted to disk for recovery after restart.

    Example:
        ```python
        manager = WebhookManager(persistence_path="./webhooks.json")

        # Register webhook
        config = await manager.register_webhook(
            url="https://example.com/webhook",
            events=["memory.created", "consolidation.completed"],
            secret="my_shared_secret"
        )

        # Get delivery history
        history = await manager.get_delivery_history(
            webhook_id=config.id,
            limit=100
        )
        ```
    """

    def __init__(
        self,
        persistence_path: Optional[str] = None,
        max_history_size: int = 10000,
        http_session: Optional[aiohttp.ClientSession] = None,
    ):
        """
        Initialize WebhookManager.

        Args:
            persistence_path: Path to save webhook configurations
            max_history_size: Maximum delivery records to keep
            http_session: Optional aiohttp session for HTTP requests
        """
        self._persistence_path = Path(persistence_path) if persistence_path else None
        self._max_history_size = max_history_size

        # Webhook storage
        self._webhooks: Dict[str, WebhookConfig] = {}

        # Delivery history: {webhook_id: [WebhookDelivery]}
        self._history: Dict[str, List[WebhookDelivery]] = {}

        # HTTP session
        self._session = http_session

        # Background retry queue
        self._retry_queue: asyncio.Queue[WebhookDelivery] = asyncio.Queue()
        self._retry_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self._deliveries_total = 0
        self._deliveries_success = 0
        self._deliveries_failed = 0

    # ======================================================================
    # Lifecycle
    # ======================================================================

    async def start(self) -> None:
        """Start the webhook manager and retry processor."""
        if self._running:
            return

        self._running = True

        # Load persisted webhooks
        await self._load_webhooks()

        # Start retry processor
        self._retry_task = asyncio.create_task(self._retry_loop())

        logger.info("[WebhookManager] Started")

    async def stop(self) -> None:
        """Stop the webhook manager."""
        if not self._running:
            return

        self._running = False

        # Stop retry processor
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass

        # Close HTTP session if we created it
        if self._session:
            await self._session.close()
            self._session = None

        logger.info("[WebhookManager] Stopped")

    # ======================================================================
    # Webhook Registration
    # ======================================================================

    async def register_webhook(
        self,
        url: str,
        events: Optional[List[str]] = None,
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        enabled: bool = True,
        retry_config: Optional[RetryConfig] = None,
    ) -> WebhookConfig:
        """
        Register a new webhook endpoint.

        Args:
            url: Target URL for webhook delivery
            events: List of event types (empty = all events)
            secret: Shared secret for HMAC (auto-generated if None)
            headers: Additional HTTP headers
            enabled: Whether webhook is initially active
            retry_config: Retry behavior configuration

        Returns:
            The created WebhookConfig
        """
        webhook_id = f"wh_{uuid.uuid4().hex[:12]}"

        if secret is None:
            # Auto-generate secure random secret
            import secrets
            secret = secrets.token_urlsafe(32)

        config = WebhookConfig(
            id=webhook_id,
            url=url,
            events=events or [],
            secret=secret,
            headers=headers or {},
            enabled=enabled,
            retry_config=retry_config or RetryConfig(),
        )

        self._webhooks[webhook_id] = config
        self._history[webhook_id] = []

        # Persist to disk
        await self._save_webhooks()

        logger.info(
            f"[WebhookManager] Registered webhook {webhook_id} "
            f"to {url} (events: {events or 'all'})"
        )

        return config

    async def update_webhook(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[str]] = None,
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        enabled: Optional[bool] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> Optional[WebhookConfig]:
        """
        Update an existing webhook configuration.

        Args:
            webhook_id: ID of webhook to update
            url: New URL (optional)
            events: New event list (optional)
            secret: New secret (optional)
            headers: New headers (optional)
            enabled: New enabled state (optional)
            retry_config: New retry config (optional)

        Returns:
            Updated WebhookConfig or None if not found
        """
        if webhook_id not in self._webhooks:
            return None

        config = self._webhooks[webhook_id]

        if url is not None:
            config.url = url
        if events is not None:
            config.events = events
        if secret is not None:
            config.secret = secret
        if headers is not None:
            config.headers = headers
        if enabled is not None:
            config.enabled = enabled
        if retry_config is not None:
            config.retry_config = retry_config

        config.updated_at = datetime.now(timezone.utc)

        await self._save_webhooks()

        logger.info(f"[WebhookManager] Updated webhook {webhook_id}")

        return config

    async def delete_webhook(self, webhook_id: str) -> bool:
        """
        Delete a webhook.

        Args:
            webhook_id: ID of webhook to delete

        Returns:
            True if webhook was deleted
        """
        if webhook_id not in self._webhooks:
            return False

        del self._webhooks[webhook_id]
        del self._history[webhook_id]

        await self._save_webhooks()

        logger.info(f"[WebhookManager] Deleted webhook {webhook_id}")

        return True

    def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get webhook configuration by ID."""
        return self._webhooks.get(webhook_id)

    def list_webhooks(
        self,
        enabled_only: bool = False,
        event_type: Optional[str] = None,
    ) -> List[WebhookConfig]:
        """
        List registered webhooks with optional filtering.

        Args:
            enabled_only: Only return enabled webhooks
            event_type: Only return webhooks subscribed to this event

        Returns:
            List of WebhookConfig objects
        """
        webhooks = list(self._webhooks.values())

        if enabled_only:
            webhooks = [w for w in webhooks if w.enabled]

        if event_type:
            webhooks = [w for w in webhooks if w.matches_event(event_type)]

        return webhooks

    # ======================================================================
    # Event Delivery
    # ======================================================================

    async def send_webhooks_for_event(self, event: Event) -> List[WebhookDelivery]:
        """
        Deliver an event to all matching webhooks.

        Called automatically by EventBus integration.

        Args:
            event: The event to deliver

        Returns:
            List of delivery records created
        """
        # Find matching webhooks
        matching = [
            w for w in self._webhooks.values()
            if w.enabled and w.matches_event(event.type)
        ]

        if not matching:
            return []

        # Deliver to all webhooks concurrently
        tasks = [
            self._deliver_to_webhook(event, webhook)
            for webhook in matching
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return delivery records
        deliveries = [
            r for r in results
            if isinstance(r, WebhookDelivery)
        ]

        return deliveries

    async def _deliver_to_webhook(
        self,
        event: Event,
        webhook: WebhookConfig,
    ) -> WebhookDelivery:
        """
        Deliver event to a single webhook with retry logic.

        Args:
            event: Event to deliver
            webhook: Webhook configuration

        Returns:
            Delivery record
        """
        delivery_id = f"dlv_{uuid.uuid4().hex[:12]}"

        delivery = WebhookDelivery(
            id=delivery_id,
            webhook_id=webhook.id,
            event_id=event.id,
            attempt=1,
        )

        # Attempt delivery with retries
        for attempt in range(1, webhook.retry_config.max_attempts + 1):
            delivery.attempt = attempt

            try:
                success = await self._attempt_delivery(event, webhook, delivery)

                if success:
                    delivery.status = "success"
                    delivery.completed_at = datetime.now(timezone.utc)
                    self._deliveries_success += 1
                    break
                else:
                    # Non-success response, retry if configured
                    if attempt < webhook.retry_config.max_attempts:
                        delivery.status = "retrying"
                        delay = webhook.retry_config.get_delay(attempt)
                        logger.info(
                            f"[WebhookManager] Retrying {webhook.id} "
                            f"(attempt {attempt}/{webhook.retry_config.max_attempts}) "
                            f"after {delay}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        delivery.status = "failed"
                        delivery.completed_at = datetime.now(timezone.utc)
                        self._deliveries_failed += 1

            except Exception as e:
                delivery.error_message = str(e)
                delivery.status = "failed"
                delivery.completed_at = datetime.now(timezone.utc)
                self._deliveries_failed += 1
                logger.error(
                    f"[WebhookManager] Delivery {delivery_id} failed: {e}",
                    exc_info=True
                )
                break

        # Record delivery
        await self._add_to_history(webhook.id, delivery)

        self._deliveries_total += 1

        return delivery

    async def _attempt_delivery(
        self,
        event: Event,
        webhook: WebhookConfig,
        delivery: WebhookDelivery,
    ) -> bool:
        """
        Attempt a single HTTP delivery to a webhook.

        Args:
            event: Event to deliver
            webhook: Webhook configuration
            delivery: Delivery record to update

        Returns:
            True if delivery was successful (2xx status)
        """
        # Build payload
        payload = self._build_payload(event, webhook)
        payload_str = json.dumps(payload)

        # Generate signature
        signature = WebhookSignature.sign(payload_str, webhook.secret)

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "MnemoCore-Webhook/1.0",
            WebhookSignature.SIGNATURE_HEADER: signature,
            **webhook.headers,
        }

        # Create session if needed
        close_session = False
        session = self._session
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            # Send HTTP POST
            async with session.post(
                webhook.url,
                data=payload_str,
                headers=headers,
                timeout=aiohttp.ClientSession(total=webhook.retry_config.timeout_seconds),
            ) as response:
                delivery.http_status = response.status

                # Read response body (limited size)
                try:
                    body = await response.text(limit=10000)
                    delivery.response_body = body[:1000]  # Truncate
                except Exception:
                    pass

                # 2xx status = success
                success = 200 <= response.status < 300

                if success:
                    logger.debug(
                        f"[WebhookManager] Delivered {event.id} to {webhook.id} "
                        f"(status={response.status})"
                    )
                else:
                    logger.warning(
                        f"[WebhookManager] Webhook {webhook.id} returned "
                        f"status {response.status}"
                    )

                return success

        except asyncio.TimeoutError:
            delivery.error_message = "Timeout"
            return False
        except aiohttp.ClientError as e:
            delivery.error_message = f"HTTP error: {e}"
            return False
        finally:
            if close_session:
                await session.close()

    def _build_payload(self, event: Event, webhook: WebhookConfig) -> Dict[str, Any]:
        """Build webhook payload with event data and metadata."""
        return {
            "id": event.id,
            "type": event.type,
            "data": event.data,
            "metadata": event.metadata,
            "webhook_delivery": {
                "webhook_id": webhook.id,
                "sent_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    # ======================================================================
    # Delivery History
    # ======================================================================

    async def _add_to_history(
        self,
        webhook_id: str,
        delivery: WebhookDelivery,
    ) -> None:
        """Add delivery to history with size limiting."""
        if webhook_id not in self._history:
            self._history[webhook_id] = []

        history = self._history[webhook_id]
        history.append(delivery)

        # Trim to max size
        if len(history) > self._max_history_size:
            self._history[webhook_id] = history[-self._max_history_size:]

    async def get_delivery_history(
        self,
        webhook_id: str,
        limit: int = 100,
        status: Optional[str] = None,
    ) -> List[WebhookDelivery]:
        """
        Get delivery history for a webhook.

        Args:
            webhook_id: Webhook ID
            limit: Maximum records to return
            status: Filter by status (optional)

        Returns:
            List of delivery records (most recent first)
        """
        history = self._history.get(webhook_id, [])

        if status:
            history = [d for d in history if d.status == status]

        return list(reversed(history[-limit:]))

    async def get_webhook_status(self, webhook_id: str) -> Dict[str, Any]:
        """Get delivery statistics for a webhook."""
        if webhook_id not in self._webhooks:
            return {}

        history = self._history.get(webhook_id, [])

        # Calculate statistics
        total = len(history)
        success = sum(1 for d in history if d.status == "success")
        failed = sum(1 for d in history if d.status == "failed")
        pending = sum(1 for d in history if d.status in ("pending", "retrying"))

        # Recent deliveries (last 24 hours)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent = [d for d in history if d.created_at >= cutoff]
        recent_success = sum(1 for d in recent if d.status == "success")

        return {
            "webhook_id": webhook_id,
            "total_deliveries": total,
            "successful_deliveries": success,
            "failed_deliveries": failed,
            "pending_deliveries": pending,
            "success_rate": success / total if total > 0 else 0.0,
            "recent_deliveries_24h": len(recent),
            "recent_success_rate_24h": recent_success / len(recent) if recent else 0.0,
            "last_delivery": history[-1].to_dict() if history else None,
        }

    # ======================================================================
    # Retry Loop
    # ======================================================================

    async def _retry_loop(self) -> None:
        """Background loop for retrying failed deliveries."""
        logger.debug("[WebhookManager] Retry loop started")

        while self._running:
            try:
                # Get next delivery to retry (with timeout)
                delivery = await asyncio.wait_for(
                    self._retry_queue.get(),
                    timeout=5.0
                )

                webhook = self._webhooks.get(delivery.webhook_id)
                if not webhook or not webhook.enabled:
                    continue

                # Get the original event (we'd need to store events)
                # For now, skip retry since we don't have event persistence
                logger.debug(f"[WebhookManager] Skipping retry for {delivery.id}")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"[WebhookManager] Retry loop error: {e}")

        logger.debug("[WebhookManager] Retry loop stopped")

    # ======================================================================
    # Persistence
    # ======================================================================

    async def _load_webhooks(self) -> None:
        """Load webhook configurations from disk."""
        if not self._persistence_path or not self._persistence_path.exists():
            return

        try:
            import aiofiles

            async with aiofiles.open(self._persistence_path, "r") as f:
                content = await f.read()
                data = json.loads(content)

            for webhook_data in data.get("webhooks", []):
                config = WebhookConfig.from_dict(webhook_data)
                self._webhooks[config.id] = config
                self._history[config.id] = []

            logger.info(
                f"[WebhookManager] Loaded {len(self._webhooks)} webhooks "
                f"from {self._persistence_path}"
            )

        except Exception as e:
            logger.error(f"[WebhookManager] Failed to load webhooks: {e}")

    async def _save_webhooks(self) -> None:
        """Save webhook configurations to disk."""
        if not self._persistence_path:
            return

        try:
            import aiofiles

            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": "1.0",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "webhooks": [
                    w.to_dict(scrub_secret=False)
                    for w in self._webhooks.values()
                ],
            }

            async with aiofiles.open(self._persistence_path, "w") as f:
                await f.write(json.dumps(data, indent=2))

        except Exception as e:
            logger.error(f"[WebhookManager] Failed to save webhooks: {e}")

    # ======================================================================
    # Metrics
    # ======================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get webhook manager metrics."""
        return {
            "total_webhooks": len(self._webhooks),
            "enabled_webhooks": sum(1 for w in self._webhooks.values() if w.enabled),
            "total_deliveries": self._deliveries_total,
            "successful_deliveries": self._deliveries_success,
            "failed_deliveries": self._deliveries_failed,
            "success_rate": (
                self._deliveries_success / self._deliveries_total
                if self._deliveries_total > 0
                else 0.0
            ),
        }


# =============================================================================
# Global Singleton
# =============================================================================

_WEBHOOK_MANAGER: Optional[WebhookManager] = None


def get_webhook_manager(
    persistence_path: Optional[str] = None,
) -> WebhookManager:
    """
    Get or create the global WebhookManager singleton.

    Args:
        persistence_path: Path to save webhook configurations

    Returns:
        The shared WebhookManager instance
    """
    global _WEBHOOK_MANAGER

    if _WEBHOOK_MANAGER is None:
        _WEBHOOK_MANAGER = WebhookManager(persistence_path=persistence_path)

    return _WEBHOOK_MANAGER


def reset_webhook_manager() -> None:
    """Reset the global WebhookManager singleton (useful for testing)."""
    global _WEBHOOK_MANAGER
    _WEBHOOK_MANAGER = None
