import json

import pytest

from mnemocore.events.event_bus import Event
from mnemocore.events.webhook_manager import (
    UnsafeWebhookPersistenceError,
    WebhookDelivery,
    WebhookManager,
    WebhookSignature,
)


class _Response:
    status = 204

    async def text(self) -> str:
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _Session:
    def __init__(self) -> None:
        self.headers = None
        self.payload = None

    def post(self, _url, *, data, headers):
        self.payload = data
        self.headers = headers
        return _Response()


@pytest.mark.asyncio
async def test_persistence_stores_only_secret_reference(tmp_path):
    path = tmp_path / "webhooks.json"
    secret = "must-never-reach-disk"
    manager = WebhookManager(persistence_path=str(path))

    config = await manager.register_webhook(
        url="https://example.test/hook",
        secret=secret,
        secret_ref="MNEMOCORE_WEBHOOK_TEST_SECRET",
    )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    saved = persisted["webhooks"][0]
    assert saved["secret_ref"] == "MNEMOCORE_WEBHOOK_TEST_SECRET"
    assert "secret" not in saved
    assert secret not in path.read_text(encoding="utf-8")
    assert config.secret == secret


@pytest.mark.asyncio
async def test_persistent_inline_secret_requires_reference(tmp_path):
    manager = WebhookManager(persistence_path=str(tmp_path / "webhooks.json"))

    with pytest.raises(ValueError, match="secret_ref"):
        await manager.register_webhook(
            url="https://example.test/hook",
            secret="inline-only",
        )


@pytest.mark.asyncio
async def test_legacy_plaintext_persistence_is_rejected_without_secret_disclosure(tmp_path):
    path = tmp_path / "webhooks.json"
    secret = "legacy-plaintext-value"
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "webhooks": [
                    {
                        "id": "wh_legacy",
                        "url": "https://example.test/hook",
                        "secret": secret,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manager = WebhookManager(persistence_path=str(path))

    with pytest.raises(UnsafeWebhookPersistenceError) as exc_info:
        await manager.start()

    assert secret not in str(exc_info.value)
    assert manager.get_webhook("wh_legacy") is None


@pytest.mark.asyncio
async def test_delivery_resolves_referenced_secret_at_send_time(tmp_path):
    resolved_secret = "resolved-at-delivery"
    session = _Session()
    references = []

    def resolve(secret_ref: str) -> str:
        references.append(secret_ref)
        return resolved_secret

    manager = WebhookManager(
        persistence_path=str(tmp_path / "webhooks.json"),
        http_session=session,
        secret_resolver=resolve,
    )
    webhook = await manager.register_webhook(
        url="https://example.test/hook",
        secret_ref="vault://webhooks/customer-a",
    )
    event = Event(id="evt_1", type="memory.created", data={}, metadata={})
    delivery = WebhookDelivery(
        id="delivery_1",
        webhook_id=webhook.id,
        event_id=event.id,
    )

    assert await manager._attempt_delivery(event, webhook, delivery)
    assert references == ["vault://webhooks/customer-a"]
    assert WebhookSignature.verify(
        session.payload,
        resolved_secret,
        session.headers[WebhookSignature.SIGNATURE_HEADER],
    )


@pytest.mark.asyncio
async def test_non_persistent_inline_secret_remains_supported():
    manager = WebhookManager()

    webhook = await manager.register_webhook(
        url="https://example.test/hook",
        secret="existing-inline-secret",
    )

    assert webhook.secret == "existing-inline-secret"
    assert webhook.secret_ref is None


@pytest.mark.asyncio
async def test_secret_resolution_errors_are_redacted(tmp_path):
    secret = "resolver-leaked-secret"

    def fail(_secret_ref: str) -> str:
        raise RuntimeError(f"backend exposed {secret}")

    manager = WebhookManager(
        persistence_path=str(tmp_path / "webhooks.json"),
        http_session=_Session(),
        secret_resolver=fail,
    )
    webhook = await manager.register_webhook(
        url="https://example.test/hook",
        secret_ref="vault://webhooks/customer-a",
    )
    event = Event(id="evt_1", type="memory.created", data={}, metadata={})
    delivery = WebhookDelivery(
        id="delivery_1",
        webhook_id=webhook.id,
        event_id=event.id,
    )

    assert not await manager._attempt_delivery(event, webhook, delivery)
    assert delivery.error_message == "Webhook signing secret unavailable"
    assert secret not in delivery.error_message
