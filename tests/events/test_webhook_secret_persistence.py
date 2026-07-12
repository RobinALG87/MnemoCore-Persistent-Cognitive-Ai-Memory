import json

import pytest

from mnemocore.events.event_bus import Event
from mnemocore.events.webhook_manager import (
    SensitiveWebhookHeaderError,
    UnsafeWebhookPersistenceError,
    WebhookDelivery,
    WebhookManager,
    WebhookPersistenceError,
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header_name",
    [
        "Authorization",
        "proxy-authorization",
        "X-API-Key",
        "api-key",
        "Cookie",
        "set-cookie",
        "X-Customer-Token",
    ],
)
async def test_persistent_webhooks_reject_sensitive_headers(tmp_path, header_name):
    manager = WebhookManager(
        persistence_path=str(tmp_path / "webhooks.json"),
        sensitive_header_names={"X-Customer-Token"},
    )

    with pytest.raises(SensitiveWebhookHeaderError, match="secret references"):
        await manager.register_webhook(
            url="https://example.test/hook",
            secret_ref="SIGNING_SECRET",
            headers={header_name: "must-not-persist"},
        )


@pytest.mark.asyncio
async def test_persistent_webhooks_allow_non_sensitive_headers(tmp_path):
    manager = WebhookManager(persistence_path=str(tmp_path / "webhooks.json"))

    config = await manager.register_webhook(
        url="https://example.test/hook",
        secret_ref="SIGNING_SECRET",
        headers={"X-Webhook-Version": "2"},
    )

    assert config.headers == {"X-Webhook-Version": "2"}


@pytest.mark.asyncio
@pytest.mark.parametrize("document", [{}, {"version": "3.0", "webhooks": []}])
async def test_load_rejects_missing_or_unknown_persistence_version(tmp_path, document):
    path = tmp_path / "webhooks.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    manager = WebhookManager(persistence_path=str(path))

    with pytest.raises(WebhookPersistenceError, match="version 2.0"):
        await manager.start()


@pytest.mark.asyncio
async def test_load_rejects_malformed_json(tmp_path):
    path = tmp_path / "webhooks.json"
    path.write_text('{"version": "2.0",', encoding="utf-8")

    with pytest.raises(WebhookPersistenceError):
        await WebhookManager(persistence_path=str(path)).start()


@pytest.mark.asyncio
async def test_load_rejects_sensitive_persisted_headers(tmp_path):
    path = tmp_path / "webhooks.json"
    path.write_text(
        json.dumps(
            {
                "version": "2.0",
                "webhooks": [
                    {
                        "id": "wh_unsafe",
                        "url": "https://example.test/hook",
                        "secret_ref": "SIGNING_SECRET",
                        "headers": {"Authorization": "plaintext-credential"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SensitiveWebhookHeaderError):
        await WebhookManager(persistence_path=str(path)).start()


@pytest.mark.asyncio
async def test_malformed_load_is_fail_closed_without_partial_state(tmp_path):
    path = tmp_path / "webhooks.json"
    path.write_text(
        json.dumps(
            {
                "version": "2.0",
                "webhooks": [
                    {
                        "id": "wh_valid",
                        "url": "https://example.test/valid",
                        "secret_ref": "VALID_SECRET",
                    },
                    {"id": "wh_invalid"},
                ],
            }
        ),
        encoding="utf-8",
    )
    manager = WebhookManager(persistence_path=str(path))

    with pytest.raises(WebhookPersistenceError):
        await manager.start()

    assert manager.get_webhook("wh_valid") is None


@pytest.mark.asyncio
async def test_register_rolls_back_when_persistence_fails(tmp_path, monkeypatch):
    manager = WebhookManager(persistence_path=str(tmp_path / "webhooks.json"))

    def fail_write(_content):
        raise OSError("disk full")

    monkeypatch.setattr(manager, "_atomic_write", fail_write)
    with pytest.raises(WebhookPersistenceError):
        await manager.register_webhook(
            url="https://example.test/hook",
            secret_ref="SIGNING_SECRET",
        )

    assert manager.list_webhooks() == []


@pytest.mark.asyncio
async def test_update_and_delete_roll_back_when_persistence_fails(tmp_path, monkeypatch):
    manager = WebhookManager(persistence_path=str(tmp_path / "webhooks.json"))
    config = await manager.register_webhook(
        url="https://example.test/original",
        secret_ref="SIGNING_SECRET",
    )

    def fail_write(_content):
        raise OSError("disk full")

    monkeypatch.setattr(manager, "_atomic_write", fail_write)
    with pytest.raises(WebhookPersistenceError):
        await manager.update_webhook(config.id, url="https://example.test/changed")
    assert manager.get_webhook(config.id).url == "https://example.test/original"

    with pytest.raises(WebhookPersistenceError):
        await manager.delete_webhook(config.id)
    assert manager.get_webhook(config.id) is not None

    reopened = WebhookManager(persistence_path=str(tmp_path / "webhooks.json"))
    await reopened._load_webhooks()
    assert reopened.get_webhook(config.id).url == "https://example.test/original"


@pytest.mark.asyncio
async def test_concurrent_registration_persists_every_webhook(tmp_path):
    import asyncio

    path = tmp_path / "webhooks.json"
    manager = WebhookManager(persistence_path=str(path))

    await asyncio.gather(
        *(
            manager.register_webhook(
                url=f"https://example.test/hook/{index}",
                secret_ref=f"SIGNING_SECRET_{index}",
            )
            for index in range(10)
        )
    )

    reloaded = WebhookManager(persistence_path=str(path))
    await reloaded._load_webhooks()
    assert len(reloaded.list_webhooks()) == 10


@pytest.mark.asyncio
async def test_registration_log_redacts_url_credentials_and_query(tmp_path):
    from loguru import logger

    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), level="INFO")
    try:
        manager = WebhookManager(persistence_path=str(tmp_path / "webhooks.json"))
        await manager.register_webhook(
            url="https://alice:password@example.test/hook?token=query-secret",
            secret_ref="SIGNING_SECRET",
        )
    finally:
        logger.remove(sink)

    output = "".join(messages)
    assert "example.test" in output
    assert "alice" not in output
    assert "password" not in output
    assert "query-secret" not in output
