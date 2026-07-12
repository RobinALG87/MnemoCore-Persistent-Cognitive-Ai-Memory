from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mnemocore.api.routes.health import router


def _client(*, engine=None, container=None) -> TestClient:
    app = FastAPI()
    if engine is not None:
        app.state.engine = engine
    if container is not None:
        app.state.container = container
    app.include_router(router)
    return TestClient(app)


def test_health_remains_live_when_runtime_is_not_initialized():
    response = _client().get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["engine_ready"] is False


def test_ready_accepts_initialized_local_runtime_without_redis():
    container = SimpleNamespace(redis_storage=None)

    response = _client(engine=object(), container=container).get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["engine_ready"] is True


def test_ready_rejects_runtime_without_engine():
    response = _client(container=SimpleNamespace(redis_storage=None)).get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_health_reports_redis_without_making_it_a_local_readiness_dependency():
    redis = SimpleNamespace(check_health=AsyncMock(return_value=False))
    container = SimpleNamespace(redis_storage=redis)

    health = _client(engine=object(), container=container).get("/health")
    ready = _client(engine=object(), container=container).get("/ready")

    assert health.status_code == 200
    assert health.json()["redis_connected"] is False
    assert ready.status_code == 200

