from fastapi.testclient import TestClient


class _HeaderScopeAuthorizer:
    """Deterministic stand-in for an auth integration bound to a credential."""

    async def authorize(self, request, scope):
        return request.headers.get("X-Test-Subject") == scope.user_id


def _client(sqlite_path, *, authorizer=None):
    from mnemocore.api.v3_app import create_v3_app

    return TestClient(create_v3_app(sqlite_path, scope_authorizer=authorizer))


def _scope(**overrides):
    return {
        "tenant_id": "tenant-a",
        "user_id": "user-a",
        "agent_id": "agent-a",
        "project_id": "project-a",
        **overrides,
    }


def test_v3_memory_lifecycle_is_exact_scope_and_reports_scores(tmp_path):
    client = _client(tmp_path / "v3.sqlite", authorizer=_HeaderScopeAuthorizer())
    headers = {"X-Test-Subject": "user-a"}
    create = client.post(
        "/v3/memories",
        headers=headers,
        json={**_scope(), "content": "The launch window is Tuesday."},
    )
    assert create.status_code == 201
    created = create.json()
    assert created["scope_key"] == '["tenant-a","user-a","agent-a","project-a",null]'
    memory_id = created["memory"]["id"]

    recall = client.post(
        "/v3/memories/recall",
        headers=headers,
        json={**_scope(), "query": "launch window"},
    )
    assert recall.status_code == 200
    result = recall.json()["results"][0]
    assert result["memory"]["id"] == memory_id
    assert result["scoring_version"] == "hybrid-lexical-binary-hdv-v2"
    assert set(result["score_components"]) == {"lexical", "hdv", "hybrid"}

    foreign = client.get(
        f"/v3/memories/{memory_id}", headers=headers, params=_scope(user_id="user-b")
    )
    assert foreign.status_code == 403
    assert foreign.json()["detail"] == "scope is not authorized"

    forgotten = client.delete(
        f"/v3/memories/{memory_id}", headers=headers, params=_scope()
    )
    assert forgotten.status_code == 200
    assert forgotten.json()["memory"]["status"] == "forgotten"


def test_v3_requires_every_required_scope_identifier(tmp_path):
    client = _client(tmp_path / "v3.sqlite", authorizer=_HeaderScopeAuthorizer())
    response = client.post(
        "/v3/memories", json={"user_id": "u", "agent_id": "a", "content": "x"}
    )
    assert response.status_code == 422


def test_v3_remember_rejects_deep_or_reserved_metadata(tmp_path):
    client = _client(tmp_path / "v3.sqlite", authorizer=_HeaderScopeAuthorizer())
    headers = {"X-Test-Subject": "user-a"}

    deep = client.post(
        "/v3/memories",
        headers=headers,
        json={
            **_scope(),
            "content": "must not persist",
            "metadata": {"one": {"two": {"three": {"four": "too deep"}}}},
        },
    )
    reserved = client.post(
        "/v3/memories",
        headers=headers,
        json={
            **_scope(),
            "content": "must not persist",
            "metadata": {"internal_state": "forbidden"},
        },
    )

    assert deep.status_code == 422
    assert reserved.status_code == 422


def test_v3_remember_accepts_bounded_nested_metadata(tmp_path):
    client = _client(tmp_path / "v3.sqlite", authorizer=_HeaderScopeAuthorizer())
    response = client.post(
        "/v3/memories",
        headers={"X-Test-Subject": "user-a"},
        json={
            **_scope(),
            "content": "release metadata",
            "metadata": {"source": {"name": "brief", "tags": ["release"]}},
        },
    )

    assert response.status_code == 201
    assert response.json()["memory"]["metadata"] == {
        "source": {"name": "brief", "tags": ["release"]}
    }


def test_v3_is_disabled_without_a_scope_authorizer(tmp_path):
    client = _client(tmp_path / "v3.sqlite")

    response = client.post(
        "/v3/memories", json={**_scope(), "content": "must not be stored"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "v3 scope authorization is not configured"


def test_v3_only_app_lifespan_does_not_initialize_legacy_haim(tmp_path, monkeypatch):
    from mnemocore.api.v3_app import create_v3_app

    def fail_if_called(*args, **kwargs):
        raise AssertionError("legacy HAIM container must not be initialized")

    monkeypatch.setattr("mnemocore.core.container.build_container", fail_if_called)
    monkeypatch.setattr("mnemocore.core.engine.HAIMEngine.initialize", fail_if_called)

    with TestClient(
        create_v3_app(tmp_path / "v3.sqlite", scope_authorizer=_HeaderScopeAuthorizer())
    ) as client:
        response = client.post(
            "/v3/memories",
            headers={"X-Test-Subject": "user-a"},
            json={**_scope(), "content": "v3 starts without HAIM"},
        )

    assert response.status_code == 201
