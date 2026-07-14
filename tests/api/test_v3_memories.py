from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(sqlite_path):
    from mnemocore.api.routes.v3_memories import router

    app = FastAPI()
    app.state.v3_sqlite_path = sqlite_path
    app.include_router(router)
    return TestClient(app)


def _scope(**overrides):
    return {
        "tenant_id": "tenant-a",
        "user_id": "user-a",
        "agent_id": "agent-a",
        "project_id": "project-a",
        **overrides,
    }


def test_v3_memory_lifecycle_is_exact_scope_and_reports_scores(tmp_path):
    client = _client(tmp_path / "v3.sqlite")
    create = client.post("/v3/memories", json={**_scope(), "content": "The launch window is Tuesday."})
    assert create.status_code == 201
    created = create.json()
    assert created["scope_key"] == '["tenant-a","user-a","agent-a","project-a",null]'
    memory_id = created["memory"]["id"]

    recall = client.post("/v3/memories/recall", json={**_scope(), "query": "launch window"})
    assert recall.status_code == 200
    result = recall.json()["results"][0]
    assert result["memory"]["id"] == memory_id
    assert result["scoring_version"] == "hybrid-lexical-binary-hdv-v1"
    assert set(result["score_components"]) == {"lexical", "hdv", "hybrid"}

    foreign = client.get(f"/v3/memories/{memory_id}", params=_scope(user_id="user-b"))
    assert foreign.status_code == 404
    assert foreign.json()["detail"] == "memory not found"

    forgotten = client.delete(f"/v3/memories/{memory_id}", params=_scope())
    assert forgotten.status_code == 200
    assert forgotten.json()["memory"]["status"] == "forgotten"


def test_v3_requires_every_required_scope_identifier(tmp_path):
    client = _client(tmp_path / "v3.sqlite")
    response = client.post("/v3/memories", json={"user_id": "u", "agent_id": "a", "content": "x"})
    assert response.status_code == 422
