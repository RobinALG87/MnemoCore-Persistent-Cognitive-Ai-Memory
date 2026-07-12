from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_ci_summary_is_fail_closed_for_all_required_jobs() -> None:
    workflow = read(".github/workflows/ci.yml")
    summary = workflow.split("build-status:", 1)[1]

    assert "needs.docker.result" in summary
    assert "needs.packaging.result" in summary
    assert 'if [ "$job" != "success" ]' in summary


def test_compose_requires_secrets_and_uses_api_port_for_metrics() -> None:
    compose = read("docker-compose.yml")

    assert "changeme" not in compose
    assert "REDIS_PASSWORD:?" in compose
    assert "QDRANT_API_KEY:?" in compose
    assert "9090" not in compose
    assert '"127.0.0.1:8100:8100"' in compose


def test_helm_uses_single_http_listener_and_http_probes() -> None:
    values = read("helm/mnemocore/values.yaml")
    deployment = read("helm/mnemocore/templates/deployment.yaml")
    service = read("helm/mnemocore/templates/service.yaml")

    assert "metrics: 8100" in values
    assert "metricsPort: 8100" in values
    assert 'existingSecret: "mnemocore-api-key"' in values
    assert "path: /health" in deployment
    assert "path: /ready" in deployment
    assert "targetPort: http" in service
    assert "name: {{ .Values.redis.existingSecret }}" in deployment
    assert "name: {{ .Values.qdrant.existingSecret }}" in deployment


def test_helm_optional_dependencies_are_disabled_by_default() -> None:
    values = read("helm/mnemocore/values.yaml")
    redis = values.split("redis:", 1)[1].split("# Qdrant Configuration", 1)[0]
    qdrant = values.split("qdrant:", 1)[1].split("# Network Policies", 1)[0]

    assert "enabled: false" in redis
    assert "enabled: false" in qdrant
