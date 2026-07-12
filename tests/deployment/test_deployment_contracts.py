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


def test_ci_docker_job_runs_the_prototype_and_checks_public_endpoints() -> None:
    workflow = read(".github/workflows/ci.yml")
    docker_job = workflow.split("  docker:", 1)[1].split("  build-status:", 1)[0]

    assert "docker run --rm" in docker_job
    assert "docker run --detach" in docker_job
    assert "--publish 127.0.0.1:8100:8100" in docker_job
    assert "wait_for_endpoint /health" in docker_job
    assert "wait_for_endpoint /ready" in docker_job
    assert "wait_for_endpoint /metrics/" in docker_job
    assert "http://127.0.0.1:8100/metrics/" in docker_job
    assert "grep -Eq '^# (HELP|TYPE) '" in docker_job
    assert docker_job.count("if: always()") >= 2
    assert 'docker logs "$CONTAINER_NAME" || true' in docker_job
    assert 'docker rm --force "$CONTAINER_NAME" || true' in docker_job


def test_compose_requires_secrets_and_uses_api_port_for_metrics() -> None:
    compose = read("docker-compose.yml")

    assert "changeme" not in compose
    assert "HAIM_API_KEY:?" in compose
    assert "REDIS_PASSWORD:?" in compose
    assert "QDRANT_API_KEY:?" in compose
    assert "9090" not in compose
    assert '"127.0.0.1:8100:8100"' in compose


def test_compose_qdrant_healthcheck_uses_available_bash_http_probe() -> None:
    compose = read("docker-compose.yml")
    qdrant = compose.split("  qdrant:", 1)[1].split("# Networks", 1)[0]

    assert "curl" not in qdrant
    assert "/bin/bash" in qdrant
    assert "/dev/tcp/127.0.0.1/6333" in qdrant
    assert "GET /healthz HTTP/1.1" in qdrant
    assert "grep -q '200 OK'" in qdrant


def test_helm_uses_single_http_listener_and_http_probes() -> None:
    values = read("helm/mnemocore/values.yaml")
    deployment = read("helm/mnemocore/templates/deployment.yaml")
    service = read("helm/mnemocore/templates/service.yaml")

    assert "metrics: 8100" in values
    assert "replicaCount: 1" in values
    autoscaling = values.split("  autoscaling:", 1)[1].split(
        "  # Pod Disruption Budget", 1
    )[0]
    assert "enabled: false" in autoscaling
    assert 'existingSecret: "mnemocore-api-key"' in values
    assert "path: /health" in deployment
    assert "path: /ready" in deployment
    assert "targetPort: http" in service
    assert "metricsPort" not in values
    assert "name: metrics" not in service
    service_monitor = read("helm/mnemocore/templates/servicemonitor.yaml")
    assert "- port: http" in service_monitor
    assert "path: /metrics/" in service_monitor
    assert "name: {{ .Values.redis.existingSecret }}" in deployment
    assert "name: {{ .Values.qdrant.existingSecret }}" in deployment


def test_helm_optional_dependencies_are_disabled_by_default() -> None:
    values = read("helm/mnemocore/values.yaml")
    redis = values.split("redis:", 1)[1].split("# Qdrant Configuration", 1)[0]
    qdrant = values.split("qdrant:", 1)[1].split("# Network Policies", 1)[0]

    assert "enabled: false" in redis
    assert "enabled: false" in qdrant


def test_helm_chart_has_no_unresolved_subcharts_and_uses_public_metadata() -> None:
    chart = read("helm/mnemocore/Chart.yaml")
    repository = (
        "https://github.com/RobinALG87/MnemoCore-Persistent-Cognitive-Ai-Memory"
    )

    assert "dependencies:" not in chart
    assert "charts.bitnami.com" not in chart
    assert "qdrant.github.io" not in chart
    assert f"home: {repository}" in chart
    assert f"  - {repository}" in chart
    assert "github.com/your-org" not in chart


def test_helm_deployment_has_no_dead_secret_template_checksum() -> None:
    deployment = read("helm/mnemocore/templates/deployment.yaml")

    assert "checksum/secret" not in deployment
    assert 'Template.BasePath "/secret.yaml"' not in deployment
    assert 'Template.BasePath "/configmap.yaml"' in deployment


def test_helm_ignore_uses_only_supported_patterns() -> None:
    helmignore = read("helm/mnemocore/.helmignore")

    assert "negation" not in helmignore
    assert not any(line.startswith("!") for line in helmignore.splitlines())


def test_embedded_deployment_gates_close_before_next_document() -> None:
    for template in ("deployment-redis.yaml", "deployment-qdrant.yaml"):
        content = read(f"helm/mnemocore/templates/{template}")
        deployment_document = content.split("\n---", 1)[0]

        assert deployment_document.rstrip().endswith("{{- end }}")
