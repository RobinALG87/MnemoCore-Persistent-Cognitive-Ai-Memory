from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_image_installs_wheel_and_starts_public_api_module():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "pip wheel" in dockerfile
    assert "pip install" in dockerfile
    assert "mnemocore.api.main:app" in dockerfile
    assert "src.api.main:app" not in dockerfile
    assert "EXPOSE 8100" in dockerfile
    assert "COPY pyproject.toml README.md LICENSE config.yaml ./" in dockerfile
    assert "COPY pyproject.toml README.md LICENSE CHANGELOG.md" not in dockerfile
    assert "HAIM_API_KEY=" not in dockerfile
    assert "sed -i 's/\\r$//' /entrypoint.sh" in dockerfile
    assert 'ENTRYPOINT ["/entrypoint.sh"]' in dockerfile
    assert (
        'CMD ["uvicorn", "mnemocore.api.main:app", "--host", "0.0.0.0", '
        '"--port", "8100", "--workers", "1", "--log-level", "info"]'
        in dockerfile
    )
