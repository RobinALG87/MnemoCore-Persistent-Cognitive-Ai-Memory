from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_image_installs_wheel_and_starts_public_api_module():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "pip wheel" in dockerfile
    assert "pip install" in dockerfile
    assert "mnemocore.api.main:app" in dockerfile
    assert "src.api.main:app" not in dockerfile
    assert "EXPOSE 8100" in dockerfile
