"""Release-pipeline contracts kept independent of GitHub Actions execution."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_ghcr_image_name_is_lowercase_and_not_derived_from_mixed_case_repository() -> None:
    workflow = read(".github/workflows/docker-publish.yml")

    assert "IMAGE_NAME: robinalg87/mnemocore-persistent-cognitive-ai-memory" in workflow
    assert "IMAGE_NAME: ${{ github.repository }}" not in workflow
    assert "attestations: write" in workflow
    assert "id-token: write" in workflow


def test_published_release_pipeline_builds_then_publishes_to_pypi_and_huggingface() -> None:
    workflow = read(".github/workflows/release-publish.yml")

    assert "release:" in workflow
    assert "python -m build" in workflow
    assert "twine check dist/*" in workflow
    assert "tests/agent_memory tests/integrations" in workflow
    assert "benchmarks/test_agent_memory_baseline.py" in workflow
    assert "pypa/gh-action-pypi-publish" in workflow
    assert "secrets.PYPI_API_TOKEN" in workflow
    assert "Granis87/MnemoCore" in workflow
    assert "secrets.HF_TOKEN" in workflow
