# Contributing to MnemoCore

Thank you for your interest in contributing to MnemoCore!

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/RobinALG87/MnemoCore-Infrastructure-for-Persistent-Cognitive-Memory.git
cd MnemoCore-Infrastructure-for-Persistent-Cognitive-Memory

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

### 2. For Reproducible Builds

For production deployments or when you need exact dependency versions:

```bash
pip install -r requirements.txt
```

**IMPORTANT:** If you add or update dependencies:
1. Update `pyproject.toml` (the canonical source of truth)
2. Sync `requirements.txt` for Docker compatibility
3. Optionally regenerate a pinned lockfile: `pip freeze > requirements.lock`

## Dependency Structure

| File | Purpose |
|------|---------|
| `pyproject.toml` | **Canonical source of truth** for all dependencies |
| `requirements.txt` | Runtime dependencies (for Docker/legacy compatibility) |
| `requirements-dev.txt` | Development and testing dependencies |
| `requirements-optional.txt` | Optional visualization and LLM dependencies |

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/mnemocore --cov-report=html

# Run specific test file
pytest tests/core/test_engine.py

# Run with verbose output
pytest -v tests/

# Run only fast tests (skip slow integration tests)
pytest -v -m "not slow" tests/
```

### Test Requirements

- All new features must include unit tests
- Bug fixes should include regression tests
- Coverage threshold: 80% for new code
- Use pytest fixtures for common setup (see `tests/conftest.py`)

### Test Fixture Patterns

```python
# Use the engine fixture for integration tests
async def test_my_feature(engine: HAIMEngine):
    node_id = await engine.store("test content")
    assert node_id is not None

# Use mock fixtures for unit tests
def test_hdv_operations(mock_hdv):
    result = mock_hdv.xor_bind(other_hdv)
    assert result is not None
```

## Code Quality

We use pre-commit hooks to maintain code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
```

### Pre-commit Checks

- **ruff**: Linting and formatting (replaces flake8, isort, black)
- **mypy**: Type checking
- **bandit**: Security scanning
- File format checks (YAML, JSON, TOML)
- Trailing whitespace and end-of-file fixes

### Code Style Guidelines

- Follow PEP 8 guidelines
- Use type hints for all public functions
- Write Google-style docstrings for classes and public methods
- Keep line length under 100 characters
- Use descriptive variable names (avoid single-letter except in loops)

### Docstring Convention (Google-style)

```python
def store_memory(self, content: str, metadata: Optional[Dict] = None) -> str:
    """Store a new memory in the system.

    Args:
        content: The text content to store.
        metadata: Optional dictionary of metadata to attach.

    Returns:
        The unique ID of the stored memory node.

    Raises:
        StorageError: If the memory cannot be persisted.
        ValueError: If content is empty.
    """
    pass
```

## Version Numbers

All version numbers should be synchronized to a single value. The canonical version is defined in `pyproject.toml`:

```toml
[project]
version = "5.1.0"
```

At runtime, use `importlib.metadata.version("mnemocore")` to get the version.

## Pull Request Process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** with clear commit messages:
   ```bash
   git commit -m "feat: add new memory consolidation algorithm"
   ```

3. **Ensure all tests pass**:
   ```bash
   pytest
   ```

4. **Run pre-commit hooks**:
   ```bash
   pre-commit run --all-files
   ```

5. **Update documentation** if needed:
   - API changes: Update docstrings and `docs/API.md`
   - Architecture changes: Update `docs/ARCHITECTURE.md`
   - New config options: Update `docs/CONFIGURATION.md`

6. **Submit a pull request** with:
   - Clear description of changes
   - Link to any related issues
   - Test coverage for new code

### PR Checklist

- [ ] Tests pass locally
- [ ] Pre-commit hooks pass
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG.md updated (for user-facing changes)
- [ ] Commit messages follow conventional format

## Commit Message Format

We use conventional commits:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding/updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add SM-2 spaced repetition algorithm
fix: resolve memory leak in tier promotion
docs: update deployment guide with Helm instructions
test: add regression test for issue #42
```

## Project Structure

```
mnemocore/
  core/           # Core engine, HDV, tier management
  storage/        # Backup, export, compression
  cognitive/      # Reconstructive recall, associations
  events/         # Event bus, webhooks
  subconscious/   # Background processing, dream pipeline
  meta/           # Goal tracking, learning journal
  mcp/            # MCP server implementation
  api/            # FastAPI REST endpoints
  cli/            # Command-line interface
  llm/            # Multi-provider LLM integration
  utils/          # Shared utilities

tests/
  core/           # Core module tests
  integration/    # Integration tests
  fixtures/       # Test fixtures

docs/             # Documentation
helm/             # Kubernetes Helm charts
k8s/              # Kubernetes manifests
scripts/          # Utility scripts
```

## Getting Help

- Open an issue on GitHub for bugs, features, or questions
- Check existing issues before creating new ones
- Include reproduction steps for bug reports

## License

By contributing to MnemoCore, you agree that your contributions will be licensed under the MIT License.
