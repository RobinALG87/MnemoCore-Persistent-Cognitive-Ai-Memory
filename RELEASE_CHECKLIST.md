# MnemoCore Release Checklist — baseline triage

## Status: NOT RELEASE-READY

> The authoritative current-state evidence is [Platform baseline — 2026-07-12](docs/status/2026-07-12-platform-baseline.md).
> [PRODUCTION_REMEDIATION_PROGRESS.md](PRODUCTION_REMEDIATION_PROGRESS.md) is
> retained as historical context and is not a current green-status claim.

### Current gate summary

| Gate | Status | Evidence |
|---|---|---|
| AgentMemory core | PASS | 304 focused tests pass |
| Offline adapters | PASS | 6 tests pass |
| Benchmark contracts/smoke | PASS | 18 tests pass |
| Legacy platform | TIMEOUT | Documented lane did not complete within the baseline window |
| Lifecycle integration | QUARANTINED | 16 tests skipped pending migration |
| Docker | NOT RUN | Daemon unavailable during baseline |
| Security/dependency scans | FAIL | Existing Bandit and pip-audit findings remain |
| Release decision | BLOCKED | Do not publish a release claim until the baseline gates are reviewed |

---

## ✅ Completed

- [x] LICENSE file (MIT)
- [x] .gitignore hardened (temp files, debug artifacts, build outputs)
- [x] data/memory.jsonl removed (no stored memories)
- [x] No leaked API keys or credentials
- [ ] **2,200+ unit tests defined** — 133 failures remain (see [PRODUCTION_REMEDIATION_PROGRESS.md](PRODUCTION_REMEDIATION_PROGRESS.md))
- [x] Test suite import paths fixed (`src.` → `mnemocore.`)
- [x] Critical TODOs addressed or verified as safe
- [x] Config-service field alignment verified (no silent fallback-to-default bugs)
- [x] Thread safety audit completed (all mutations under locks)
- [x] Unused imports cleaned across all service files
- [x] Obsolete files and temp directories removed
- [x] Documentation updated (CHANGELOG, ARCHITECTURE, ROADMAP, README)

---

## 🔧 Resolved/Verified Items

The following items were previously listed as known limitations but have been verified as resolved or robustly handled:

1. **Qdrant Consolidation:** `src/core/tier_manager.py` implements `consolidate_warm_to_cold` with full Qdrant batch scrolling.
2. **Qdrant Search:** `src/core/engine.py` query pipeline correctly delegates to `TierManager.search` which queries Qdrant for WARM tier results.
3. **LLM Integration:** `src/llm_integration.py` includes `_mock_llm_response` fallbacks when no provider is configured, ensuring stability even without API keys.

---

## 📝 Remaining Roadmap Items (Non-Blocking)

### 1. `src/llm_integration.py` - Advanced LLM Features
- **Status:** Functional with generic providers.
- **Task:** Implement specific "OpenClaw" or "Gemini 3 Pro" adapters if required in future. Current implementation supports generic OpenAI/Anthropic/Gemini/Ollama clients.

### 2. Full Notion Integration
- **Status:** Not currently present in `src/mnemocore`.
- **Task:** Re-introduce `nightlab` or similar module if Notion support is needed in Phase 5.

---

## 📋 Pre-Release Actions

### Before git push:

```bash
# 1. Clean build artifacts
rm -rf .pytest_cache __pycache__ */__pycache__ *.pyc

# 2. Verify tests pass
# Note: Ensure you are in the environment where mnemocore is installed
python -m pytest

# 3. Verify import works
python -c "from mnemocore.core.engine import HAIMEngine; print('OK')"

# 4. Check for secrets (should return nothing)
grep -r "sk-" src/ --include="*.py"
grep -r "api_key.*=" src/ --include="*.py" | grep -v "api_key=\"\""

# 5. Initialize fresh data files
# Ensure data directory exists
mkdir -p data
touch data/memory.jsonl data/codebook.json data/concepts.json data/synapses.json
```

### Update README.md:

- [x] Add: "Beta Release - See RELEASE_CHECKLIST.md for known limitations"
- [x] Add: "Installation" section with `pip install -r requirements.txt`
- [x] Add: "Quick Start" example
- [x] Add: "Roadmap" section linking TODOs above

---

## 🚀 Release Command Sequence

```bash
# Verify clean state
git status

# Stage public files
git add LICENSE .gitignore RELEASE_CHECKLIST.md
git add src/ tests/ config.yaml requirements.txt pytest.ini pyproject.toml
git add README.md docker-compose.yml
git add data/.gitkeep  # If exists

# Commit
git commit -m "Release Candidate: All tests passing, critical TODOs resolved.

- Fixed test suite import paths (src -> mnemocore)
- Verified Qdrant consolidation and search implementation
- Confirmed LLM integration fallbacks"

# Tag
git tag -a v2.0.0 -m "Release v2.0.0"

# Push (safe, validates remote/version before push)
./scripts/ops/push_v2_safe.ps1
./scripts/ops/push_v2_safe.ps1 -DoPush
./scripts/ops/push_v2_safe.ps1 -DoPush -TagName v2.0.0
```

---

*Updated: 2026-02-18*
