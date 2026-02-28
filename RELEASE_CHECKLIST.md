# MnemoCore Release Checklist — v5.0.0

## Status: 🟢 GREEN

---

## ✅ Completed

- [x] LICENSE file (MIT)
- [x] .gitignore hardened (temp files, debug artifacts, build outputs)
- [x] data/memory.jsonl removed (no stored memories)
- [x] No leaked API keys or credentials
- [x] **1291+ unit tests passing** (Phase 5: 200+ tests, Phase 6: 85 tests)
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
git add LICENSE .gitignore RELEASE_CHECKLIST.md REFACTORING_TODO.md
git add src/ tests/ config.yaml requirements.txt pytest.ini pyproject.toml
git add README.md docker-compose.yml
git add data/.gitkeep  # If exists

# Commit
git commit -m "Release Candidate: All tests passing, critical TODOs resolved.

- Fixed test suite import paths (src -> mnemocore)
- Verified Qdrant consolidation and search implementation
- Confirmed LLM integration fallbacks"

# Tag
git tag -a v0.5.0-beta -m "Public Beta Release"

# Push
git push origin main --tags
```

---

*Updated: 2026-02-18*
