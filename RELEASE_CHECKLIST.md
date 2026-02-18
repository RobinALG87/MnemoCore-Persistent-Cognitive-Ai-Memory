# MnemoCore Public Beta Release Checklist

## Status: ðŸŸ  ORANGE â†’ ðŸŸ¢ GREEN

---

## âœ… Completed

- [x] LICENSE file (MIT)
- [x] .gitignore created
- [x] data/memory.jsonl removed (no stored memories)
- [x] No leaked API keys or credentials
- [x] 82 unit tests passing

---

## ðŸ”§ Code TODOs (Known Limitations)

These are documented gaps that can ship as "Phase 4 roadmap" items:

### 1. `src/core/tier_manager.py:338`
```python
pass # TODO: Implement full consolidation with Qdrant
```
**Impact:** Warmâ†’Cold tier consolidation limited
**Workaround:** Hotâ†’Warm works, Cold is filesystem-based
**Fix:** Implement Qdrant batch scroll API for full archival

### 2. `src/core/engine.py:192`
```python
# TODO: Phase 3.5 Qdrant search for WARM/COLD
```
**Impact:** Query only searches HOT tier currently
**Workaround:** Promote memories before querying
**Fix:** Add async Qdrant similarity search in query()

### 3. `src/llm_integration.py:55-57, 128-129`
```python
# TODO: Call Gemini 3 Pro via OpenClaw API
reconstruction = "TODO: Call Gemini 3 Pro"
```
**Impact:** LLM reconstruction not functional
**Workaround:** Raw vector similarity works
**Fix:** Implement LLM client or make it pluggable

### 4. `src/nightlab/engine.py:339`
```python
# TODO: Notion API integration
```
**Impact:** Session documentation not auto-pushed
**Workaround:** Written to local markdown files
**Fix:** Add optional Notion connector

---

## ðŸ“‹ Pre-Release Actions

### Before git push:

```bash
# 1. Clean build artifacts
rm -rf .pytest_cache __pycache__ */__pycache__ *.pyc

# 2. Verify tests pass
source .venv/bin/activate && python -m pytest tests/ -v

# 3. Verify import works
python -c "from mnemocore.core.engine import HAIMEngine; print('OK')"

# 4. Check for secrets (should return nothing)
grep -r "sk-" src/ --include="*.py"
grep -r "api_key.*=" src/ --include="*.py" | grep -v "api_key=\"\""

# 5. Initialize fresh data files
touch data/memory.jsonl data/codebook.json data/concepts.json data/synapses.json
```

### Update README.md:

- [ ] Add: "Beta Release - See RELEASE_CHECKLIST.md for known limitations"
- [ ] Add: "Installation" section with `pip install -r requirements.txt`
- [ ] Add: "Quick Start" example
- [ ] Add: "Roadmap" section linking TODOs above

---

## ðŸš€ Release Command Sequence

```bash
cd /home/dev-robin/Desktop/mnemocore

# Verify clean state
git status

# Stage public files (exclude .venv)
git add LICENSE .gitignore RELEASE_CHECKLIST.md
git add src/ tests/ config.yaml requirements.txt pytest.ini
git add README.md studycase.md docker-compose.yml
git add data/.gitkeep  # If exists, or create empty dirs

# Commit
git commit -m "Initial public beta release (MIT)

Known limitations documented in RELEASE_CHECKLIST.md"

# Tag
git tag -a v0.1.0-beta -m "Public Beta Release"

# Push (when ready)
git push origin main --tags
```

---

## Post-Release

- [ ] Create GitHub repository
- [ ] Add repository topics: `vsa`, `holographic-memory`, `active-inference`, `vector-symbolic-architecture`
- [ ] Enable GitHub Issues for community feedback
- [ ] Publish whitepaper/blog post

---

*Generated: 2026-02-15*

