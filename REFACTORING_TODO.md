# Refactoring TODO

Status för kodoptimering inför kommande funktionalitet.

---

## Hög Prioritet

### 1. Konsolidera HDV-implementation
**Status:** ✅ Completed (2026-02-18)

**Problem:**
- Dubbel implementation: `src/core/hdv.py` (float) + `src/core/binary_hdv.py` (binary)
- Skapar branch-logik genom hela koden
- Float HDV är legacy och bör depreceras

**Åtgärder genomförda:**
- `src/core/hdv.py` - Markerad som DEPRECATED med varning
- `src/core/__init__.py` - Exporterar nu BinaryHDV först
- `src/core/node.py` - Använder endast BinaryHDV
- `src/core/holographic.py` - Konverterad till BinaryHDV med XOR-binding
- `src/core/router.py` - Tog bort HDV-branching
- `src/core/engine.py` - Tog bort Union-typer och branching
- `src/core/tier_manager.py` - Standardiserade på BinaryHDV

---

### 2. Ofullständiga features
**Status:** Pending

**Problem:**
- Flera TODOs i produktionskod som lämnats oimplementerade

**Filer:**
- `src/llm_integration.py`

**TODOs:**
```
Line 56:  # TODO: Call Gemini 3 Pro via OpenClaw API
Line 106: # TODO: superposition_query() not implemented in HAIMEngine
Line 131: # TODO: Call Gemini 3 Pro
Line 301: # TODO: Implement concept-to-memory-ID mapping
Line 320: # TODO: orchestrate_orch_or() not implemented
```

**Åtgärd:**
- Implementera funktionerna
- Eller ta bort dödkod

---

### 3. Standardisera felhantering
**Status:** Pending

**Problem:**
- Vissa funktioner returnerar `None` vid fel
- Andra kastar exceptions
- Svårt att förutse felbeteende

**Åtgärd:**
- Definiera domän-specifika exceptions:
  - `MemoryNotFoundError`
  - `StorageError`
  - `EncodingError`
  - `ConsolidationError`
- Skapa `src/core/exceptions.py`
- Uppdatera alla moduler att använda konsistent felhantering

---

## Medelprioritet

### 4. Minska Singleton-användning
**Status:** 📋 Roadmap

**Problem:**
- `AsyncRedisStorage.get_instance()`
- `QdrantStore.get_instance()`
- Försvårar testning

**Åtgärd:**
- Inför Dependency Injection
- Passa beroenden via konstruktor

**Komplexitet:** Hög - Kräver genomgripande ändringar av instansiering

---

### 5. Bryt isär stora funktioner
**Status:** 📋 Roadmap

**Problem:**
- `engine.py:store()` - 76 rader
- `tier_manager.py:consolidate_warm_to_cold()` - 48 rader

**Åtgärd:**
- Extrahera till mindre, testbara enheter

**Komplexitet:** Hög - Refaktorering av kärnlogik

---

### 6. Konsolidera Circuit Breakers
**Status:** ✅ Completed (2026-02-18)

**Problem:**
- `src/core/resilience.py` - pybreaker implementation
- `src/core/reliability.py` - Native implementation
- Dubbel implementation

**Åtgärder genomförda:**
- `src/core/reliability.py` - Nu primär modul med pre-konfigurerade instanser
- `src/core/resilience.py` - Markerad som DEPRECATED
- `src/core/qdrant_store.py` - Uppdaterad till reliability
- `src/api/main.py` - Uppdaterad till reliability, tog bort pybreaker-beroende

---

### 7. Centralisera hårkodade sökvägar
**Status:** ✅ Completed (2026-02-18)

**Problem:**
- `"./data"` fanns hårdkodat på flera ställen

**Åtgärder genomförda:**
- `src/core/holographic.py` - Använder nu `config.paths.data_dir` som default
- Alla sökvägar centraliserade i `config.yaml` och `HAIMConfig`

---

### 8. Standardisera import-stil
**Status:** ✅ Verified (2026-02-18)

**Problem:**
- Blandning av relativa och absoluta imports
- Till och med inom samma fil

**Analys:**
- `src/core/` använder konsekvent relativa imports (`.module`)
- Övriga moduler använder absoluta imports (`src.core.module`)
- Inga filer har blandad stil

**Slutsats:**
Import-stilen följer redan rekommenderad Python-praxis. Ingen åtgärd behövs.

---

## Låg prioritet

### 9. Rensa debug-filer
- Ta bort eller flytta `debug_*.py`
- Konsolidera test-helpers

### 10. Standardisera logging
- Välj ett framework (loguru rekommenderas)
- Ta bort ad-hoc print-statements

### 11. Förbättra typsäkerhet
- Lägg till mypy i CI
- Komplettera type hints
- Använd `TypedDict` för komplexa dict-returns

---

## Förbättra testtäckning

```bash
pytest --cov=src --cov-report=html
```

Kör för att identifiera luckor i testtäckningen.

---

## Fil-prioriteringslista

| Prioritet | Fil | Anledning |
|-----------|-----|-----------|
| 1 | `src/core/engine.py` | Kärnlogik, HDV dual-mode |
| 2 | `src/core/tier_manager.py` | Stora funktioner, lagringskomplexitet |
| 3 | `src/llm_integration.py` | Flera oimplementerade TODOs |
| 4 | `src/core/resilience.py` | Duplikat circuit breaker |
| 5 | `src/core/binary_hdv.py` | Överväg extrahering till separat paket |

---

## Framsteg

- [x] Punkt 1: HDV-konsolidering ✅
- [ ] Punkt 2: Ofullständiga features
- [ ] Punkt 3: Felhantering
- [ ] Punkt 4: Singleton-reduktion 📋 Roadmap
- [ ] Punkt 5: Stora funktioner 📋 Roadmap
- [x] Punkt 6: Circuit breakers ✅
- [x] Punkt 7: Hårkodade sökvägar ✅
- [x] Punkt 8: Import-stil ✅ (redan konsekvent)

---

## Roadmap (Framtida refaktorering)

Dessa punkter kräver mer omfattande ändringar och bör planeras in senare:

| Punkt | Beskrivning | Komplexitet |
|-------|-------------|-------------|
| 4 | Minska Singleton-användning, inför DI | Hög |
| 5 | Bryt isär stora funktioner i engine/tier_manager | Hög |
