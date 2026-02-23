<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# skapa en omfattande md plan för sakerna som måste åtgärdas delge den här så ska jag skapa planneringen för det

Här är den fullständiga planen:

***

```markdown
# MnemoCore — Förbättrings- & Utvecklingsplan v1.0
> Skapad: 2026-02-23 | Status: Aktiv planering

---

## 📋 Innehållsförteckning
1. [Kritiska Refaktoreringar](#1-kritiska-refaktoreringar)
2. [Vektorsök & Datalagring](#2-vektorsök--datalagring)
3. [Minnesarkitektur & Kognition](#3-minnesarkitektur--kognition)
4. [Multi-Agent & Federering](#4-multi-agent--federering)
5. [Dreaming-modulen](#5-dreaming-modulen)
6. [Säkerhet, Privacy & GDPR](#6-säkerhet-privacy--gdpr)
7. [API & Integrationer](#7-api--integrationer)
8. [Observabilitet & Testning](#8-observabilitet--testning)
9. [SDK & Developer Experience](#9-sdk--developer-experience)
10. [Prioriteringsmatris](#10-prioriteringsmatris)

---

## 1. Kritiska Refaktoreringar

### 1.1 Dela upp `engine.py` (50KB — God Object)
**Problem:** En fil hanterar för många ansvarsområden.

**Åtgärd — dela i 3 moduler:**
- [ ] `engine_core.py` — Grundläggande store/retrieve/forget-operationer
- [ ] `engine_lifecycle.py` — Init, shutdown, health checks, garbage collection
- [ ] `engine_coordinator.py` — Orkestrering mellan sub-system och event-routing

**Krav:**
- [ ] Alla publika interface bevaras (bakåtkompatibelt)
- [ ] Dependency injection via befintlig `container.py`
- [ ] 100% testcoverage för varje ny modul
- [ ] Uppdatera `core/__init__.py` med nya exports

---

### 1.2 Dela upp `tier_manager.py` (47KB — God Object)
**Problem:** Tier-logik, eviction-policy och scoring är sammanblandade.

**Åtgärd — dela i 3 moduler:**
- [ ] `tier_storage.py` — Ren CRUD mot varje tier (working/episodic/semantic/procedural)
- [ ] `tier_eviction.py` — Eviction-policies: LRU, importance-weighted, decay-triggered
- [ ] `tier_scoring.py` — Poängsättning och promoteringsbeslut mellan tiers

**Krav:**
- [ ] Definiera ett `TierInterface` (ABC) som alla tiers implementerar
- [ ] Inga cirkulära importer
- [ ] Benchmarks för tier-transition-latens

---

### 1.3 Dela upp `subconscious_ai.py` (35KB)
**Problem:** Blandar backgroundprocessor, pattern-detection och inference.

**Åtgärd — dela i 3 moduler:**
- [ ] `subconscious_processor.py` — Background task runner och scheduling
- [ ] `subconscious_patterns.py` — Mönsterigenkänning och klusteranalys
- [ ] `subconscious_inference.py` — Slutledning och syntesoperationer

---

### 1.4 Generell kodkvalitet
- [ ] Sätt max filstorlek-lint-regel: **800 rader per fil**
- [ ] Inför `mypy --strict` för hela `src/`
- [ ] Ersätt alla `dict`-returtyper med `TypedDict` eller `dataclass`
- [ ] Ta bort alla `# type: ignore`-kommentarer och fixa korrekt
- [ ] Unified error handling — använd befintlig `exceptions.py` konsekvent

---

## 2. Vektorsök & Datalagring

### 2.1 Hybrid Search (Dense + Sparse)
**Problem:** Nuvarande `qdrant_store.py` använder bara dense vectors (semantisk sökning).  
**Konsekvens:** Exakta nyckelord och entitetssökning missar.

**Åtgärd:**
- [ ] Lägg till BM25/SPLADE sparse vector index i Qdrant
- [ ] Implementera `HybridSearchEngine` med reciprocal rank fusion (RRF)
- [ ] Konfigurerbar alpha-vikt: `alpha * dense + (1-alpha) * sparse`
- [ ] Exponera `hybrid_search(query, alpha=0.7)` i engine API

**Config:**
```yaml
search:
  mode: hybrid  # dense | sparse | hybrid
  hybrid_alpha: 0.7
  sparse_model: "naver/splade-cocondenser-ensemble-distil"
```


---

### 2.2 Embedding Version Registry

**Problem:** Om embedding-modell byts ut blir alla lagrade vektorer inkompatibla.

**Åtgärd — ny fil: `embedding_registry.py`**

- [ ] Varje vektor taggas med `embedding_model_id` + `embedding_version`
- [ ] `EmbeddingRegistry` håller en mapping av alla aktiva modeller
- [ ] `MigrationPlanner` genererar en re-embedding-plan vid modellbyte
- [ ] Background re-embedding-worker som processar i batchar (throttled)
- [ ] Rollback-stöd: behåll gamla vektorer tills migration är verifierad

**Schema-tillägg i MemoryNode:**

```python
embedding_model_id: str
embedding_version: int
embedding_checksum: str  # validering
```


---

### 2.3 Kompressionslagret för vektorer

- [ ] Implementera Product Quantization (PQ) för volymoptimering
- [ ] Scalar quantization (INT8) för minnen med lågt confidence-värde
- [ ] Auto-compress minnen äldre än konfigurerbar threshold

---

### 2.4 Backup \& Snapshotting

- [ ] Automatiska Qdrant-snapshots (schema + vektorer) schemalagda
- [ ] Inkrementell backup med WAL (Write-Ahead Log)
- [ ] `MemoryExporter` — exportera minnen till JSON/Parquet för portabilitet
- [ ] `MemoryImporter` med schema-validering och dedup

---

## 3. Minnesarkitektur \& Kognition

### 3.1 Rekonstruktivt Minne

**Problem:** Recall returnerar lagrade items direkt — inte biologiskt korrekt.
**Konsekvens:** Inget stöd för att "fylla i luckor" under recall.

**Åtgärd — ny fil: `memory_reconstructor.py`**

- [ ] `ReconstructiveRecall.recall(query)` — hämtar fragment och synthesiserar svar
- [ ] Integration med `gap_detector.py` och `gap_filler.py` (redan finns!)
- [ ] Confidence-score för rekonstruerade vs lagrade minnen
- [ ] Flagga rekonstruerade minnen med `is_reconstructed: bool`

---

### 3.2 Episodic Future Thinking (EFT)

**Problem:** Minnet är helt bakåtblickande — ingen prediktiv kapacitet.

**Åtgärd — ny fil: `future_thinking.py`**

- [ ] Bygg på befintlig `prediction_store.py` och `anticipatory.py`
- [ ] `EpisodeFutureSimulator.simulate(context)` — genererar troliga framtida scenarios
- [ ] Scenarios lagras i `prediction_store` med decay om de inte inträffar
- [ ] Integration med `attention.py` för att prioritera troliga scenarios

---

### 3.3 Context Window Prioritizer

**Problem:** `llm_integration.py` saknar strategi när kontextfönstret är fullt.

**Åtgärd — ny fil: `context_optimizer.py`**

- [ ] `ContextWindowPrioritizer.rank(memories, token_budget)`
- [ ] Ranking-formel: `score = relevance × recency_weight × importance × (1/token_cost)`
- [ ] Chunk-splitting för långa minnen med semantisk koherens
- [ ] Token counting med `tiktoken` per modell

---

### 3.4 Förbättra Forgetting Curve

**Problem:** `forgetting_curve.py` saknar individuell inlärningsprofil.

**Åtgärd:**

- [ ] Per-agent `LearningProfile` med personlig decay-konstant
- [ ] Spaced repetition-integration: minnen som retrieves förstärks (SM-2 algoritm)
- [ ] Emotionella minnen (via `emotional_tag.py`) får lägre decay — biologiskt korrekt
- [ ] `ForgettingAnalytics` — dashboard för att visualisera minnesförfall

---

### 3.5 Associationsnätverk

- [ ] Grafbaserad representation av minnesassociationer (NetworkX eller Neo4j)
- [ ] `AssociationStrengthener.reinforce(node_a, node_b)` vid co-retrieval
- [ ] Exportera kunskapsgrafen som GraphQL-endpoint
- [ ] Visualiseringsverktyg för associationsnät

---

## 4. Multi-Agent \& Federering

### 4.1 Shared Memory med Konflikthantering

**Problem:** Arkitekturen stödjer en agent. Multi-agent kräver delat minne.

**Åtgärd — ny fil: `federated_memory.py`**

- [ ] `SharedMemorySpace` med läs/skriv-access per agent
- [ ] CRDT-baserad merge för konkurrerande skrivningar (Last-Write-Wins + merge-strategies)
- [ ] Optimistic locking för kritiska minnesnoder
- [ ] `MemoryOwnership` — spåra vilken agent som skapade/äger ett minne

---

### 4.2 Memory Sharing Protocol

- [ ] `MemoryShareRequest` — agent A ber agent B dela ett specifikt minne
- [ ] Permissions-modell: `public | private | team | owner-only`
- [ ] Selective memory sync (bara minnen med relevans > threshold)
- [ ] Audit log för alla delningsoperationer

---

### 4.3 Konsensusminnen

- [ ] `ConsensusMechanism` — flera agenter röstar om sanningshalten i ett minne
- [ ] Quorum-baserad verifiering för viktiga faktaminnen
- [ ] `DisagreementStore` — loggar när agenter har motstridiga minnen

---

## 5. Dreaming-modulen

### 5.1 Offline Konsolideringsschema

**Bakgrund:** Det mest naturliga nästa steget — emulerar biologisk sömnkonsolidering.
**Kärna:** Under idle-tid kör bakgrundsprocesser som aktivt förbättrar minneskvaliteten.

**Åtgärd — ny fil: `dream_scheduler.py`**

- [ ] `IdleDetector` — detekterar när ingen aktiv interaktion sker (konfigurerbar threshold)
- [ ] `DreamSession` — en konsoliderings-cykel med start/stopp och loggning
- [ ] Schemalägg sessioner med cron-liknande syntax i `config.yaml`
- [ ] Graceful shutdown — avbryt dream-session om ny interaktion börjar

---

### 5.2 Dream Processing Pipeline

**Åtgärd — ny fil: `dream_pipeline.py`**

```
[Episodic Cluster] 
    → [Pattern Extractor]          # Hitta dolda samband
    → [Recursive Synthesizer]      # Bygg på befintlig recursive_synthesizer.py
    → [Contradiction Resolver]     # Använd befintlig contradiction.py
    → [Semantic Promoter]          # Flytta värdefulla insikter till semantic tier
    → [Dream Report Generator]     # Logg av vad som konsoliderades
```

**Steg:**

- [ ] `EpisodicClusterBuilder` — grupperar relaterade episodiska minnen
- [ ] `DreamPatternExtractor` — identifierar upprepade mönster och anomalier
- [ ] `ContradictionResolver` — löser sovande konflikter (bygger på `contradiction.py`)
- [ ] `SemanticPromoter` — automatisk tier-promotion av konsoliderat innehåll
- [ ] `DreamReportLogger` — sparar vad varje dream-session åstadkom

---

### 5.3 Dream Quality Metrics

- [ ] `DreamEfficiencyScore` — hur mycket nytt semantiskt innehåll skapades
- [ ] `ConsolidationRate` — episodiska minnen konsoliderade per timme
- [ ] `ContradictionResolutionRate` — lösta konflikter per session
- [ ] Exponera metrics till befintlig Grafana-dashboard

---

## 6. Säkerhet, Privacy \& GDPR


---

### 6.1 Kryptering

- [ ] **At-rest encryption**: AES-256 för alla minnesnoder i Qdrant
- [ ] **In-transit encryption**: mTLS för all kommunikation
- [ ] **Field-level encryption**: Kryptera specifika metadata-fält (t.ex. user_id)
- [ ] Key rotation utan driftstopp
- [ ] HSM-stöd (Hardware Security Module) för produktionsnyckelhantering

---

### 6.2 Access Control

- [ ] `MemoryACL` — Access Control List per minnesnod
- [ ] RBAC (Role-Based): `reader | writer | admin | owner`
- [ ] API-key rotation med expiry
- [ ] Rate limiting per API-nyckel med konfigurerbara quotas
- [ ] JWT-stöd med scope-baserade permissions

---

### 6.3 Säkerhetsaudit

- [ ] Dependency scanning (Dependabot / Snyk) i CI/CD
- [ ] SAST (Static Application Security Testing) — Bandit för Python
- [ ] Secret scanning — inga API-nycklar i kod
- [ ] Penetrationstestningsguide i `SECURITY.md`

---

## 7. API \& Integrationer

### 7.1 Komplettera MCP-verktyg

**Nuläge:** `mcp/server.py` exponerar redan flera viktiga verktyg (bl.a. `memory_store`, `memory_query`, `memory_delete`, `memory_stats`), men de drömspecifika och exporterande funktionerna saknas.

**Åtgärd — kompletta MCP tool definitions:**

- [x] `memory_store` — lagra ett minne med full metadata (Redan implementerad)
- [x] `memory_recall` (som `memory_query`) — semantisk sökning (Redan implementerad)
- [x] `memory_forget` (som `memory_delete`) — radera specificerat minne (Redan implementerad)
- [x] `memory_stats` — returnera statistik om minnessystemet (Redan implementerad)
- [ ] `memory_synthesize` — trigga explicit syntes
- [ ] `memory_dream` — manuellt trigga en dream-session
- [ ] `memory_export` — exportera minnen som JSON
- [ ] MCP-dokumentation för alla verktyg i `/docs/mcp-tools.md`

---

### 7.2 Webhook \& Event System

- [ ] `EventBus` — intern pubsub för minneshändelser
- [ ] Webhook-konfiguration: `on_consolidation`, `on_contradiction`, `on_dream_complete`
- [ ] Retry-logik för misslyckade webhook-leveranser (exponential backoff)
- [ ] Event schema i JSON Schema-format
- [ ] Webhook signature verification (HMAC)

---

### 7.3 GraphQL-endpoint

- [ ] Exposera kunskapsgrafen via GraphQL (Strawberry eller Ariadne)
- [ ] Queries: `memories(filter)`, `associations(node_id)`, `timeline(from, to)`
- [ ] Subscriptions: realtidsuppdateringar när minnen förändras
- [ ] Komplettera befintlig REST med GraphQL side-by-side

---

### 7.4 LangChain \& LlamaIndex-integration

- [ ] `MnemoCoreVectorStore` — LangChain-kompatibel VectorStore-adapter
- [ ] `MnemoCoreRetriever` — LlamaIndex-kompatibel Retriever
- [ ] Publicera som separata pip-paket: `mnemocore-langchain`, `mnemocore-llamaindex`
- [ ] Exempel-notebooks i `integrations/`

---

## 8. Observabilitet \& Testning

### 8.1 Utöka Testsviten

**Nuläge:** Tester finns men coverage är oklar.

**Mål: 85% coverage på all core-kod**

- [ ] Unit tests för varje ny modul (minimum 10 testfall per fil)
- [ ] Integration tests för tier-transition flöden
- [ ] Property-based testing med Hypothesis för HDV-operationer
- [ ] Chaos tests — vad händer om Qdrant är nere?
- [ ] Memory leak-tester för långkörande processer
- [ ] Performance regression tests med baseline-benchmark

---

### 8.2 Benchmarking

**Åtgärd — utöka `benchmarks/`**

- [ ] Latens-benchmark: store, recall, synthesize per tier
- [ ] Throughput-benchmark: minnen/sekund vid concurrent writes
- [ ] Memory footprint: RAM och disk per 10K/100K/1M minnen
- [ ] Jämförelse: MnemoCore vs MemGPT vs Zep vs LangMem
- [ ] Automatisk regression-alarm om latens ökar >10%

---

### 8.3 Grafana Dashboard

**Utöka befintligt `grafana-dashboard.json`:**

- [ ] Dream session visualisering (konsoliderings-progress)
- [ ] Tier distribution (hur minnen fördelar sig)
- [ ] Forgetting curve live-visualisering
- [ ] Contradiction rate över tid
- [ ] Embedding model usage per session

---

### 8.4 Distributed Tracing

- [ ] OpenTelemetry (OTel) integration
- [ ] Trace-id genom hela retrieve → recall → synthesize pipeline
- [ ] Jaeger/Tempo-kompatibel export
- [ ] Span-annotationer för varje kognitiv operation

---

## 9. SDK \& Developer Experience

### 9.1 Python SDK

**Nuläge:** MnemoCore (v4.5.1) finns redan uppsatt som Python-paket via `pyproject.toml` och pybreaker/hatchling är konfigurerat. Paketet har publicerats.

- [x] `pip install mnemocore` — standalone Python-paket (Redan implementerat och paketeringsklart)
- [x] Publishera till PyPI (Konfigurerat och publicerat)
- [ ] Typed client med autocomplete (Skapa en dedicerad `MnemoCoreClient` wrapper)
- [ ] Async-first design (`await client.recall(...)`) i klienten
- [ ] Context manager: `async with MnemoCore() as mc:`

---

### 9.2 JavaScript/TypeScript SDK

- [ ] `npm install @mnemocore/client`
- [ ] TypeScript definitions för alla typer
- [ ] Browser + Node.js-kompatibelt
- [ ] Publishera till npm

---

### 9.3 CLI-verktyg

- [ ] `mnemocore store "Robin gillar Python"` — lagra från terminal
- [ ] `mnemocore recall "vad gillar Robin?"` — söka från terminal
- [ ] `mnemocore dream --now` — trigga dream-session manuellt
- [ ] `mnemocore stats` — visa system-statistik
- [ ] `mnemocore export --format json > backup.json`

---

### 9.4 Dokumentation

- [ ] Komplett API-referens (autogenererad med Sphinx/MkDocs)
- [ ] "Getting Started in 5 minutes" guide
- [ ] Arkitekturdiagram (C4-modell nivå 1–3)
- [ ] Cognitive model-förklaring för icke-neuroscience-läsare
- [ ] Cookbook: 10 vanliga use cases med kodexempel
- [ ] Video-tutorial (screencasts) för grundläggande flöden

---

## 10. Prioriteringsmatris

| \# | Uppgift | Prioritet | Effekt | Komplexitet | Sprint |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | Dela engine.py → 3 moduler | 🔴 Kritisk | Hög | Medium | S1 |
| 2 | Dela tier_manager.py → 3 moduler | 🔴 Kritisk | Hög | Medium | S1 |
| 3 | Hybrid search (dense+sparse) | 🔴 Kritisk | Hög | Låg | S1 |
| 4 | Embedding version registry | 🔴 Kritisk | Hög | Medium | S1 |
| 5 | Context Window Prioritizer | 🟠 Hög | Hög | Medium | S2 |
| 7 | Dream Scheduler + Pipeline | 🟠 Hög | Unik | Medium | S2 |
| 8 | Dela subconscious_ai.py | 🟠 Hög | Medium | Medium | S2 |
| 9 | Nya MCP-verktyg (Dream/Synthesize) | 🟠 Hög | Hög | Låg | S2 |
| 10 | Kryptering at-rest | 🟠 Hög | Hög | Medium | S2 |
| 11 | Rekonstruktivt minne | 🟡 Medium | Hög | Hög | S3 |
| 12 | Grafbaserade associationer | 🟡 Medium | Hög | Hög | S3 |
| 13 | Multi-agent shared memory | 🟡 Medium | Hög | Hög | S3 |
| 14 | Python SDK (Klientförbättringar) | 🟡 Medium | Adoption | Låg | S3 |
| 15 | Webhook/Event system | 🟡 Medium | Medium | Medium | S3 |
| 16 | Episodic Future Thinking | 🟢 Låg | Unik | Hög | S4 |
| 17 | GraphQL endpoint | 🟢 Låg | Medium | Medium | S4 |
| 18 | TypeScript SDK | 🟢 Låg | Adoption | Medium | S4 |
| 19 | Distributed Tracing (OTel) | 🟢 Låg | Ops | Medium | S4 |
| 20 | Fullständig dokumentation | 🟢 Låg | Adoption | Låg | S4 |


---

## Sprint-definition (förslag)

```
Sprint 1 (S1) — Stabilisering & Search      ~ 3 veckor
Sprint 2 (S2) — Kognition & Säkerhet        ~ 3 veckor
Sprint 3 (S3) — Skalning & Ekosystem        ~ 4 veckor
Sprint 4 (S4) — Innovation & Polish         ~ 4 veckor

Total estimat: ~14 veckor (3.5 månader) till v2.0
```


---

## Definition of Done

För varje uppgift gäller:

- [ ] Kod skriven och PR skapad
- [ ] Code review godkänd
- [ ] Unit tests skrivna (täcker happy path + 2 error cases minimum)
- [ ] Befintliga tester passerar
- [ ] `mypy` rapporterar inga fel
- [ ] Dokumentation uppdaterad (docstrings + README om relevant)
- [ ] CHANGELOG.md uppdaterad

---

*Plan version 1.0 | Robin ALG87 | MnemoCore Persistent Cognitive AI Memory*

```

***

Planen är redo att klistras in direkt i repot som t.ex. `IMPROVEMENT_PLAN.md`. Den är strukturerad för att fungera direkt i GitHub Projects, Jira eller Notion — varje `- [ ]`-checkbox är en enskild task. Vill du att jag bryter ner någon specifik sprint till enskilda GitHub Issues-format, eller ska vi börja med att koda en specifik del?```

