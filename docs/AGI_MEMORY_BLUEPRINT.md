# MnemoCore AGI Memory Blueprint
### Toward a True Cognitive Memory Substrate for Agentic Systems

> This document defines the **Phase 5 “AGI Memory” architecture** for MnemoCore – transforming it from a high‑end hyperdimensional memory engine into a **general cognitive substrate** for autonomous AI agents.

---

## 0. Goals & Non‑Goals

### 0.1 Core Goals

- Provide a **plug‑and‑play cognitive memory system** that any agent framework can mount as its “mind”:
  - Solves **context window** limits by offloading long‑term structure and recall.
  - Solves **memory management** by autonomous consolidation, forgetting, and self‑repair.
  - Provides **new thoughts, associations and suggestions** rather than only retrieval.
- Implement an explicit, formal model of:
  - **Working / Short‑Term Memory (WM/STM)**
  - **Episodic Memory**
  - **Semantic Memory**
  - **Procedural / Skill Memory**
  - **Meta‑Memory & Self‑Model**
- Maintain:
  - `pip install mnemocore` **zero‑infra dev mode** (SQLite / in‑process vector store).
  - Full infra path (Redis, Qdrant, k8s, MCP, OpenClaw live memory integration).[cite:436][cite:437]
- Provide **clean public APIs** (Python + HTTP + MCP) that:
  - Give agents a minimal but powerful surface: `observe / recall / reflect / propose_change`.
  - Are stable enough to build higher‑level frameworks on (LangGraph, AutoGen, OpenAI Agents, OpenClaw, custom stacks).

### 0.2 Non‑Goals

- MnemoCore is **not**:
  - En LLM eller policy‑generator.
  - En komplett agentram – det är **minnet + kognitiva processer**.
- MnemoCore ska **inte** hårdkoda specifika LLM‑providers.
  - LLM används via abstraherad integration (`SubconsciousAI`, `LLMIntegration`) så att byte av motor är trivialt.

---

## 1. Cognitive Architecture Overview

### 1.1 High‑Level Mental Model

Systemet ska exponera en internt konsekvent kognitiv modell:

- **Working Memory (WM)**  
  - Korttidsbuffert per agent / samtal / uppgift.
  - Håller aktuella mål, senaste steg, delresultat.
  - Living in RAM, med explicit API.

- **Episodic Memory (EM)**  
  - Sekvens av *episodes*: “agent X gjorde Y i kontext Z och fick utfallet U”.
  - Tidsstämplad, med länkar mellan episoder (kedjor).
  - Riktad mot “vad hände när, i vilken ordning”.

- **Semantic Memory (SM)**  
  - Abstraherade, konsoliderade representationer (concepts, prototypes).
  - Sammanfattningar av hundratals episoder → en “semantic anchor”.
  - Bra för svar på “vad vet jag generellt om X?”.

- **Procedural Memory (PM)**  
  - Skills, planer, recept: “för att lösa typ‑X problem, gör följande steg …”.
  - Kan hålla både mänsklig läsbar text och exekverbar kod (snippets, tools).

- **Meta‑Memory (MM)**  
  - Självmodell för MnemoCore själv: prestanda, reliability, konfiguration, kända svagheter.
  - Driver **självförbättringsloopen**.

Alla dessa lever ovanpå din befintliga HDV/VSA‑kärna, tier manager, synapse index, subconscious loop osv.[cite:436][cite:437]

### 1.2 New Core Services

Föreslagna nya Python‑moduler (under `src/mnemocore/core`):

- `memory_model.py`  
  - Typed dataklasser för WM/EM/SM/PM/MM entities.
- `working_memory.py`  
  - WM implementation per agent/task med snabb caching.
- `episodic_store.py`  
  - Episodisk tidsserie, sekvens‑API.
- `semantic_store.py`  
  - Wrapper ovanpå befintlig vektorstore (Qdrant/HDV/HNSW) + consolidation hooks.
- `procedural_store.py`  
  - Lagret för skills, scripts, tool definitions.
- `meta_memory.py`  
  - Självmodell, logik för self‑improvement proposals.
- `pulse.py`  
  - “Heartbeat”‑loop: driver subtle thoughts, consolidation ticks, gap detection, self‑reflection.
- `agent_profile.py`  
  - Persistent profil per agent: preferenser, styrkor/svagheter, quirks.

---

## 2. Data Model

### 2.1 Working Memory (WM)

```python
# src/mnemocore/core/memory_model.py

@dataclass
class WorkingMemoryItem:
    id: str
    agent_id: str
    created_at: datetime
    ttl_seconds: int
    content: str
    kind: Literal["thought", "observation", "goal", "plan_step", "meta"]
    importance: float  # 
    tags: list[str]
    hdv: BinaryHDV | None

@dataclass
class WorkingMemoryState:
    agent_id: str
    items: list[WorkingMemoryItem]
    max_items: int
```

**Invariantes:**

- WM är *liten* (t.ex. 32–128 items per agent).
- WM ligger primärt i RAM; kan serialiseras till Redis/SQLite för persistens.
- Access är O(1)/O(log n); LRU + importance‑vägning vid evicering.

### 2.2 Episodic Memory (EM)

```python
@dataclass
class EpisodeEvent:
    timestamp: datetime
    kind: Literal["observation", "action", "thought", "reward", "error"]
    content: str
    metadata: dict[str, Any]
    hdv: BinaryHDV

@dataclass
class Episode:
    id: str
    agent_id: str
    started_at: datetime
    ended_at: datetime | None
    goal: str | None
    context: str | None   # project / environment
    events: list[EpisodeEvent]
    outcome: Literal["success", "failure", "partial", "unknown"]
    reward: float | None
    links_prev: list[str]  # previous episode IDs
    links_next: list[str]  # next episode IDs
    ltp_strength: float
    reliability: float
```

### 2.3 Semantic Memory (SM)

```python
@dataclass
class SemanticConcept:
    id: str
    label: str                     # "fastapi-request-validation"
    description: str
    tags: list[str]
    prototype_hdv: BinaryHDV
    support_episode_ids: list[str] # episodes som gav upphov
    reliability: float
    last_updated_at: datetime
    metadata: dict[str, Any]
```

Kopplas direkt mot consolidation/semantic_consolidation + codebook/immunology.[cite:436][cite:437]

### 2.4 Procedural Memory (PM)

```python
@dataclass
class ProcedureStep:
    order: int
    instruction: str
    code_snippet: str | None
    tool_call: dict[str, Any] | None

@dataclass
class Procedure:
    id: str
    name: str
    description: str
    created_by_agent: str | None
    created_at: datetime
    updated_at: datetime
    steps: list[ProcedureStep]
    trigger_pattern: str           # "if user asks about X and Y"
    success_count: int
    failure_count: int
    reliability: float
    tags: list[str]
```

Procedurer kan genereras av LLM (SubconsciousAI), testas i episodiskt minne, och sedan promotas/demotas med reliability‑loop.

### 2.5 Meta‑Memory (MM)

```python
@dataclass
class SelfMetric:
    name: str         # "hot_tier_hit_rate", "avg_query_latency_ms"
    value: float
    window: str       # "5m", "1h", "24h"
    updated_at: datetime

@dataclass
class SelfImprovementProposal:
    id: str
    created_at: datetime
    author: Literal["system", "agent", "human"]
    title: str
    description: str
    rationale: str
    expected_effect: str
    status: Literal["pending", "accepted", "rejected", "implemented"]
    metadata: dict[str, Any]
```

MM lagras delvis i vanlig storage (SM/PM) men har egen API‑yta.

---

## 3. Service Layer Design

### 3.1 Working Memory Service

**Fil:** `src/mnemocore/core/working_memory.py`

Ansvar:

- Hålla en per‑agent WM‑state.
- Explicita operationer:
  - `push_item(agent_id, item: WorkingMemoryItem)`
  - `get_state(agent_id) -> WorkingMemoryState`
  - `clear(agent_id)`
  - `prune(agent_id)` – enligt importance + LRU.
- Integrera med engine/query:
  - Vid varje query: WM får en snapshot av top‑K resultat som “context items”.
  - Vid svar: agent kan markera vilka items som var relevanta.

### 3.2 Episodic Store Service

**Fil:** `src/mnemocore/core/episodic_store.py`

Ansvar:

- Skapa och uppdatera Episodes:
  - `start_episode(agent_id, goal, context) -> episode_id`
  - `append_event(episode_id, kind, content, metadata)`
  - `end_episode(episode_id, outcome, reward)`
- Query:
  - `get_episode(id)`
  - `get_recent(agent_id, limit, context)`
  - `find_similar_episodes(hdv, top_k)`
- Koppling till befintlig HDV + tier manager:
  - Varje Episode får en “episode_hdv” (bundle över event‑HDVs).
  - LTP + reliabilitet följer samma formel som övrig LTP.

### 3.3 Semantic Store Service

**Fil:** `src/mnemocore/core/semantic_store.py`

Ansvar:

- Hålla SemanticConcepts + codebook.
- API:
  - `upsert_concept(concept: SemanticConcept)`
  - `find_nearby_concepts(hdv, top_k)`
  - `get_concept(id)`
- Hookar mot:
  - `semantic_consolidation.py` → abstraktioner / anchors.
  - `immunology.py` → attractor cleanup.
  - `recursive_synthesizer.py` → djup konceptsyntes.

### 3.4 Procedural Store Service

**Fil:** `src/mnemocore/core/procedural_store.py`

Ansvar:

- Lagra och hämta Procedures.
- API:
  - `store_procedure(proc: Procedure)`
  - `get_procedure(id)`
  - `find_applicable_procedures(query, agent_id)`
  - `record_procedure_outcome(id, success: bool)`
- Integrera med:
  - SubconsciousAI → generera nya procedurer från pattern i EM/SM.
  - Reliability‑loopen → promota “verified” skills.

### 3.5 Meta Memory Service

**Fil:** `src/mnemocore/core/meta_memory.py`

Ansvar:

- Hålla SelfMetrics + SelfImprovementProposals.
- API:
  - `record_metric(metric: SelfMetric)`
  - `list_metrics(filter...)`
  - `create_proposal(...)`
  - `update_proposal_status(id, status)`
- Integrera med:
  - Pulse → skanna metrics och föreslå ändringar.
  - LLM → generera förslagstexter (“self‑reflection reports”).

---

## 4. Pulse & Subtle Thoughts

### 4.1 Pulse Definition

**Fil:** `src/mnemocore/core/pulse.py`

“Pulsen” är en central loop (async task, cron, eller k8s CronJob) som:

- Kör med konfigurerbart intervall (t.ex. var 10:e sekund–var 5:e minut).
- Har ett definierat set “ticks”:

```python
class PulseTick(Enum):
    WM_MAINTENANCE = "wm_maintenance"
    EPISODIC_CHAINING = "episodic_chaining"
    SEMANTIC_REFRESH = "semantic_refresh"
    GAP_DETECTION = "gap_detection"
    INSIGHT_GENERATION = "insight_generation"
    PROCEDURE_REFINEMENT = "procedure_refinement"
    META_SELF_REFLECTION = "meta_self_reflection"
```

Pulse orchestrerar:

- **WM_MAINTENANCE**  
  - Prune WM per agent.
  - Lyfta nyligen viktiga items (“keep in focus”).

- **EPISODIC_CHAINING**  
  - Skapa/länka episodiska sekvenser (prev/next).
  - “Temporala narrativ”.

- **SEMANTIC_REFRESH**  
  - Uppdatera semantic concepts baserat på nya episoder.
  - Trigga immunology cleanup för drift.

- **GAP_DETECTION**  
  - Kör `GapDetector` över EM/SM sista N minuter/timmar.
  - Producera strukturerade knowledge gaps.

- **INSIGHT_GENERATION**  
  - Kör SubconsciousAI/LLM över utvalda kluster.
  - Skapar nya SemanticConcepts, Procedures, eller MetaProposals.

- **PROCEDURE_REFINEMENT**  
  - Uppdatera reliability över PM.
  - Flagga outdated/farliga procedures.

- **META_SELF_REFLECTION**  
  - Sammanfattar senaste metriker, gap, failures → SelfImprovementProposals.

### 4.2 Pulse Implementation Sketch

```python
# src/mnemocore/core/pulse.py

class Pulse:
    def __init__(self, container, config):
        self.container = container
        self.config = config
        self._running = False

    async def start(self):
        self._running = True
        while self._running:
            start = datetime.utcnow()
            await self.tick()
            elapsed = (datetime.utcnow() - start).total_seconds()
            await asyncio.sleep(max(0, self.config.pulse_interval_seconds - elapsed))

    async def tick(self):
        await self._wm_maintenance()
        await self._episodic_chaining()
        await self._semantic_refresh()
        await self._gap_detection()
        await self._insight_generation()
        await self._procedure_refinement()
        await self._meta_self_reflection()
```

Konfiguration i `config.yaml`:

```yaml
haim:
  pulse:
    enabled: true
    interval_seconds: 30
    max_agents_per_tick: 50
    max_episodes_per_tick: 200
```

---

## 5. Agent‑Facing APIs (Python & HTTP & MCP)

### 5.1 High‑Level Python API

**Fil:** `src/mnemocore/agent_interface.py`

Syfte: ge agent‑kod ett ENKELT API:

```python
class CognitiveMemoryClient:
    def __init__(self, engine: HAIMEngine, wm, episodic, semantic, procedural, meta):
        ...

    # --- Observation & WM ---
    def observe(self, agent_id: str, content: str, **meta) -> str: ...
    def get_working_context(self, agent_id: str, limit: int = 16) -> list[WorkingMemoryItem]: ...

    # --- Episodic ---
    def start_episode(self, agent_id: str, goal: str, context: str | None = None) -> str: ...
    def append_event(self, episode_id: str, kind: str, content: str, **meta) -> None: ...
    def end_episode(self, episode_id: str, outcome: str, reward: float | None = None) -> None: ...

    # --- Semantic / Retrieval ---
    def recall(self, agent_id: str, query: str, context: str | None = None,
               top_k: int = 8, modes: tuple[str, ...] = ("episodic","semantic")) -> list[dict]: ...

    # --- Procedural ---
    def suggest_procedures(self, agent_id: str, query: str, top_k: int = 5) -> list[Procedure]: ...
    def record_procedure_outcome(self, proc_id: str, success: bool) -> None: ...

    # --- Meta / Self-awareness ---
    def get_knowledge_gaps(self, agent_id: str, lookback_hours: int = 24) -> list[dict]: ...
    def get_self_improvement_proposals(self) -> list[SelfImprovementProposal]: ...
```

### 5.2 HTTP Layer Additions

Utöver befintliga `/store`, `/query`, `/feedback`, osv.[cite:437]

Nya endpoints:

- `POST /wm/observe`
- `GET /wm/{agent_id}`
- `POST /episodes/start`
- `POST /episodes/{id}/event`
- `POST /episodes/{id}/end`
- `GET /episodes/{id}`
- `GET /agents/{agent_id}/episodes`
- `GET /agents/{agent_id}/context`
- `GET /agents/{agent_id}/knowledge-gaps`
- `GET /procedures/search`
- `POST /procedures/{id}/feedback`
- `GET /meta/proposals`
- `POST /meta/proposals`

### 5.3 MCP Tools

Utöka `mnemocore.mcp.server` med nya verktyg:

- `store_observation`
- `recall_context`
- `start_episode`, `end_episode`
- `query_memory`
- `get_knowledge_gaps`
- `get_self_improvement_proposals`

Så att Claude/GPT‑agenter kan:

- “Titta in” i agentens egen historik.
- Få WM + relevanta episoder + semantic concepts innan svar.
- Få gaps och self‑reflection prompts.

---

## 6. Self‑Improvement Loop

### 6.1 Loop Definition

Målet: MnemoCore ska **ständigt förbättra sig**:

1. Samlar **metrics** (performance + quality).
2. Upptäcker systematiska brister (höga felrates, gap‑clusters).
3. Genererar SelfImprovementProposals via LLM.
4. Låter människa eller meta‑agent granska & appliera.

### 6.2 Pipeline

1. **Metrics Collection**
   - Utnyttja befintlig `metrics.py` + Prometheus.[cite:436][cite:437]
   - Exempelmetriker:
     - `query_hit_rate`, `retrieval_latency_ms`
     - `feedback_success_rate`, `feedback_failure_rate`
     - `hot_tier_size`, `tier_promotion_rate`
     - `gap_detection_count`, `gap_fill_count`

2. **Issue Detection (Rule‑Based)**
   - Batchjobb (Pulse) kör enkla regler:
     - Om `feedback_failure_rate > X` för en viss tag (t.ex. “fastapi”) → skapa “knowledge area weak” flagg.
     - Om `hot_tier_hit_rate < threshold` → dålig context‑masking eller fel tuned thresholds.

3. **Proposal Generation (LLM)**
   - `SubconsciousAI` får inputs:
     - Metrics, knowledge gaps, failure cases, config snapshot.
   - Prompt genererar:
     - `SelfImprovementProposal.title/description/rationale`.

4. **Review & Execution**
   - API / UI för att lista proposals.
   - Människa/agent accepterar/rejectar.
   - Vid accept:
     - Kan trigga config ändringar (med patch PR).
     - Kan skapa GitHub issues/PR mallar.

### 6.3 API

- `GET /meta/proposals`
- `POST /meta/proposals/{id}/status`

---

## 7. Association & “Subtle Thoughts”

### 7.1 Association Engine

Målet: Systemet ska **själv föreslå**:

- Analogier (“det här liknar när vi gjorde X i annat projekt”).
- Relaterade koncept (“du pratar om Y, men Z har varit viktigt tidigare”).
- Långsiktiga teman och lärdomar.

Bygg vidare på:

- `synapse_index.py` (hebbian connections).[cite:436]
- `ripple_context.py` (kaskader).[cite:436]
- `recursive_synthesizer.py` (konceptsyntes).[cite:436]

Nya pattern:

- Vid varje Pulse:
  - Hämta senaste N episoder.
  - Kör k‑NN i semantic concept space.
  - Kör ripple over synapses.
  - Generera en uppsättning **CandidateAssociations**:

```python
@dataclass
class CandidateAssociation:
    id: str
    agent_id: str
    created_at: datetime
    source_episode_ids: list[str]
    related_concept_ids: list[str]
    suggestion_text: str
    confidence: float
```

Lagra i SM/EM så att agent/LLM kan hämta “subtle thoughts” innan svar:

- `GET /agents/{agent_id}/subtle-thoughts`

---

## 8. Storage Backends & Profiles

### 8.1 Profiles

Behåll pip‑enkelheten via profiler:

- **Lite Profile** (default, no extra deps):
  - WM: in‑process dict
  - EM: SQLite
  - SM: in‑process HDV + mmap
  - PM/MM: SQLite/JSON
- **Standard Profile**:
  - WARM: Redis
  - COLD: filesystem
- **Scale Profile**:
  - WARM: Redis
  - COLD: Qdrant (eller annan vector DB)
  - Optionellt: S3 archive

Konfigurationsexempel:

```yaml
haim:
  profile: "lite"  # "lite" | "standard" | "scale"
```

---

## 9. OpenClaw & External Agents

### 9.1 Designprincip för integration

För OpenClaw / liknande orchestrators:

- En agent definieras genom:
  - `agent_id`
  - `capabilities` (tools etc.)
- MnemoCore ska behandla `agent_id` som primär nyckel för:
  - WM
  - Episoder
  - Preferenser
  - Procedurer som agenten själv skapat

### 9.2 “Live Memory” Pattern

- När OpenClaw kör:
  - Varje observation → `observe(agent_id, content, meta)`  
  - Varje tool call / action → episod event.
  - Före varje beslut:
    - Hämta:
      - `WM`
      - `recent episodes`
      - `relevant semantic concepts`
      - `subtle thoughts` / associations
      - `knowledge gaps` (om agenten vill använda dessa som frågor).

---

## 10. Testing & Evaluation Plan

### 10.1 Unit & Integration Tests

Nya testfiler:

- `tests/test_working_memory.py`
- `tests/test_episodic_store.py`
- `tests/test_semantic_store.py`
- `tests/test_procedural_store.py`
- `tests/test_meta_memory.py`
- `tests/test_pulse.py`
- `tests/test_agent_interface.py`

Fokus:

- Invariantes (max WM size, LTP thresholds, reliability‑update).
- Episodic chaining korrekt.
- Semantic consolidation integration med nya SM‑API:t.
- Pulse tick ordering & time budget.

### 10.2 Behavioural Benchmarks

Skapa `benchmarks/AGI_MEMORY_SCENARIOS.md`:

- Multi‑session tasks där agent måste:
  - Minnas user preferences över dagar.
  - Lära sig av failed attempts (feedback).
  - Använda analogier över domäner.

Mät:

- Context reuse rate.
- Time‑to‑solve vs “no memory” baseline.
- Antal genererade self‑improvement proposals som faktiskt förbättrar outcomes.

---

## 11. Implementation Roadmap

### Phase 5.0 – Core Structure

1. Introduce `memory_model.py`, `working_memory.py`, `episodic_store.py`, `semantic_store.py`, `procedural_store.py`, `meta_memory.py`, `pulse.py`.
2. Wire everything in `container.py` (new providers).
3. Add `CognitiveMemoryClient` + minimal tests.

### Phase 5.1 – WM/EM/SM in Engine

4. Integrate WM into engine query/store paths.
5. Integrate EM creation in API (store/query/feedback).
6. Adapt semantic_consolidation/immunology to new SM service.

### Phase 5.2 – Procedural & Association

7. Implement procedural store + reliability integration.
8. Build association engine + subtle thoughts endpoints.

### Phase 5.3 – Self‑Improvement

9. Wire metrics → meta_memory → proposals via SubconsciousAI.
10. Add endpoints & optional small UI for proposals.

### Phase 5.4 – Hardening & Agents

11. Harden profiles (lite/standard/scale).
12. Build reference integrations (OpenClaw, LangGraph, AutoGen).

---

## 12. Developer Notes

- Håll **backwards compatibility** på API där det går:
  - Nya endpoints → prefix `v2` om nödvändigt.
  - Python API kan vara “ny high‑level layer” ovanpå befintlig `HAIMEngine`.
- All ny funktionalitet **feature‑flaggas i config**:
  - `haim.pulse.enabled`
  - `haim.episodic.enabled`
  - `haim.procedural.enabled`
  - etc.
- Strikt logging / metrics för allt nytt:
  - `haim_pulse_tick_duration_seconds`
  - `haim_wm_size`
  - `haim_episode_count`
  - `haim_procedure_success_rate`
  - `haim_self_proposals_pending`

---

*This blueprint is the contract between MnemoCore, its agents, and its contributors. The intention is to let autonomous AI agents, human developers, and MnemoCore itself co‑evolve toward a truly cognitive memory substrate – one that remembers, forgets, reflects, and grows.*
