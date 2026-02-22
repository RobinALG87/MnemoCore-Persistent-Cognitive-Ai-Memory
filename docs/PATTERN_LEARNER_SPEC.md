# MnemoCore Pattern Learner — Specification Draft

**Version:** 0.1-draft  
**Date:** 2026-02-20  
**Status:** Draft for Review  
**Author:** Omega (GLM-5) for Robin Granberg

---

## Executive Summary

Pattern Learner är en MnemoCore-modul som lär sig från användarinteraktioner **utan att lagra persondata**. Den extraherar statistiska mönster, topic clustering och kvalitetsmetrics som kan användas för att förbättra chatbot-performance över tid.

**Key principle:** Learn patterns, forget people.

---

## Problem Statement

### Healthcare Chatbot Challenges

| Utmaning | Konsekvens |
|----------|------------|
| GDPR/HIPAA compliance | Kan inte lagra konversationer |
| Multitenancy | Data får inte läcka mellan kliniker |
| Quality improvement | Behöver veta vad som fungerar |
| Knowledge gaps | Behöver identifiera vad som saknas i docs |

### Current Solutions (Limitations)

- **Stateless RAG:** Ingen inlärning alls
- **Full memory:** GDPR-risk, sekretessproblem
- **Manual analytics:** Tidskrävande, inte real-time

---

## Solution: Pattern Learner

### Core Concept

```
User Query ──► Anonymize ──► Extract Pattern ──► Aggregate
                  │
                  └── PII removed before storage
```

**What IS stored:**
- Topic clusters (anonymized)
- Query frequency distributions
- Response quality aggregates
- Knowledge gap indicators

**What is NOT stored:**
- User identities
- Clinic associations
- Patient data
- Raw conversations

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Pattern Learner Module                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Anonymizer │───►│Topic Extractor│───►│  Aggregator  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                    │          │
│         │                   ▼                    ▼          │
│         │           ┌──────────────┐    ┌──────────────┐   │
│         │           │Topic Embedder│    │ Stats Store  │   │
│         │           │  (MnemoCore) │    │  (Encrypted) │   │
│         │           └──────────────┘    └──────────────┘   │
│         │                   │                    │          │
│         └───────────────────┴────────────────────┘          │
│                             │                               │
│                             ▼                               │
│                    ┌──────────────┐                        │
│                    │  Insights API│                        │
│                    └──────────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Components

#### 1. Anonymizer

**Purpose:** Remove all PII before processing

**Methods:**
- Named Entity Recognition (NER) for person names
- Pattern matching for phone numbers, addresses
- Clinic/organization detection
- Session ID hashing

```python
class Anonymizer:
    """Remove PII from queries before pattern extraction"""
    
    def __init__(self):
        self.ner_model = load_ner_model("sv")  # Swedish
        self.patterns = {
            "phone": r"\+?\d{1,3}[\s-]?\d{2,4}[\s-]?\d{2,4}[\s-]?\d{2,4}",
            "email": r"[\w\.-]+@[\w\.-]+\.\w+",
            "personal_number": r"\d{6,8}[-\s]?\d{4}",
        }
    
    def anonymize(self, text: str) -> str:
        """Remove all PII from text"""
        
        # 1. NER for names
        entities = self.ner_model.extract(text)
        for entity in entities:
            if entity.type in ["PER", "ORG"]:
                text = text.replace(entity.text, "[ANON]")
        
        # 2. Pattern matching
        for pattern_type, pattern in self.patterns.items():
            text = re.sub(pattern, f"[{pattern_type.upper()}]", text)
        
        # 3. Remove clinic names (configurable blacklist)
        for clinic_name in self.clinic_blacklist:
            text = text.replace(clinic_name, "[KLINIK]")
        
        return text
```

---

#### 2. Topic Extractor

**Purpose:** Extract semantic topics from anonymized queries

**Methods:**
- Keyword extraction (TF-IDF)
- Topic modeling (LDA, BERTopic)
- Embedding-based clustering

```python
class TopicExtractor:
    """Extract topics from anonymized queries"""
    
    def __init__(self, mnemocore_engine):
        self.engine = mnemocore_engine
        self.topic_threshold = 0.5
    
    async def extract_topics(self, query: str) -> List[str]:
        """Extract topics from anonymized query"""
        
        # 1. Get keywords
        keywords = self._extract_keywords(query)
        
        # 2. Find similar topics in MnemoCore
        similar = await self.engine.query(query, top_k=5)
        
        # 3. Cluster into topics
        topics = []
        for memory_id, similarity in similar:
            if similarity > self.topic_threshold:
                memory = await self.engine.get_memory(memory_id)
                topics.extend(memory.metadata.get("topics", []))
        
        # 4. Deduplicate
        return list(set(topics + keywords))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using TF-IDF"""
        # Simple implementation
        words = text.lower().split()
        return [w for w in words if len(w) > 3 and w not in STOPWORDS_SV]
```

---

#### 3. Aggregator

**Purpose:** Store statistical patterns without PII

**Data structures:**

```python
@dataclass
class TopicStats:
    """Statistics for a topic"""
    topic: str
    count: int = 0
    first_seen: datetime = None
    last_seen: datetime = None
    trend: float = 0.0  # Recent increase/decrease

@dataclass
class ResponseQuality:
    """Aggregated response quality (no individual ratings)"""
    response_signature: str  # Hash of response template
    avg_rating: float = 0.5
    sample_count: int = 0
    last_updated: datetime = None

@dataclass
class KnowledgeGap:
    """Topics with no good answers"""
    topic: str
    query_count: int = 0
    failure_rate: float = 1.0  # % of queries that got "I don't know"
    suggested_action: str = ""  # "add documentation", "improve answer"
```

**Storage:**

```python
class PatternStore:
    """Store patterns (encrypted, no PII)"""
    
    def __init__(self, encryption_key: bytes):
        self.key = encryption_key
        self.topics: Dict[str, TopicStats] = {}
        self.qualities: Dict[str, ResponseQuality] = {}
        self.gaps: Dict[str, KnowledgeGap] = {}
    
    def record_topic(self, topic: str):
        """Record that a topic was queried"""
        if topic not in self.topics:
            self.topics[topic] = TopicStats(
                topic=topic,
                first_seen=datetime.utcnow()
            )
        
        stats = self.topics[topic]
        stats.count += 1
        stats.last_seen = datetime.utcnow()
    
    def record_quality(self, response_sig: str, rating: int):
        """Record response quality (aggregated)"""
        if response_sig not in self.qualities:
            self.qualities[response_sig] = ResponseQuality(
                response_signature=response_sig
            )
        
        q = self.qualities[response_sig]
        # Exponential moving average
        q.avg_rating = 0.9 * q.avg_rating + 0.1 * (rating / 5.0)
        q.sample_count += 1
        q.last_updated = datetime.utcnow()
    
    def record_gap(self, topic: str, had_answer: bool):
        """Record knowledge gap"""
        if topic not in self.gaps:
            self.gaps[topic] = KnowledgeGap(topic=topic)
        
        gap = self.gaps[topic]
        gap.query_count += 1
        if not had_answer:
            gap.failure_rate = (gap.failure_rate * (gap.query_count - 1) + 1) / gap.query_count
        else:
            gap.failure_rate = (gap.failure_rate * (gap.query_count - 1)) / gap.query_count
```

---

#### 4. Insights API

**Purpose:** Provide actionable insights to admins/developers

**Endpoints:**

```python
# GET /insights/topics?top_k=10
{
    "topics": [
        {"topic": "implantat", "count": 1250, "trend": 0.15},
        {"topic": "rotfyllning", "count": 980, "trend": -0.02},
        {"topic": "priser", "count": 850, "trend": 0.30}
    ],
    "period": "30d"
}

# GET /insights/gaps
{
    "knowledge_gaps": [
        {
            "topic": "tandreglering vuxna",
            "query_count": 145,
            "failure_rate": 0.85,
            "suggested_action": "add documentation"
        },
        {
            "topic": "akut tandvård",
            "query_count": 89,
            "failure_rate": 0.72,
            "suggested_action": "improve answer"
        }
    ]
}

# GET /insights/quality
{
    "top_responses": [
        {"signature": "abc123", "avg_rating": 4.8, "sample_count": 520},
        {"signature": "def456", "avg_rating": 4.5, "sample_count": 340}
    ],
    "worst_responses": [
        {"signature": "xyz789", "avg_rating": 2.1, "sample_count": 45}
    ]
}
```

---

## MnemoCore Integration

### Usage Pattern

```python
from mnemocore import HAIMEngine
from mnemocore.pattern_learner import PatternLearner

# Initialize MnemoCore (stores topic embeddings)
engine = HAIMEngine(dimension=16384)
await engine.initialize()

# Initialize Pattern Learner
learner = PatternLearner(
    engine=engine,
    encryption_key=get_encryption_key(),
    anonymizer=Anonymizer()
)

# Process a query (automatic learning)
async def handle_query(user_query: str, tenant_id: str):
    # 1. Anonymize
    anon_query = learner.anonymize(user_query)
    
    # 2. Extract patterns (no PII)
    topics = await learner.extract_topics(anon_query)
    
    # 3. Record topic usage
    for topic in topics:
        learner.record_topic(topic)
    
    # 4. Get answer from RAG
    answer = await rag_lookup(anon_query)
    
    # 5. Record if we had an answer
    learner.record_gap(
        topic=topics[0] if topics else "unknown",
        had_answer=(answer is not None)
    )
    
    return answer

# Get insights (admin only)
async def get_dashboard():
    top_topics = learner.get_top_topics(10)
    gaps = learner.get_knowledge_gaps()
    quality = learner.get_response_quality()
    
    return {
        "popular_topics": top_topics,
        "needs_documentation": gaps,
        "response_performance": quality
    }
```

---

## GDPR Compliance

### Data Minimization

| Data Type | Stored? | Justification |
|-----------|---------|---------------|
| Raw queries | ❌ | PII risk |
| User IDs | ❌ | Not needed |
| Session IDs | ❌ | Not needed |
| Clinic IDs | ❌ | Not needed |
| **Topic labels** | ✅ | Anonymized |
| **Topic counts** | ✅ | Statistical |
| **Quality scores** | ✅ | Aggregated |
| **Gap indicators** | ✅ | Anonymized |

### Right to Erasure (GDPR Art 17)

Since no PII is stored, right to erasure is **automatically satisfied**.

### Data Retention

```python
# Configurable retention
retention_policy = {
    "topic_stats": "365d",  # Keep for 1 year
    "quality_scores": "90d",  # Keep for 3 months
    "gap_indicators": "30d",  # Refresh monthly
}

# Automatic cleanup
async def cleanup_old_patterns():
    cutoff = datetime.utcnow() - timedelta(days=retention_policy["topic_stats"])
    for topic, stats in learner.topics.items():
        if stats.last_seen < cutoff:
            del learner.topics[topic]
```

---

## Security Considerations

### Encryption

- All pattern data encrypted at rest (AES-256)
- Encryption keys managed via HSM or Azure Key Vault
- Per-tenant encryption optional (for multi-tenant isolation)

### Access Control

```python
# Insights API requires admin role
@app.get("/insights/topics")
@require_role("admin")
async def get_topics():
    return learner.get_top_topics(10)
```

### Audit Logging

```python
# Log all pattern access (not the patterns themselves)
async def log_access(user_id: str, endpoint: str, timestamp: datetime):
    await audit_log.store({
        "user_id": user_id,
        "endpoint": endpoint,
        "timestamp": timestamp.isoformat(),
        # No pattern data logged
    })
```

---

## Implementation Roadmap

### Phase 1: MVP (2 weeks)

- [ ] Anonymizer with Swedish NER
- [ ] Basic topic extraction (keywords)
- [ ] Topic counter (no MnemoCore yet)
- [ ] Simple insights API

### Phase 2: MnemoCore Integration (2 weeks)

- [ ] Topic embedding storage in MnemoCore
- [ ] Semantic topic clustering
- [ ] Gap detection using similarity search

### Phase 3: Quality Metrics (2 weeks)

- [ ] Response quality tracking
- [ ] Feedback integration
- [ ] Quality dashboard

### Phase 4: Production Hardening (2 weeks)

- [ ] Encryption at rest
- [ ] Access control
- [ ] Audit logging
- [ ] Performance optimization

---

## Business Value

### For Healthcare Organizations

| Value | Metric |
|-------|--------|
| **Documentation gaps** | Know what to add to knowledge base |
| **Popular topics** | Prioritize documentation efforts |
| **Response quality** | Improve user satisfaction |
| **Trend analysis** | Identify emerging needs |

### For Opus Dental (Competitive Advantage)

| Advantage | Value |
|-----------|-------|
| **Continuous improvement** | Chatbot gets smarter without storing PII |
| **Customer insights** | Know what clinics need |
| **Compliance by design** | GDPR-safe from day 1 |
| **Unique selling point** | "Learning chatbot" vs competitors |

---

## Technical Requirements

### Dependencies

```
mnemocore>=4.5.0
spacy[sv]>=3.7.0  # Swedish NER
numpy>=1.24.0
cryptography>=41.0.0  # Encryption
```

### Infrastructure

- MnemoCore instance (can be shared or per-tenant)
- Encrypted storage (Azure SQL, PostgreSQL with TDE)
- Optional: Azure Key Vault for key management

### Performance

- Topic extraction: <50ms per query
- Insights API: <200ms
- Storage: ~1KB per unique topic (highly efficient)

---

## Open Questions

1. **Topic granularity:** How specific should topics be? "Implantat" vs "Implantat pris" vs "Implantat komplikationer"

2. **Trend detection:** What time window for trend analysis? 7d? 30d?

3. **Multi-language:** Support for Finnish/Norwegian in addition to Swedish?

4. **Tenant isolation:** Should patterns be shared across tenants (anonymized) or kept separate?

5. **Feedback mechanism:** How to collect ratings? Thumbs up/down? 1-5 stars?

---

## Conclusion

Pattern Learner enables **continuous improvement** of healthcare chatbots **without GDPR risk**. It learns what users ask about, which answers work, and where documentation is missing — all without storing any personal data.

**Key innovation:** Transform "memory" into "patterns" — compliance-safe learning.

---

## Next Steps

1. Review this spec
2. Decide on open questions
3. Prioritize MVP features
4. Start implementation

---

*Draft by Omega (GLM-5) for Robin Granberg*  
*2026-02-20*
