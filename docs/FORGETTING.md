# MnemoCore Forgetting & Spaced Repetition System

> **Version**: 5.1.0 &nbsp;|&nbsp; **Source**: `src/mnemocore/core/forgetting/`

MnemoCore includes a biologically-inspired forgetting system based on Ebbinghaus's forgetting curve and the SM-2 (SuperMemo 2) spaced repetition algorithm. This system manages memory retention, schedules reviews, and supports per-agent learning profiles.

---

## Table of Contents

- [Overview](#overview)
- [SM-2 Algorithm](#sm-2-algorithm)
- [Learning Profiles](#learning-profiles)
- [Forgetting Curve Manager](#forgetting-curve-manager)
- [Review Scheduling](#review-scheduling)
- [Emotional Memory Integration](#emotional-memory-integration)
- [Analytics Dashboard](#analytics-dashboard)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)

---

## Overview

The forgetting system models how memories naturally decay and provides mechanisms to strengthen them:

```
New Memory (LTP = 0.5)
    │
    ▼
Forgetting Curve (exponential decay)
    │
    ├── Retention < threshold? ──> Schedule Review
    │                                    │
    │                              Review Quality (0–5)
    │                                    │
    │                              SM-2 Update
    │                                    │
    │                              New Interval + EF
    │
    ├── Emotional Memory? ──> Slower Decay (50% reduction)
    │
    ├── Retention < min_eig? ──> Candidate for Eviction
    │
    └── Retention > permanence? ──> Permanent (no decay)
```

---

## SM-2 Algorithm

The SuperMemo 2 algorithm schedules reviews based on recall quality. Each memory maintains an SM-2 state:

### SM-2 State

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `memory_id` | `str` | — | Memory identifier |
| `repetitions` | `int` | `0` | Number of successful reviews |
| `interval` | `float` | `0.0` | Current review interval (days) |
| `easiness_factor` | `float` | `2.5` | Ease of recall (min 1.3) |
| `last_review_quality` | `int` | `0` | Last review grade (0–5) |
| `last_review_date` | `datetime?` | `None` | When last reviewed |
| `next_review_date` | `datetime?` | `None` | When next review is due |

### Review Quality Scale

| Grade | Name | Description |
|-------|------|-------------|
| **0** | Complete Blackout | No memory of the item |
| **1** | Incorrect, but Recognized | Wrong answer, but recognized the content |
| **2** | Incorrect, Easy Recall | Wrong answer, but it felt familiar |
| **3** | Correct, Difficult | Correct answer with significant effort |
| **4** | Correct, Hesitation | Correct answer with some hesitation |
| **5** | Perfect Recall | Instant, confident recall |

### Interval Calculation

After each review, the next interval is calculated as:

$$I(n) = \begin{cases} 1 & \text{if } n = 1 \\ 6 & \text{if } n = 2 \\ I(n-1) \times EF & \text{if } n > 2 \end{cases}$$

Where $EF$ (Easiness Factor) is adjusted after each review:

$$EF' = EF + (0.1 - (5 - q) \times (0.08 + (5 - q) \times 0.02))$$

$EF$ is clamped to a minimum of 1.3. If quality $q < 3$, repetitions reset to 0.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `SM2_MIN_EASINESS` | `1.3` | Minimum easiness factor |
| `SM2_DEFAULT_EASINESS` | `2.5` | Default easiness factor |
| `SM2_QUALITY_MIN` | `0` | Minimum review quality |
| `SM2_QUALITY_MAX` | `5` | Maximum review quality |

---

## Learning Profiles

Each agent can have a personalized learning profile that modifies decay rates, SM-2 parameters, and review frequency.

### Profile Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_id` | `str` | — | Agent identifier |
| `base_decay` | `float` | `1.0` | Base decay rate multiplier |
| `easiness_factor` | `float` | `2.5` | Agent-specific EF |
| `review_frequency_multiplier` | `float` | `1.0` | Review frequency modifier |
| `emotional_sensitivity` | `float` | `0.7` | How much emotions affect retention |
| `learning_style` | `str` | `"balanced"` | Learning style preset |
| `sm2_min_easiness` | `float` | `1.3` | Agent-specific min EF |
| `sm2_interval_modifier` | `float` | `1.0` | Interval scaling factor |

### Learning Styles

| Style | Description | Decay | Review Freq |
|-------|-------------|-------|-------------|
| `"fast"` | Rapid learner, shorter reviews | 0.8× | 1.2× |
| `"slow"` | Gradual learner, more repetition | 1.2× | 0.8× |
| `"visual"` | Visual learner (balanced) | 1.0× | 1.0× |
| `"analytical"` | Methodical, thorough reviews | 0.9× | 1.1× |
| `"balanced"` | Default balanced approach | 1.0× | 1.0× |

### Creating Profiles

```python
from mnemocore.core.forgetting.profile import LearningProfile

# Using factory method
profile = LearningProfile.for_agent("agent-01", learning_style="fast")

# Custom profile
profile = LearningProfile(
    agent_id="agent-02",
    base_decay=0.9,
    emotional_sensitivity=0.8,
    learning_style="analytical"
)
```

---

## Forgetting Curve Manager

The `ForgettingCurveManager` orchestrates all forgetting, review scheduling, and SM-2 operations.

### Initialization

```python
from mnemocore.core.forgetting.manager import ForgettingCurveManager

manager = ForgettingCurveManager(
    engine=haim_engine,
    target_retention=0.70,           # Target 70% retention
    min_eig_to_consolidate=0.3,      # Min EIG for consolidation
    persistence_path="./data/forgetting_state.json"
)
```

### Key Operations

```python
# Get or create a learning profile for an agent
profile = manager.get_or_create_profile("agent-01", learning_style="balanced")

# Update SM-2 state after a review
sm2_state = manager.update_sm2_state(
    memory_id="mem_123",
    quality=4,           # Correct with hesitation
    agent_id="agent-01"
)

# Calculate current retention for a memory
retention = manager.calculate_sm2_retention("mem_123")

# Get the number of days until next review
days = manager.next_review_days(memory_node, profile)

# Schedule reviews for a batch of memories
entries = manager.schedule_reviews(memory_nodes, agent_id="agent-01")

# Get due reviews
due = manager.get_due_reviews(agent_id="agent-01", limit=50)

# Run a full forgetting cycle
stats = await manager.run_once(agent_id="agent-01")

# Record a review result
sm2_state = await manager.record_review_result(
    memory_id="mem_123",
    quality=5,  # Perfect recall
    agent_id="agent-01"
)
```

---

## Review Scheduling

Reviews are represented as `ReviewEntry` objects:

| Field | Type | Description |
|-------|------|-------------|
| `memory_id` | `str` | Memory to review |
| `agent_id` | `str` | Agent responsible |
| `due_at` | `datetime` | When the review is due |
| `current_retention` | `float` | Current retention level |
| `stability` | `float` | Memory stability |
| `sm2_state` | `SM2State?` | Full SM-2 state |
| `emotional_salience` | `float` | Emotional weight |
| `action` | `str` | Action type |

### Review Actions

| Action | Description |
|--------|-------------|
| `review` | Standard review — test recall and update SM-2 |
| `consolidate` | Memory is strong enough to consolidate/merge |
| `evict` | Memory retention too low — candidate for removal |
| `boost` | Memory needs reinforcement beyond standard review |

### Querying the Schedule

```python
# Get full schedule for an agent
schedule = manager.get_schedule(agent_id="agent-01")

# Get only due reviews
due = manager.get_due_reviews(agent_id="agent-01", limit=20)

# Get entries by action type
to_evict = manager.get_actions_by_type("evict", agent_id="agent-01")
to_consolidate = manager.get_actions_by_type("consolidate")

# Remove a memory from the schedule
manager.remove_entry("mem_123")
```

---

## Emotional Memory Integration

Emotional memories decay slower, reflecting how emotionally significant experiences persist longer in human memory.

### Emotional Decay Modifiers

| Condition | Effect |
|-----------|--------|
| High salience (> 0.5) | Decay reduced by 50% |
| Low salience | Standard decay rate |
| Agent emotional sensitivity | Scales the emotional effect |

### How It Works

```python
# Check if a memory is emotional
is_emotional = manager.is_emotional_memory(memory_node)

# Get the emotional salience score
salience = manager.get_emotional_salience(memory_node)

# Apply emotional modifier to retention
adjusted_retention = manager.apply_emotional_decay_modifier(
    retention=0.65,
    node=memory_node,
    profile=agent_profile
)
```

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `EMOTIONAL_DECAY_REDUCTION` | `0.5` | Decay rate reduction for emotional memories |
| `HIGH_SALIENCE_THRESHOLD` | `0.5` | Threshold for "emotional" classification |

---

## Analytics Dashboard

The `ForgettingAnalytics` module provides visualization data and performance metrics.

### Dashboard Data

```python
from mnemocore.core.forgetting.analytics import ForgettingAnalytics

analytics = ForgettingAnalytics(manager=manager, engine=engine)

# Get full dashboard data
dashboard = analytics.get_dashboard_data(agent_id="agent-01")
# Returns: {
#   "retention_curve": [...],
#   "review_summary": {...},
#   "sm2_performance": {...},
#   "emotional_distribution": {...},
#   "schedule_stats": {...}
# }
```

### Retention Curve Export

```python
# Export retention curve as CSV
csv_data = analytics.export_retention_curve_csv(agent_id="agent-01")
# "time_days,retention,memory_count\n0.0,1.0,100\n1.0,0.85,95\n..."
```

### Agent Comparison

```python
# Compare learning performance across agents
comparison = analytics.get_agent_comparison()
# Returns per-agent: total_memories, avg_retention, review_count, ...
```

### Learning Progress

```python
# Track learning progress over time
progress = analytics.get_learning_progress_chart(
    agent_id="agent-01",
    memory_id="mem_123"  # Optional: specific memory
)
```

---

## Configuration

### Forgetting Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `TARGET_RETENTION` | `0.70` | Default target retention level |
| `MIN_EIG_TO_CONSOLIDATE` | `0.3` | Min EIG for consolidation eligibility |
| `ANALYTICS_HISTORY_SIZE` | `1000` | Max analytics history entries |

### In config.yaml

The forgetting system uses LTP config values from the main configuration:

```yaml
ltp:
  initial_importance: 0.5      # Starting LTP for new memories
  decay_lambda: 0.01           # Exponential decay rate
  permanence_threshold: 0.95   # "Permanent memory" LTP threshold
  half_life_days: 30.0         # Decay half-life

self_improvement:
  enabled: false               # Self-improvement also uses forgetting data
  dry_run: true
```

---

## Usage Examples

### Basic Review Cycle

```python
from mnemocore.core.forgetting.manager import ForgettingCurveManager

manager = ForgettingCurveManager(engine=engine)

# Create a profile for an agent
profile = manager.get_or_create_profile("student-01", learning_style="balanced")

# Schedule reviews for all memories
entries = manager.schedule_reviews(engine.get_all_nodes(), agent_id="student-01")

# Get due reviews
due = manager.get_due_reviews(agent_id="student-01")

for entry in due:
    print(f"Review: {entry.memory_id} (retention: {entry.current_retention:.2f})")
    
    # After testing recall:
    quality = 4  # e.g., correct with hesitation
    await manager.record_review_result(entry.memory_id, quality, "student-01")
```

### Automated Forgetting Cycle

```python
# Run a full automated cycle (schedule + update + evict)
stats = await manager.run_once(agent_id="student-01")
print(f"Reviewed: {stats.get('reviewed', 0)}")
print(f"Evicted: {stats.get('evicted', 0)}")
print(f"Consolidated: {stats.get('consolidated', 0)}")
```

### Multi-Agent Learning Comparison

```python
analytics = ForgettingAnalytics(manager=manager, engine=engine)

# Create different profiles
manager.get_or_create_profile("fast-learner", learning_style="fast")
manager.get_or_create_profile("slow-learner", learning_style="slow")

# After some review cycles, compare
comparison = analytics.get_agent_comparison()
for agent in comparison:
    print(f"{agent['agent_id']}: avg retention = {agent['avg_retention']:.2f}")
```

---

*See [GLOSSARY.md](GLOSSARY.md) for term definitions. See [CONFIGURATION.md](CONFIGURATION.md) for all settings.*
