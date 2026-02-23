# Embedding Version Registry - Phase 6.0

## Overview

The Embedding Version Registry provides comprehensive lifecycle management for embedding models, enabling seamless transitions between different vector encoding strategies without data loss.

## Architecture

### Core Components

1. **EmbeddingRegistry** - Central registry for all embedding models
2. **MigrationPlanner** - Creates and manages migration plans
3. **ReEmbeddingWorker** - Background worker for throttled re-embedding
4. **EmbeddingVersionManager** - High-level API for version management

### Data Models

#### EmbeddingModelSpec
Immutable specification identifying an embedding model:
- `model_id`: Unique model identifier (e.g., "binary_hdv", "sentence_transformer_all-MiniLM-L6-v2")
- `version`: Monotonically increasing version number
- `dimension`: Vector dimensionality
- `checksum`: SHA-256 hash of model configuration for verification

#### MigrationPlan
A complete migration plan for multiple nodes:
- Tasks with individual node migration status
- Progress tracking (percentage complete)
- Throttling configuration (batch size, delay)
- Rollback support

#### MigrationTask
Single node migration task:
- Source and target model specs
- Status (PENDING, IN_PROGRESS, COMPLETED, FAILED, ROLLED_BACK)
- Priority level (CRITICAL, HIGH, NORMAL, LOW)
- Retry tracking with error messages

## MemoryNode Integration

Each `MemoryNode` now includes embedding version fields:

```python
@dataclass
class MemoryNode:
    # ... existing fields ...
    embedding_model_id: str = "binary_hdv"
    embedding_version: int = 1
    embedding_checksum: str = ""
```

### New Methods

- `get_embedding_info()` - Get embedding metadata as dict
- `is_embedding_compatible(target_spec)` - Check if vector matches target
- `needs_migration(target_spec)` - Determine if migration is required
- `update_embedding_info(model_id, version, checksum)` - Update after re-embedding
- `embedding_qualified_id` - Property returning "{model_id}:v{version}:{checksum[:8]}"

## Configuration

New configuration section in `config.yaml`:

```yaml
embedding_registry:
  enabled: true
  registry_db_path: null  # Defaults to ./data/embedding_registry.sqlite
  auto_migrate: false
  migration_batch_size: 100
  migration_throttle_ms: 10
  max_parallel_workers: 2
  backup_old_vectors: true
  max_retries: 3
  worker_enabled: true
```

## Environment Variables

- `HAIM_EMBEDDING_REGISTRY_ENABLED` - Enable/disable registry
- `HAIM_EMBEDDING_REGISTRY_AUTO_MIGRATE` - Auto-migrate on model switch
- `HAIM_EMBEDDING_REGISTRY_MIGRATION_BATCH_SIZE` - Batch size for migrations
- `HAIM_EMBEDDING_REGISTRY_MIGRATION_THROTTLE_MS` - Delay between batches
- `HAIM_EMBEDDING_REGISTRY_MAX_PARALLEL_WORKERS` - Max concurrent workers
- `HAIM_EMBEDDING_REGISTRY_BACKUP_OLD_VECTORS` - Enable rollback support
- `HAIM_EMBEDDING_REGISTRY_MAX_RETRIES` - Max retry attempts
- `HAIM_EMBEDDING_REGISTRY_WORKER_ENABLED` - Background worker toggle

## Usage Examples

### Basic Model Registration

```python
from mnemocore.core import EmbeddingVersionManager

manager = EmbeddingVersionManager()

# Register a new embedding model
spec = manager.register_model(
    model_id="sentence_transformer_all-MiniLM-L6-v2",
    version=1,
    dimension=384,
    checksum="abc123...",
    name="All-MiniLM-L6-v2",
    description="Sentence Transformer multilingual model"
)
```

### Switching Active Model

```python
# Switch to a new model (with migration)
plan = manager.switch_active_model(
    target_model_id="new_model",
    target_version=2,
    migrate_existing=True
)

# Start the background worker
await manager.start_worker()
```

### Creating a Migration Plan

```python
plan = manager.create_migration(
    node_ids=["node1", "node2", "node3"],
    target_model_id="target_model",
    target_version=1,
    priority=Priority.HIGH
)

print(f"Migration plan: {plan.plan_id}")
print(f"Progress: {plan.progress_percent:.1f}%")
```

### Checking Node Compatibility

```python
from mnemocore.core import MemoryNode, verify_vector_compatibility

node = MemoryNode(id="test", hdv=hdv, content="test")
target_spec = EmbeddingModelSpec("model", 1, 1024, "hash")

if node.needs_migration(target_spec):
    print("Node needs migration")
```

### Rollback

```python
# Rollback a failed migration
success = await manager.rollback_migration(plan_id="plan123")

# Or rollback specific node
success = await manager.rollback_migration(
    plan_id="plan123",
    node_id="node456"
)
```

## Storage Integration

### Qdrant Payload Indexes

New indexes are created for filtering by embedding version:

```python
# Created automatically in QdrantStore.ensure_collections()
await client.create_payload_index(
    collection_name=collection,
    field_name="embedding_model_id",
    field_schema=PayloadSchemaType.KEYWORD,
)
await client.create_payload_index(
    collection_name=collection,
    field_name="embedding_version",
    field_schema=PayloadSchemaType.INTEGER,
)
```

## Database Schema

### models table
- `model_id` - Primary key
- `version` - Model version
- `dimension` - Vector dimensionality
- `checksum` - Configuration checksum
- `name` - Human-readable name
- `description` - Model description
- `created_at` - ISO timestamp
- `is_active` - Boolean flag

### migration_plans table
- `plan_id` - Primary key
- `source_model_id`, `source_version`, `source_checksum`
- `target_model_id`, `target_version`, `target_checksum`
- `status` - Migration status
- `total_nodes`, `completed_nodes`, `failed_nodes`
- `created_at`, `started_at`, `completed_at`
- `batch_size`, `throttle_delay_ms`, `max_parallel_workers`
- `plan_data` - JSON serialized plan

### migration_tasks table
- `task_id` - Primary key
- `plan_id` - Foreign key
- `node_id` - Memory node ID
- `source_model_id`, `source_version`, `source_checksum`
- `target_model_id`, `target_version`, `target_checksum`
- `status`, `priority`
- `created_at`, `started_at`, `completed_at`
- `error_message`, `retry_count`, `max_retries`
- `old_vector_backup` - BLOB for rollback
- `can_rollback` - Boolean flag

## API Exports

All components are exported from `mnemocore.core`:

```python
from mnemocore.core import (
    EmbeddingModelSpec,
    MigrationTask,
    MigrationPlan,
    MigrationStatus,
    Priority,
    EmbeddingRegistry,
    MigrationPlanner,
    ReEmbeddingWorker,
    EmbeddingVersionManager,
    create_vector_metadata,
    verify_vector_compatibility,
)
```

## Testing

Run tests with:

```bash
pytest tests/test_embedding_registry.py -v
```

Test coverage includes:
- Model spec creation and validation
- Migration plan and task management
- Registry operations
- Planner functionality
- Worker lifecycle
- MemoryNode integration
- Utility functions
