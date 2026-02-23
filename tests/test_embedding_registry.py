"""
Tests for Phase 6.0: Embedding Version Registry
================================================
"""

import asyncio
import sqlite3
import tempfile
from pathlib import Path

import pytest

from mnemocore.core.embedding_registry import (
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
from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.node import MemoryNode


class TestEmbeddingModelSpec:
    """Tests for EmbeddingModelSpec dataclass."""

    def test_create_spec(self):
        """Test creating a model specification."""
        spec = EmbeddingModelSpec(
            model_id="test_model",
            version=1,
            dimension=1024,
            checksum="abc123",
        )
        assert spec.model_id == "test_model"
        assert spec.version == 1
        assert spec.dimension == 1024
        assert spec.checksum == "abc123"

    def test_qualified_id(self):
        """Test qualified_id property."""
        spec = EmbeddingModelSpec(
            model_id="model",
            version=2,
            dimension=512,
            checksum="a" * 64,
        )
        qid = spec.qualified_id
        assert "model" in qid
        assert "v2" in qid
        assert qid.endswith("aaaaaaaa")

    def test_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        spec = EmbeddingModelSpec(
            model_id="test",
            version=1,
            dimension=256,
            checksum="xyz",
            name="Test Model",
            description="A test embedding model",
        )
        data = spec.to_dict()
        restored = EmbeddingModelSpec.from_dict(data)
        assert restored.model_id == spec.model_id
        assert restored.version == spec.version
        assert restored.dimension == spec.dimension
        assert restored.checksum == spec.checksum
        assert restored.name == spec.name
        assert restored.description == spec.description

    def test_validation(self):
        """Test validation constraints."""
        with pytest.raises(ValueError):
            EmbeddingModelSpec(
                model_id="",  # Empty model_id
                version=1,
                dimension=100,
                checksum="x",
            )

        with pytest.raises(ValueError):
            EmbeddingModelSpec(
                model_id="test",
                version=0,  # Version must be >= 1
                dimension=100,
                checksum="x",
            )

        with pytest.raises(ValueError):
            EmbeddingModelSpec(
                model_id="test",
                version=1,
                dimension=0,  # Dimension must be positive
                checksum="x",
            )


class TestMigrationTask:
    """Tests for MigrationTask dataclass."""

    def test_create_task(self):
        """Test creating a migration task."""
        source = EmbeddingModelSpec("m1", 1, 100, "a")
        target = EmbeddingModelSpec("m2", 1, 100, "b")

        task = MigrationTask(
            node_id="node123",
            source_spec=source,
            target_spec=target,
            priority=Priority.HIGH,
        )

        assert task.node_id == "node123"
        assert task.status == MigrationStatus.PENDING
        assert task.priority == Priority.HIGH
        assert task.retry_count == 0

    def test_task_serialization(self):
        """Test MigrationTask serialization."""
        source = EmbeddingModelSpec("m1", 1, 100, "a")
        target = EmbeddingModelSpec("m2", 1, 100, "b")

        task = MigrationTask(
            node_id="node1",
            source_spec=source,
            target_spec=target,
            status=MigrationStatus.IN_PROGRESS,
            priority=Priority.NORMAL,
        )

        data = task.to_dict()
        restored = MigrationTask.from_dict(data)

        assert restored.node_id == task.node_id
        assert restored.status == task.status
        assert restored.priority == task.priority


class TestMigrationPlan:
    """Tests for MigrationPlan dataclass."""

    def test_create_plan(self):
        """Test creating a migration plan."""
        source = EmbeddingModelSpec("m1", 1, 100, "a")
        target = EmbeddingModelSpec("m2", 1, 100, "b")

        plan = MigrationPlan(
            plan_id="plan1",
            source_spec=source,
            target_spec=target,
            batch_size=50,
        )

        assert plan.plan_id == "plan1"
        assert plan.total_nodes == 0
        assert plan.progress_percent == 0.0
        # Empty plan (no nodes) is considered complete
        assert plan.is_complete

    def test_add_tasks(self):
        """Test adding tasks to a plan."""
        source = EmbeddingModelSpec("m1", 1, 100, "a")
        target = EmbeddingModelSpec("m2", 1, 100, "b")

        plan = MigrationPlan(
            plan_id="plan1",
            source_spec=source,
            target_spec=target,
        )

        plan.add_task(MigrationTask("n1", source, target))
        plan.add_task(MigrationTask("n2", source, target))

        assert plan.total_nodes == 2
        assert len(plan.tasks) == 2

    def test_progress_tracking(self):
        """Test progress tracking in plans."""
        source = EmbeddingModelSpec("m1", 1, 100, "a")
        target = EmbeddingModelSpec("m2", 1, 100, "b")

        plan = MigrationPlan(
            plan_id="plan1",
            source_spec=source,
            target_spec=target,
        )

        for i in range(10):
            plan.add_task(MigrationTask(f"n{i}", source, target))

        # Mark 5 as complete
        for task in plan.tasks[:5]:
            task.status = MigrationStatus.COMPLETED
        plan.completed_nodes = 5

        assert plan.progress_percent == 50.0
        assert not plan.is_complete

        # Mark rest as complete
        for task in plan.tasks[5:]:
            task.status = MigrationStatus.COMPLETED
        plan.completed_nodes = 10

        assert plan.progress_percent == 100.0
        assert plan.is_complete


class TestEmbeddingRegistry:
    """Tests for EmbeddingRegistry."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        try:
            Path(db_path).unlink(missing_ok=True)
        except Exception:
            pass

    @pytest.fixture
    def registry(self, temp_db):
        """Create a registry with temp database."""
        return EmbeddingRegistry(db_path=temp_db, default_dimension=1024)

    def test_registry_initialization(self, registry):
        """Test registry creates default model."""
        active = registry.get_active_model()
        assert active is not None
        assert active.model_id == "binary_hdv"
        assert active.version == 1

    def test_register_model(self, registry):
        """Test registering a new model."""
        spec = EmbeddingModelSpec(
            model_id="new_model",
            version=1,
            dimension=512,
            checksum="abc123",
            name="New Model",
        )

        result = registry.register_model(spec, set_active=False)
        assert result is True

        retrieved = registry.get_model(spec.qualified_id)
        assert retrieved is not None
        assert retrieved.model_id == "new_model"

    def test_set_active_model(self, registry):
        """Test setting a model as active."""
        spec = EmbeddingModelSpec(
            model_id="active_model",
            version=1,
            dimension=256,
            checksum="xyz",
        )

        registry.register_model(spec, set_active=True)

        active = registry.get_active_model()
        assert active.model_id == "active_model"

    def test_list_models(self, registry):
        """Test listing all registered models."""
        models = registry.list_models()
        initial_count = len(models)

        # Add more models
        for i in range(3):
            spec = EmbeddingModelSpec(
                model_id=f"model_{i}",
                version=1,
                dimension=100,
                checksum=f"hash{i}",
            )
            registry.register_model(spec)

        models = registry.list_models()
        assert len(models) >= initial_count + 3

    def test_get_compatible_models(self, registry):
        """Test filtering models by dimensionality."""
        # Add models with different dimensions
        registry.register_model(EmbeddingModelSpec("d100", 1, 100, "a"))
        registry.register_model(EmbeddingModelSpec("d200", 1, 200, "b"))
        registry.register_model(EmbeddingModelSpec("d100_v2", 2, 100, "c"))

        compatible = registry.get_compatible_models(dimension=100)
        model_ids = {m.model_id for m in compatible}

        assert "d100" in model_ids
        assert "d100_v2" in model_ids
        assert "d200" not in model_ids


class TestMigrationPlanner:
    """Tests for MigrationPlanner."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name
        yield db_path
        try:
            Path(db_path).unlink(missing_ok=True)
        except Exception:
            pass

    @pytest.fixture
    def registry(self, temp_db):
        """Create a registry."""
        return EmbeddingRegistry(db_path=temp_db)

    @pytest.fixture
    def planner(self, registry):
        """Create a planner."""
        return MigrationPlanner(registry)

    def test_create_migration_plan(self, planner):
        """Test creating a migration plan."""
        target = EmbeddingModelSpec("target", 1, 1024, "target_hash")

        node_ids = ["node1", "node2", "node3"]
        plan = planner.create_migration_plan(
            target_spec=target,
            node_ids=node_ids,
            priority=Priority.HIGH,
        )

        assert plan is not None
        assert plan.target_spec.model_id == "target"
        assert len(plan.tasks) == 3
        assert plan.total_nodes == 3

    def test_no_migration_for_same_model(self, planner, registry):
        """Test that migration to same model is rejected."""
        active = registry.get_active_model()

        plan = planner.create_migration_plan(
            target_spec=active,
            node_ids=["node1"],
        )

        assert plan is None  # No plan created for same model

    def test_get_plan(self, planner):
        """Test retrieving a plan."""
        target = EmbeddingModelSpec("target", 1, 1024, "hash")
        plan = planner.create_migration_plan(
            target_spec=target,
            node_ids=["node1"],
        )

        retrieved = planner.get_plan(plan.plan_id)
        assert retrieved is not None
        assert retrieved.plan_id == plan.plan_id

    def test_update_task_status(self, planner):
        """Test updating task status."""
        target = EmbeddingModelSpec("target", 1, 1024, "hash")
        plan = planner.create_migration_plan(
            target_spec=target,
            node_ids=["node1", "node2"],
        )

        planner.update_task_status(
            plan.plan_id,
            "node1",
            MigrationStatus.COMPLETED,
        )

        updated = planner.get_plan(plan.plan_id)
        assert updated.completed_nodes == 1


class TestMemoryNodeEmbeddingFields:
    """Tests for MemoryNode embedding version fields."""

    def test_node_has_embedding_fields(self):
        """Test that MemoryNode has embedding fields."""
        hdv = BinaryHDV.random(dimension=1024)
        node = MemoryNode(
            id="test_node",
            hdv=hdv,
            content="test content",
        )

        assert hasattr(node, "embedding_model_id")
        assert hasattr(node, "embedding_version")
        assert hasattr(node, "embedding_checksum")

        # Check defaults
        assert node.embedding_model_id == "binary_hdv"
        assert node.embedding_version == 1

    def test_node_get_embedding_info(self):
        """Test get_embedding_info method."""
        hdv = BinaryHDV.random(dimension=1024)
        node = MemoryNode(
            id="test_node",
            hdv=hdv,
            content="test",
            embedding_model_id="custom_model",
            embedding_version=2,
            embedding_checksum="abc123",
        )

        info = node.get_embedding_info()
        assert info["embedding_model_id"] == "custom_model"
        assert info["embedding_version"] == 2
        assert info["embedding_checksum"] == "abc123"

    def test_node_is_embedding_compatible(self):
        """Test is_embedding_compatible method."""
        hdv = BinaryHDV.random(dimension=1024)
        node = MemoryNode(
            id="test_node",
            hdv=hdv,
            content="test",
            embedding_model_id="model_a",
            embedding_version=1,
            embedding_checksum="hash1",
        )

        # Compatible match
        assert node.is_embedding_compatible("model_a", 1, "hash1")

        # Different model
        assert not node.is_embedding_compatible("model_b", 1, "hash1")

        # Different version
        assert not node.is_embedding_compatible("model_a", 2, "hash1")

        # Different checksum
        assert not node.is_embedding_compatible("model_a", 1, "hash2")

    def test_node_needs_migration(self):
        """Test needs_migration method."""
        hdv = BinaryHDV.random(dimension=1024)
        node = MemoryNode(
            id="test_node",
            hdv=hdv,
            content="test",
            embedding_model_id="old_model",
            embedding_version=1,
            embedding_checksum="old_hash",
        )

        old_spec = EmbeddingModelSpec("old_model", 1, 1024, "old_hash")
        new_spec = EmbeddingModelSpec("new_model", 1, 1024, "new_hash")

        assert not node.needs_migration(old_spec)
        assert node.needs_migration(new_spec)

    def test_node_update_embedding_info(self):
        """Test update_embedding_info method."""
        hdv = BinaryHDV.random(dimension=1024)
        node = MemoryNode(
            id="test_node",
            hdv=hdv,
            content="test",
        )

        node.update_embedding_info("new_model", 3, "new_hash")

        assert node.embedding_model_id == "new_model"
        assert node.embedding_version == 3
        assert node.embedding_checksum == "new_hash"

    def test_node_qualified_id(self):
        """Test embedding_qualified_id property."""
        hdv = BinaryHDV.random(dimension=1024)
        node = MemoryNode(
            id="test_node",
            hdv=hdv,
            content="test",
            embedding_model_id="model",
            embedding_version=2,
            embedding_checksum="a" * 64,
        )

        qid = node.embedding_qualified_id
        assert "model" in qid
        assert "v2" in qid
        assert qid.endswith("aaaaaaaa")


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_vector_metadata(self):
        """Test create_vector_metadata function."""
        spec = EmbeddingModelSpec("model", 1, 100, "hash")
        metadata = create_vector_metadata(model_spec=spec)

        assert metadata["embedding_model_id"] == "model"
        assert metadata["embedding_version"] == "1"
        assert metadata["embedding_checksum"] == "hash"

    def test_verify_vector_compatibility(self):
        """Test verify_vector_compatibility function."""
        spec = EmbeddingModelSpec("model", 1, 100, "hash")
        metadata = {
            "embedding_model_id": "model",
            "embedding_version": "1",
            "embedding_checksum": "hash",
        }

        assert verify_vector_compatibility(metadata, spec) is True

        # Incompatible
        metadata["embedding_version"] = "2"
        assert verify_vector_compatibility(metadata, spec) is False


class TestEmbeddingVersionManager:
    """Tests for EmbeddingVersionManager."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name
        yield db_path
        try:
            Path(db_path).unlink(missing_ok=True)
        except Exception:
            pass

    @pytest.fixture
    def manager(self, temp_db):
        """Create a version manager."""
        return EmbeddingVersionManager(db_path=temp_db)

    def test_manager_initialization(self, manager):
        """Test manager creates registry and planner."""
        assert manager.registry is not None
        assert manager.planner is not None
        assert manager.worker is not None

    def test_register_model_via_manager(self, manager):
        """Test registering a model through the manager."""
        spec = manager.register_model(
            model_id="test_model",
            version=1,
            dimension=512,
            checksum="hash123",
            name="Test Model",
        )

        assert spec is not None
        assert spec.model_id == "test_model"

    def test_get_active_model_via_manager(self, manager):
        """Test getting active model through manager."""
        active = manager.get_active_model()
        assert active is not None

    def test_list_models_via_manager(self, manager):
        """Test listing models through manager."""
        models = manager.list_models()
        assert len(models) > 0

    def test_create_migration_via_manager(self, manager):
        """Test creating migration through manager."""
        # First register a target model
        target = manager.register_model(
            model_id="target_model",
            version=1,
            dimension=1024,
            checksum="target_hash",
        )

        plan = manager.create_migration(
            node_ids=["node1", "node2"],
            target_model_id="target_model",
            target_version=1,
        )

        assert plan is not None
        assert plan.total_nodes == 2


@pytest.mark.asyncio
class TestReEmbeddingWorker:
    """Tests for ReEmbeddingWorker async operations."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = f.name
        yield db_path
        try:
            Path(db_path).unlink(missing_ok=True)
        except Exception:
            pass

    @pytest.fixture
    def worker(self, temp_db):
        """Create a worker instance."""
        registry = EmbeddingRegistry(db_path=temp_db)
        planner = MigrationPlanner(registry)
        return ReEmbeddingWorker(registry, planner)

    async def test_worker_lifecycle(self, worker):
        """Test starting and stopping worker."""
        assert not worker._running

        await worker.start()
        assert worker._running

        await worker.stop()
        assert not worker._running

    async def test_worker_statistics(self, worker):
        """Test getting worker statistics."""
        stats = worker.get_statistics()
        assert "processed_count" in stats
        assert "failed_count" in stats
        assert "rolled_back_count" in stats
        assert "is_running" in stats
