"""
Concurrency Tests for File Safety
==================================

Tests for file persistence safety under concurrent access. Validates that
concurrent save operations do not corrupt files and that atomic writes
work correctly.

Test Categories:
    - Concurrent holographic.save() -> file not corrupted
    - Concurrent strategy_bank._persist_to_disk() -> file valid
    - Concurrent procedural_store._persist_to_disk() -> file valid
"""

import asyncio
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from mnemocore.core.holographic import ConceptualMemory
from mnemocore.core.binary_hdv import BinaryHDV
from mnemocore.core.strategy_bank import (
    StrategyBankService,
    Strategy,
    StrategyOutcome,
    RetrievalJudge,
)
from mnemocore.core.procedural_store import ProceduralStoreService
from mnemocore.core.memory_model import Procedure, ProcedureStep
from mnemocore.core.config import StrategyBankConfig, ProceduralConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary directory for test storage."""
    storage_dir = tmp_path / "file_safety_test"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@pytest.fixture
def conceptual_memory(temp_storage_dir):
    """Create a ConceptualMemory instance for testing."""
    memory = ConceptualMemory(
        dimension=1024,  # Smaller for faster tests
        storage_dir=str(temp_storage_dir)
    )
    yield memory


@pytest.fixture
def strategy_bank(temp_storage_dir):
    """Create a StrategyBankService with persistence enabled."""
    config = MagicMock()
    config.max_strategies = 1000
    config.max_outcomes_per_strategy = 100
    config.target_negative_ratio = 0.4
    config.min_confidence_threshold = 0.3
    config.persistence_path = str(temp_storage_dir / "strategies.json")
    config.auto_persist = False  # We'll control persistence manually
    config.flush_interval_seconds = 1.0

    bank = StrategyBankService(config=config)
    yield bank
    bank.close()


@pytest.fixture
def procedural_store(temp_storage_dir):
    """Create a ProceduralStoreService with persistence enabled."""
    config = MagicMock()
    config.persistence_path = str(temp_storage_dir / "procedures.json")
    config.min_reliability_threshold = 0.1
    config.reliability_boost_on_success = 0.05
    config.reliability_penalty_on_failure = 0.10

    store = ProceduralStoreService(config=config)
    yield store


# =============================================================================
# Test: Concurrent Holographic Save
# =============================================================================

class TestConcurrentHolographicSave:
    """
    Tests for concurrent ConceptualMemory save operations.

    Validates that:
    - Concurrent save() calls don't corrupt the file
    - Atomic writes prevent partial data
    - File remains readable after concurrent saves
    """

    @pytest.mark.asyncio
    async def test_conceptual_memory_concurrent_saves_no_corruption(
        self, conceptual_memory, temp_storage_dir
    ):
        """
        Test that concurrent save() operations don't corrupt the file.

        Multiple threads/coroutines saving at the same time should not
        result in corrupted JSON data.
        """
        num_saves = 50

        async def save_concepts(batch_id: int):
            """Store concepts and trigger save."""
            for i in range(5):
                conceptual_memory.store_concept(
                    f"concept_{batch_id}_{i}",
                    {
                        f"attr_{batch_id}": f"value_{i}",
                        "batch": str(batch_id)
                    }
                )
            # Trigger synchronous save
            conceptual_memory.save()

        # Run concurrent save operations
        tasks = [save_concepts(i) for i in range(num_saves)]
        await asyncio.gather(*tasks)

        # Verify file is valid JSON
        codebook_path = os.path.join(
            conceptual_memory.storage_dir, "codebook.json"
        )
        concepts_path = os.path.join(
            conceptual_memory.storage_dir, "concepts.json"
        )

        # Files should exist and be valid JSON
        assert os.path.exists(codebook_path), "Codebook file not created"
        assert os.path.exists(concepts_path), "Concepts file not created"

        with open(codebook_path, 'r') as f:
            codebook_data = json.load(f)
        assert isinstance(codebook_data, dict)

        with open(concepts_path, 'r') as f:
            concepts_data = json.load(f)
        assert isinstance(concepts_data, dict)

    @pytest.mark.asyncio
    async def test_conceptual_memory_atomic_write_consistency(
        self, conceptual_memory, temp_storage_dir
    ):
        """
        Test that atomic writes maintain file consistency.

        Even under concurrent writes, the file should always contain
        either the old or new complete data, never partial data.
        """
        concepts_path = os.path.join(
            conceptual_memory.storage_dir, "concepts.json"
        )

        async def write_and_verify(iteration: int):
            """Write concepts and verify file is valid."""
            # Store a unique concept
            conceptual_memory.store_concept(
                f"atomic_test_{iteration}",
                {"iteration": str(iteration), "timestamp": datetime.now().isoformat()}
            )
            conceptual_memory.save()

            # Immediately verify file is valid JSON
            with open(concepts_path, 'r') as f:
                data = json.load(f)
            assert isinstance(data, dict), f"Invalid JSON on iteration {iteration}"

        tasks = [write_and_verify(i) for i in range(30)]
        await asyncio.gather(*tasks)

        # Final verification
        with open(concepts_path, 'r') as f:
            final_data = json.load(f)
        assert isinstance(final_data, dict)
        # Should have at least some concepts
        assert len(final_data) > 0

    @pytest.mark.asyncio
    async def test_conceptual_memory_async_flush_safety(
        self, conceptual_memory, temp_storage_dir
    ):
        """
        Test that async flush operations are safe under concurrency.

        The async flush method should properly handle concurrent calls.
        """
        num_flushes = 20

        async def store_and_flush(i: int):
            """Store a concept and trigger async flush."""
            conceptual_memory.store_concept(
                f"flush_test_{i}",
                {"flush_index": str(i)}
            )
            conceptual_memory._dirty = True
            await conceptual_memory.flush()

        tasks = [store_and_flush(i) for i in range(num_flushes)]
        await asyncio.gather(*tasks)

        # Verify final state is consistent
        concepts_path = os.path.join(
            conceptual_memory.storage_dir, "concepts.json"
        )
        with open(concepts_path, 'r') as f:
            data = json.load(f)

        # All flush_test concepts should be present
        flush_concepts = [k for k in data.keys() if k.startswith("flush_test_")]
        assert len(flush_concepts) > 0, "No flush_test concepts saved"


# =============================================================================
# Test: Concurrent Strategy Bank Persistence
# =============================================================================

class TestConcurrentStrategyBankPersistence:
    """
    Tests for concurrent StrategyBank persistence operations.

    Validates that:
    - Concurrent _persist_to_disk() calls don't corrupt the file
    - Strategies are not lost during concurrent writes
    - File remains valid JSON after concurrent operations
    """

    @pytest.mark.asyncio
    async def test_strategy_bank_concurrent_persist_no_corruption(
        self, strategy_bank, temp_storage_dir
    ):
        """
        Test that concurrent persist operations don't corrupt the file.

        Multiple concurrent persist calls should result in a valid file.
        """
        num_strategies = 50

        # Create strategies
        for i in range(num_strategies):
            strategy = Strategy(
                name=f"Test Strategy {i}",
                trigger_pattern=f"pattern_{i}",
                approach=f"Approach for strategy {i}",
                category="test",
                tags=["concurrent_test"],
            )
            strategy_bank.store_strategy(strategy)

        # Trigger concurrent persists
        def persist_op():
            strategy_bank._persist_to_disk()

        threads = [threading.Thread(target=persist_op) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify file is valid JSON
        persist_path = temp_storage_dir / "strategies.json"
        assert persist_path.exists(), "Persistence file not created"

        with open(persist_path, 'r') as f:
            data = json.load(f)

        assert "strategies" in data
        assert isinstance(data["strategies"], list)
        # All strategies should be present
        assert len(data["strategies"]) == num_strategies

    @pytest.mark.asyncio
    async def test_strategy_bank_concurrent_distill_and_persist(
        self, strategy_bank, temp_storage_dir
    ):
        """
        Test concurrent distillation and persistence operations.

        Creating strategies while persisting should not cause issues.
        """
        num_operations = 30

        def distill_op(i: int):
            strategy_bank.distill_from_episode(
                episode_id=f"ep_{i}",
                trigger_pattern=f"trigger_{i}",
                approach=f"approach_{i}",
                outcome="success" if i % 2 == 0 else "failure",
                quality_score=0.5 + (i % 10) * 0.05,
                name=f"Concurrent Strategy {i}",
            )

        def persist_op():
            strategy_bank._persist_to_disk()

        # Mix of distill and persist operations
        threads = []
        for i in range(num_operations):
            threads.append(threading.Thread(target=distill_op, args=(i,)))
            if i % 5 == 0:
                threads.append(threading.Thread(target=persist_op))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Final persist
        strategy_bank._persist_to_disk()

        # Verify file is valid
        persist_path = temp_storage_dir / "strategies.json"
        with open(persist_path, 'r') as f:
            data = json.load(f)

        # All strategies should be present
        assert len(data["strategies"]) == num_operations

    @pytest.mark.asyncio
    async def test_strategy_bank_judge_and_persist(
        self, strategy_bank, temp_storage_dir
    ):
        """
        Test that judge operations with persistence work concurrently.

        Judge updates should persist correctly under concurrent access.
        """
        # Create a strategy
        strategy = Strategy(
            name="Judge Test Strategy",
            trigger_pattern="test pattern",
            approach="test approach",
        )
        strategy_id = strategy_bank.store_strategy(strategy)

        def judge_and_persist(i: int):
            strategy_bank.judge_retrieval(
                strategy_id=strategy_id,
                query=f"test query {i}",
                retrieved_content="test content",
                outcome="success" if i % 2 == 0 else "failure",
                retrieval_score=0.5,
            )
            if i % 10 == 0:
                strategy_bank._persist_to_disk()

        threads = [threading.Thread(target=judge_and_persist, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Final persist and verify
        strategy_bank._persist_to_disk()

        persist_path = temp_storage_dir / "strategies.json"
        with open(persist_path, 'r') as f:
            data = json.load(f)

        # Find our strategy
        our_strategy = next(
            (s for s in data["strategies"] if s["id"] == strategy_id),
            None
        )
        assert our_strategy is not None
        # Should have outcomes
        assert len(our_strategy["outcomes"]) == 50


# =============================================================================
# Test: Concurrent Procedural Store Persistence
# =============================================================================

class TestConcurrentProceduralStorePersistence:
    """
    Tests for concurrent ProceduralStore persistence operations.

    Validates that:
    - Concurrent _persist_to_disk_async() calls don't corrupt the file
    - Procedures are not lost during concurrent writes
    - File remains valid after concurrent operations
    """

    @pytest.mark.asyncio
    async def test_procedural_store_concurrent_persist_no_corruption(
        self, procedural_store, temp_storage_dir
    ):
        """
        Test that concurrent persist operations don't corrupt the file.

        Multiple concurrent persist calls should result in a valid file.
        """
        num_procedures = 30

        # Create procedures
        for i in range(num_procedures):
            proc = Procedure(
                id=f"proc_{i}",
                name=f"Test Procedure {i}",
                description=f"Description {i}",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                steps=[
                    ProcedureStep(order=1, instruction=f"Step 1 for proc {i}"),
                    ProcedureStep(order=2, instruction=f"Step 2 for proc {i}"),
                ],
                trigger_pattern=f"trigger_{i}",
                success_count=0,
                failure_count=0,
                reliability=0.5,
                tags=["concurrent_test"],
            )
            await procedural_store.store_procedure(proc)

        # Trigger concurrent persists
        async def persist_op():
            await procedural_store.flush()

        tasks = [persist_op() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Verify file is valid JSON
        persist_path = temp_storage_dir / "procedures.json"
        assert persist_path.exists(), "Persistence file not created"

        with open(persist_path, 'r') as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == num_procedures

    @pytest.mark.asyncio
    async def test_procedural_store_concurrent_store_and_persist(
        self, procedural_store, temp_storage_dir
    ):
        """
        Test concurrent store and persist operations.

        Creating procedures while persisting should not cause issues.
        """
        num_operations = 40

        async def store_procedure_op(i: int):
            proc = Procedure(
                id=f"proc_concurrent_{i}",
                name=f"Concurrent Procedure {i}",
                description=f"Created concurrently {i}",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                steps=[ProcedureStep(order=1, instruction="Do something")],
                trigger_pattern=f"concurrent_trigger_{i}",
                reliability=0.5,
                tags=[],
            )
            await procedural_store.store_procedure(proc)

        async def persist_op():
            await procedural_store.flush()

        # Mix operations
        tasks = []
        for i in range(num_operations):
            tasks.append(store_procedure_op(i))
            if i % 5 == 0:
                tasks.append(persist_op())

        await asyncio.gather(*tasks)

        # Final persist
        await procedural_store.flush()

        # Verify file is valid
        persist_path = temp_storage_dir / "procedures.json"
        with open(persist_path, 'r') as f:
            data = json.load(f)

        assert len(data) == num_operations

    @pytest.mark.asyncio
    async def test_procedural_store_outcome_recording_concurrently(
        self, procedural_store, temp_storage_dir
    ):
        """
        Test that concurrent outcome recording works with persistence.

        Recording outcomes while persisting should maintain consistency.
        """
        # Create a procedure
        proc = Procedure(
            id="proc_outcome_test",
            name="Outcome Test Procedure",
            description="Test for concurrent outcomes",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            steps=[ProcedureStep(order=1, instruction="Test step")],
            trigger_pattern="outcome_test",
            reliability=0.5,
            tags=[],
        )
        await procedural_store.store_procedure(proc)

        async def record_outcome(i: int):
            success = i % 2 == 0
            await procedural_store.record_procedure_outcome("proc_outcome_test", success)
            if i % 10 == 0:
                await procedural_store.flush()

        tasks = [record_outcome(i) for i in range(50)]
        await asyncio.gather(*tasks)

        # Final persist
        await procedural_store.flush()

        # Verify
        final_proc = await procedural_store.get_procedure("proc_outcome_test")
        assert final_proc.success_count + final_proc.failure_count == 50

        # Verify file is valid
        persist_path = temp_storage_dir / "procedures.json"
        with open(persist_path, 'r') as f:
            data = json.load(f)

        proc_data = next((p for p in data if p["id"] == "proc_outcome_test"), None)
        assert proc_data is not None
        assert proc_data["success_count"] + proc_data["failure_count"] == 50


# =============================================================================
# Test: Cross-Service File Safety
# =============================================================================

class TestCrossServiceFileSafety:
    """
    Tests for file safety when multiple services persist concurrently.

    Validates that:
    - Multiple services can persist to different files concurrently
    - No file cross-contamination occurs
    """

    @pytest.mark.asyncio
    async def test_multiple_services_concurrent_persistence(
        self, conceptual_memory, strategy_bank, procedural_store, temp_storage_dir
    ):
        """
        Test that multiple services can persist concurrently without issues.

        Each service should maintain its own file integrity.
        """
        async def conceptual_ops():
            for i in range(10):
                conceptual_memory.store_concept(
                    f"cross_test_concept_{i}",
                    {"index": str(i)}
                )
            conceptual_memory.save()

        def strategy_ops():
            for i in range(10):
                strategy_bank.distill_from_episode(
                    episode_id=f"cross_ep_{i}",
                    trigger_pattern=f"cross_trigger_{i}",
                    approach=f"cross_approach_{i}",
                    outcome="success",
                    name=f"Cross Strategy {i}",
                )
            strategy_bank._persist_to_disk()

        async def procedural_ops():
            for i in range(10):
                proc = Procedure(
                    id=f"cross_proc_{i}",
                    name=f"Cross Procedure {i}",
                    description="Cross test",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    steps=[],
                    trigger_pattern=f"cross_proc_trigger_{i}",
                    reliability=0.5,
                    tags=[],
                )
                await procedural_store.store_procedure(proc)
            await procedural_store.flush()

        # Run all concurrently
        loop = asyncio.get_event_loop()
        await asyncio.gather(
            conceptual_ops(),
            loop.run_in_executor(None, strategy_ops),
            procedural_ops(),
        )

        # Verify all files are valid
        codebook_path = os.path.join(conceptual_memory.storage_dir, "codebook.json")
        with open(codebook_path, 'r') as f:
            codebook = json.load(f)
        assert isinstance(codebook, dict)

        strategies_path = temp_storage_dir / "strategies.json"
        with open(strategies_path, 'r') as f:
            strategies = json.load(f)
        assert "strategies" in strategies

        procedures_path = temp_storage_dir / "procedures.json"
        with open(procedures_path, 'r') as f:
            procedures = json.load(f)
        assert isinstance(procedures, list)


# =============================================================================
# Test: Stress Testing
# =============================================================================

class TestFileSafetyStress:
    """
    Stress tests for file safety under high concurrent load.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_high_volume_concurrent_persistence(
        self, strategy_bank, temp_storage_dir
    ):
        """
        Stress test with high volume of concurrent persist operations.

        Marked as 'slow' - run with: pytest -m slow
        """
        num_strategies = 200
        num_persists = 50

        # Create many strategies
        for i in range(num_strategies):
            strategy = Strategy(
                name=f"Stress Strategy {i}",
                trigger_pattern=f"stress_{i}",
                approach=f"Stress approach {i}",
            )
            strategy_bank.store_strategy(strategy)

        # Many concurrent persists
        def persist_op():
            strategy_bank._persist_to_disk()

        threads = [threading.Thread(target=persist_op) for _ in range(num_persists)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify file is still valid
        persist_path = temp_storage_dir / "strategies.json"
        with open(persist_path, 'r') as f:
            data = json.load(f)

        assert len(data["strategies"]) == num_strategies


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
