import asyncio
import json
import os
import sys
import time
from unittest.mock import MagicMock, patch
import pytest

# --- Mocking Infrastructure ---
import types
def mock_module(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

# Mock dependencies if they are not importable
if "src.core.engine" not in sys.modules:
    mock_module("src.core")
    mock_module("src.core.engine")
    sys.modules["src.core.engine"].HAIMEngine = MagicMock()

if "src.core.async_storage" not in sys.modules:
    mock_module("src.core.async_storage")
    sys.modules["src.core.async_storage"].AsyncRedisStorage = MagicMock()

if "src.meta.learning_journal" not in sys.modules:
    mock_module("src.meta")
    mock_module("src.meta.learning_journal")
    sys.modules["src.meta.learning_journal"].LearningJournal = MagicMock()

if "aiohttp" not in sys.modules:
    mock_module("aiohttp")
    sys.modules["aiohttp"].ClientSession = MagicMock()

# Now we can safely import daemon
sys.path.insert(0, os.path.abspath("."))
from src.subconscious.daemon import SubconsciousDaemon

async def _async_test_save_evolution_state_non_blocking():
    """
    Async test logic that verifies _save_evolution_state does not block the event loop.
    We simulate slow I/O by patching json.dump.
    """

    # 1. Setup Daemon
    daemon = SubconsciousDaemon()

    # Use a temp path for the state file to avoid permission issues
    with patch("src.subconscious.daemon.EVOLUTION_STATE_PATH", "/tmp/test_evolution_perf.json"):

        # 2. Patch json.dump to be slow (simulate blocking I/O)
        # We need to patch it where it is used. daemon.py imports json.
        # So we patch json.dump.
        original_dump = json.dump

        def slow_dump(*args, **kwargs):
            time.sleep(0.2) # Block for 200ms
            return original_dump(*args, **kwargs)

        with patch("json.dump", side_effect=slow_dump):

            # 3. Create a background task (ticker) to measure loop blocking
            # If the loop is blocked, this task won't get a chance to run

            loop_blocked_duration = 0
            ticker_running = True

            async def ticker():
                nonlocal loop_blocked_duration
                while ticker_running:
                    start = time.time()
                    await asyncio.sleep(0.01) # Yield control
                    diff = time.time() - start
                    # If sleep(0.01) took significantly longer, the loop was blocked
                    if diff > 0.05:
                        loop_blocked_duration = max(loop_blocked_duration, diff)

            ticker_task = asyncio.create_task(ticker())

            # Allow ticker to start
            await asyncio.sleep(0.05)

            # 4. Run the method under test
            # If it is synchronous, it will block the loop, and ticker won't run until it finishes.
            # If it is asynchronous and properly non-blocking (awaiting in executor), ticker should run in between.

            start_time = time.time()
            if asyncio.iscoroutinefunction(daemon._save_evolution_state):
                await daemon._save_evolution_state()
            else:
                daemon._save_evolution_state()
            end_time = time.time()

            # Cleanup
            ticker_running = False
            try:
                await ticker_task
            except asyncio.CancelledError:
                pass

            # 5. Assertions
            print(f"Operation took: {end_time - start_time:.4f}s")
            print(f"Max loop block: {loop_blocked_duration:.4f}s")

            # If the operation was truly non-blocking, the ticker should have run frequently,
            # and the max loop block should be close to 0.01s (maybe up to 0.05s tolerance).
            # If it was blocking (synchronous sleep(0.2)), the ticker would be delayed by ~0.2s.

            # We fail if loop was blocked for more than 100ms
            if loop_blocked_duration >= 0.1:
                raise AssertionError(f"Event loop was blocked for {loop_blocked_duration:.4f}s")

def test_save_evolution_state_non_blocking():
    asyncio.run(_async_test_save_evolution_state_non_blocking())

if __name__ == "__main__":
    test_save_evolution_state_non_blocking()
