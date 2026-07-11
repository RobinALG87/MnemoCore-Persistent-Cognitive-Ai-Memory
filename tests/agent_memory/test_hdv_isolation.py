import json
import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_hdv_rerank_stays_inside_agent_memory_dependency_boundary(tmp_path):
    script = r'''
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import os
import threading
from pathlib import Path

from mnemocore.agent_memory import AgentMemory, MemoryScope

async def main():
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    memory = await AgentMemory.open(
        Path(os.environ["HDV_TEST_DB"]),
        scope=MemoryScope(
            tenant_id="test-tenant",
            user_id="test-user",
            agent_id="test-agent",
            project_id="test-project",
        ),
    )
    try:
        await memory.remember("sharedterm alpha deterministic record")
        first = await memory.recall("sharedterm alpha", use_hdv_rerank=True)
        second = await memory.recall("sharedterm alpha", use_hdv_rerank=True)
        forbidden = [
            name
            for name in (
                "mnemocore.core.engine",
                "qdrant_client",
                "redis",
                "fastapi",
                "faiss",
            )
            if name in sys.modules
        ]
        return {
            "orders_match": [item.memory.id for item in first] == [
                item.memory.id for item in second
            ],
            "component_ranges": all(
                0.0 <= item.score_components["hdv"] <= 1.0
                and 0.0 <= item.score_components["fusion"] <= 1.0
                for item in first + second
            ),
            "forbidden": forbidden,
        }
    finally:
        await memory.close()
        executor.shutdown(wait=True, cancel_futures=True)

result = asyncio.run(main())
result["threads_after_shutdown"] = [thread.name for thread in threading.enumerate()]
print(json.dumps(result))
sys.stdout.flush()
os._exit(0)
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["HDV_TEST_DB"] = str(tmp_path / "memory.db")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "orders_match": True,
        "component_ranges": True,
        "forbidden": [],
        "threads_after_shutdown": ["MainThread"],
    }
