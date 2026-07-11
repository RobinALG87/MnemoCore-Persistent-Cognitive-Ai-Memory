import json
import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_hdv_rerank_stays_inside_agent_memory_dependency_boundary(tmp_path):
    script = r'''
import asyncio
import json
import sys
import os
from pathlib import Path

from mnemocore.agent_memory import AgentMemory, MemoryScope

async def main():
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
        await memory.remember("sharedterm beta deterministic record")
        results = await memory.recall("sharedterm alpha", use_hdv_rerank=True)
        components = results[0].score_components
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
        print(json.dumps({
            "has_hdv": "hdv" in components,
            "has_fusion": "fusion" in components,
            "forbidden": forbidden,
        }))
    finally:
        await memory.close()
        await asyncio.get_running_loop().shutdown_default_executor()

asyncio.run(main())
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
    assert result == {"has_hdv": True, "has_fusion": True, "forbidden": []}
