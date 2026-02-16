import importlib
import sys
from pathlib import Path

src_path = Path("src").absolute()
sys.path.insert(0, str(src_path))

modules = [
    "core.config",
    "core.hdv",
    "core.binary_hdv",
    "core.node",
    "core.synapse",
    "core.qdrant_store",
    "core.async_storage",
    "core.tier_manager",
    "core.engine",
    "core.router",
]

for mod in modules:
    print(f"Importing {mod}...", end="", flush=True)
    try:
        importlib.import_module(mod)
        print(" OK")
    except Exception as e:
        print(f" FAILED: {e}")
