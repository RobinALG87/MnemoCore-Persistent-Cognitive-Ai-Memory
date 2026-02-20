#!/usr/bin/env python3
"""
MnemoCore Bridge — Universal CLI
=================================
Lightweight bridge between MnemoCore REST API and any AI CLI tool.
No heavy dependencies: only stdlib + requests.

Usage:
    python mnemo_bridge.py context [--query TEXT] [--top-k 5] [--ctx CTX_ID]
    python mnemo_bridge.py store TEXT [--source SOURCE] [--tags TAG1,TAG2] [--ctx CTX_ID]
    python mnemo_bridge.py health

Environment variables:
    MNEMOCORE_URL       Base URL of MnemoCore API  (default: http://localhost:8100)
    MNEMOCORE_API_KEY   API key (same as HAIM_API_KEY)
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("MNEMOCORE_URL", "http://localhost:8100").rstrip("/")
API_KEY  = os.getenv("MNEMOCORE_API_KEY") or os.getenv("HAIM_API_KEY", "")
TIMEOUT  = int(os.getenv("MNEMOCORE_TIMEOUT", "5"))

HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(path: str) -> Optional[Dict]:
    try:
        r = requests.get(f"{BASE_URL}{path}", headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return None
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        return None


def _post(path: str, payload: Dict) -> Optional[Dict]:
    try:
        r = requests.post(f"{BASE_URL}{path}", headers=HEADERS,
                          json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return None
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        return None


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_health() -> int:
    data = _get("/health")
    if data is None:
        print("MnemoCore is OFFLINE (could not connect)", file=sys.stderr)
        return 1
    status = data.get("status", "unknown")
    print(f"MnemoCore status: {status}")
    return 0 if status == "ok" else 1


def cmd_store(text: str, source: str, tags: List[str], ctx: Optional[str]) -> int:
    metadata: Dict[str, Any] = {"source": source}
    if tags:
        metadata["tags"] = tags

    payload: Dict[str, Any] = {"content": text, "metadata": metadata}
    if ctx:
        payload["agent_id"] = ctx

    data = _post("/store", payload)
    if data is None:
        print("Failed to store memory (MnemoCore offline or error)", file=sys.stderr)
        return 1

    memory_id = data.get("id") or data.get("memory_id", "?")
    print(f"Stored: {memory_id}")
    return 0


def cmd_context(query: Optional[str], top_k: int, ctx: Optional[str]) -> int:
    """
    Fetch relevant memories and print them as a markdown block
    suitable for injection into any AI tool's system prompt.
    """
    payload: Dict[str, Any] = {
        "query": query or "recent work context decisions bugs fixes",
        "top_k": top_k,
    }
    if ctx:
        payload["agent_id"] = ctx

    data = _post("/query", payload)
    if data is None:
        # Silently return empty — don't break the calling tool's startup
        return 0

    results: List[Dict] = data.get("results", [])
    if not results:
        return 0

    lines = [
        "<!-- MnemoCore: Persistent Memory Context -->",
        "## Relevant memory from previous sessions\n",
    ]
    for r in results:
        content  = r.get("content", "").strip()
        score    = r.get("score", 0.0)
        meta     = r.get("metadata", {})
        source   = meta.get("source", "unknown")
        tags     = meta.get("tags", [])
        tag_str  = f" [{', '.join(tags)}]" if tags else ""

        lines.append(f"- **[{source}{tag_str}]** (relevance {score:.2f}): {content}")

    lines.append("\n<!-- End MnemoCore Context -->")
    print("\n".join(lines))
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="MnemoCore universal CLI bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # health
    sub.add_parser("health", help="Check MnemoCore connectivity")

    # store
    p_store = sub.add_parser("store", help="Store a memory")
    p_store.add_argument("text", help="Memory content")
    p_store.add_argument("--source", default="cli", help="Source label")
    p_store.add_argument("--tags", default="", help="Comma-separated tags")
    p_store.add_argument("--ctx",  default=None, help="Context/project ID")

    # context
    p_ctx = sub.add_parser("context", help="Fetch context as markdown")
    p_ctx.add_argument("--query", default=None, help="Semantic query string")
    p_ctx.add_argument("--top-k", type=int, default=5, help="Number of results")
    p_ctx.add_argument("--ctx",  default=None, help="Context/project ID")

    args = parser.parse_args()

    if args.cmd == "health":
        return cmd_health()

    if args.cmd == "store":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        return cmd_store(args.text, args.source, tags, args.ctx)

    if args.cmd == "context":
        return cmd_context(args.query, args.top_k, args.ctx)

    return 1


if __name__ == "__main__":
    sys.exit(main())
