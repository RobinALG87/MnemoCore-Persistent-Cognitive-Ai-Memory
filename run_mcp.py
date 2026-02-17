#!/usr/bin/env python3
import sys
import os

# Add the current directory to sys.path to ensure src is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the mcp object from the server implementation
try:
    from src.mcp_server import mcp
except ImportError as e:
    print(f"Error importing mcp_server: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("Starting MnemoCore MCP Server...", file=sys.stderr)
    try:
        # mcp.run() automatically parses command line arguments (e.g. 'stdio', 'sse')
        # Default is usually stdio if run directly, but explicit argument is better if known.
        # However, for MCP servers intended to be used by Claude Desktop, they are often run with `python run_mcp.py`
        # which defaults to stdio if not specified.
        mcp.run()
    except KeyboardInterrupt:
        print("Stopping server...", file=sys.stderr)
