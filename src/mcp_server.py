import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP, Context

# Initialize MnemoCore Engine
# We need to make sure we are in the right directory or set the path correctly
import sys
import os

# Add src to path if needed (though running from root usually works if PYTHONPATH is set)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.engine import HAIMEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

# Initialize Engine
# We use a global engine instance. In a real deployment, we might want to manage lifecycle better.
# For MCP, the server process starts and stays running.
try:
    engine = HAIMEngine()
    logger.info("HAIMEngine initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize HAIMEngine: {e}")
    sys.exit(1)

# Initialize MCP Server
mcp = FastMCP("MnemoCore")

# --- Tools ---

@mcp.tool()
def store_memory(content: str, metadata: Optional[Dict[str, Any]] = None, goal_id: Optional[str] = None) -> str:
    """
    Store a new memory in MnemoCore.

    Args:
        content: The text content of the memory.
        metadata: Optional dictionary of metadata (key-value pairs).
        goal_id: Optional ID of a goal context to bind this memory to.

    Returns:
        The ID of the stored memory.
    """
    try:
        mem_id = engine.store(content, metadata=metadata, goal_id=goal_id)
        return f"Stored memory with ID: {mem_id}"
    except Exception as e:
        return f"Error storing memory: {str(e)}"

@mcp.tool()
def query_memories(query: str, top_k: int = 5) -> str:
    """
    Query memories using semantic search (Holographic Active Inference).

    Args:
        query: The search query text.
        top_k: Number of results to return (default 5).

    Returns:
        A JSON string containing the list of matching memories with scores.
    """
    try:
        results = engine.query(query, top_k=top_k)
        formatted_results = []
        for mem_id, score in results:
            node = engine.get_memory(mem_id)
            if node:
                formatted_results.append({
                    "id": mem_id,
                    "content": node.content,
                    "score": float(score),
                    "metadata": node.metadata
                })
        return json.dumps(formatted_results, indent=2)
    except Exception as e:
        return f"Error querying memories: {str(e)}"

@mcp.tool()
def get_memory(memory_id: str) -> str:
    """
    Retrieve a specific memory by its ID.

    Args:
        memory_id: The ID of the memory to retrieve.

    Returns:
        The memory content and metadata as a JSON string.
    """
    node = engine.get_memory(memory_id)
    if node:
        data = {
            "id": node.id,
            "content": node.content,
            "metadata": node.metadata,
            "created_at": node.created_at.isoformat(),
            "tier": getattr(node, "tier", "unknown")
        }
        return json.dumps(data, indent=2)
    else:
        return f"Memory not found: {memory_id}"

@mcp.tool()
def delete_memory(memory_id: str) -> str:
    """
    Delete a memory by its ID.

    Args:
        memory_id: The ID of the memory to delete.

    Returns:
        Confirmation message.
    """
    try:
        engine.delete_memory(memory_id)
        return f"Deleted memory: {memory_id}"
    except Exception as e:
        return f"Error deleting memory: {str(e)}"

@mcp.tool()
def reason_by_analogy(source_concept: str, source_value: str, target_concept: str) -> str:
    """
    Perform analogical reasoning: A is to B as C is to ?

    Args:
        source_concept: The first concept (A).
        source_value: The value associated with A (B).
        target_concept: The second concept (C).

    Returns:
        The predicted value for C (D) and score.
    """
    try:
        results = engine.reason_by_analogy(source_concept, source_value, target_concept)
        # Format results
        output = [f"{v} (score: {s:.4f})" for v, s in results[:5]]
        return f"Analogy: {source_concept}:{source_value} :: {target_concept}:?\nResults:\n" + "\n".join(output)
    except Exception as e:
        return f"Error reasoning by analogy: {str(e)}"

# --- Resources ---

@mcp.resource("mnemocore://memories/recent")
def get_recent_memories() -> str:
    """
    Get a list of the most recent memories in the HOT tier.
    """
    try:
        recent = engine.get_recent_memories(n=20)
        output = []
        for node in recent:
            output.append({
                "id": node.id,
                "content": node.content,
                "created_at": node.created_at.isoformat()
            })
        return json.dumps(output, indent=2)
    except Exception as e:
        return f"Error retrieving recent memories: {str(e)}"

@mcp.resource("mnemocore://stats")
def get_stats() -> str:
    """
    Get current statistics of the MnemoCore engine.
    """
    try:
        stats = engine.get_stats()
        return json.dumps(stats, indent=2)
    except Exception as e:
        return f"Error retrieving stats: {str(e)}"

# --- Prompts ---

@mcp.prompt()
def recall(topic: str) -> str:
    """
    Create a prompt to help the user recall information about a topic.
    """
    return f"""I need to recall details about '{topic}' from my long-term memory.
Please use the 'query_memories' tool to search for '{topic}' and any related concepts.
Then, summarize what you find."""

@mcp.prompt()
def dream() -> str:
    """
    Trigger a 'dreaming' session to consolidate memories and find hidden connections.
    """
    return """Please assume the role of the Subconscious.
Review the recent memories using 'mnemocore://memories/recent'.
Then, use 'reason_by_analogy' or 'query_memories' to find unexpected connections between them.
If you find strong connections, suggest storing a new insight using 'store_memory'."""

if __name__ == "__main__":
    # mcp.run() handles argument parsing (e.g. 'stdio', 'sse')
    mcp.run()
