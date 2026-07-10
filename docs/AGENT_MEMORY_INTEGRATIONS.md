# Agent-memory integrations

MnemoCore exposes four small adapters around the same controlled contract:

1. recall bounded context before work;
2. observe/remember during work;
3. keep writes behind one explicit policy.

The adapters do not import LangGraph, CrewAI, OpenClaw, or MCP at module import
time. Install only the framework you use. All adapters delegate scope and
validation to `AgentMemory`.

## Shared bridge

```python
from mnemocore.agent_memory import AgentMemory, MemoryScope
from mnemocore.integrations import AgentMemoryBridge, IntegrationPolicy

memory = await AgentMemory.open(
    "./data/agent-memory.db",
    scope=MemoryScope(user_id="robin", agent_id="codex", project_id="demo"),
)
bridge = AgentMemoryBridge(
    memory,
    policy=IntegrationPolicy(max_context_tokens=1200, include_ancestors=True),
)
brief = await bridge.context("fix retrieval")
await bridge.observe("retrieval failed because the query was underspecified")
```

Set `allow_writes=False` for read-only agents. Requests above the configured
token budget are rejected instead of silently expanding context.

## LangGraph

`LangGraphMemory.context_node` is an ordinary async node returning a state
update, so it can be inserted without a LangGraph-specific dependency:

```python
from mnemocore.integrations import LangGraphMemory

memory_nodes = LangGraphMemory(bridge, query_key="goal")
graph.add_node("memory_context", memory_nodes.context_node)
graph.add_node("memory_observe", memory_nodes.observation_node)
```

The input state needs `goal` (or `query`) for context and `observation` for the
write node. The output keys are `memory_context` and `memory_observation`.

## CrewAI

CrewAI uses synchronous tools, so pass a `SyncAgentMemory` client:

```python
from mnemocore.agent_memory import SyncAgentMemory
from mnemocore.integrations import CrewAIMemoryTools

with SyncAgentMemory.open("./data/agent-memory.db", scope=scope) as memory:
    tools = CrewAIMemoryTools(memory, max_context_tokens=1200)
    agent = Agent(
        role="reviewer",
        goal="use durable project context",
        tools=list(tools.as_tools()),
    )
```

If CrewAI is installed, `as_tools()` returns decorated tools. Without it, it
returns the same plain Python callables for easy testing.

## OpenClaw

The OpenClaw adapter uses a deliberately small event shape, avoiding a hard
dependency on a particular OpenClaw release:

```python
from mnemocore.integrations import OpenClawMemory

openclaw = OpenClawMemory(bridge)
before = await openclaw.before_turn({"goal": "fix retrieval"})
after = await openclaw.after_turn({
    "observation": "the first approach failed its scope test",
})
```

`before_turn` returns `memory_context`; `after_turn` returns
`memory_observation`. Unknown or blank event fields fail visibly.

## MCP

MCP is lazy and allowlisted to two tools by default:

```python
from mnemocore.integrations import create_mcp_server

server = create_mcp_server(memory, name="MnemoCore Memory")
server.run(transport="stdio")
```

The exposed tools are `memory_recall` and `memory_remember`. Destructive,
export, and unrestricted tools are not registered by this adapter. Install the
optional MCP package before starting the server.

## Controlled deployment checklist

- Use a distinct `MemoryScope` per tenant/user/agent/project.
- Keep `include_ancestors=False` for strict base clients unless inheritance is intended.
- Set a small integration budget first (for example 800–1200 tokens).
- Use `allow_writes=False` for analysis-only agents.
- Treat receipts and evidence ids as part of the agent audit trail.
