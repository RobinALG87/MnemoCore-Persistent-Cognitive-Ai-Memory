"""
MnemoCore CLI - Main Entry Point

Command-line interface for MnemoCore memory operations.

Usage:
    mnemocore store "Robin gillar Python"      # Store a memory
    mnemocore recall "vad gillar Robin?"       # Recall/search memories
    mnemocore dream --now                      # Trigger dream session
    mnemocore stats                            # Show system statistics
    mnemocore export --format json > backup.json  # Export memories
"""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Optional, Callable, Any

import click

from loguru import logger


# ============================================================================
# Engine Lifecycle Decorator / Context Manager
# ============================================================================

@asynccontextmanager
async def engine_context(config_path: Optional[Path] = None):
    """
    Async context manager for HAIMEngine lifecycle.

    Handles:
    - Loading config
    - Creating engine
    - Initializing
    - Cleanup on exit

    Usage:
        async with engine_context(config_path) as engine:
            result = await engine.query("test")
    """
    from mnemocore.core.engine import HAIMEngine
    from mnemocore.core.config import load_config

    # Load config
    config = load_config(config_path) if config_path and config_path.exists() else None

    # Create engine
    engine = HAIMEngine(config=config)

    try:
        # Initialize
        await engine.initialize()
        yield engine
    finally:
        # Cleanup
        await engine.close()


def with_engine(func: Callable) -> Callable:
    """
    Decorator that handles the complete engine lifecycle for CLI commands.

    This decorator:
    1. Extracts config_path from click context
    2. Creates and initializes the engine
    3. Passes the engine to the wrapped function
    4. Handles cleanup on exit
    5. Runs the async function via asyncio.run()

    The wrapped function should be async and accept 'engine' as first argument after ctx.

    Usage:
        @cli.command()
        @click.pass_context
        @with_engine
        async def my_command(ctx, engine, ...other_args...):
            result = await engine.query("test")
            # No need to close engine - decorator handles it

    Args:
        func: Async function to wrap. Should accept (ctx, engine, ...) or (engine, ...)

    Returns:
        Wrapped function that handles engine lifecycle
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Find click context in args
        ctx = None
        for arg in args:
            if isinstance(arg, click.Context):
                ctx = arg
                break

        if ctx is None:
            ctx = click.get_current_context()

        # Get config path from context
        config_path = Path(ctx.obj["config_path"]) if ctx.obj.get("config_path") else None

        async def run_with_engine():
            async with engine_context(config_path) as engine:
                # Check if function expects ctx as first arg
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                if params and params[0] in ('ctx', 'context'):
                    return await func(*args, engine=engine, **kwargs)
                else:
                    # Inject engine as first arg after ctx
                    new_args = list(args)
                    # Find position to insert engine
                    if 'engine' in kwargs:
                        return await func(*args, **kwargs)
                    else:
                        return await func(engine, *args[1:] if len(args) > 1 and isinstance(args[0], click.Context) else args, **kwargs)

        return asyncio.run(run_with_engine())

    return wrapper


def with_engine_simple(func: Callable) -> Callable:
    """
    Simplified decorator for commands that only need the engine.

    Usage:
        @cli.command()
        @click.pass_context
        @with_engine_simple
        async def my_command(ctx, engine):
            result = await engine.query("test")
            click.echo(str(result))

    The decorated function must be async and accept (ctx, engine, ...).
    """
    @wraps(func)
    def wrapper(ctx, *args, **kwargs):
        config_path = Path(ctx.obj["config_path"]) if ctx.obj.get("config_path") else None

        async def run():
            async with engine_context(config_path) as engine:
                return await func(ctx, engine, *args, **kwargs)

        return asyncio.run(run())

    return wrapper


# ============================================================================
# CLI Group and Main Entry
# ============================================================================

@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to config.yaml file",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--data-dir",
    "-d",
    type=click.Path(),
    default="./data",
    help="Data directory path",
)
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool, data_dir: str):
    """
    MnemoCore - Persistent Cognitive Memory CLI

    A hierarchical AI memory engine with hot/warm/cold tiers,
    vector search, and subconscious consolidation.
    """
    ctx.ensure_object(dict)

    # Set up context
    ctx.obj["verbose"] = verbose
    ctx.obj["config_path"] = config
    ctx.obj["data_dir"] = Path(data_dir)

    # Configure logging
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")


# ============================================================================
# CLI Commands
# ============================================================================

@cli.command()
@click.argument("content", required=True)
@click.option(
    "--metadata",
    "-m",
    help="JSON metadata as string",
)
@click.option(
    "--tags",
    "-t",
    multiple=True,
    help="Tags to attach (can use multiple times)",
)
@click.option(
    "--importance",
    "-i",
    type=float,
    default=0.5,
    help="Importance score (0.0-1.0)",
)
@click.option(
    "--category",
    "-c",
    help="Memory category",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
@click.pass_context
def store(ctx, content: str, metadata: Optional[str], tags: tuple, importance: float, category: Optional[str], output_json: bool):
    """
    Store a new memory.

    Example:
        mnemocore store "Robin gillar Python programmering"
    """
    @with_engine_simple
    async def _store(ctx, engine):
        # Build metadata
        meta = {}
        if metadata:
            try:
                meta.update(json.loads(metadata))
            except json.JSONDecodeError:
                click.echo(f"Error: Invalid JSON metadata: {metadata}", err=True)
                return

        if tags:
            meta["tags"] = list(tags)

        if category:
            meta["category"] = category

        meta["cli_stored"] = True
        meta["stored_at"] = datetime.now(timezone.utc).isoformat()

        try:
            memory_id = await engine.store(content, metadata=meta if meta else None)

            if output_json:
                result = {
                    "success": True,
                    "memory_id": memory_id,
                    "content": content,
                    "metadata": meta,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                click.echo(json.dumps(result, indent=2))
            else:
                click.echo(f"Stored memory: {memory_id}")
                click.echo(f"Content: {content[:100]}{'...' if len(content) > 100 else ''}")
                if tags:
                    click.echo(f"Tags: {', '.join(tags)}")

        except Exception as e:
            if output_json:
                error_result = {"success": False, "error": str(e)}
                click.echo(json.dumps(error_result, indent=2))
            else:
                click.echo(f"Error storing memory: {e}", err=True)

    return _store(ctx)


@cli.command()
@click.argument("query", required=True)
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=5,
    help="Number of results to return",
)
@click.option(
    "--min-score",
    "-s",
    type=float,
    default=0.0,
    help="Minimum similarity score (0.0-1.0)",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
@click.option(
    "--show-content",
    "-C",
    is_flag=True,
    help="Show full content in results",
)
@click.pass_context
def recall(ctx, query: str, top_k: int, min_score: float, output_json: bool, show_content: bool):
    """
    Recall/search memories.

    Example:
        mnemocore recall "vad gillar Robin?"
        mnemocore recall "Python" -k 10 --json
    """
    @with_engine_simple
    async def _recall(ctx, engine):
        try:
            # Query memories
            results = await engine.query(query, top_k=top_k)

            # Filter by score
            filtered = [(mid, score) for mid, score in results if score >= min_score]

            if output_json:
                # Get full memory details
                memories = []
                for memory_id, score in filtered:
                    node = await engine.get_memory(memory_id)
                    if node:
                        memories.append({
                            "id": memory_id,
                            "score": round(score, 4),
                            "content": node.content,
                            "created_at": node.created_at.isoformat() if node.created_at else None,
                            "tier": node.tier,
                            "metadata": node.metadata,
                        })

                result = {
                    "query": query,
                    "count": len(memories),
                    "memories": memories,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                if not filtered:
                    click.echo(f"No memories found matching: {query}")
                    return

                click.echo(f"Found {len(filtered)} memories for: {query}")
                click.echo()

                for i, (memory_id, score) in enumerate(filtered[:top_k], 1):
                    node = await engine.get_memory(memory_id)
                    if node:
                        content_preview = node.content[:80] + "..." if len(node.content) > 80 else node.content
                        click.echo(f"{i}. [{memory_id}] (score: {score:.2f}) [{node.tier.upper()}]")
                        click.echo(f"   {content_preview}")

                        if show_content:
                            click.echo(f"   Full: {node.content}")

                        if node.metadata and node.metadata.get("tags"):
                            click.echo(f"   Tags: {', '.join(node.metadata['tags'])}")
                        click.echo()

        except Exception as e:
            if output_json:
                error_result = {"success": False, "error": str(e)}
                click.echo(json.dumps(error_result, indent=2))
            else:
                click.echo(f"Error recalling memories: {e}", err=True)

    return _recall(ctx)


@cli.command()
@click.option(
    "--now",
    is_flag=True,
    help="Run dream session immediately",
)
@click.option(
    "--report-path",
    "-o",
    type=click.Path(),
    help="Path to save dream report",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
@click.pass_context
def dream(ctx, now: bool, report_path: Optional[str], output_json: bool):
    """
    Trigger a dream session for memory consolidation.

    Dream sessions perform:
    - Episodic clustering
    - Pattern extraction
    - Recursive synthesis
    - Contradiction resolution
    - Semantic promotion

    Example:
        mnemocore dream --now
        mnemocore dream --now --report-path dream_report.json
    """
    if not now:
        click.echo("Use --now flag to trigger dream session immediately.")
        click.echo("Example: mnemocore dream --now")
        return

    @with_engine_simple
    async def _dream(ctx, engine):
        from mnemocore.subconscious.dream_pipeline import DreamPipeline, DreamPipelineConfig

        try:
            click.echo("Starting dream session...", err=True)

            # Create dream pipeline config
            dream_config = DreamPipelineConfig(
                enable_episodic_clustering=True,
                enable_pattern_extraction=True,
                enable_recursive_synthesis=True,
                enable_contradiction_resolution=True,
                enable_semantic_promotion=True,
                enable_dream_report=True,
            )

            # Run dream pipeline
            pipeline = DreamPipeline(engine, dream_config)
            result = await pipeline.run()

            # Save report if path provided
            if report_path and result.get("success"):
                report_file = Path(report_path)
                report_file.parent.mkdir(parents=True, exist_ok=True)
                with open(report_file, "w") as f:
                    json.dump(result, f, indent=2)
                click.echo(f"Dream report saved to: {report_path}", err=True)

            if output_json:
                click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                if result.get("success"):
                    click.echo("Dream session completed successfully")
                    click.echo()
                    click.echo(f"Duration: {result.get('duration_seconds', 0):.1f} seconds")
                    click.echo(f"Memories processed: {result.get('memories_processed', 0)}")
                    click.echo(f"Episodic clusters: {result.get('episodic_clusters_count', 0)}")
                    click.echo(f"Patterns extracted: {result.get('patterns_extracted_count', 0)}")
                    click.echo(f"Synthesis results: {result.get('synthesis_results_count', 0)}")
                    click.echo(f"Contradictions found: {result.get('contradictions_found', 0)}")
                    click.echo(f"Contradictions resolved: {result.get('contradictions_resolved', 0)}")
                    click.echo(f"Semantic promotions: {result.get('semantic_promotions', 0)}")

                    # Show recommendations if available
                    report = result.get("dream_report")
                    if report and report.get("recommendations"):
                        click.echo()
                        click.echo("Recommendations:")
                        for rec in report["recommendations"]:
                            click.echo(f"  - {rec}")
                else:
                    click.echo(f"Dream session failed: {result.get('error', 'Unknown error')}", err=True)

        except Exception as e:
            if output_json:
                error_result = {"success": False, "error": str(e)}
                click.echo(json.dumps(error_result, indent=2))
            else:
                click.echo(f"Error running dream session: {e}", err=True)

    return _dream(ctx)


@cli.command()
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
@click.option(
    "--tier",
    "-t",
    type=click.Choice(["hot", "warm", "cold", "all"]),
    default="all",
    help="Filter by tier",
)
@click.pass_context
def stats(ctx, output_json: bool, tier: str):
    """
    Show system statistics.

    Example:
        mnemocore stats
        mnemocore stats --json
        mnemocore stats -t hot
    """
    @with_engine_simple
    async def _stats(ctx, engine):
        try:
            # Get stats
            stats_data = await engine.get_stats()

            if output_json:
                click.echo(json.dumps(stats_data, indent=2, ensure_ascii=False))
            else:
                # Format stats for display
                click.echo("=" * 50)
                click.echo("MnemoCore System Statistics")
                click.echo("=" * 50)
                click.echo()

                click.echo(f"Engine Version: {stats_data.get('engine_version', 'unknown')}")
                click.echo(f"Dimension: {stats_data.get('dimension', 'unknown')}")
                click.echo(f"Encoding: {stats_data.get('encoding', 'unknown')}")
                click.echo()

                # Tier stats
                tiers = stats_data.get("tiers", {})
                click.echo("Memory Tiers:")
                if tiers:
                    if tier in ("hot", "all"):
                        hot = tiers.get("hot", {})
                        click.echo(f"  HOT:   {hot.get('count', 0):>6} memories (max: {hot.get('max', 'N/A')})")
                    if tier in ("warm", "all"):
                        warm = tiers.get("warm", {})
                        click.echo(f"  WARM:  {warm.get('count', 0):>6} memories (max: {warm.get('max', 'N/A')})")
                    if tier in ("cold", "all"):
                        cold = tiers.get("cold", {})
                        click.echo(f"  COLD:  {cold.get('count', 0):>6} memories")
                click.echo()

                # Concept stats
                click.echo(f"Concepts:    {stats_data.get('concepts_count', 0):>6}")
                click.echo(f"Symbols:     {stats_data.get('symbols_count', 0):>6}")
                click.echo(f"Synapses:    {stats_data.get('synapses_count', 0):>6}")
                click.echo()

                # Background workers
                workers = stats_data.get("background_workers", {})
                if workers:
                    click.echo("Background Workers:")
                    for name, worker in workers.items():
                        status_str = "running" if worker.get("running") else "stopped"
                        click.echo(f"  {name}: {status_str}")
                    click.echo()

                # Gap detector
                gap = stats_data.get("gap_detector", {})
                if gap:
                    click.echo("Gap Detection:")
                    click.echo(f"  Gaps detected: {gap.get('gaps_detected', 0)}")
                    click.echo(f"  Gaps filled:    {gap.get('gaps_filled', 0)}")
                    click.echo()

                # Subconscious queue
                backlog = stats_data.get("subconscious_backlog", 0)
                if backlog > 0:
                    click.echo(f"Subconscious Queue: {backlog} pending")

                click.echo("=" * 50)

        except Exception as e:
            if output_json:
                error_result = {"success": False, "error": str(e)}
                click.echo(json.dumps(error_result, indent=2))
            else:
                click.echo(f"Error getting stats: {e}", err=True)

    return _stats(ctx)


@cli.command()
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "jsonl"]),
    default="json",
    help="Export format",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output file path",
)
@click.option(
    "--collection",
    "-c",
    type=click.Choice(["hot", "warm", "all"]),
    default="all",
    help="Collection to export",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    help="Maximum number of memories to export",
)
@click.option(
    "--include-vectors",
    is_flag=True,
    help="Include vector embeddings in export",
)
@click.pass_context
def export(ctx, format: str, output: str, collection: str, limit: Optional[int], include_vectors: bool):
    """
    Export memories to a file.

    Example:
        mnemocore export --format json -o backup.json
        mnemocore export -f jsonl -o backup.jsonl -c hot --limit 1000
    """
    @with_engine_simple
    async def _export(ctx, engine):
        try:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            click.echo(f"Exporting memories from {collection} tier(s)...", err=True)

            # Collect memories to export
            memories_to_export = []
            tiers_to_export = ["hot", "warm"] if collection == "all" else [collection]

            for tier_name in tiers_to_export:
                if tier_name == "hot":
                    tier_memories = await engine.tier_manager.get_all_hot()
                else:  # warm
                    tier_memories = await engine.tier_manager.get_hot_recent(limit or 10000)

                for mem in tier_memories:
                    memories_to_export.append({
                        "id": mem.id,
                        "content": mem.content,
                        "created_at": mem.created_at.isoformat() if mem.created_at else None,
                        "tier": mem.tier,
                        "ltp_strength": mem.ltp_strength,
                        "access_count": getattr(mem, "access_count", 0),
                        "metadata": mem.metadata or {},
                    })

                    if include_vectors:
                        memories_to_export[-1]["vector"] = mem.hdv.tolist() if hasattr(mem.hdv, "tolist") else list(mem.hdv)

                    if limit and len(memories_to_export) >= limit:
                        break

                if limit and len(memories_to_export) >= limit:
                    break

            # Write export
            exported_count = 0
            with open(output_path, "w", encoding="utf-8") as f:
                if format == "json":
                    json.dump(memories_to_export, f, indent=2, ensure_ascii=False)
                    exported_count = len(memories_to_export)
                else:  # jsonl
                    for mem in memories_to_export:
                        f.write(json.dumps(mem, ensure_ascii=False) + "\n")
                        exported_count += 1

            file_size = output_path.stat().st_size
            size_mb = file_size / (1024 * 1024)

            click.echo(f"Exported {exported_count} memories to {output_path}", err=True)
            click.echo(f"File size: {size_mb:.2f} MB", err=True)

        except Exception as e:
            click.echo(f"Error exporting memories: {e}", err=True)

    return _export(ctx)


@cli.command()
@click.argument("memory_id", required=True)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force deletion without confirmation",
)
@click.pass_context
def delete(ctx, memory_id: str, force: bool):
    """
    Delete a memory by ID.

    Example:
        mnemocore delete mem_abc123
        mnemocore delete mem_abc123 --force
    """
    @with_engine_simple
    async def _delete(ctx, engine):
        try:
            # Get memory info first
            node = await engine.get_memory(memory_id)

            if not node:
                click.echo(f"Memory not found: {memory_id}", err=True)
                return

            if not force:
                click.echo(f"Memory to delete: {memory_id}")
                click.echo(f"Content: {node.content[:100]}...")
                click.confirm("Do you want to delete this memory?", abort=True)

            # Delete memory
            await engine.delete_memory(memory_id)
            click.echo(f"Deleted memory: {memory_id}")

        except Exception as e:
            click.echo(f"Error deleting memory: {e}", err=True)

    return _delete(ctx)


@cli.command()
@click.argument("memory_id", required=True)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
@click.pass_context
def get(ctx, memory_id: str, output_json: bool):
    """
    Get a memory by ID.

    Example:
        mnemocore get mem_abc123
        mnemocore get mem_abc123 --json
    """
    @with_engine_simple
    async def _get(ctx, engine):
        try:
            node = await engine.get_memory(memory_id)

            if not node:
                click.echo(f"Memory not found: {memory_id}", err=True)
                return

            if output_json:
                result = {
                    "id": node.id,
                    "content": node.content,
                    "created_at": node.created_at.isoformat() if node.created_at else None,
                    "tier": node.tier,
                    "ltp_strength": node.ltp_strength,
                    "access_count": getattr(node, "access_count", 0),
                    "metadata": node.metadata or {},
                }
                click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                click.echo(f"ID: {node.id}")
                click.echo(f"Tier: {node.tier.upper()}")
                click.echo(f"LTP Strength: {node.ltp_strength:.3f}")
                click.echo(f"Created: {node.created_at.isoformat() if node.created_at else 'N/A'}")
                click.echo()
                click.echo("Content:")
                click.echo(node.content)
                click.echo()
                if node.metadata:
                    click.echo("Metadata:")
                    for key, value in node.metadata.items():
                        click.echo(f"  {key}: {value}")

        except Exception as e:
            click.echo(f"Error getting memory: {e}", err=True)

    return _get(ctx)


@cli.command()
@click.pass_context
def health(ctx):
    """
    Check system health.

    Example:
        mnemocore health
    """
    @with_engine_simple
    async def _health(ctx, engine):
        try:
            health_data = await engine.health_check()

            click.echo("MnemoCore Health Status")
            click.echo("=" * 40)
            click.echo()

            status = health_data.get("status", "unknown")
            click.echo(f"Status: {status}")
            click.echo(f"Initialized: {health_data.get('initialized', False)}")
            click.echo(f"Timestamp: {health_data.get('timestamp', 'N/A')}")
            click.echo()

            # Tier stats
            tiers = health_data.get("tiers", {})
            if tiers:
                click.echo("Tiers:")
                total_memories = 0
                for tier_name, tier_stats in tiers.items():
                    if isinstance(tier_stats, dict):
                        count = tier_stats.get("count", 0)
                        total_memories += count
                        click.echo(f"  {tier_name.upper()}: {count} memories")
                click.echo(f"  Total: {total_memories} memories")
                click.echo()

            # Background workers
            workers = health_data.get("background_workers", {})
            if workers:
                click.echo("Background Workers:")
                for name, worker in workers.items():
                    running = worker.get("running", False)
                    status_str = "Running" if running else "Stopped"
                    click.echo(f"  {name}: {status_str}")
                click.echo()

            # Qdrant
            qdrant = health_data.get("qdrant")
            if qdrant:
                if "error" in qdrant:
                    click.echo(f"Qdrant: Error - {qdrant['error']}")
                else:
                    click.echo(f"Qdrant: Connected")

        except Exception as e:
            click.echo(f"Error checking health: {e}", err=True)

    return _health(ctx)


# Import and register additional commands
from .commands import register_commands
register_commands(cli)


if __name__ == "__main__":
    cli()
