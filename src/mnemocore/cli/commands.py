"""
Additional CLI Commands for MnemoCore

Provides specialized commands for:
- Associative queries
- Concept management
- Memory binding
- Batch operations
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List

import click

from loguru import logger


@click.group()
def associations():
    """Memory association and connection commands."""
    pass


@associations.command()
@click.argument("memory_id", required=True)
@click.option(
    "--depth",
    "-d",
    type=int,
    default=2,
    help="Max hop depth for associations",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=20,
    help="Max number of associations to return",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
@click.pass_context
def find(ctx, memory_id: str, depth: int, limit: int, output_json: bool):
    """
    Find associated memories following synaptic connections.

    Example:
        mnemocore-cli associations find mem_abc123 --depth 2
    """
    async def _find():
        from mnemocore.core.engine import HAIMEngine
        from mnemocore.core.config import load_config

        # Load config
        config_path = ctx.obj.get("config_path")
        config = load_config(Path(config_path)) if config_path and Path(config_path).exists() else None

        # Create engine
        engine = HAIMEngine(config=config)
        await engine.initialize()

        try:
            # Get synaptic path
            path = await engine.get_synaptic_path(memory_id, max_depth=depth, limit=limit)

            if output_json:
                result = {
                    "source_memory": memory_id,
                    "associations": [
                        {
                            "id": mem_id,
                            "strength": strength,
                            "path": path_str,
                        }
                        for mem_id, strength, path_str in path
                    ],
                    "count": len(path),
                }
                click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                if not path:
                    click.echo(f"No associations found for memory: {memory_id}")
                    return

                click.echo(f"Found {len(path)} associations for {memory_id}:")
                click.echo()

                for i, (mem_id, strength, path_str) in enumerate(path, 1):
                    node = await engine.tier_manager.get_memory(mem_id)
                    content = node.content[:50] + "..." if node and len(node.content) > 50 else (node.content if node else "N/A")
                    click.echo(f"{i}. [{mem_id}] (strength: {strength:.2f})")
                    click.echo(f"   Content: {content}")
                    click.echo(f"   Path: {' -> '.join(path_str)}")
                    click.echo()

        except Exception as e:
            click.echo(f"Error finding associations: {e}", err=True)
        finally:
            await engine.close()

    asyncio.run(_find())


@click.group()
def concepts():
    """Concept and symbolic memory commands."""
    pass


@concepts.command()
@click.argument("name", required=True)
@click.argument("description", required=True)
@click.option(
    "--examples",
    "-e",
    multiple=True,
    help="Example items for this concept",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
@click.pass_context
def define(ctx, name: str, description: str, examples: tuple, output_json: bool):
    """
    Define a conceptual symbol.

    Example:
        mnemocore-cli concepts define "programming" "Writing code to solve problems" -e "Python" -e "JavaScript"
    """
    async def _define():
        from mnemocore.core.engine import HAIMEngine
        from mnemocore.core.config import load_config

        # Load config
        config_path = ctx.obj.get("config_path")
        config = load_config(Path(config_path)) if config_path and Path(config_path).exists() else None

        # Create engine
        engine = HAIMEngine(config=config)
        await engine.initialize()

        try:
            # Define concept
            concept_id = await asyncio.to_thread(
                engine.define_concept,
                name,
                description,
                examples=list(examples) if examples else None,
            )

            if output_json:
                result = {
                    "success": True,
                    "concept_id": concept_id,
                    "name": name,
                    "description": description,
                    "examples": list(examples),
                }
                click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                click.echo(f"Defined concept: {name} ({concept_id})")
                if examples:
                    click.echo(f"Examples: {', '.join(examples)}")

        except Exception as e:
            if output_json:
                error_result = {"success": False, "error": str(e)}
                click.echo(json.dumps(error_result, indent=2))
            else:
                click.echo(f"Error defining concept: {e}", err=True)
        finally:
            await engine.close()

    asyncio.run(_define())


@concepts.command()
@click.argument("query", required=True)
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=5,
    help="Number of results",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
@click.pass_context
def inspect(ctx, query: str, top_k: int, output_json: bool):
    """
    Inspect a concept or related concepts.

    Example:
        mnemocore-cli concepts inspect "programming"
    """
    async def _inspect():
        from mnemocore.core.engine import HAIMEngine
        from mnemocore.core.config import load_config

        # Load config
        config_path = ctx.obj.get("config_path")
        config = load_config(Path(config_path)) if config_path and Path(config_path).exists() else None

        # Create engine
        engine = HAIMEngine(config=config)
        await engine.initialize()

        try:
            # Inspect concept
            result = await asyncio.to_thread(
                engine.inspect_concept,
                query,
                top_k=top_k,
            )

            if output_json:
                click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                if not result:
                    click.echo(f"No concept found matching: {query}")
                    return

                click.echo(f"Concept: {result.get('name', 'N/A')}")
                click.echo(f"Description: {result.get('description', 'N/A')}")
                click.echo()

                related = result.get("related_concepts", [])
                if related:
                    click.echo("Related concepts:")
                    for i, rel in enumerate(related[:top_k], 1):
                        click.echo(f"  {i}. {rel.get('name', 'N/A')} (similarity: {rel.get('similarity', 0):.2f})")

        except Exception as e:
            if output_json:
                error_result = {"success": False, "error": str(e)}
                click.echo(json.dumps(error_result, indent=2))
            else:
                click.echo(f"Error inspecting concept: {e}", err=True)
        finally:
            await engine.close()

    asyncio.run(_inspect())


@click.group()
def batch():
    """Batch operations on memories."""
    pass


@batch.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "jsonl"]),
    default="jsonl",
    help="Input file format",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
@click.pass_context
def store(ctx, file: str, format: str, output_json: bool):
    """
    Batch store memories from a file.

    File should contain memories with 'content' and optional 'metadata'.

    Example:
        mnemocore-cli batch store memories.jsonl
    """
    async def _batch_store():
        from mnemocore.core.engine import HAIMEngine
        from mnemocore.core.config import load_config

        # Load config
        config_path = ctx.obj.get("config_path")
        config = load_config(Path(config_path)) if config_path and Path(config_path).exists() else None

        # Create engine
        engine = HAIMEngine(config=config)
        await engine.initialize()

        try:
            file_path = Path(file)
            stored_count = 0
            failed_count = 0
            results = []

            with open(file_path, "r", encoding="utf-8") as f:
                if format == "json":
                    memories = json.load(f)
                    if isinstance(memories, dict):
                        memories = [memories]
                else:  # jsonl
                    memories = [json.loads(line) for line in f if line.strip()]

                click.echo(f"Storing {len(memories)} memories...", err=True)

                for i, mem in enumerate(memories, 1):
                    try:
                        content = mem.get("content", "")
                        if not content:
                            failed_count += 1
                            continue

                        metadata = mem.get("metadata", {})
                        metadata["batch_import"] = True

                        memory_id = await asyncio.to_thread(
                            engine.store,
                            content,
                            metadata=metadata,
                        )

                        stored_count += 1
                        results.append({
                            "index": i,
                            "memory_id": memory_id,
                            "success": True,
                        })

                        if i % 100 == 0:
                            click.echo(f"Progress: {i}/{len(memories)}", err=True)

                    except Exception as e:
                        failed_count += 1
                        results.append({
                            "index": i,
                            "error": str(e),
                            "success": False,
                        })

            if output_json:
                result = {
                    "success": True,
                    "total": len(memories),
                    "stored": stored_count,
                    "failed": failed_count,
                    "results": results,
                }
                click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                click.echo()
                click.echo(f"Batch store complete:")
                click.echo(f"  Total: {len(memories)}")
                click.echo(f"  Stored: {stored_count}")
                click.echo(f"  Failed: {failed_count}")

        except Exception as e:
            if output_json:
                error_result = {"success": False, "error": str(e)}
                click.echo(json.dumps(error_result, indent=2))
            else:
                click.echo(f"Error in batch store: {e}", err=True)
        finally:
            await engine.close()

    asyncio.run(_batch_store())


@batch.command()
@click.argument("query", required=True)
@click.argument("file", type=click.Path())
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "jsonl"]),
    default="jsonl",
    help="Output file format",
)
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=100,
    help="Maximum number of results",
)
@click.pass_context
def export(ctx, query: str, file: str, format: str, top_k: int):
    """
    Batch export memories matching a query.

    Example:
        mnemocore-cli batch export "Python" python_memories.jsonl
    """
    async def _batch_export():
        from mnemocore.core.engine import HAIMEngine
        from mnemocore.core.config import load_config

        # Load config
        config_path = ctx.obj.get("config_path")
        config = load_config(Path(config_path)) if config_path and Path(config_path).exists() else None

        # Create engine
        engine = HAIMEngine(config=config)
        await engine.initialize()

        try:
            # Query memories
            results = await asyncio.to_thread(
                engine.query,
                query,
                top_k=top_k,
            )

            output_path = Path(file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            exported_count = 0

            with open(output_path, "w", encoding="utf-8") as f:
                if format == "json":
                    memories = []
                    for memory_id, score in results:
                        node = await engine.tier_manager.get_memory(memory_id)
                        if node:
                            memories.append({
                                "id": memory_id,
                                "content": node.content,
                                "score": score,
                                "created_at": node.created_at.isoformat() if node.created_at else None,
                                "tier": node.tier,
                                "metadata": node.metadata or {},
                            })
                            exported_count += 1
                    json.dump(memories, f, indent=2, ensure_ascii=False)
                else:  # jsonl
                    for memory_id, score in results:
                        node = await engine.tier_manager.get_memory(memory_id)
                        if node:
                            mem_data = {
                                "id": memory_id,
                                "content": node.content,
                                "score": score,
                                "created_at": node.created_at.isoformat() if node.created_at else None,
                                "tier": node.tier,
                                "metadata": node.metadata or {},
                            }
                            f.write(json.dumps(mem_data, ensure_ascii=False) + "\n")
                            exported_count += 1

            click.echo(f"Exported {exported_count} memories to {output_path}", err=True)

        except Exception as e:
            click.echo(f"Error in batch export: {e}", err=True)
        finally:
            await engine.close()

    asyncio.run(_batch_export())


@click.command()
@click.argument("memory_id_1", required=True)
@click.argument("memory_id_2", required=True)
@click.option(
    "--strength",
    "-s",
    type=float,
    default=0.5,
    help="Connection strength (0.0-1.0)",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
@click.pass_context
def bind(ctx, memory_id_1: str, memory_id_2: str, strength: float, output_json: bool):
    """
    Create a synaptic connection between two memories.

    Example:
        mnemocore-cli bind mem_abc123 mem_def456 --strength 0.8
    """
    async def _bind():
        from mnemocore.core.engine import HAIMEngine
        from mnemocore.core.config import load_config

        # Load config
        config_path = ctx.obj.get("config_path")
        config = load_config(Path(config_path)) if config_path and Path(config_path).exists() else None

        # Create engine
        engine = HAIMEngine(config=config)
        await engine.initialize()

        try:
            # Bind memories
            await asyncio.to_thread(
                engine.bind_memories,
                memory_id_1,
                memory_id_2,
                strength=strength,
            )

            if output_json:
                result = {
                    "success": True,
                    "memory_id_1": memory_id_1,
                    "memory_id_2": memory_id_2,
                    "strength": strength,
                }
                click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                click.echo(f"Bound memories: {memory_id_1} <-> {memory_id_2} (strength: {strength})")

        except Exception as e:
            if output_json:
                error_result = {"success": False, "error": str(e)}
                click.echo(json.dumps(error_result, indent=2))
            else:
                click.echo(f"Error binding memories: {e}", err=True)
        finally:
            await engine.close()

    asyncio.run(_bind())


@click.command()
@click.argument("query", required=True)
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=5,
    help="Number of results",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
@click.pass_context
def associative(ctx, query: str, top_k: int, output_json: bool):
    """
    Perform associative query (follows synaptic connections).

    Example:
        mnemocore-cli associative "Python programming"
    """
    async def _associative():
        from mnemocore.core.engine import HAIMEngine
        from mnemocore.core.config import load_config

        # Load config
        config_path = ctx.obj.get("config_path")
        config = load_config(Path(config_path)) if config_path and Path(config_path).exists() else None

        # Create engine
        engine = HAIMEngine(config=config)
        await engine.initialize()

        try:
            # Associative query
            results = await engine.associative_query(query, top_k=top_k)

            if output_json:
                result = {
                    "query": query,
                    "results": [
                        {
                            "id": mem_id,
                            "content": content,
                            "score": score,
                            "associations": associations,
                        }
                        for mem_id, content, score, associations in results
                    ],
                    "count": len(results),
                }
                click.echo(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                if not results:
                    click.echo(f"No associative results for: {query}")
                    return

                click.echo(f"Associative results for: {query}")
                click.echo()

                for i, (mem_id, content, score, associations) in enumerate(results, 1):
                    content_preview = content[:60] + "..." if len(content) > 60 else content
                    click.echo(f"{i}. [{mem_id}] (score: {score:.2f})")
                    click.echo(f"   {content_preview}")
                    if associations:
                        assoc_str = ", ".join(associations[:3])
                        click.echo(f"   Associations: {assoc_str}")
                    click.echo()

        except Exception as e:
            if output_json:
                error_result = {"success": False, "error": str(e)}
                click.echo(json.dumps(error_result, indent=2))
            else:
                click.echo(f"Error in associative query: {e}", err=True)
        finally:
            await engine.close()

    asyncio.run(_associative())


# Add all commands to the main CLI
def register_commands(cli_group: click.Group) -> None:
    """Register all additional commands to the main CLI group."""
    # Add associations group
    assoc_group = associations
    cli_group.add_command(assoc_group, name="associations")

    # Add concepts group
    concepts_group = concepts
    cli_group.add_command(concepts_group, name="concepts")

    # Add batch group
    batch_group = batch
    cli_group.add_command(batch_group, name="batch")

    # Add individual commands
    cli_group.add_command(bind)
    cli_group.add_command(associative)
