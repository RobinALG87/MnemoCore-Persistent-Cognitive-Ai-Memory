"""
Shared utility functions for HAIMEngine modules.

This module provides common utilities used across engine_core.py, engine_coordinator.py,
and engine_lifecycle.py to avoid code duplication.
"""

import asyncio
import functools
import hashlib
from typing import Callable, Optional, TypeVar, ParamSpec

import numpy as np
from loguru import logger

P = ParamSpec('P')
T = TypeVar('T')


# =============================================================================
# Thread Pool Executor Helper
# =============================================================================

async def run_in_thread(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """
    Run a blocking function in a thread pool executor.

    This is a utility for running CPU-bound or I/O-bound blocking operations
    without blocking the async event loop.

    Args:
        func: The blocking function to run.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The return value of the function.

    Example:
        result = await run_in_thread(some_blocking_func, arg1, arg2, kwarg=value)
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


# =============================================================================
# Token Vector Generation (Legacy Compatibility)
# =============================================================================

def get_token_vector(token: str, dimension: int) -> np.ndarray:
    """
    Generate a deterministic token vector using a cached approach.

    This function creates deterministic pseudo-random vectors for tokens,
    which is useful for legacy compatibility with older encoding methods.

    Note:
        This is decorated with lru_cache at the call site if caching is needed.
        The cache is typically applied per-engine instance to avoid memory leaks.

    Args:
        token: The token string to generate a vector for.
        dimension: The dimensionality of the output vector.

    Returns:
        A numpy array of shape (dimension,) with values in {-1, 1}.
    """
    seed_bytes = hashlib.shake_256(token.encode()).digest(4)
    seed = int.from_bytes(seed_bytes, 'little')
    return np.random.RandomState(seed).choice([-1, 1], size=dimension)


# =============================================================================
# Async Task Exception Handling
# =============================================================================

def log_task_exception(task: asyncio.Task) -> None:
    """
    Callback to log exceptions from fire-and-forget asyncio tasks.

    Use this with asyncio.ensure_future() to prevent silent exception loss:

        task = asyncio.ensure_future(some_coro())
        task.add_done_callback(log_task_exception)

    Args:
        task: The completed asyncio.Task to check for exceptions.
    """
    try:
        exc = task.exception()
        if exc is not None:
            logger.error(
                f"Async task {task.get_name()} failed with exception: {exc}",
                exc_info=exc,
            )
    except asyncio.CancelledError:
        # Task was cancelled, this is not an error
        logger.debug(f"Async task {task.get_name()} was cancelled")
    except Exception as e:
        # Unexpected error retrieving the exception
        logger.error(f"Error retrieving exception from task: {e}")


def safe_ensure_future(coro, *, name: Optional[str] = None) -> asyncio.Task:
    """
    Create an asyncio.Task with automatic exception logging.

    This wraps asyncio.ensure_future() with a done callback that logs
    any exceptions, preventing silent failure of fire-and-forget tasks.

    Args:
        coro: The coroutine to schedule.
        name: Optional name for the task (for debugging).

    Returns:
        The scheduled asyncio.Task with exception logging attached.

    Example:
        safe_ensure_future(some_background_operation(), name="bg_op")
    """
    task = asyncio.ensure_future(coro)
    if name:
        try:
            task.set_name(name)
        except AttributeError:
            pass  # Python < 3.8 doesn't have set_name
    task.add_done_callback(log_task_exception)
    return task
