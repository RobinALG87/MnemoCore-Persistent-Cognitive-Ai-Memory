"""
Process utilities for MnemoCore optimizations.
"""
import os
from loguru import logger

try:
    import psutil
except ImportError:
    psutil = None

def lower_process_priority():
    """
    Lowers the current process priority to yield CPU to the main thread and OS.
    Critical for background daemons like Subconscious AI ensuring they don't starve
    the main API or edge device hardware (e.g. Raspberry Pi).
    """
    try:
        if os.name == 'nt':
            if psutil is None:
                logger.debug("psutil not installed, could not lower Windows process priority")
                return
            p = psutil.Process(os.getpid())
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            logger.debug("Lowered Windows process priority to BELOW_NORMAL")
        else:
            # Unix-like systems
            os.nice(10)
            logger.debug("Lowered Unix Unix process niceness +10")
    except Exception as e:
        logger.debug(f"Failed to lower process priority: {e}")
