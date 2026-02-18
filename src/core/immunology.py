"""
Auto-Associative Cleanup Loop — Vector Immunology (Phase 4.0)
==============================================================
Inspired by Biological Immune System & Hopfield-network attractor dynamics.

The "vector immune system" of MnemoCore continuously monitors the HOT tier for:

  1. Semantic Drift: memories whose stored HDV has drifted far from its own
     nearest-neighbor cluster (i.e. it is now an isolated outlier).
     → Action: re-encode from stored content string, or flag for review.

  2. Corrupted / Low-Signal Nodes: nodes with near-zero LTP AND high HDV entropy
     (effectively random vectors with no semantic content).
     → Action: quarantine (move to COLD) or prune.

  3. Stale Synaptic Noise: decayed synapses that waste adjacency-list memory.
     → Action: (delegates to engine.cleanup_decay)

  4. HDV Auto-correction (Attractor Convergence):
     A simplified Hopfield-style convergence step:
       v_clean = sign(W · v)
     where W = superposition of the k nearest clean prototype vectors.
     This "snaps" a slightly noisy vector to its nearest attractor basin.

Biological analogy:
  - Innate immune response  → fast outlier / corruption detection
  - Adaptive immune response → targeted re-encoding of drifted memories
  - Memory T-cells (long-lived) → proto-memories (semantic consolidation)

Public API:
    loop = ImmunologyLoop(engine)
    await loop.start()   # background task
    await loop.sweep()   # single on-demand sweep
    await loop.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import numpy as np

from .binary_hdv import BinaryHDV, majority_bundle
from .node import MemoryNode

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Configuration                                                      #
# ------------------------------------------------------------------ #

@dataclass
class ImmunologyConfig:
    """Tunable parameters for the immunology sweep."""
    sweep_interval_seconds: float = 300.0   # how often to run (default 5 min)
    drift_threshold: float = 0.40           # Hamming dist > this → drifted
    entropy_threshold: float = 0.48         # bit-balance entropy > this → corrupted
    min_ltp_to_keep: float = 0.05           # nodes below this AND corrupted → quarantine
    attractor_k: int = 5                    # neighbours used for attractor convergence
    attractor_enabled: bool = True          # run Hopfield attractor step
    re_encode_drifted: bool = True          # re-encode drifted nodes from content
    quarantine_corrupted: bool = True       # move corrupted nodes to COLD
    enabled: bool = True


# ------------------------------------------------------------------ #
#  Entropy helper                                                     #
# ------------------------------------------------------------------ #

def _bit_entropy(hdv: BinaryHDV) -> float:
    """
    Balance entropy of a binary vector (0 = all same bits, 0.5 = perfect balance).
    Defined as H = -p*log2(p) - (1-p)*log2(1-p)  where p = fraction of 1-bits.

    A healthy semantic vector should be close to 0.5 (≈ random yet meaningful).
    A corrupted vector may be severely imbalanced (entropy near 0).
    """
    bits = np.unpackbits(hdv.data)
    p = float(bits.sum()) / len(bits)
    if p <= 0 or p >= 1:
        return 0.0
    # normalised Shannon entropy ÷ max = 1 for p=0.5
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# ------------------------------------------------------------------ #
#  Main immunology loop                                               #
# ------------------------------------------------------------------ #

class ImmunologyLoop:
    """
    Autonomous background sweep that detects and neutralises
    corrupted / drifted vectors in the HOT tier.
    """

    def __init__(self, engine, config: Optional[ImmunologyConfig] = None):
        self.engine = engine
        self.cfg = config or ImmunologyConfig()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.last_sweep: Optional[datetime] = None
        self.cumulative_stats: Dict = {
            "sweeps": 0,
            "drifted_corrected": 0,
            "corrupted_quarantined": 0,
            "synapses_cleaned": 0,
        }

    # ---- Lifecycle ----------------------------------------------- #

    async def start(self) -> None:
        if not self.cfg.enabled:
            logger.info("ImmunologyLoop disabled by config.")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="immunology_loop")
        logger.info(
            f"ImmunologyLoop started — sweep every {self.cfg.sweep_interval_seconds}s"
        )

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ImmunologyLoop stopped.")

    # ---- Main loop ----------------------------------------------- #

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.cfg.sweep_interval_seconds)
                if self._running:
                    await self.sweep()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"ImmunologyLoop sweep error: {exc}", exc_info=True)
                await asyncio.sleep(30)

    # ---------------------------------------------------------------- #
    #  Core sweep                                                       #
    # ---------------------------------------------------------------- #

    async def sweep(self) -> Dict:
        """
        Run a single immunology sweep over the HOT tier.

        Returns:
            Stats dict for this sweep.
        """
        t0 = time.monotonic()
        nodes: List[MemoryNode] = await self.engine.tier_manager.get_hot_snapshot()

        if not nodes:
            return {}

        # Build reference matrix (all HOT node vectors)
        vecs = np.stack([n.hdv.data for n in nodes])  # (N, D/8)

        drifted_corrected = 0
        corrupted_quarantined = 0

        for i, node in enumerate(nodes):
            action = await self._assess_node(node, i, nodes, vecs)
            if action == "corrected":
                drifted_corrected += 1
            elif action == "quarantined":
                corrupted_quarantined += 1

        # Delegate stale synapse cleanup to the engine's own method
        await self.engine.cleanup_decay(threshold=0.05)

        elapsed = time.monotonic() - t0
        sweep_stats = {
            "nodes_scanned": len(nodes),
            "drifted_corrected": drifted_corrected,
            "corrupted_quarantined": corrupted_quarantined,
            "elapsed_seconds": round(elapsed, 3),
        }

        # Accumulate
        self.cumulative_stats["sweeps"] += 1
        self.cumulative_stats["drifted_corrected"] += drifted_corrected
        self.cumulative_stats["corrupted_quarantined"] += corrupted_quarantined
        self.last_sweep = datetime.now(timezone.utc)

        if drifted_corrected or corrupted_quarantined:
            logger.info(
                f"Immunology sweep — nodes={len(nodes)} "
                f"corrected={drifted_corrected} quarantined={corrupted_quarantined} "
                f"({elapsed*1000:.0f}ms)"
            )

        return sweep_stats

    # ---- Per-node assessment ------------------------------------- #

    async def _assess_node(
        self,
        node: MemoryNode,
        idx: int,
        all_nodes: List[MemoryNode],
        vecs: np.ndarray,
    ) -> str:
        """
        Assess a single node and take corrective action if needed.

        Returns:
            "ok" | "corrected" | "quarantined"
        """
        # --- 1. Entropy check (corruption detection) ---
        entropy = _bit_entropy(node.hdv)
        is_corrupted = entropy < (1.0 - self.cfg.entropy_threshold)

        if is_corrupted and node.ltp_strength < self.cfg.min_ltp_to_keep:
            if self.cfg.quarantine_corrupted:
                logger.warning(
                    f"Quarantining corrupted node {node.id[:8]} "
                    f"(entropy={entropy:.3f} ltp={node.ltp_strength:.3f})"
                )
                # Move to COLD by deleting from HOT/WARM and cold-archiving
                await self.engine.tier_manager.delete_memory(node.id)
                return "quarantined"

        # --- 2. Drift detection (proximity to nearest cluster) ---
        if self.cfg.re_encode_drifted or self.cfg.attractor_enabled:
            # Compute distances to all other nodes (vectorised XOR popcount)
            xor_all = np.bitwise_xor(vecs[idx : idx + 1], vecs)  # (1, D/8) vs (N, D/8)
            hamming_all = np.unpackbits(xor_all, axis=1).sum(axis=1).astype(np.float32)
            hamming_all /= vecs.shape[1] * 8
            hamming_all[idx] = 1.0  # exclude self

            # k nearest neighbours
            k = min(self.cfg.attractor_k, len(all_nodes) - 1)
            if k < 1:
                return "ok"

            nn_indices = np.argpartition(hamming_all, k)[:k]
            nn_min_dist = float(hamming_all[nn_indices].min())
            nn_mean_dist = float(hamming_all[nn_indices].mean())

            is_drifted = nn_min_dist > self.cfg.drift_threshold

            if is_drifted:
                if self.cfg.re_encode_drifted and node.content:
                    # Re-encode from source text to restore semantic fidelity
                    new_hdv = await asyncio.get_running_loop().run_in_executor(
                        None, self.engine.encode_content, node.content
                    )
                    node.hdv = new_hdv
                    # Update the packed vector in our local array
                    vecs[idx] = new_hdv.data
                    node.metadata["immune_re_encoded_at"] = datetime.now(timezone.utc).isoformat()
                    logger.debug(f"Re-encoded drifted node {node.id[:8]} (nn_min={nn_min_dist:.3f})")
                    return "corrected"

                elif self.cfg.attractor_enabled:
                    # Hopfield attractor: new_v = sign(bundle(neighbours))
                    nn_vecs = [all_nodes[i].hdv for i in nn_indices]
                    proto = majority_bundle(nn_vecs)
                    # Soft convergence: XOR blend – bits that agree with proto are kept
                    node.hdv = proto
                    vecs[idx] = proto.data
                    node.metadata["immune_attractor_at"] = datetime.now(timezone.utc).isoformat()
                    logger.debug(f"Attractor-converged drifted node {node.id[:8]}")
                    return "corrected"

        return "ok"

    @property
    def stats(self) -> Dict:
        return {
            **self.cumulative_stats,
            "last_sweep": self.last_sweep.isoformat() if self.last_sweep else None,
        }
