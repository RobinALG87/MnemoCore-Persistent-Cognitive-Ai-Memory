import numpy as np
import pytest

from mnemocore.core.synapse import SynapticConnection
from mnemocore.core.synapse_index import SynapseIndex


def test_get_multi_hop_neighbors():
    index = SynapseIndex()

    # Create nodes A, B, C, D
    # A <-> B (weight 0.8)
    # B <-> C (weight 0.5)
    # C <-> D (weight 0.9)
    # A <-> C (weight 0.1)

    syn_ab = index.add_or_fire("A", "B")
    syn_ab.strength = 0.8

    syn_bc = index.add_or_fire("B", "C")
    syn_bc.strength = 0.5

    syn_cd = index.add_or_fire("C", "D")
    syn_cd.strength = 0.9

    syn_ac = index.add_or_fire("A", "C")
    syn_ac.strength = 0.1

    # 1 hop from A
    hops_1 = index.get_multi_hop_neighbors("A", depth=1)
    assert hops_1["B"] == pytest.approx(0.8)
    assert hops_1["C"] == pytest.approx(0.1)
    assert "D" not in hops_1

    hops_2 = index.get_multi_hop_neighbors("A", depth=2)
    assert hops_2["B"] == pytest.approx(0.8)
    assert hops_2["C"] == pytest.approx(0.4)
    assert hops_2["D"] == pytest.approx(0.09)

    hops_3 = index.get_multi_hop_neighbors("A", depth=3)
    assert hops_3["D"] == pytest.approx(0.36)
