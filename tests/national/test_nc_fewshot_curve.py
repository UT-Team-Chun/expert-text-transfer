"""Unit test for the borehole-holdout construction of the few-shot curve
(P-T6): whole boreholes only, deterministic, disjoint pool."""

from __future__ import annotations

import numpy as np

from scripts.nc_fewshot_curve import borehole_holdout


def test_holdout_is_borehole_atomic_and_disjoint() -> None:
    groups = np.repeat([f"b{i}" for i in range(20)], 5)
    hold, pool = borehole_holdout(groups, frac=0.5, seed=0)
    held_b = set(np.unique(groups[hold]))
    # atomic: every row of a held borehole is held
    for b in held_b:
        assert hold[groups == b].all()
    # disjoint pool
    assert held_b.isdisjoint(set(pool))
    assert len(held_b) + len(pool) == 20
    assert len(held_b) == 10


def test_holdout_deterministic() -> None:
    groups = np.repeat([f"b{i}" for i in range(30)], 3)
    h1, p1 = borehole_holdout(groups, seed=7)
    h2, p2 = borehole_holdout(groups, seed=7)
    np.testing.assert_array_equal(h1, h2)
    np.testing.assert_array_equal(p1, p2)
