"""Tests for the borehole-block permutation primitive (prereg P-T1/T3)."""

from __future__ import annotations

import numpy as np
import pytest

from national.evaluation.grouped_null import block_permutation_indices


def _toy(n_boreholes: int = 6, rows_per: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups, depth, strata = [], [], []
    for b in range(n_boreholes):
        for r in range(rows_per):
            groups.append(f"b{b}")
            depth.append(float(r))
            strata.append("east" if b < n_boreholes // 2 else "west")
    return (np.asarray(groups), np.asarray(depth), np.asarray(strata),
            np.random.default_rng(seed))


def test_blocks_move_whole(seed: int = 1) -> None:
    """Every row of one borehole must receive text from ONE donor borehole."""
    groups, depth, strata, rng = _toy(seed=seed)
    idx = block_permutation_indices(groups, depth, None, rng)
    for g in np.unique(groups):
        donor_groups = np.unique(groups[idx[groups == g]])
        assert len(donor_groups) == 1, f"{g} received text from {donor_groups}"


def test_depth_rank_preserved() -> None:
    """Within a borehole, donated rows must arrive shallow-to-deep."""
    groups, depth, strata, rng = _toy()
    idx = block_permutation_indices(groups, depth, None, rng)
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]
        order = np.argsort(depth[rows])
        donated_depth = depth[idx[rows[order]]]
        assert np.all(np.diff(donated_depth) >= 0)


def test_strata_respected() -> None:
    """With strata, donors must come from the recipient's own stratum."""
    groups, depth, strata, rng = _toy(n_boreholes=10)
    idx = block_permutation_indices(groups, depth, strata, rng)
    for i in range(len(groups)):
        assert strata[idx[i]] == strata[i]


def test_unequal_block_lengths_do_not_clip_or_duplicate() -> None:
    """Boreholes of different lengths must NOT be block-swapped by clipping.

    This test previously asserted the opposite -- that a 5-layer recipient
    receiving a 2-layer donor reuses the donor's deepest row three times. The
    2026-08-13 audit established that this makes the index map a non-injective
    mapping rather than a permutation (42.7% duplicated rows on the real P-T1
    population), so the observed statistic is not exchangeable with the null
    draws. Length-mismatched boreholes are now handled rank-wise instead, and
    the map is always a bijection.
    """
    groups = np.asarray(["a"] * 5 + ["b"] * 2)
    depth = np.asarray([0, 1, 2, 3, 4, 0, 1], dtype=float)
    for seed in range(20):
        idx = block_permutation_indices(groups, depth, None,
                                        np.random.default_rng(seed))
        assert np.array_equal(np.sort(idx), np.arange(len(groups))), (
            "must be a permutation for every draw, not only on average")
        # ranks 0 and 1 exist in both boreholes and so may be exchanged;
        # ranks 2-4 exist only in "a" and must therefore stay put.
        assert set(idx[2:5]) == {2, 3, 4}
        # no row may be duplicated
        assert len(set(idx)) == len(groups)


def test_permutation_is_reproducible() -> None:
    groups, depth, strata, _ = _toy()
    i1 = block_permutation_indices(groups, depth, strata, np.random.default_rng(7))
    i2 = block_permutation_indices(groups, depth, strata, np.random.default_rng(7))
    np.testing.assert_array_equal(i1, i2)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        block_permutation_indices(np.asarray(["a"]), np.asarray([0.0, 1.0]), None,
                                  np.random.default_rng(0))


# ---------------------------------------------------------- bijectivity
# These guard the defect found in the 2026-08-13 audit: the previous donor
# assignment clipped a long recipient's block to its shorter donor, so 42.7%
# of null-arm rows on the real P-T1 population were duplicates of the donor's
# deepest layer and 22,547 source rows were never used. That is not a
# permutation, and a null draw carrying less text diversity than the observed
# arm is not exchangeable with it.


def _ragged_frame(n_boreholes=60, seed=0, n_strata=3):
    """Boreholes with deliberately UNEQUAL block lengths, which is the case
    the old implementation got wrong."""
    rng = np.random.default_rng(seed)
    g, d, s = [], [], []
    for b in range(n_boreholes):
        L = int(rng.integers(1, 15))          # 1..14 layers: many singletons
        st = f"s{b % n_strata}"
        for k in range(L):
            g.append(f"b{b}")
            d.append(1.0 + k)
            s.append(st)
    return np.array(g), np.array(d, dtype=float), np.array(s)


def test_block_permutation_is_a_bijection_on_ragged_blocks():
    from national.evaluation.grouped_null import block_permutation_indices
    g, d, s = _ragged_frame()
    for seed in range(5):
        idx = block_permutation_indices(g, d, s, np.random.default_rng(seed))
        assert np.array_equal(np.sort(idx), np.arange(len(g))), (
            "index map must be a permutation: every embedding used exactly once")


def test_block_permutation_is_a_bijection_without_strata():
    from national.evaluation.grouped_null import block_permutation_indices
    g, d, _ = _ragged_frame()
    idx = block_permutation_indices(g, d, None, np.random.default_rng(1))
    assert np.array_equal(np.sort(idx), np.arange(len(g)))


def test_donor_is_always_inside_the_stratum():
    from national.evaluation.grouped_null import block_permutation_indices
    g, d, s = _ragged_frame()
    idx = block_permutation_indices(g, d, s, np.random.default_rng(2))
    assert (s[idx] == s).all(), "a row must never receive text from another stratum"


def test_length_matched_blocks_keep_their_depth_order_and_change_borehole():
    """For a borehole handled by step 1, the whole block comes from ONE other
    borehole, in the same shallow-to-deep order."""
    from national.evaluation.grouped_null import block_permutation_indices
    # every borehole has 6 layers, so step 1 handles all of them
    g = np.array([f"b{b}" for b in range(20) for _ in range(6)])
    d = np.array([1.0 + k for _ in range(20) for k in range(6)])
    idx = block_permutation_indices(g, d, None, np.random.default_rng(3))
    assert np.array_equal(np.sort(idx), np.arange(len(g)))
    for b in range(20):
        rows = np.flatnonzero(g == f"b{b}")
        donors = g[idx[rows]]
        assert len(set(donors)) == 1, "a matched block must come from one donor"
        assert donors[0] != f"b{b}", "a derangement must not return own text"
        # depth order preserved: donor rows are consecutive and ascending
        assert np.all(np.diff(idx[rows]) == 1)


def test_diagnostics_account_for_every_row():
    from national.evaluation.grouped_null import block_permutation_indices
    g, d, s = _ragged_frame()
    idx, diag = block_permutation_indices(
        g, d, s, np.random.default_rng(4), return_diagnostics=True)
    assert diag["is_bijection"] is True
    total = (diag["frac_block_matched"] + diag["frac_rank_fallback"]
             + diag["frac_self_retained"])
    assert abs(total - 1.0) < 1e-6, f"paths must partition the rows, got {total}"
    # a row keeps its own embedding only where the diagnostics say so
    # diagnostics are rounded to 6 dp for the provenance JSON
    assert abs((idx == np.arange(len(g))).mean()
               - diag["frac_self_retained"]) < 1e-6


def test_same_seed_reproduces_the_same_draw():
    from national.evaluation.grouped_null import block_permutation_indices
    g, d, s = _ragged_frame()
    a = block_permutation_indices(g, d, s, np.random.default_rng(7))
    b = block_permutation_indices(g, d, s, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_legacy_clipped_permutation_is_not_a_bijection():
    """Regression guard. If someone reinstates the clipping rule, this fails."""
    from national.evaluation.grouped_null import legacy_clipped_block_permutation
    g, d, s = _ragged_frame()
    idx = legacy_clipped_block_permutation(g, d, s, np.random.default_rng(0))
    assert len(np.unique(idx)) < len(g), (
        "the legacy routine is retained only to quantify its bias; it duplicates "
        "rows by construction and must never be used for inference")
