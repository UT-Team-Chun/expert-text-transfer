"""Borehole-block permutation for shuffled-embedding nulls (prereg P-T1/T3).

The historical null permutes embedding ROWS independently, which breaks the
within-borehole block structure of the text: layers of one borehole share a
logger, a template, a project and a stratigraphic context, so row-level
shuffling over-fragments that structure and can make the null too easy to
beat (the pre-review's P0-5). The block permutation implemented here severs
the text<->borehole link at the BOREHOLE level instead:

- every borehole is assigned a DONOR borehole drawn by permutation within a
  stratum (e.g. its region, or region x lithology class);
- each of the recipient's rows takes the donor's embedding at the same
  within-borehole depth rank, preserving the shallow-to-deep ordering of the
  donated text block.

Capacity, dimensionality, per-fold PCA structure, the number of text-bearing
rows AND the block granularity of the text are all preserved; only the
assignment of whole text blocks to boreholes is randomised.

**The reassignment must be a bijection on rows.** An earlier version of this
module assigned donors by a free permutation within the stratum and then
CLIPPED the donor's block to its own length, so a recipient longer than its
donor received the donor's deepest layer repeated. Measured on the P-T1
population that made 42.7% of null-arm rows duplicates, left 22,547 source
rows unused, and reused one source row 77 times. Such a draw is not a
permutation of the embedding matrix: the null arm carried systematically less
text diversity than the observed arm, so the observed statistic was not
exchangeable with the null draws and the measured effect was biased. The
defective routine is kept below as ``legacy_clipped_block_permutation`` for the
sole purpose of quantifying that bias, and is not used by any analysis.

The scheme implemented by :func:`block_permutation_indices` restores
bijectivity in two steps:

1. **Length-matched block swap.** Boreholes are grouped by (stratum, number of
   logged layers); within each group of two or more, a random DERANGEMENT
   reassigns whole blocks. Donor and recipient have equal length by
   construction, so ranks align exactly, no row is duplicated or dropped, and
   no borehole receives its own text.
2. **Rank-wise residual.** Boreholes whose (stratum, length) group has a single
   member are pooled per stratum and permuted RANK BY RANK: the layers sitting
   at a given within-borehole depth rank are deranged among the pooled
   boreholes that have that rank. This is also a bijection; it sacrifices some
   block coherence for the minority of boreholes with an unusual length.

A row can retain its own embedding only when it is the sole row at its rank in
its stratum's residual pool -- unavoidable, strictly conservative (the null
keeps real text), and counted in the diagnostics.

Measured coverage of step 1 (rows): 99.5% on the full text-bearing population
under a region stratum and 97.6% under region x lithology-macro; 91.1% and
70.3% respectively on the 500-files-per-region subsample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["block_permutation_indices"]


def _derangement(m: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation of ``range(m)`` with no fixed point.

    Rejection sampling: a uniform permutation is a derangement with probability
    ~1/e, so this accepts after ~2.7 draws on average. ``m == 1`` has no
    derangement and returns the identity; callers must treat that case as
    "cannot permute" and account for it.
    """
    if m <= 1:
        return np.zeros(m, dtype=np.int64)
    if m == 2:
        return np.array([1, 0], dtype=np.int64)
    while True:
        p = rng.permutation(m)
        if not np.any(p == np.arange(m)):
            return p.astype(np.int64)


def block_permutation_indices(
    groups: np.ndarray,
    depth: np.ndarray,
    strata: np.ndarray | None,
    rng: np.random.Generator,
    *,
    return_diagnostics: bool = False,
):
    """Row-index map implementing a bijective borehole-block text permutation.

    Args:
        groups: per-row borehole identifier (any hashable dtype).
        depth:  per-row depth, defining the within-borehole rank order.
        strata: optional per-row stratum label; boreholes are only permuted
            within a stratum. ``None`` = one global stratum. A borehole's
            stratum is its first row's label.
        rng: the permutation source.
        return_diagnostics: also return a dict describing how the draw was
            realised (see module docstring).

    Returns:
        ``idx`` such that ``emb[idx]`` is the block-permuted embedding matrix.
        ``idx`` is a PERMUTATION of ``arange(len(groups))``: every embedding is
        used exactly once.
    """
    n = len(groups)
    if len(depth) != n or (strata is not None and len(strata) != n):
        raise ValueError("groups / depth / strata must have equal length")

    df = pd.DataFrame({
        "g": np.asarray(groups),
        "d": np.asarray(depth, dtype=np.float64),
        "row": np.arange(n),
    })
    df["rank"] = df.groupby("g", sort=False)["d"].rank(method="first").astype(int) - 1

    if strata is None:
        b_strat = pd.Series(0, index=df["g"].unique())
    else:
        b_strat = pd.Series(np.asarray(strata), index=df.index).groupby(
            df["g"], sort=False).first()

    rows_by_g = {g: sub.sort_values("rank")["row"].to_numpy()
                 for g, sub in df.groupby("g", sort=False)}
    length_of = {g: len(r) for g, r in rows_by_g.items()}

    idx = np.arange(n, dtype=np.int64)
    n_block, n_rank, n_self = 0, 0, 0

    # ---- step 1: length-matched whole-block derangement --------------------
    key = pd.DataFrame({"g": b_strat.index, "st": b_strat.to_numpy()})
    key["L"] = key["g"].map(length_of)
    residual: dict = {}
    for (_st, _L), sub in key.groupby(["st", "L"], sort=False):
        gs = sub["g"].to_numpy()
        if len(gs) < 2:
            residual.setdefault(_st, []).append(gs[0])
            continue
        perm = _derangement(len(gs), rng)
        for g, dg in zip(gs, gs[perm]):
            idx[rows_by_g[g]] = rows_by_g[dg]
            n_block += len(rows_by_g[g])

    # ---- step 2: rank-wise derangement over the residual pool --------------
    for _st, gs in residual.items():
        if not gs:
            continue
        maxlen = max(length_of[g] for g in gs)
        for k in range(maxlen):
            at_rank = [g for g in gs if length_of[g] > k]
            if len(at_rank) < 2:
                if at_rank:
                    n_self += 1
                continue
            src = np.array([rows_by_g[g][k] for g in at_rank], dtype=np.int64)
            idx[src] = src[_derangement(len(at_rank), rng)]
            n_rank += len(at_rank)

    if return_diagnostics:
        uniq = len(np.unique(idx))
        diag = {
            "n_rows": n,
            "is_bijection": bool(uniq == n),
            "frac_block_matched": round(n_block / n, 6) if n else 0.0,
            "frac_rank_fallback": round(n_rank / n, 6) if n else 0.0,
            "frac_self_retained": round(n_self / n, 6) if n else 0.0,
        }
        return idx, diag
    return idx


def legacy_clipped_block_permutation(
    groups: np.ndarray,
    depth: np.ndarray,
    strata: np.ndarray | None,
    rng: np.random.Generator,
    *,
    return_diagnostics: bool = False,
):
    """DEFECTIVE predecessor of :func:`block_permutation_indices`.

    Assigns donors by a free permutation within the stratum and clips the
    donor's block to the recipient's length, duplicating the donor's deepest
    layer whenever the recipient is longer. The returned index map is NOT a
    permutation. Retained solely so the resulting bias can be measured and
    reported; no analysis may use it.
    """
    n = len(groups)
    if len(depth) != n or (strata is not None and len(strata) != n):
        raise ValueError("groups / depth / strata must have equal length")
    df = pd.DataFrame({
        "g": np.asarray(groups),
        "d": np.asarray(depth, dtype=np.float64),
        "row": np.arange(n),
    })
    df["rank"] = df.groupby("g", sort=False)["d"].rank(method="first").astype(int) - 1
    if strata is None:
        b_strat = {g: 0 for g in df["g"].unique()}
    else:
        s = pd.Series(np.asarray(strata), index=df.index)
        b_strat = s.groupby(df["g"], sort=False).first().to_dict()
    rows_by_g = {g: sub.sort_values("rank")["row"].to_numpy()
                 for g, sub in df.groupby("g", sort=False)}
    donor_of: dict = {}
    strat_to_gs: dict = {}
    for g, st in b_strat.items():
        strat_to_gs.setdefault(st, []).append(g)
    for st, gs in strat_to_gs.items():
        gs_arr = np.asarray(gs, dtype=object)
        perm = rng.permutation(len(gs_arr))
        for g, dg in zip(gs_arr, gs_arr[perm]):
            donor_of[g] = dg
    idx = np.empty(n, dtype=np.int64)
    for g, rows in rows_by_g.items():
        donor_rows = rows_by_g[donor_of[g]]
        take = np.minimum(np.arange(len(rows)), len(donor_rows) - 1)
        idx[rows] = donor_rows[take]
    if return_diagnostics:
        uniq = len(np.unique(idx))
        return idx, {
            "n_rows": n,
            "is_bijection": bool(uniq == n),
            "frac_duplicated_rows": round(1.0 - uniq / n, 6) if n else 0.0,
            "scheme": "LEGACY clipped (defective; measurement only)",
        }
    return idx
