#!/usr/bin/env python
"""Grouped-null content effect: the pre-registered primary analysis (P-T1/T2/T3).

Implements the protocol frozen in
``docs/research/2026-08-11_nc_text_preregistration.md``:

- **Baseline** (richest non-text): depth + elevation + river/coast distance +
  AIST regime(8)/litho-macro(14)/era(9) one-hots + a train-side KNN spatial
  prior (own borehole excluded) [+ raw lat/lon kept out -- the baseline is the
  ladder's ``plus_geology`` + KNN, not the thin rung].
- **Text arm**: strength-stripped (v2) lithology-only embedding, per-fold PCA
  fit on training regions only.
- **Null**: borehole-BLOCK permutation (``national.evaluation.grouped_null``),
  strata = training-region label, >=``--n-perm-block`` draws per fold,
  HGB seed cycled through ``--seeds`` so real and null share the seed set.
  A row-level null (``--n-perm-row`` draws) is run alongside for the P-T3
  sensitivity comparison.
- **Inference**: per-fold permutation p with the (1+r)/(1+n) correction;
  folds combined by the BONFERRONI-CORRECTED MINIMUM and the CAUCHY
  combination (never by averaging p, and not by Stouffer -- Stouffer assumes
  independent folds, and leave-one-region-out folds share most of their
  training rows, so it is anti-conservative here; it is still written to the
  artefact as a secondary figure). Borehole-block BCa interval over paired
  per-borehole losses, full leave-one-out jackknife, null accumulated over
  all draws (no refits); the 10^4 resample over the 8 held-out regions is
  reported as a descriptive spread, not as a confidence interval.

Covariates are attached by EXACT borehole identity from the v4id parquet
(``boring_file``), not by the legacy nearest-neighbour coordinate match.

CLI::

    cd backend
    .venv/bin/python -m scripts.nc_grouped_null --domain japan \
        --per-region-files 500 --sample-seed 42 \
        --n-perm-block 1000 --n-perm-row 200 \
        --out ../docs/research/2026-08-12_grouped_null_japan_s42.json

P-T5 (proper-noun strip + template normalisation) is the same protocol with a
different text arm -- add
``--strip-mode lithology_only_depersonalised``; the embedding cache key is
content-hashed, so the control gets its own cache entry automatically.
"""
from __future__ import annotations

# OpenMP cap BEFORE any sklearn import: HGB's parallel splitter spin-waits at
# barriers on sub-100k-row fits (observed 6.6 effective threads at ~0 speedup).
# threadpoolctl cannot fix this reliably -- libomp is loaded lazily after the
# limiter enumerates libraries -- so pin it at process start. Override with
# OMP_NUM_THREADS in the environment for large-data runs.
import os
# Measured on the 46k x 134 fit this protocol actually runs (M-series,
# 16 cores): 1 thread 5.35 s, 2 threads 9.59 s, 4 threads 9.96 s,
# 8 threads 15.80 s. `sample` shows the multi-thread time is spent in
# __kmp_barrier -> cthread_yield, i.e. the histogram/splitting kernels
# hit the OpenMP barrier far more often than they do useful work at this
# size. One thread per process is both faster and lets us fan the
# embarrassingly parallel region shards across the cores instead.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")


import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from national.evaluation.grouped_null import block_permutation_indices
from scripts.text_leakage_controls import (
    DEPERSONALISE_MODE,
    STRIP_MODES,
    STRIP_VOCAB_VERSION,
    apply_strip_mode,
    cache_tag,
    load_domain,
)
from scripts.uk_transfer_test import embed_texts

LOG = logging.getLogger("nc_grouped_null")
REPO = Path(__file__).resolve().parents[2]

K_NEIGHBOURS = 10


# ------------------------------------------------------------ features


def attach_v4id_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Exact identity join of the v4id per-borehole covariates.

    Brings absolute_elevation (surface part), river/coast distance and the
    three AIST codes onto the transfer frame by ``boring_file`` -- no
    nearest-neighbour tolerance, no rounded keys. Elevation in v4id is
    per-row (mouth - depth); we reconstruct the per-borehole collar
    elevation as ``absolute_elevation + depth_from_surface`` (constant
    within a borehole) and re-derive the per-row value on the transfer side.
    """
    cols = ["boring_file", "absolute_elevation", "depth_from_surface",
            "river_distance_km", "coast_distance_km", "regime_code",
            "aist_litho_macro_code", "aist_era_code"]
    v4 = pd.read_parquet(REPO / "data/features/borings_japan_v4id.parquet",
                         columns=cols)
    v4["boring_file"] = v4["boring_file"].astype(str)
    v4["collar_elevation"] = v4["absolute_elevation"] + v4["depth_from_surface"]
    per_file = (
        v4.groupby("boring_file", observed=True)
        .agg(collar_elevation=("collar_elevation", "first"),
             river_distance_km=("river_distance_km", "first"),
             coast_distance_km=("coast_distance_km", "first"),
             regime_code=("regime_code", "first"),
             aist_litho_macro_code=("aist_litho_macro_code", "first"),
             aist_era_code=("aist_era_code", "first"))
        .reset_index()
    )
    out = df.copy()
    out["boring_file"] = out["boring_file"].astype(str)
    out = out.merge(per_file, on="boring_file", how="left", validate="m:1")
    out["absolute_elevation"] = out["collar_elevation"] - out["depth_from_surface"]
    return out.drop(columns=["collar_elevation"])


def build_rich_features(df: pd.DataFrame, domain: str) -> tuple[pd.DataFrame, list[str]]:
    """Rich non-text baseline columns per the prereg."""
    if domain == "japan":
        joined = attach_v4id_covariates(df)
        cov = joined["regime_code"].notna().to_numpy()
        LOG.info("v4id covariate coverage: %.3f (%d/%d rows)",
                 cov.mean(), int(cov.sum()), len(joined))
        joined = joined[cov].reset_index(drop=True)
        oh = pd.concat(
            [pd.get_dummies(joined["regime_code"].astype("Int64"), prefix="rg"),
             pd.get_dummies(joined["aist_litho_macro_code"].astype("Int64"), prefix="lt"),
             pd.get_dummies(joined["aist_era_code"].astype("Int64"), prefix="er")],
            axis=1).astype(np.float32)
        out = pd.concat([joined, oh], axis=1)
        base = (["depth_from_surface", "absolute_elevation",
                 "river_distance_km", "coast_distance_km"] + list(oh.columns))
        # river/coast can be NaN where the geometry inputs were absent; HGB
        # handles NaN natively, so keep them.
        return out, base
    # UK: depth + ground elevation (no AIST equivalents; no river/coast).
    base = [c for c in ("depth_from_surface", "ground_level") if c in df.columns]
    return df.reset_index(drop=True), base


def knn_prior(df: pd.DataFrame, tr: np.ndarray) -> np.ndarray:
    """[IDW mean-N of K nearest TRAIN boreholes (own excluded), min distance].

    Same construction as nc_knn_prior but keyed on boring_file (identity)
    rather than coordinates, so co-located boreholes stay distinct.
    """
    from scipy.spatial import cKDTree
    key = df.groupby("boring_file", sort=False)
    bmap = {k: i for i, k in enumerate(key.groups.keys())}
    bid = df["boring_file"].map(bmap).to_numpy()
    coords = key[["latitude_deg", "longitude_deg"]].first().to_numpy()
    bmean = key["n_value"].mean().to_numpy()
    tr_b = np.unique(bid[tr])
    tree = cKDTree(coords[tr_b])
    q = df[["latitude_deg", "longitude_deg"]].to_numpy(np.float64)
    k_query = min(K_NEIGHBOURS + 1, len(tr_b))
    dist, idx = tree.query(q, k=k_query)
    if k_query == 1:  # keep 2-D shape for the loop below
        dist, idx = dist[:, None], idx[:, None]
    out = np.zeros((len(df), 2))
    for i in range(len(df)):
        nb = tr_b[idx[i]]
        dd = dist[i]
        own = nb == bid[i]
        nb, dd = nb[~own][:K_NEIGHBOURS], dd[~own][:K_NEIGHBOURS]
        if len(nb) == 0:  # degenerate toy folds
            out[i] = [np.nan, np.nan]
            continue
        w = 1.0 / np.maximum(dd, 1e-6)
        out[i, 0] = float(np.sum(w * bmean[nb]) / np.sum(w))
        out[i, 1] = float(dd.min())
    return out


# ------------------------------------------------------------ inference


def _stouffer(pvals: list[float]) -> float:
    from scipy.stats import norm
    z = norm.isf(np.clip(pvals, 1e-12, 1 - 1e-12))
    zc = float(np.sum(z) / np.sqrt(len(z)))
    return float(norm.sf(zc))


def _bca_interval(stats: np.ndarray, theta_hat: float, jack: np.ndarray,
                  alpha: float = 0.05) -> tuple[float, float]:
    """BCa interval from bootstrap replicates + jackknife values."""
    from scipy.stats import norm
    stats = np.asarray(stats, dtype=np.float64)
    z0 = norm.ppf(np.clip((stats < theta_hat).mean(), 1e-9, 1 - 1e-9))
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a = 0.0 if den == 0 else num / den
    z = norm.ppf([alpha / 2, 1 - alpha / 2])
    adj = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.quantile(stats, adj)
    return float(lo), float(hi)


# ------------------------------------------------------------ evaluation


def evaluate_grouped(
    df: pd.DataFrame,
    base: list[str],
    emb: np.ndarray,
    *,
    seeds: list[int],
    n_perm_block: int,
    n_perm_row: int,
    use_knn: bool = True,
    pca_dim: int = 64,
    per_row_errors: bool = True,
    model_factory=None,
    fold_col: str = "region",
    strata_col: str | list[str] = "region",
    fold_labels: list[str] | None = None,
    regions_filter: list[str] | None = None,
    return_parts: bool = False,
) -> dict:
    """Per-fold arms + grouped/row nulls + combined inference.

    ``regions_filter`` restricts the loop to a subset of held-out regions so the
    run can be sharded across processes. Sharding is statistically inert here:
    every permutation draws its own generator from ``100_000 * seed + p``
    (see below), which depends on neither the region nor the loop order, so a
    sharded region's numbers are bit-identical to the same region computed in a
    full sequential run. ``return_parts`` additionally returns the pieces the
    cross-region combination needs, so shards can be combined afterwards by
    ``combine_parts`` -- the same code path the single-process run uses.

    ``model_factory(seed) -> estimator`` defaults to the protocol's HGB
    (max_iter=400, lr=0.05); tests inject a light model to exercise the
    permutation/inference mechanics without paying 400-tree fits.
    """
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor

    if model_factory is None:
        def model_factory(seed):  # noqa: F811 - protocol default
            return HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.05, max_depth=None,
                random_state=seed)

    # The pre-registration freezes the null's stratum as region x
    # lithology-macro. ``strata_col`` therefore accepts a list; columns absent
    # from the frame (the UK archive carries no AIST lithology code) are
    # dropped and the columns actually used are recorded in the result.
    want = [strata_col] if isinstance(strata_col, str) else list(strata_col)
    strata_used = [c for c in want if c in df.columns]
    if not strata_used:
        raise SystemExit(f"none of the requested strata columns {want} are in "
                         f"the frame; refusing to run an unstratified null")

    def _strata_labels(frame: pd.DataFrame) -> np.ndarray:
        if len(strata_used) == 1:
            return frame[strata_used[0]].to_numpy()
        return frame[strata_used].astype(str).agg("|".join, axis=1).to_numpy()

    regions = sorted(fold_labels) if fold_labels is not None \
        else sorted(df[fold_col].unique())
    if regions_filter is not None:
        want = set(regions_filter)
        unknown = want - set(regions)
        if unknown:
            raise SystemExit(f"unknown region(s) {sorted(unknown)}; "
                             f"available: {regions}")
        regions = [r for r in regions if r in want]
    y = df["n_value"].to_numpy(np.float64)
    groups = df["boring_file"].to_numpy()
    depth = df["depth_from_surface"].to_numpy(np.float64)

    def _fit_pred(Xtr, ytr, Xte, seed):
        m = model_factory(seed)
        m.fit(Xtr.astype(np.float64), ytr)
        return m.predict(Xte.astype(np.float64))

    per: dict = {}
    fold_p_block: list[float] = []
    fold_p_row: list[float] = []
    borehole_losses: list[pd.DataFrame] = []

    for r in regions:
        te = (df[fold_col] == r).to_numpy()
        tr = ~te
        if te.sum() < 30 or tr.sum() < 100:
            continue
        Xb = df[base].to_numpy(np.float64)
        if use_knn:
            kf = knn_prior(df, tr)
            Xb = np.hstack([Xb, kf])
        k = min(pca_dim, emb.shape[1], int(tr.sum()))
        pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(emb[tr])
        red_tr, red_te = pca.transform(emb[tr]), pca.transform(emb[te])

        # real arms, seed-averaged
        rms_no, rms_tx = [], []
        se_tx = None
        for s in seeds:
            pred_no = _fit_pred(Xb[tr], y[tr], Xb[te], s)
            rms_no.append(float(np.sqrt(np.mean((pred_no - y[te]) ** 2))))
            pred = _fit_pred(np.hstack([Xb[tr], red_tr]), y[tr],
                             np.hstack([Xb[te], red_te]), s)
            rms_tx.append(float(np.sqrt(np.mean((pred - y[te]) ** 2))))
            if per_row_errors:
                se = (pred - y[te]) ** 2
                se_tx = se if se_tx is None else se_tx + se
        text_mean = float(np.mean(rms_tx))

        # nulls
        perm_diag: list[dict] = []

        def _null_rmses(n_perm: int, block: bool, collect_se: bool = False):
            rmses = np.empty(n_perm)
            se_acc = None
            strata_tr = _strata_labels(df.loc[tr])
            for p in range(n_perm):
                s = seeds[p % len(seeds)]
                rng = np.random.default_rng(100_000 * s + p)
                if block:
                    itr, dg = block_permutation_indices(
                        groups[tr], depth[tr], strata_tr, rng,
                        return_diagnostics=True)
                    perm_diag.append(dg)
                    ite = block_permutation_indices(groups[te], depth[te],
                                                    None, rng)
                else:
                    itr = rng.permutation(int(tr.sum()))
                    ite = rng.permutation(int(te.sum()))
                pred = _fit_pred(np.hstack([Xb[tr], red_tr[itr]]), y[tr],
                                 np.hstack([Xb[te], red_te[ite]]), s)
                rmses[p] = float(np.sqrt(np.mean((pred - y[te]) ** 2)))
                if collect_se:
                    # Audit 2026-08-13: this used to accumulate only the first
                    # len(seeds) draws, so the BCa interval was built on a
                    # 3-draw estimate of the null loss while the point estimate
                    # used 1000. Accumulate every draw; the per-borehole null
                    # loss is then an expectation over the full permutation
                    # distribution, which is the quantity the interval is
                    # about. Cost is one add per draw.
                    se = (pred - y[te]) ** 2
                    se_acc = se if se_acc is None else se_acc + se
            return rmses, se_acc

        blk, se_blk = _null_rmses(n_perm_block, block=True, collect_se=True)
        row, _ = _null_rmses(n_perm_row, block=False)

        p_blk = (1.0 + float((blk <= text_mean).sum())) / (1.0 + len(blk))
        p_row = (1.0 + float((row <= text_mean).sum())) / (1.0 + len(row))
        fold_p_block.append(p_blk)
        fold_p_row.append(p_row)

        per[r] = {
            "no_text": [float(np.mean(rms_no)), float(np.std(rms_no))],
            "text": [text_mean, float(np.std(rms_tx))],
            "block_null_mean": float(blk.mean()),
            "block_null_std": float(blk.std()),
            "row_null_mean": float(row.mean()),
            "row_null_std": float(row.std()),
            "content_pct_block": round(100.0 * (text_mean - blk.mean()) / blk.mean(), 3),
            "content_pct_row": round(100.0 * (text_mean - row.mean()) / row.mean(), 3),
            "perm_p_block": p_blk,
            "perm_p_row": p_row,
            "n_te": int(te.sum()),
            # Evidence that the null draw was a permutation, carried in the
            # artefact so a reader never has to take it on trust. The
            # predecessor of this null duplicated 42.7% of rows by clipping a
            # long recipient's block to its shorter donor; had these fields
            # existed then, is_bijection would have read False on every draw.
            "null_permutation": {
                "strata_columns": list(strata_used),
                "all_draws_bijective": bool(
                    perm_diag and all(d["is_bijection"] for d in perm_diag)),
                "n_draws_checked": len(perm_diag),
                "frac_block_matched_mean": round(float(np.mean(
                    [d["frac_block_matched"] for d in perm_diag])), 5)
                    if perm_diag else None,
                "frac_rank_fallback_mean": round(float(np.mean(
                    [d["frac_rank_fallback"] for d in perm_diag])), 5)
                    if perm_diag else None,
                "frac_self_retained_mean": round(float(np.mean(
                    [d["frac_self_retained"] for d in perm_diag])), 5)
                    if perm_diag else None,
            },
        }
        LOG.info("%-16s text %.3f | block null %.3f (p=%.4g) | row null %.3f "
                 "| content(block) %+.2f%%", r, text_mean, blk.mean(), p_blk,
                 row.mean(), per[r]["content_pct_block"])

        if per_row_errors and se_blk is not None:
            n_se = n_perm_block
            borehole_losses.append(pd.DataFrame({
                "region": r,
                "boring_file": groups[te],
                "se_text": se_tx / len(seeds),
                "se_null": se_blk / n_se,
            }))

    if return_parts:
        return {"per_region": per, "fold_p_block": fold_p_block,
                "fold_p_row": fold_p_row,
                "borehole_losses": borehole_losses}
    return combine_parts(per, fold_p_block, fold_p_row,
                         borehole_losses)


def _cauchy_combination(ps: list[float]) -> float:
    """Cauchy combination test (Liu & Xie 2020).

    Stouffer assumes the fold p-values are independent. They are not: with
    leave-one-region-out over 8 regions, any two folds share 6/8 of their
    training rows, so Stouffer is anti-conservative here. The Cauchy statistic
    sum(tan((0.5 - p) * pi)) / k has an approximately standard Cauchy null tail
    under ARBITRARY dependence, which is the assumption we can actually
    defend. Reported alongside the Bonferroni-corrected minimum, which is valid
    under arbitrary dependence too and is the most conservative of the three.
    """
    if not ps:
        return float("nan")
    q = np.clip(np.asarray(ps, dtype=float), 1e-15, 1 - 1e-15)
    t = float(np.mean(np.tan((0.5 - q) * np.pi)))
    from scipy.stats import cauchy
    return float(cauchy.sf(t))


def _bonferroni_min(ps: list[float]) -> float:
    """Smallest p-value, Bonferroni-corrected. Valid under any dependence."""
    if not ps:
        return float("nan")
    return float(min(1.0, len(ps) * min(ps)))


def _null_provenance(per: dict) -> dict:
    """Describe the null AS RUN, read back from the per-region artefacts.

    This string used to be a hardcoded literal. It said "strata=region ...
    Stouffer across folds" and kept saying it after the stratum became
    region x lithology-macro and after Stouffer was demoted to a secondary
    figure, so the blob a reviewer reads first contradicted the per-region
    truth beneath it. Deriving it from the artefact makes that class of drift
    impossible.
    """
    strata, bij, checked = set(), True, []
    for d in per.values():
        npm = d.get("null_permutation") or {}
        strata.update(npm.get("strata_columns") or [])
        bij = bij and bool(npm.get("all_draws_bijective", False))
        if npm.get("n_draws_checked") is not None:
            checked.append(npm["n_draws_checked"])
    return {
        "unit": "borehole block (length-matched derangement; rank-wise for "
                "boreholes with an unshared length)",
        "strata_columns_used": sorted(strata),
        "every_draw_was_a_permutation": bij,
        "draws_checked_per_region": sorted(set(checked)),
        "pairing": "real and null share the HGB seed set (seed-paired)",
        "p_value": "(1+r)/(1+n) corrected per fold",
        "fold_combination": "Bonferroni-corrected minimum and Cauchy "
                            "combination are primary (valid under arbitrary "
                            "dependence, which leave-one-region-out folds have "
                            "because they share most of their training rows); "
                            "Stouffer reported as a secondary figure only",
        "interval": "borehole-block BCa over paired per-borehole losses, full "
                    "leave-one-out jackknife; the 8-region resample is a "
                    "descriptive spread, not a CI",
    }


def combine_parts(per: dict, fold_p_block: list[float],
                  fold_p_row: list[float],
                  borehole_losses: list[pd.DataFrame]) -> dict:
    """Cross-region combination: Bonferroni-min + Cauchy, region spread, BCa.

    Factored out so a sharded run (one process per held-out region) is
    combined by exactly the code a single-process run uses -- there is no
    second implementation that could drift from it.
    """
    # ---- combined inference -------------------------------------------
    regions_used = sorted(per.keys())
    cb = np.array([per[r]["content_pct_block"] for r in regions_used])
    cr = np.array([per[r]["content_pct_row"] for r in regions_used])

    # Region-level resampling over 8 units. A percentile bootstrap on 8
    # observations under-covers badly, and on this data it demonstrably failed
    # to contain the full-population value, so it is reported as a DESCRIPTIVE
    # spread of the per-region effects, not as an inferential interval. The
    # inferential interval is the borehole-block BCa below.
    rngb = np.random.default_rng(42)
    boots = np.array([
        cb[rngb.integers(0, len(cb), len(cb))].mean() for _ in range(10_000)
    ])
    region_ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    # borehole-block bootstrap over paired per-borehole squared errors
    borehole_ci = None
    if borehole_losses:
        bl = pd.concat(borehole_losses, ignore_index=True)
        agg = bl.groupby("boring_file", observed=True)[["se_text", "se_null"]].mean()
        a_tx = agg["se_text"].to_numpy()
        a_nl = agg["se_null"].to_numpy()
        nb = len(agg)

        def _theta(ix: np.ndarray) -> float:
            return float(100.0 * (np.sqrt(a_tx[ix].mean())
                                  - np.sqrt(a_nl[ix].mean()))
                         / np.sqrt(a_nl[ix].mean()))

        theta_hat = _theta(np.arange(nb))
        rngc = np.random.default_rng(43)
        stats = np.array([
            _theta(rngc.integers(0, nb, nb)) for _ in range(10_000)
        ])
        # FULL leave-one-out jackknife, in closed form.
        #
        # This used to be thinned to ~200 points, which silently breaks the
        # BCa acceleration: a = sum(d^3) / (6 * (sum(d^2))^1.5) scales like
        # m^(-1/2) in the number of jackknife points m, so computing it on m
        # of n points inflates |a| by sqrt(n/m) -- a factor of 4.5 at
        # nb = 4,000 and ~57 at the full-population nb. theta is a ratio of
        # means, so every leave-one-out value follows from the two running
        # sums and no loop is needed; the exact jackknife is also FASTER than
        # the thinned loop it replaces.
        s_tx, s_nl = a_tx.sum(), a_nl.sum()
        m_tx = (s_tx - a_tx) / (nb - 1)
        m_nl = (s_nl - a_nl) / (nb - 1)
        jack = 100.0 * (np.sqrt(m_tx) - np.sqrt(m_nl)) / np.sqrt(m_nl)
        borehole_ci = {
            "theta_hat_pct": round(theta_hat, 3),
            "bca_95": [round(v, 3) for v in
                       _bca_interval(stats, theta_hat, jack)],
            "percentile_95": [round(float(np.percentile(stats, 2.5)), 3),
                              round(float(np.percentile(stats, 97.5)), 3)],
            "n_boreholes": int(nb),
            "jackknife": "full leave-one-out (closed form), not thinned",
        }

    return {
        "per_region": per,
        "summary": {
            "content_pct_block_mean": round(float(cb.mean()), 3),
            "content_pct_row_mean": round(float(cr.mean()), 3),
            "delta_block_minus_row_pp": round(float(cb.mean() - cr.mean()), 3),
            "regions_negative_block": f"{int((cb < 0).sum())}/{len(cb)}",
            "regions_negative_row": f"{int((cr < 0).sum())}/{len(cr)}",
            # Primary: valid under arbitrary dependence between folds.
            "bonferroni_min_p_block": _bonferroni_min(fold_p_block),
            "bonferroni_min_p_row": _bonferroni_min(fold_p_row),
            "cauchy_p_block": _cauchy_combination(fold_p_block),
            "cauchy_p_row": _cauchy_combination(fold_p_row),
            # Secondary, retained for continuity with the pre-registration,
            # which named Stouffer before the fold dependence was quantified.
            # Anti-conservative here: LORO folds share 6/8 of their training
            # rows, so the fold p-values are positively dependent.
            "stouffer_p_block": _stouffer(fold_p_block),
            "stouffer_p_row": _stouffer(fold_p_row),
            "fold_p_block": [round(x, 6) for x in fold_p_block],
            "fold_p_row": [round(x, 6) for x in fold_p_row],
            "region_spread_block": {
                "per_region_min": round(float(cb.min()), 3),
                "per_region_max": round(float(cb.max()), 3),
                "resample_2p5_97p5": [round(v, 3) for v in region_ci],
                "note": "descriptive spread over 8 regions, NOT a 95% CI: a "
                        "percentile bootstrap on 8 units under-covers. Use "
                        "borehole_block_bootstrap for inference.",
            },
            "borehole_block_bootstrap": borehole_ci,
        },
    }


# ------------------------------------------------------------ driver


def _hash_texts(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def embedding_cache_path(cache_dir: Path, domain: str, texts: list[str],
                         strip_mode: str = "lithology_only") -> Path:
    """Content-hashed embedding cache path.

    The historical fixed-name cache silently returned the WRONG embeddings
    when the subsample changed, so the stripped texts themselves are hashed
    into the name. That also means a NEW strip mode automatically gets its own
    cache entry (different texts -> different hash); the mode tag is in the
    name too, so the entry is identifiable without re-deriving the hash.
    """
    return cache_dir / (f"grouped_{domain}_{cache_tag(strip_mode)}_"
                        f"{_hash_texts(texts)}_e5.npy")


def run(domain: str, out: Path, cache_dir: Path, *,
        per_region_files: int, sample_seed: int, seeds: list[int],
        n_perm_block: int, n_perm_row: int, use_knn: bool,
        regions: list[str] | None = None, shard: bool = False,
        strata_col: str | list[str] = ("region", "aist_litho_macro_code"),
        strip_mode: str = "lithology_only") -> dict:
    df, _thin_base, _ = load_domain(domain, cache_dir,
                                    per_region_files=per_region_files,
                                    sample_seed=sample_seed)
    df, base = build_rich_features(df, domain)
    # P-T5 runs the SAME protocol on depersonalised text; every other mode is
    # the frozen lithology-only arm and is byte-identical to the historical run.
    texts, strip_stats = apply_strip_mode(
        df["text"].tolist(), domain, strip_mode,
        boring_files=df["boring_file"].astype(str).tolist())
    LOG.info("strip %s: %s", strip_mode,
             json.dumps({k: v for k, v in strip_stats.items()
                         if k in ("per_op", "total", "header_coverage",
                                  "n_projects_with_template", "n_empty_out",
                                  "frac_chars_removed")},
                        ensure_ascii=False))
    cache = embedding_cache_path(cache_dir, domain, texts, strip_mode)
    emb = embed_texts(texts, cache)

    parts = evaluate_grouped(df, base, emb, seeds=seeds,
                             n_perm_block=n_perm_block, n_perm_row=n_perm_row,
                             use_knn=use_knn, regions_filter=regions,
                             strata_col=strata_col, return_parts=True)
    if shard:
        # One process per held-out region. Persist the raw parts; the
        # cross-region combination happens once, in --combine, using the same
        # combine_parts() the single-process path calls.
        out.parent.mkdir(parents=True, exist_ok=True)
        losses = parts.pop("borehole_losses")
        if losses:
            pd.concat(losses, ignore_index=True).to_parquet(
                out.with_suffix(".losses.parquet"))
        # Self-describing shards: --combine keeps the first shard's config, so
        # a P-T5 shard set is never mistaken for the frozen P-T1 arm.
        parts["config"] = {"domain": domain, "strip_mode": strip_mode,
                           "strip_stats": strip_stats,
                           "embedding_cache": cache.name,
                           "per_region_files": per_region_files,
                           "sample_seed": sample_seed, "seeds": seeds,
                           "n_perm_block": n_perm_block,
                           "n_perm_row": n_perm_row, "use_knn": use_knn}
        out.write_text(json.dumps(parts, indent=2))
        LOG.info("wrote shard %s (regions=%s)", out,
                 sorted(parts["per_region"]))
        return parts

    res = combine_parts(parts["per_region"], parts["fold_p_block"],
                        parts["fold_p_row"], parts["borehole_losses"])
    res["config"] = {
        "domain": domain, "n_rows": int(len(df)),
        "n_boreholes": int(df["boring_file"].nunique()),
        "per_region_files": per_region_files, "sample_seed": sample_seed,
        "seeds": seeds, "n_perm_block": n_perm_block, "n_perm_row": n_perm_row,
        "regions": regions,
        "baseline": base if len(base) < 12 else f"{len(base)} cols (rich rung)",
        "use_knn": use_knn,
        "strip": f"{strip_mode} {STRIP_VOCAB_VERSION}",
        "strip_mode": strip_mode,
        "strip_stats": strip_stats,
        "embedding_cache": cache.name,
        "prereg": ("docs/research/2026-08-11_nc_text_preregistration.md "
                   + ("P-T5" if strip_mode == DEPERSONALISE_MODE
                      else "P-T1/T2/T3")),
    }
    res["_provenance"] = {
        "purpose": "Pre-registered grouped-null primary analysis (P-T1/T2/T3)",
        "null": _null_provenance(res["per_region"]),
        "script": "backend/scripts/nc_grouped_null.py",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    LOG.info("wrote %s", out)
    LOG.info("SUMMARY %s", json.dumps(res["summary"], indent=2))
    return res


def combine(shard_dir: Path, out: Path, pattern: str = "*.json") -> dict:
    """Combine per-region shards written by ``--shard`` into the final result.

    Per-region numbers are bit-identical to a sequential run (every permutation
    seeds its own generator from ``100_000 * seed + p``, independent of region
    and loop order), so this reproduces the unsharded output exactly.
    """
    per: dict = {}
    fold_p_block: list[float] = []
    fold_p_row: list[float] = []
    losses: list[pd.DataFrame] = []
    config: dict | None = None
    shards = sorted(q for q in shard_dir.glob(pattern)
                    if not q.name.endswith(".combined.json"))
    if not shards:
        raise SystemExit(f"no shards matching {pattern} under {shard_dir}")
    for q in shards:
        d = json.loads(q.read_text())
        dup = set(d["per_region"]) & set(per)
        if dup:
            raise SystemExit(f"region(s) {sorted(dup)} appear in more than one "
                             f"shard; refusing to double-count")
        per.update(d["per_region"])
        fold_p_block += d["fold_p_block"]
        fold_p_row += d["fold_p_row"]
        lp = q.with_suffix(".losses.parquet")
        if lp.exists():
            losses.append(pd.read_parquet(lp))
        config = config or d.get("config")
    res = combine_parts(per, fold_p_block, fold_p_row, losses)
    res["config"] = config or {}
    res["config"]["combined_from_shards"] = [q.name for q in shards]
    res["_provenance"] = {
        "purpose": "Pre-registered grouped-null primary analysis (P-T1/T2/T3), "
                   "combined from per-region shards",
        "null": _null_provenance(per),
        "sharding": "statistically inert: each permutation seeds its own "
                    "generator from 100_000 * seed + permutation index, which "
                    "depends on neither region nor loop order",
        "script": "backend/scripts/nc_grouped_null.py",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    LOG.info("wrote %s from %d shards (%d regions)", out, len(shards), len(per))
    LOG.info("SUMMARY %s", json.dumps(res["summary"], indent=2))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", choices=("japan", "uk"), default="japan")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path,
                    default=REPO / "data/features/derived/nc_cache")
    ap.add_argument("--per-region-files", type=int, default=500,
                    help="Japan subsample budget; 0 = full population.")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-perm-block", type=int, default=1000)
    ap.add_argument("--n-perm-row", type=int, default=200)
    ap.add_argument("--no-knn", action="store_true")
    ap.add_argument("--strata-col", nargs="+",
                    default=["region", "aist_litho_macro_code"],
                    help="Null stratum columns. Default is the pre-registered "
                         "region x lithology-macro; columns absent from the "
                         "frame (the UK archive has no AIST code) are dropped "
                         "and the ones actually used are recorded in the "
                         "result.")
    ap.add_argument("--regions", nargs="+", default=None,
                    help="Restrict to these held-out regions (sharding).")
    ap.add_argument("--shard", action="store_true",
                    help="Write raw per-region parts for later --combine "
                         "instead of the combined result.")
    ap.add_argument("--combine", type=Path, default=None,
                    help="Directory of shard JSONs to combine into --out.")
    ap.add_argument("--strip-mode", choices=STRIP_MODES,
                    default="lithology_only",
                    help="Text arm representation. Default = the frozen "
                         "pre-registered strength-stripped arm (P-T1). "
                         f"{DEPERSONALISE_MODE!r} additionally removes "
                         "identity tokens (header substrings, place names) and "
                         "per-project boilerplate -- the P-T5 control.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.combine is not None:
        combine(args.combine, args.out)
        return 0
    run(args.domain, args.out, args.cache_dir,
        per_region_files=args.per_region_files, sample_seed=args.sample_seed,
        seeds=args.seeds, n_perm_block=args.n_perm_block,
        n_perm_row=args.n_perm_row, use_knn=not args.no_knn,
        regions=args.regions, shard=args.shard, strata_col=args.strata_col,
        strip_mode=args.strip_mode)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(main())
