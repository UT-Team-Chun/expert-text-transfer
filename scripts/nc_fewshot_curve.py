#!/usr/bin/env python
"""Cross-archive few-shot learning curve, borehole-budgeted (P-T6).

Supersedes the row-budgeted few-shot arm of ``nc_cross_archive``: sampling
target ROWS puts sibling layers of one borehole on both sides of the
adaptation/evaluation split (a borehole shares stratigraphy, logger and
location), which inflates the adaptation numbers -- the published
rho=0.506/0.461 at n=1,000 rows is suspected of exactly this. Here:

- the evaluation HOLDOUT is a fixed, seed-0, 50% split of target BOREHOLES,
  shared by every budget, arm and seed (curve points are comparable);
- adaptation budgets are counted in BOREHOLES {0, 10, 25, 50, 100, 300},
  drawn per seed from the non-holdout pool only;
- at every budget both a depth-only and a depth+text adaptation arm run, so
  the curve answers "how many target-archive borings does the text channel
  save?" (the reviewer's importance framing);
- the zero-shot decomposition (depth / +text / +shuffled) is reported on the
  same holdout AND on the full target archive (continuity with the published
  numbers), in BOTH directions symmetrically.

Bar (prereg P-T6): text few-shot > no-text few-shot at every budget in both
directions; the row->borehole attenuation is reported, not hidden.

CLI::

    cd backend
    .venv/bin/python -m scripts.nc_fewshot_curve \
        --out ../docs/research/2026-08-12_fewshot_borehole_curve.json
"""
from __future__ import annotations

# OpenMP cap BEFORE any sklearn import: HGB's parallel splitter spin-waits at
# barriers on sub-100k-row fits (observed 6.6 effective threads at ~0 speedup).
# threadpoolctl cannot fix this reliably -- libomp is loaded lazily after the
# limiter enumerates libraries -- so pin it at process start. Override with
# OMP_NUM_THREADS in the environment for large-data runs.
import os
# Measured on the fit this protocol actually runs (M-series, 16 cores):
# 1 thread 5.35 s, 2 threads 9.59 s, 4 threads 9.96 s, 8 threads 15.80 s.
# `sample` shows the multi-thread time goes into __kmp_barrier ->
# cthread_yield: at this data size the histogram/splitting kernels hit the
# OpenMP barrier far more often than they do useful work. One thread per
# process is faster, and leaves the cores free to run shards in parallel.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")


import argparse
import json
import logging
from pathlib import Path

import numpy as np

from scripts.text_leakage_controls import (
    STRIP_VOCAB_VERSION,
    load_domain,
    strip_text,
)
from scripts.uk_transfer_test import embed_texts

LOG = logging.getLogger("nc_fewshot_curve")
REPO = Path(__file__).resolve().parents[2]


def _fit_predict(Xtr, ytr, Xte, seed=42):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                      max_depth=None, random_state=seed)
    m.fit(Xtr.astype(np.float64), ytr)
    return m.predict(Xte.astype(np.float64))


def _metrics(pred, y):
    from scipy.stats import spearmanr
    rho = float(spearmanr(pred, y).statistic)
    z = (y - y.mean()) / y.std()
    zp = (pred - pred.mean()) / (pred.std() + 1e-9)
    return {"spearman_rho": round(rho, 4),
            "z_rmse": round(float(np.sqrt(np.mean((zp - z) ** 2))), 4)}


def _load(domain: str, cache_dir: Path):
    df, _, _ = load_domain(domain, cache_dir)
    texts = [strip_text(t, domain, "lithology_only") for t in df["text"].tolist()]
    emb = embed_texts(
        texts,
        cache_dir / f"ncnull_{domain}_lithology_only_{STRIP_VOCAB_VERSION}_e5.npy")
    depth = df["depth_from_surface"].to_numpy(np.float64).reshape(-1, 1)
    y = df["n_value"].to_numpy(np.float64)
    groups = df["boring_file"].astype(str).to_numpy()
    return depth, emb, y, groups


def borehole_holdout(groups: np.ndarray, frac: float = 0.5,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fixed borehole-level holdout: (row mask, non-holdout borehole pool).

    Deterministic in ``seed``; whole boreholes only -- no row of a holdout
    borehole may enter any adaptation set.
    """
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    hold_b = set(rng.permutation(uniq)[: int(len(uniq) * frac)])
    hold = np.isin(groups, list(hold_b))
    pool_b = np.asarray([b for b in uniq if b not in hold_b])
    return hold, pool_b


def run_direction(src: str, tgt: str, cache_dir: Path, seeds: list[int],
                  budgets: list[int]) -> dict:
    from sklearn.decomposition import PCA
    Xs_d, Es, ys, _gs = _load(src, cache_dir)
    Xt_d, Et, yt, gt = _load(tgt, cache_dir)
    pca = PCA(n_components=64, svd_solver="randomized", random_state=0).fit(Es)
    Ps, Pt = pca.transform(Es), pca.transform(Et)

    hold, pool_b = borehole_holdout(gt, frac=0.5, seed=0)
    hold_b = set(np.unique(gt[hold]))
    out: dict = {
        "n_source_rows": int(len(ys)), "n_target_rows": int(len(yt)),
        "n_target_boreholes": int(len(np.unique(gt))),
        "n_holdout_rows": int(hold.sum()), "n_holdout_boreholes": len(hold_b),
    }

    # target-trained depth-only reference, borehole-grouped split
    ref_pred = _fit_predict(Xt_d[~hold], yt[~hold], Xt_d[hold])
    out["reference_target_trained_depth_only"] = _metrics(ref_pred, yt[hold])

    # zero-shot decomposition, both on the shared holdout and full target
    for scope, mask in (("holdout", hold), ("full_target", np.ones(len(yt), bool))):
        arms = {}
        for arm in ("depth_only", "depth_text", "depth_shuffled"):
            per_seed = []
            for s in seeds:
                if arm == "depth_only":
                    Xtr, Xte = Xs_d, Xt_d[mask]
                else:
                    Pt_arm = Pt
                    if arm == "depth_shuffled":
                        g = np.random.default_rng(s)
                        Pt_arm = Pt[g.permutation(len(Pt))]
                    Xtr = np.hstack([Xs_d, Ps])
                    Xte = np.hstack([Xt_d[mask], Pt_arm[mask]])
                pred = _fit_predict(Xtr, ys, Xte, seed=s)
                per_seed.append(_metrics(pred, yt[mask]))
            arms[arm] = {k: round(float(np.mean([m[k] for m in per_seed])), 4)
                         for k in per_seed[0]}
        out[f"zero_shot_{scope}"] = arms

    # borehole-budget few-shot curve on the shared holdout
    curve: dict[str, dict] = {}
    for n_b in budgets:
        if n_b > len(pool_b):
            LOG.warning("budget %d > pool %d; skipped", n_b, len(pool_b))
            continue
        arms = {"depth_only": [], "depth_text": []}
        for s in seeds:
            g = np.random.default_rng(s)
            adapt_b = set(g.permutation(pool_b)[:n_b]) if n_b else set()
            ad = np.isin(gt, list(adapt_b)) if n_b else np.zeros(len(yt), bool)
            # depth-only adaptation
            Xtr = np.vstack([Xs_d, Xt_d[ad]]) if n_b else Xs_d
            ytr = np.concatenate([ys, yt[ad]]) if n_b else ys
            pred = _fit_predict(Xtr, ytr, Xt_d[hold], seed=s)
            arms["depth_only"].append(_metrics(pred, yt[hold]))
            # depth+text adaptation
            Xtr = (np.vstack([np.hstack([Xs_d, Ps]),
                              np.hstack([Xt_d[ad], Pt[ad]])])
                   if n_b else np.hstack([Xs_d, Ps]))
            pred = _fit_predict(Xtr, ytr, np.hstack([Xt_d[hold], Pt[hold]]),
                                seed=s)
            arms["depth_text"].append(_metrics(pred, yt[hold]))
        curve[f"boreholes={n_b}"] = {
            arm: {
                "spearman_rho_mean": round(float(np.mean(
                    [m["spearman_rho"] for m in v])), 4),
                "spearman_rho_std": round(float(np.std(
                    [m["spearman_rho"] for m in v])), 4),
                "z_rmse_mean": round(float(np.mean([m["z_rmse"] for m in v])), 4),
            }
            for arm, v in arms.items()
        }
        LOG.info("  n=%3d boreholes: depth %.3f | +text %.3f", n_b,
                 curve[f"boreholes={n_b}"]["depth_only"]["spearman_rho_mean"],
                 curve[f"boreholes={n_b}"]["depth_text"]["spearman_rho_mean"])
    out["fewshot_curve"] = curve
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=REPO / "data/features/derived")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[0, 10, 25, 50, 100, 300])
    a = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    res = {"config": {
        "representation": "lithology_only, multilingual-e5, source-fit PCA-64",
        "model": "HGB(400)", "seeds": a.seeds, "budgets_boreholes": a.budgets,
        "holdout": "fixed 50% of target BOREHOLES (seed 0), shared across "
                   "budgets/arms/seeds",
        "prereg": "docs/research/2026-08-11_nc_text_preregistration.md P-T6",
        "supersedes": "row-budgeted few-shot of nc_cross_archive "
                      "(2026-07-04_cross_archive_transfer.json)"},
        "_provenance": {
            "purpose": "P-T6 borehole-budgeted few-shot curve; kills the "
                       "sibling-layer leak of the row-budgeted design",
            "script": "backend/scripts/nc_fewshot_curve.py"}}
    for src, tgt in (("japan", "uk"), ("uk", "japan")):
        LOG.info("direction %s -> %s", src, tgt)
        res[f"{src}_to_{tgt}"] = run_direction(src, tgt, a.cache_dir, a.seeds,
                                               a.budgets)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    LOG.info("wrote %s", a.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
