#!/usr/bin/env python
"""NC round-3 [E2b] — kriging-style structured SPATIAL prior as a baseline rung.

Completes the rich-baseline ladder (nc_rich_baseline.py) with the reviewer's remaining
structured-covariate candidate: a KNN spatial prior from surrounding boreholes -- the
strongest "structured geology" a practitioner could build without text. Leakage discipline:

  - the prior is computed from the LRO TRAINING side only (for held-out-region rows the
    neighbours are, by construction, boreholes in OTHER regions -- exactly what a spatial
    interpolator could do out-of-region);
  - for TRAINING rows the own borehole is excluded (leave-own-borehole-out), so the
    feature never contains the row's own label through its own borehole.

Feature: inverse-distance-weighted mean of the k=10 nearest (other) training boreholes'
mean N, plus the distance to the nearest training borehole (a data-density covariate).

Rungs evaluated (LM lithology_only content effect on top of each):
  plus_knn          : thin + KNN prior
  plus_all          : thin + regime + geology(litho/era) + parser + KNN  (the richest
                      structured baseline constructible from this archive without text)

Run (CPU):
  cd backend
  uv run python -m scripts.nc_knn_prior --out ../docs/research/2026-07-04_knn_prior_rung.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.nc_geo_ablation import _join_aist
from scripts.text_leakage_controls import (_structured_litho_features, load_domain,
                                           strip_text, STRIP_VOCAB_VERSION)
from scripts.uk_transfer_test import _fit_rmse, embed_texts

LOG = logging.getLogger("nc_knn_prior")
REPO = Path(__file__).resolve().parents[2]
K_NEIGHBOURS = 10


def _borehole_table(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Borehole coords, mean N per borehole, and per-row borehole id."""
    key = df.groupby(["latitude_deg", "longitude_deg"], sort=False)
    bmap = {k: i for i, k in enumerate(key.groups.keys())}
    bid = df.set_index(["latitude_deg", "longitude_deg"]).index.map(bmap).to_numpy()
    coords = np.array(list(key.groups.keys()))
    bmean = key["n_value"].mean().to_numpy()
    return coords, bmean, bid


def _knn_features(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    """[idw-mean-N of k nearest TRAINING boreholes (own excluded), dist to nearest]."""
    from scipy.spatial import cKDTree
    coords, bmean, bid = _borehole_table(df)
    tr_b = np.unique(bid[tr])
    tree = cKDTree(coords[tr_b])
    out = np.zeros((len(df), 2))
    q = df[["latitude_deg", "longitude_deg"]].to_numpy(np.float64)
    dist, idx = tree.query(q, k=K_NEIGHBOURS + 1)  # +1 to allow own-borehole exclusion
    for i in range(len(df)):
        nb = tr_b[idx[i]]
        dd = dist[i]
        own = nb == bid[i]
        nb, dd = nb[~own][:K_NEIGHBOURS], dd[~own][:K_NEIGHBOURS]
        w = 1.0 / np.maximum(dd, 1e-6)
        out[i, 0] = float(np.sum(w * bmean[nb]) / np.sum(w))
        out[i, 1] = float(dd.min())
    return out


def run(out: Path, cache_dir: Path, seeds: list[int]) -> dict:
    df, base, _ = load_domain("japan", cache_dir)
    raw = df["text"].tolist()
    texts = [strip_text(t, "japan", "lithology_only") for t in raw]
    emb = embed_texts(texts, cache_dir / f"ncnull_japan_lithology_only_{STRIP_VOCAB_VERSION}_e5.npy")
    joined = _join_aist(df)
    parser = _structured_litho_features(raw, df, "japan").astype(np.float32)
    oh = pd.concat([pd.get_dummies(joined["regime_code"].astype("Int64"), prefix="rg"),
                    pd.get_dummies(joined["aist_litho_macro_code"].astype("Int64"), prefix="lt"),
                    pd.get_dummies(joined["aist_era_code"].astype("Int64"), prefix="er")],
                   axis=1).astype(np.float32).to_numpy()
    Xthin = df[base].to_numpy(np.float64)
    y = df["n_value"].to_numpy(np.float64)
    region = df["region"].to_numpy()

    from sklearn.decomposition import PCA
    res: dict = {"config": {"k": K_NEIGHBOURS, "seeds": seeds,
                            "note": "KNN prior train-side only; own borehole excluded for "
                                    "training rows; ladder rungs plus_knn / plus_all"}}
    for rung in ("plus_knn", "plus_all"):
        per = {}
        for r in sorted(set(region)):
            te = region == r
            tr = ~te
            if te.sum() < 30:
                continue
            knn = _knn_features(df, tr, te)
            Xb = np.hstack([Xthin, knn]) if rung == "plus_knn" else np.hstack([Xthin, oh, parser, knn])
            k = min(64, emb.shape[1], int(tr.sum()))
            pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(emb[tr])
            red_tr, red_te = pca.transform(emb[tr]), pca.transform(emb[te])
            nt, tx, sh = [], [], []
            for s in seeds:
                nt.append(_fit_rmse(Xb[tr], y[tr], Xb[te], y[te], s))
                tx.append(_fit_rmse(np.hstack([Xb[tr], red_tr]), y[tr],
                                    np.hstack([Xb[te], red_te]), y[te], s))
                g = np.random.default_rng(s)
                sh.append(_fit_rmse(np.hstack([Xb[tr], red_tr[g.permutation(len(red_tr))]]), y[tr],
                                    np.hstack([Xb[te], red_te[g.permutation(len(red_te))]]), y[te], s))
            per[r] = {"no_text": float(np.mean(nt)), "text": float(np.mean(tx)),
                      "shuffled": float(np.mean(sh))}
        m = lambda k_: float(np.mean([v[k_] for v in per.values()]))
        n_neg = sum(v["text"] < v["shuffled"] for v in per.values())
        res[rung] = {"baseline_rmse": round(m("no_text"), 3), "text_rmse": round(m("text"), 3),
                     "shuffled_rmse": round(m("shuffled"), 3),
                     "content_pct": round(100 * (m("text") - m("shuffled")) / m("shuffled"), 2),
                     "regions_negative": f"{n_neg}/{len(per)}"}
        LOG.info("%s: baseline %.2f | content %+.1f%% (%s)", rung,
                 res[rung]["baseline_rmse"], res[rung]["content_pct"],
                 res[rung]["regions_negative"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({r: res[r] for r in ("plus_knn", "plus_all")}, indent=2))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=REPO / "data/features/derived")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    a = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    run(a.out, a.cache_dir, a.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
