#!/usr/bin/env python
"""NC round-3 [E3] — text-missingness characterisation + IPW population estimate.

Reviewer: "The large text effect may be driven by WHICH boreholes have text, not by
text content" (Japan is 57.7% text-bearing). Two deliverables:

1. CHARACTERISATION: per-row text presence on the full 2.66M corpus (borehole-location
   nearest-neighbour join of soil_text.csv), then standardized mean differences (SMD)
   between text-bearing and non-text-bearing rows over depth, N, elevation, river/coast
   distance, regime, lithology-macro, era, and region. This is the "who has text" table.

2. IPW POPULATION ESTIMATE: a propensity model P(has_text | depth, elevation, regime,
   region) fit on the full corpus; the leave-region-out content effect re-estimated on the
   text-bearing transfer sample with HGB sample_weight = 1/p-hat, reweighting the
   text-bearing rows toward the full-corpus covariate distribution. If the weighted
   content effect matches the unweighted one, the effect is not an artefact of the
   text-bearing subpopulation's covariate profile.

Estimand naming (used in the manuscript): the "text-bearing estimand" (effect where text
exists) vs the "population estimand" (full corpus; bounded by the has_text-controlled
-7.3% decomposition and approximated here by IPW).

Run (CPU):
  cd backend
  uv run python -m scripts.nc_missingness --out ../docs/research/2026-07-04_missingness_ipw.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from national.evaluation.leave_region_out import DEFAULT_REGIONS
from scripts.text_leakage_controls import load_domain, strip_text
from scripts.uk_transfer_test import embed_texts

LOG = logging.getLogger("nc_missingness")
REPO = Path(__file__).resolve().parents[2]


def _region_of(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    out = np.array(["other"] * len(lat), dtype=object)
    for name, (lat0, lat1, lon0, lon1) in DEFAULT_REGIONS.items():
        m = (lat >= lat0) & (lat < lat1) & (lon >= lon0) & (lon < lon1)
        out[m] = name
    return out


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference between groups a and b."""
    sp = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2)
    return float((np.nanmean(a) - np.nanmean(b)) / (sp + 1e-12))


def characterise(v4: pd.DataFrame) -> dict:
    ht = v4["has_text"].to_numpy(bool)
    out = {"frac_text_bearing": round(float(ht.mean()), 4), "n_total": len(v4)}
    smds = {}
    for c in ("depth_from_surface", "n_value", "absolute_elevation",
              "river_distance_km", "coast_distance_km"):
        smds[c] = round(_smd(v4.loc[ht, c].to_numpy(float), v4.loc[~ht, c].to_numpy(float)), 3)
    out["smd_continuous"] = smds
    cats = {}
    for c in ("regime_code", "aist_litho_macro_code", "aist_era_code", "region"):
        p1 = v4.loc[ht, c].value_counts(normalize=True)
        p0 = v4.loc[~ht, c].value_counts(normalize=True)
        keys = sorted(set(p1.index) | set(p0.index), key=str)
        tv = 0.5 * sum(abs(p1.get(k, 0.0) - p0.get(k, 0.0)) for k in keys)
        cats[c] = {"total_variation_distance": round(float(tv), 3),
                   "share_text": {str(k): round(float(p1.get(k, 0.0)), 3) for k in keys},
                   "share_notext": {str(k): round(float(p0.get(k, 0.0)), 3) for k in keys}}
    out["categorical"] = cats
    return out


def _weighted_lro(df, base, emb, weights, seeds, pca_dim=64) -> dict:
    """_evaluate_lro variant with per-row sample_weight in the HGB fit."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    regions = sorted(df["region"].unique())
    Xb = df[base].to_numpy(np.float64)
    y = df["n_value"].to_numpy(np.float64)
    per = {}
    for r in regions:
        te = (df["region"] == r).to_numpy(); tr = ~te
        if te.sum() < 30 or tr.sum() < 100:
            continue
        k = min(pca_dim, emb.shape[1], int(tr.sum()))
        pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(emb[tr])
        red_tr, red_te = pca.transform(emb[tr]), pca.transform(emb[te])
        w_tr = weights[tr]
        modes = {"text": [], "shuffled": []}
        for s in seeds:
            for mode in ("text", "shuffled"):
                rtr = red_tr
                if mode == "shuffled":
                    g = np.random.default_rng(s)
                    rtr = red_tr[g.permutation(len(red_tr))]
                m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                  max_depth=None, random_state=s)
                m.fit(np.hstack([Xb[tr], rtr]), y[tr], sample_weight=w_tr)
                pred = m.predict(np.hstack([Xb[te], red_te]))
                modes[mode].append(float(np.sqrt(np.mean((pred - y[te]) ** 2))))
        per[r] = {m_: float(np.mean(v)) for m_, v in modes.items()}
    tx = np.mean([v["text"] for v in per.values()])
    sh = np.mean([v["shuffled"] for v in per.values()])
    n_neg = sum(v["text"] < v["shuffled"] for v in per.values())
    return {"content_pct": round(100 * (tx - sh) / sh, 2),
            "regions_negative": f"{n_neg}/{len(per)}",
            "per_region": {r: round(100 * (v["text"] - v["shuffled"]) / v["shuffled"], 1)
                           for r, v in per.items()}}


def run(out: Path, cache_dir: Path, seeds: list[int]) -> dict:
    from scipy.spatial import cKDTree
    from sklearn.linear_model import LogisticRegression

    v4 = pd.read_parquet(REPO / "data/features/borings_japan_v4.parquet")
    # Restrict to the ANALYSIS population (measured SPT rows, matching every transfer
    # experiment's filter): rows with n_value == 0 are non-measurement padding whose
    # near-total lack of text would otherwise fabricate a huge outcome SMD (caught by
    # the 2026-07-05 sanity check: notext-group mean N was 0.71 before this filter).
    v4 = v4[(v4["n_value"] > 0) & (v4["n_value"] <= 100)].reset_index(drop=True)
    # has_text must be the LAYER-DEPTH-MATCHED definition (the 57.7% construct): a row
    # is text-bearing iff a narrative layer at the same borehole covers the row's depth.
    # (Borehole-level presence is ~100% -- nearly every KuniJiban file has SOME narrative
    # -- which is the wrong construct and yields a single-class propensity target.)
    ly = pd.read_csv(REPO / "data/features/derived/soil_text_layers.csv",
                     usecols=["latitude_deg", "longitude_deg", "depth_top_m",
                              "depth_bottom_m"]).dropna()
    locs = ly[["latitude_deg", "longitude_deg"]].drop_duplicates().reset_index(drop=True)
    tree = cKDTree(locs.to_numpy())
    d, bid = tree.query(v4[["latitude_deg", "longitude_deg"]].to_numpy(np.float64), k=1)
    # per-borehole interval table
    lkey = locs.reset_index().rename(columns={"index": "bid"})
    ly = ly.merge(lkey, on=["latitude_deg", "longitude_deg"], how="left")
    iv_by_b = {b: g[["depth_top_m", "depth_bottom_m"]].to_numpy()
               for b, g in ly.groupby("bid")}
    depth = v4["depth_from_surface"].to_numpy()
    has = np.zeros(len(v4), bool)
    near = d < 1e-3
    order = np.argsort(bid)
    for b, g_idx in pd.Series(np.arange(len(v4))[order]).groupby(bid[order]):
        if not near[g_idx.to_numpy()].any():
            continue
        iv = iv_by_b.get(b)
        if iv is None:
            continue
        dd = depth[g_idx.to_numpy()]
        cover = ((dd[:, None] >= iv[None, :, 0]) & (dd[:, None] <= iv[None, :, 1])).any(1)
        has[g_idx.to_numpy()] = cover & near[g_idx.to_numpy()]
    v4["has_text"] = has
    v4["region"] = _region_of(v4["latitude_deg"].to_numpy(), v4["longitude_deg"].to_numpy())
    LOG.info("full corpus %d rows; text-bearing %.1f%%", len(v4), 100 * v4["has_text"].mean())

    res = {"characterisation": characterise(v4)}

    # propensity: P(has_text | depth, elevation, regime one-hot, region one-hot)
    feats = pd.concat([v4[["depth_from_surface", "absolute_elevation"]],
                       pd.get_dummies(v4["regime_code"].astype("Int64"), prefix="rg"),
                       pd.get_dummies(v4["region"], prefix="re")], axis=1).astype(np.float64)
    lr = LogisticRegression(max_iter=1000).fit(feats, v4["has_text"])
    auc_proxy = float(lr.score(feats, v4["has_text"]))
    res["propensity"] = {"model": "logistic(depth, elevation, regime, region)",
                         "train_accuracy": round(auc_proxy, 3)}

    # transfer sample + weights
    df, base, _ = load_domain("japan", cache_dir)
    texts = [strip_text(t, "japan", "lithology_only") for t in df["text"].tolist()]
    from scripts.text_leakage_controls import STRIP_VOCAB_VERSION
    emb = embed_texts(texts, cache_dir / f"ncnull_japan_lithology_only_{STRIP_VOCAB_VERSION}_e5.npy")
    # nearest v4 row -> propensity features for each transfer row
    vtree = cKDTree(v4[["latitude_deg", "longitude_deg"]].to_numpy(np.float64))
    _, idx = vtree.query(df[["latitude_deg", "longitude_deg"]].to_numpy(np.float64), k=1)
    fsub = feats.iloc[idx].copy()
    fsub["depth_from_surface"] = df["depth_from_surface"].to_numpy()  # row's own depth
    p = lr.predict_proba(fsub.to_numpy(np.float64))[:, 1].clip(0.05, 0.999)
    w = (1.0 / p); w = w / w.mean()
    res["ipw_weights"] = {"min": round(float(w.min()), 2), "max": round(float(w.max()), 2),
                          "effective_sample_frac": round(float((w.sum() ** 2) / (len(w) * (w ** 2).sum())), 3)}

    res["content_effect"] = {
        "unweighted_text_bearing_estimand": _weighted_lro(df, base, emb, np.ones(len(df)), seeds),
        "ipw_population_estimand": _weighted_lro(df, base, emb, w, seeds),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({"frac_text": res["characterisation"]["frac_text_bearing"],
                      "smd": res["characterisation"]["smd_continuous"],
                      "content_unweighted": res["content_effect"]["unweighted_text_bearing_estimand"]["content_pct"],
                      "content_ipw": res["content_effect"]["ipw_population_estimand"]["content_pct"]}, indent=2))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=REPO / "data/features/derived")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    a = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    run(a.out, a.cache_dir, a.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
