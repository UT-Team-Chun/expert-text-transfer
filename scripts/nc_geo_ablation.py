#!/usr/bin/env python
"""NC revision — geological-fold flagship-LM (#5) + regime-redundancy ablation (#3).

Both reuse the PROVEN harness scripts.uk_transfer_test._evaluate_lro on the Japanese transfer
subsample, after joining AIST surface-geology codes (regime_code, aist_litho_macro_code) from
borings_japan_v4 by rounded (lat, lon) -- the same ~53%-coverage join used in the round-2
geological-province split. The headline representation is the frozen-LM lithology_only embedding
(cache shared with nc_null_controls).

#5  [MAJOR, geoscience]  The flagship LM number is only ever shown under ADMINISTRATIVE folds.
    Run the frozen-LM lithology_only content effect under a genuinely GEOLOGICAL partition
    (leave-lithology-macro-out, and leave-era-out), so the headline representation -- not just the
    embedding-free parser stand-in -- survives a geological fold.

#3  [MAJOR]  Measure (not assert) the regime-redundancy component of the -21.1% -> -7.3% gap:
    on the SAME joined subsample, run the LM content effect with base = thin [depth,lat,lon] vs
    base = +regime [.. + AIST-regime one-hot]. The drop when the regime baseline is added is the
    part of the headline content that overlaps the AIST surface-regime covariate. (The has_text
    and full-corpus-vs-subsample components are controlled in the round-2 conservative run and are
    not re-derivable on the text-bearing-only subsample, which is stated, not re-measured here.)

Run (CPU; launch after the nc_null_controls Japan embedding cache exists):
  cd backend
  uv run python -m scripts.nc_geo_ablation --out ../docs/research/2026-06-24_geo_fold_lm_and_regime_ablation.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.text_leakage_controls import load_domain, strip_text
from scripts.uk_transfer_test import _evaluate_lro, embed_texts

LOG = logging.getLogger("nc_geo_ablation")
REPO = Path(__file__).resolve().parents[2]


def _content(per: dict) -> dict:
    regions = sorted(per.keys())
    tx = {r: per[r]["text"][0] for r in regions}
    sh = {r: per[r]["shuffled"][0] for r in regions}
    nt = {r: per[r]["no_text"][0] for r in regions}
    diffs = [tx[r] - sh[r] for r in regions]
    n_neg = sum(d < 0 for d in diffs)
    m = lambda d: float(np.mean(list(d.values())))
    return {"n_groups": len(regions), "no_text": round(m(nt), 3), "text": round(m(tx), 3),
            "shuffled": round(m(sh), 3),
            "content_pct": round(100 * (m(tx) - m(sh)) / m(sh), 2),
            "groups_negative": f"{n_neg}/{len(regions)}",
            "per_group_content": {str(r): round(100 * (tx[r] - sh[r]) / sh[r], 1) for r in regions}}


def _join_aist(df: pd.DataFrame, tol_deg: float = 1e-3) -> pd.DataFrame:
    """Join regime_code / aist_litho_macro_code / aist_era_code from v4 onto the transfer df by a
    NEAREST-NEIGHBOUR spatial match. v4 stores coords as float32 and the transfer df as float64, so
    a rounded-key equi-join misses; AIST codes are per-location (constant with depth), so we match
    each transfer borehole to the nearest v4 location within ``tol_deg`` (~100 m)."""
    from scipy.spatial import cKDTree
    codes = ["regime_code", "aist_litho_macro_code", "aist_era_code"]
    v4 = pd.read_parquet(REPO / "data/features/borings_japan_v4.parquet",
                         columns=["latitude_deg", "longitude_deg", *codes])
    v4 = v4.dropna(subset=codes).astype({"latitude_deg": "float64", "longitude_deg": "float64"})
    loc = v4.drop_duplicates(subset=["latitude_deg", "longitude_deg"]).reset_index(drop=True)
    tree = cKDTree(loc[["latitude_deg", "longitude_deg"]].to_numpy())
    q = df[["latitude_deg", "longitude_deg"]].to_numpy(np.float64)
    dist, idx = tree.query(q, k=1)
    hit = dist < tol_deg
    out = df.copy()
    for c in codes:
        vals = loc[c].to_numpy()[idx].astype(float)
        out[c] = np.where(hit, vals, np.nan)
    return out


def run(out: Path, cache_dir: Path, seeds: list[int]) -> dict:
    df, base, _ = load_domain("japan", cache_dir)
    texts = [strip_text(t, "japan", "lithology_only") for t in df["text"].tolist()]
    from scripts.text_leakage_controls import STRIP_VOCAB_VERSION
    emb = embed_texts(texts, cache_dir / f"ncnull_japan_lithology_only_{STRIP_VOCAB_VERSION}_e5.npy")
    joined = _join_aist(df)
    hit = joined["aist_litho_macro_code"].notna().to_numpy()
    LOG.info("AIST join coverage %.3f (%d/%d rows)", hit.mean(), hit.sum(), len(df))
    sub = joined[hit].reset_index(drop=True)
    emb_sub = emb[hit]
    res = {"config": {"representation": "lithology_only", "seeds": seeds,
                      "join_coverage": round(float(hit.mean()), 3), "n_joined": int(hit.sum()),
                      "n_full": len(df)}, "geological_fold_lm": {}, "regime_ablation": {}}

    # ---- #5 geological folds: leave-lithology-macro-out and leave-era-out (LM, global null) ----
    for foldname, col in (("leave_litho_macro_out", "aist_litho_macro_code"),
                          ("leave_era_out", "aist_era_code")):
        grp = sub[col].astype(int).astype(str)            # group label per joined row
        vc = grp.value_counts()
        rowmask = grp.isin(vc[vc >= 60].index).to_numpy()  # keep groups with >=60 rows
        d = sub[rowmask].copy().reset_index(drop=True)
        d["region"] = grp[rowmask].to_numpy()              # _evaluate_lro folds on df["region"]
        c = _content(_evaluate_lro(d, base, emb_sub[rowmask], seeds))
        res["geological_fold_lm"][foldname] = c
        LOG.info("#5 %s: LM content %.1f%% (%s) over %d groups", foldname,
                 c["content_pct"], c["groups_negative"], c["n_groups"])

    # ---- #3 regime-redundancy: thin base vs +AIST-regime one-hot (administrative folds, LM) ----
    reg_oh = pd.get_dummies(sub["regime_code"].astype(int), prefix="reg").astype(np.float32)
    sub_reg = pd.concat([sub, reg_oh], axis=1)
    for label, b in (("thin_base", base), ("plus_regime", base + list(reg_oh.columns))):
        c = _content(_evaluate_lro(sub_reg, b, emb_sub, seeds))
        res["regime_ablation"][label] = c
        LOG.info("#3 %s (LM, admin folds): content %.1f%% (%s)", label,
                 c["content_pct"], c["groups_negative"])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({"geo_fold_lm": {k: v["content_pct"] for k, v in res["geological_fold_lm"].items()},
                      "regime_ablation": {k: v["content_pct"] for k, v in res["regime_ablation"].items()}},
                     indent=2))
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=REPO / "data/features/derived")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run(a.out, a.cache_dir, a.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
