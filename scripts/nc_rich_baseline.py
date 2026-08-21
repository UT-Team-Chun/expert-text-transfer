#!/usr/bin/env python
"""NC round-3 [E2] — rich structured-geology baseline ladder.

The reviewer's most important remaining objection: "text helps because your
structured geology baseline is too crude" (8-way surface regime). This script
measures the LM content effect against progressively RICHER structured baselines,
ending with the best structured description available -- the 61-D rule-based
lithology parser of the SAME text -- inside the baseline:

  Japan ladder (administrative LRO, leak-proof per-fold PCA, 5 seeds):
    thin        : depth, lat, lon
    +regime     : + AIST 8-way surface-regime one-hot
    +geology    : + AIST granular lithology-macro (14) + geological-era (9) one-hots
    +parser     : + 61-D structured lithology parser of the description itself
  UK ladder:
    thin        : depth, ground elevation, lat, lon
    +parser     : + 61-D BS5930 parser features (no archive-level geology codes in
                  the AGS extract; the parser is the best structured description)

At every rung the content effect = (real LM embedding) vs (shuffled LM embedding)
ON TOP of that baseline. If the effect survives the richest rung, the frozen-LM
channel carries information beyond the best structured geology we can construct;
if it shrinks toward zero, the message becomes "free text is a cheap route to
structured geological covariates locked in unstructured archives" -- the paper is
strengthened either way (and we report whichever is true).

Run (CPU, reuses caches):
  cd backend
  uv run python -m scripts.nc_rich_baseline \
      --out ../docs/research/2026-07-04_rich_baseline_ladder.json
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
                                           strip_text)
from scripts.uk_transfer_test import _evaluate_lro, embed_texts

LOG = logging.getLogger("nc_rich_baseline")
REPO = Path(__file__).resolve().parents[2]


def _content(per: dict) -> dict:
    regions = sorted(per.keys())
    tx = {r: per[r]["text"][0] for r in regions}
    sh = {r: per[r]["shuffled"][0] for r in regions}
    nt = {r: per[r]["no_text"][0] for r in regions}
    n_neg = sum(tx[r] - sh[r] < 0 for r in regions)
    m = lambda d: float(np.mean(list(d.values())))
    return {"baseline_rmse": round(m(nt), 3), "shuffled_rmse": round(m(sh), 3),
            "text_rmse": round(m(tx), 3),
            "content_pct": round(100 * (m(tx) - m(sh)) / m(sh), 2),
            "regions_negative": f"{n_neg}/{len(regions)}"}


def run(out: Path, cache_dir: Path, seeds: list[int]) -> dict:
    res: dict = {"config": {"seeds": seeds, "representation": "lithology_only e5",
                            "protocol": "administrative LRO, leak-proof per-fold PCA"}}

    # ---------------- Japan ----------------
    df, base, _ = load_domain("japan", cache_dir)
    raw = df["text"].tolist()
    texts = [strip_text(t, "japan", "lithology_only") for t in raw]
    from scripts.text_leakage_controls import STRIP_VOCAB_VERSION
    emb = embed_texts(texts, cache_dir / f"ncnull_japan_lithology_only_{STRIP_VOCAB_VERSION}_e5.npy")
    joined = _join_aist(df)
    parser = _structured_litho_features(raw, df, "japan").astype(np.float32)

    oh_regime = pd.get_dummies(joined["regime_code"].astype("Int64"), prefix="rg").astype(np.float32)
    oh_litho = pd.get_dummies(joined["aist_litho_macro_code"].astype("Int64"), prefix="lt").astype(np.float32)
    oh_era = pd.get_dummies(joined["aist_era_code"].astype("Int64"), prefix="er").astype(np.float32)
    parser_cols = [f"pf{i}" for i in range(parser.shape[1])]
    dfj = pd.concat([joined.reset_index(drop=True), oh_regime, oh_litho, oh_era,
                     pd.DataFrame(parser, columns=parser_cols)], axis=1)

    ladders = {
        "thin": list(base),
        "plus_regime": list(base) + list(oh_regime.columns),
        "plus_geology": list(base) + list(oh_regime.columns) + list(oh_litho.columns) + list(oh_era.columns),
        "plus_parser": (list(base) + list(oh_regime.columns) + list(oh_litho.columns)
                        + list(oh_era.columns) + parser_cols),
    }
    res["japan"] = {}
    for name, cols in ladders.items():
        c = _content(_evaluate_lro(dfj, cols, emb, seeds))
        res["japan"][name] = {"n_base_features": len(cols), **c}
        LOG.info("JP %-13s (%3d feats): content %+.1f%% (%s)", name, len(cols),
                 c["content_pct"], c["regions_negative"])

    # ---------------- UK ----------------
    dfu, base_u, _ = load_domain("uk", cache_dir)
    raw_u = dfu["text"].tolist()
    texts_u = [strip_text(t, "uk", "lithology_only") for t in raw_u]
    emb_u = embed_texts(texts_u, cache_dir / f"ncnull_uk_lithology_only_{STRIP_VOCAB_VERSION}_e5.npy")
    parser_u = _structured_litho_features(raw_u, dfu, "uk").astype(np.float32)
    pcols_u = [f"pf{i}" for i in range(parser_u.shape[1])]
    dfu2 = pd.concat([dfu.reset_index(drop=True),
                      pd.DataFrame(parser_u, columns=pcols_u)], axis=1)
    res["uk"] = {}
    for name, cols in (("thin", list(base_u)), ("plus_parser", list(base_u) + pcols_u)):
        c = _content(_evaluate_lro(dfu2, cols, emb_u, seeds))
        res["uk"][name] = {"n_base_features": len(cols), **c}
        LOG.info("UK %-13s (%3d feats): content %+.1f%% (%s)", name, len(cols),
                 c["content_pct"], c["regions_negative"])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({d: {k: v["content_pct"] for k, v in res[d].items()}
                      for d in ("japan", "uk")}, indent=2))
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
