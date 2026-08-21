#!/usr/bin/env python
"""NC round-3 [E1] — genuine cross-archive ZERO-SHOT transfer (Japan <-> UK).

The reviewer's sharpest point: the UK experiment is a cross-national REPLICATION
(independent model, no weight transfer), so "the only thing that can cross the
boundary is the words" overstates. This experiment tests actual transfer: train a
model on ONE archive (depth + frozen multilingual-e5 text embedding; PCA basis fit
on the SOURCE archive only), predict SPT-N in the OTHER archive with NO target
training rows (zero-shot), plus a few-shot arm (n target rows appended).

Design (per direction, e.g. Japan -> UK):
  features      : depth_from_surface (+ 64-D source-fit PCA of lithology-only e5
                  embedding). Coordinates are excluded by construction (a lat/lon
                  function learned in Japan has no meaning for UK locations) --
                  which is itself the paper's point.
  target        : raw N for training; evaluation is SCALE-FREE (Spearman rank
                  correlation on the target archive) plus RMSE on within-archive
                  z-scores (harmonisation uses only two aggregate stats, no
                  per-row labels; stated in the caveat).
  arms          : depth_only / depth+text / depth+shuffled_text (content null,
                  test-side embeddings row-permuted).
  reference     : target-trained depth_only (what local training buys).
  few-shot      : + n in {100, 1000} random target rows appended to training.

If depth+text beats depth_only zero-shot (higher rank correlation), the text
channel carries information ACROSS archive+language, not just within each -- a
true transfer claim. If not, the paper keeps the (accurate) replication language.

Run (CPU, reuses embedding caches):
  cd backend
  uv run python -m scripts.nc_cross_archive \
      --out ../docs/research/2026-07-04_cross_archive_transfer.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from scripts.text_leakage_controls import load_domain, strip_text
from scripts.uk_transfer_test import embed_texts

LOG = logging.getLogger("nc_cross_archive")
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
    # z-scored RMSE: harmonisation via two aggregate stats only
    z = (y - y.mean()) / y.std()
    zp = (pred - pred.mean()) / (pred.std() + 1e-9)
    return {"spearman_rho": round(rho, 4),
            "z_rmse": round(float(np.sqrt(np.mean((zp - z) ** 2))), 4)}


def _load(domain: str, cache_dir: Path):
    df, _, _ = load_domain(domain, cache_dir)
    texts = [strip_text(t, domain, "lithology_only") for t in df["text"].tolist()]
    from scripts.text_leakage_controls import STRIP_VOCAB_VERSION
    emb = embed_texts(texts, cache_dir / f"ncnull_{domain}_lithology_only_{STRIP_VOCAB_VERSION}_e5.npy")
    depth = df["depth_from_surface"].to_numpy(np.float64).reshape(-1, 1)
    y = df["n_value"].to_numpy(np.float64)
    return depth, emb, y


def run_direction(src: str, tgt: str, cache_dir: Path, seeds: list[int],
                  fewshot: list[int]) -> dict:
    from sklearn.decomposition import PCA
    Xs_d, Es, ys = _load(src, cache_dir)
    Xt_d, Et, yt = _load(tgt, cache_dir)
    # PCA basis: SOURCE archive only (no target information enters the representation)
    pca = PCA(n_components=64, svd_solver="randomized", random_state=0).fit(Es)
    Ps, Pt = pca.transform(Es), pca.transform(Et)
    out: dict = {"n_source": len(ys), "n_target": len(yt)}

    # reference: what local training buys (target-trained depth-only, 5-fold-ish holdout skipped;
    # trained on target and evaluated in-sample would leak -- use 50/50 split)
    rng = np.random.default_rng(0)
    half = rng.permutation(len(yt))[: len(yt) // 2]
    mask = np.zeros(len(yt), bool); mask[half] = True
    ref_pred = _fit_predict(Xt_d[mask], yt[mask], Xt_d[~mask])
    out["reference_target_trained_depth_only"] = _metrics(ref_pred, yt[~mask])

    arms: dict[str, dict] = {}
    for arm in ("depth_only", "depth_text", "depth_shuffled"):
        per_seed = []
        for s in seeds:
            if arm == "depth_only":
                Xtr, Xte = Xs_d, Xt_d
            else:
                Pt_arm = Pt
                if arm == "depth_shuffled":
                    g = np.random.default_rng(s)
                    Pt_arm = Pt[g.permutation(len(Pt))]
                Xtr = np.hstack([Xs_d, Ps])
                Xte = np.hstack([Xt_d, Pt_arm])
            pred = _fit_predict(Xtr, ys, Xte, seed=s)
            per_seed.append(_metrics(pred, yt))
        arms[arm] = {k: round(float(np.mean([m[k] for m in per_seed])), 4)
                     for k in per_seed[0]}
    out["zero_shot"] = arms

    # few-shot: append n target rows (features incl. text) to the training set
    fs: dict[str, dict] = {}
    for n in fewshot:
        per_seed = []
        for s in seeds:
            g = np.random.default_rng(s)
            idx = g.permutation(len(yt))[:n]
            hold = np.ones(len(yt), bool); hold[idx] = False
            Xtr = np.vstack([np.hstack([Xs_d, Ps]), np.hstack([Xt_d[idx], Pt[idx]])])
            ytr = np.concatenate([ys, yt[idx]])
            pred = _fit_predict(Xtr, ytr, np.hstack([Xt_d[hold], Pt[hold]]), seed=s)
            per_seed.append(_metrics(pred, yt[hold]))
        fs[f"n={n}"] = {k: round(float(np.mean([m[k] for m in per_seed])), 4)
                        for k in per_seed[0]}
    out["few_shot_depth_text"] = fs
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=REPO / "data/features/derived")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--fewshot", type=int, nargs="+", default=[100, 1000])
    a = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    res = {"config": {"representation": "lithology_only, multilingual-e5, source-fit PCA-64",
                      "model": "HGB(400)", "seeds": a.seeds,
                      "note": "zero-shot: no target rows in training; metrics scale-free "
                              "(Spearman) + z-RMSE (two aggregate stats only)"}}
    for src, tgt in (("japan", "uk"), ("uk", "japan")):
        LOG.info("direction %s -> %s", src, tgt)
        res[f"{src}_to_{tgt}"] = run_direction(src, tgt, a.cache_dir, a.seeds, a.fewshot)
        z = res[f"{src}_to_{tgt}"]["zero_shot"]
        LOG.info("  zero-shot rho: depth %.3f | +text %.3f | +shuffled %.3f",
                 z["depth_only"]["spearman_rho"], z["depth_text"]["spearman_rho"],
                 z["depth_shuffled"]["spearman_rho"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    print(json.dumps({d: res[d]["zero_shot"] for d in ("japan_to_uk", "uk_to_japan")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
