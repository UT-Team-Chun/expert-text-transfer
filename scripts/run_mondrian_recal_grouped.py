#!/usr/bin/env python
"""Mondrian conformal re-evaluation under grouped calibration splits (P-T8).

The published target-region conformal figures split calibration/evaluation
by RANDOM ROW at seed 42 -- with ~9.3 layers per borehole, sibling layers of
one borehole land on both sides, sharing borehole-specific error and making
coverage easy. This script re-evaluates the same prediction artefacts under
three split units:

- ``row``      : the legacy split (reproduces the published numbers)
- ``borehole`` : whole boreholes to one side (identity from the v4id parquet,
                 positionally aligned -- verified y_true/regime byte-equality)
- ``site``     : ~500 m coordinate cells (coarser; nearby boreholes of one
                 construction site move together)

and reports, per split unit and alpha in {0.5, 0.8, 0.95}: marginal
coverage, per-regime coverage, MEAN INTERVAL WIDTH and the Winkler
(interval) score -- coverage alone can be bought with width, so both are
shown (the review's P0-8).

Cheap: quantile fitting only, no model fits.

CLI::

    cd backend
    .venv/bin/python -m scripts.run_mondrian_recal_grouped \
        --run dkl_national_full_v2 \
        --out ../docs/research/2026-08-12_conformal_grouped_split.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from national.evaluation.calibration import ConformalCalibrator
from national.evaluation.leave_region_out import DEFAULT_REGIONS

LOG = logging.getLogger("run_mondrian_recal_grouped")
REPO = Path(__file__).resolve().parents[2]

ALPHAS = (0.5, 0.8, 0.95)


def _split_masks(unit: str, n: int, boring_file: np.ndarray,
                 lat: np.ndarray, lon: np.ndarray, seed: int,
                 cal_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if unit == "row":
        perm = rng.permutation(n)
        n_cal = int(cal_fraction * n)
        cal = np.zeros(n, bool)
        cal[perm[:n_cal]] = True
        return cal, ~cal
    if unit == "borehole":
        groups = boring_file
    elif unit == "site":
        groups = np.char.add(
            np.round(lat.astype(np.float64), 2).astype(str),
            np.char.add("_", np.round(lon.astype(np.float64), 2).astype(str)))
    else:  # pragma: no cover
        raise ValueError(unit)
    uniq = np.unique(groups)
    take = set(rng.permutation(uniq)[: int(len(uniq) * cal_fraction)])
    cal = np.isin(groups, list(take))
    return cal, ~cal


def _winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
             alpha: float) -> float:
    """Mean Winkler interval score at central coverage ``alpha``."""
    a = 1.0 - alpha
    width = hi - lo
    below = (y < lo)
    above = (y > hi)
    score = width + (2.0 / a) * ((lo - y) * below + (y - hi) * above)
    return float(score.mean())


def evaluate_split(unit: str, y: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                   regime: np.ndarray, boring_file: np.ndarray,
                   lat: np.ndarray, lon: np.ndarray, seed: int) -> dict:
    cal, ev = _split_masks(unit, len(y), boring_file, lat, lon, seed)
    calib = ConformalCalibrator().fit_mondrian(
        y[cal], mu[cal], sd[cal], regime[cal], alphas=ALPHAS, min_group_n=30)
    out: dict = {"n_cal": int(cal.sum()), "n_eval": int(ev.sum())}
    if unit != "row":
        out["n_cal_groups"] = int(len(np.unique(boring_file[cal]))) \
            if unit == "borehole" else None
    for a in ALPHAS:
        lo, hi = calib.interval_mondrian(mu[ev], sd[ev], regime[ev], a)
        cov = float(((y[ev] >= lo) & (y[ev] <= hi)).mean())
        entry = {
            "coverage": round(cov, 5),
            "gap": round(cov - a, 5),
            "mean_width": round(float((hi - lo).mean()), 4),
            "winkler": round(_winkler(y[ev], lo, hi, a), 4),
        }
        # per-regime coverage
        per_reg = {}
        for g in np.unique(regime[ev]):
            m = regime[ev] == g
            per_reg[int(g)] = round(float(
                ((y[ev][m] >= lo[m]) & (y[ev][m] <= hi[m])).mean()), 5)
        entry["per_regime_coverage"] = per_reg
        out[f"alpha_{a}"] = entry
    return out


V4ID_PARQUET = REPO / "data/features/borings_japan_v4id.parquet"


def _align_identity(y: np.ndarray, regime_npz: np.ndarray,
                    v4_path: Path | None = None):
    """Attach borehole identity (and the canonical regime) to a predictions.npz.

    Full-corpus runs align positionally with the v4id parquet, and their npz
    ``regime`` is verified byte-identical to ``regime_code`` before use.

    Leave-region-out runs hold out ONE region, whose rows are scattered through
    the parquet's row order rather than contiguous, so we select them with the
    same bounding box the splitter used and verify ``y_true`` byte-equality on
    that ordered selection. Their npz ``regime`` is NOT usable: an earlier
    version of the LRO runner saved the *training*-fold regime array (2,168,230
    rows against 495,725 predictions for kanto) -- the same quirk documented at
    ``scripts/run_tta_lro.py:143``. The Mondrian bins therefore come from the
    parquet's ``regime_code``, which is the array the npz was meant to hold.
    """
    v4 = pd.read_parquet(
        v4_path or V4ID_PARQUET,
        columns=["n_value", "regime_code", "boring_file",
                 "latitude_deg", "longitude_deg"])
    yv = v4["n_value"].to_numpy(np.float64)
    rv = v4["regime_code"].to_numpy(np.int64)
    bf = v4["boring_file"].astype(str).to_numpy()
    lat_all = v4["latitude_deg"].to_numpy(np.float64)
    lon_all = v4["longitude_deg"].to_numpy(np.float64)

    if len(y) == len(yv) and np.array_equal(y, yv):
        if not np.array_equal(regime_npz, rv):
            raise SystemExit(
                "full-corpus predictions.npz regime disagrees with the v4id "
                "parquet regime_code; refusing to guess the Mondrian bins.")
        return bf, lat_all, lon_all, rv, "positional (full corpus)"

    for name, (la0, la1, lo0, lo1) in DEFAULT_REGIONS.items():
        m = (lat_all >= la0) & (lat_all <= la1) & (lon_all >= lo0) & (lon_all <= lo1)
        if int(m.sum()) != len(y) or not np.array_equal(yv[m], y):
            continue
        how = f"held-out region bbox '{name}'"
        if len(regime_npz) == len(y):
            if not np.array_equal(regime_npz, rv[m]):
                raise SystemExit(
                    f"predictions.npz regime disagrees with regime_code on the "
                    f"'{name}' bbox selection; aborting.")
        else:
            how += (f"; regime from parquet (npz regime is train-sized, "
                    f"{len(regime_npz)} rows -- run_tta_lro.py:143 quirk)")
        return bf[m], lat_all[m], lon_all[m], rv[m], how

    raise SystemExit(
        "predictions.npz rows do not align with the v4id parquet -- cannot "
        "attach borehole identity; aborting.")


def run(run_name: str, out: Path, seeds: list[int]) -> dict:
    npz = np.load(REPO / f"data/runs/{run_name}/predictions.npz")
    y, mu, sd = (npz["y_true"].astype(np.float64),
                 npz["pred_mean"].astype(np.float64),
                 npz["pred_std"].astype(np.float64))
    boring_file, lat, lon, regime, how = _align_identity(
        y, npz["regime"].astype(np.int64))
    finite = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sd)
    n_drop = int((~finite).sum())
    if n_drop:
        LOG.warning("dropping %d non-finite prediction rows", n_drop)
        y, mu, sd, regime = y[finite], mu[finite], sd[finite], regime[finite]
        boring_file, lat, lon = boring_file[finite], lat[finite], lon[finite]
    LOG.info("identity alignment: %s (%d rows, %d boreholes)", how, len(y),
             len(np.unique(boring_file)))

    res: dict = {"run": run_name, "splits": {}, "alignment": how,
                 "n_rows": int(len(y)), "n_boreholes": int(len(np.unique(boring_file))),
                 "n_nonfinite_dropped": n_drop}
    for unit in ("row", "borehole", "site"):
        per_seed = [evaluate_split(unit, y, mu, sd, regime, boring_file,
                                   lat, lon, seed) for seed in seeds]
        agg: dict = {"per_seed": per_seed}
        for a in ALPHAS:
            key = f"alpha_{a}"
            agg[key] = {
                "coverage_mean": round(float(np.mean(
                    [p[key]["coverage"] for p in per_seed])), 5),
                "gap_mean": round(float(np.mean(
                    [p[key]["gap"] for p in per_seed])), 5),
                "mean_width": round(float(np.mean(
                    [p[key]["mean_width"] for p in per_seed])), 4),
                "winkler": round(float(np.mean(
                    [p[key]["winkler"] for p in per_seed])), 4),
            }
            LOG.info("%-9s a=%.2f: cov %.4f (gap %+0.4f) width %.2f winkler %.2f",
                     unit, a, agg[key]["coverage_mean"], agg[key]["gap_mean"],
                     agg[key]["mean_width"], agg[key]["winkler"])
        res["splits"][unit] = agg

    res["config"] = {
        "alphas": list(ALPHAS), "seeds": seeds, "cal_fraction": 0.5,
        "min_group_n": 30,
        "prereg": "docs/research/2026-08-11_nc_text_preregistration.md P-T8",
    }
    res["_provenance"] = {
        "purpose": "P-T8: conformal coverage under borehole/site-grouped "
                   "calibration splits, with width and Winkler alongside "
                   "coverage (the row split shares borehole error across "
                   "the cal/eval boundary)",
        "alignment": "predictions.npz rows verified byte-identical to the "
                     "v4id parquet on y_true before identity is attached; "
                     "full-corpus runs align positionally (regime also "
                     "checked), LRO runs match a held-out-region bbox and "
                     "take regime from the parquet because the LRO runner "
                     "saved a train-sized regime array",
        "script": "backend/scripts/run_mondrian_recal_grouped.py",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    LOG.info("wrote %s", out)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="dkl_national_full_v2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    run(args.run, args.out, args.seeds)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(main())
