#!/usr/bin/env python
"""Mondrian per-regime conformal recalibration for trained LMC national runs.

The LMC trainer (``scripts.train_lmc_national``) writes a multi-output
``predictions.npz`` with the 8 keys ``pred_mean_n``, ``pred_mean_gw``,
``pred_std_n``, ``pred_std_gw``, ``y_n``, ``y_gw``, ``mask_n``, ``mask_gw``
where the ``mask_*`` arrays indicate which rows carry an observed target for
that task (groundwater is ~81 % observed at national scale, SPT N is ~100 %).

This driver runs the existing single-output split-conformal Mondrian
calibrator (``national.evaluation.calibration.ConformalCalibrator``) once
per task, restricting each fit to the observed rows of that task. The
result is a JSON with the same structure as ``conformal_mondrian.json``
but keyed by task::

    {"task_n": {... per-α, per-regime ...}, "task_gw": {...}}

Because the two tasks are conditionally independent given the LMC posterior
mean / std, a per-task fit is the right primitive — and reusing the
single-output calibrator keeps the code path identical to Paper B's
single-task calibration story.

The LMC ``predictions.npz`` does not carry ``regime`` (we only persist the
per-task tensors), so the regime label is reconstructed by re-reading the
parquet that the LMC trainer iterated in row order. The script asserts the
row counts match before extracting ``regime_code``.

Run:
  cd backend
  .venv/bin/python -m scripts.run_mondrian_recal_lmc \\
      --run-dir ../data/runs/dkl_national_lmc_v4_m8k \\
      --parquet ../data/features/borings_japan_v4.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from national.evaluation.calibration import ConformalCalibrator

LOG = logging.getLogger("mondrian_recal_lmc")

DEFAULT_ALPHAS = (0.5, 0.8, 0.95)
REGIME_NAMES = {
    0: "ALLUVIAL",
    1: "DILUVIAL",
    2: "VOLCANIC_ASH",
    3: "SEDIMENTARY",
    4: "IGNEOUS",
    5: "METAMORPHIC",
    6: "LIMESTONE",
    7: "UNKNOWN",
}

REQUIRED_KEYS = (
    "pred_mean_n",
    "pred_mean_gw",
    "pred_std_n",
    "pred_std_gw",
    "y_n",
    "y_gw",
    "mask_n",
    "mask_gw",
)


def _load_predictions(npz_path: Path) -> dict[str, np.ndarray]:
    """Load and validate the LMC predictions.npz. Asserts all 8 keys present."""
    npz = np.load(npz_path)
    missing = [k for k in REQUIRED_KEYS if k not in npz.files]
    if missing:
        raise KeyError(
            f"predictions.npz at {npz_path} is missing required keys: {missing}. "
            f"Found: {sorted(npz.files)}"
        )
    return {k: np.asarray(npz[k]) for k in REQUIRED_KEYS}


def _load_regime(parquet_path: Path, n_expected: int) -> np.ndarray:
    """Read ``regime_code`` aligned to LMC predictions row order.

    The LMC trainer iterates the parquet in row order without re-ordering,
    so the i-th prediction corresponds to the i-th parquet row. Missing
    ``regime_code`` defaults to 7 (UNKNOWN), matching ``_build_features``.
    """
    df = pd.read_parquet(parquet_path)
    if len(df) != n_expected:
        raise ValueError(
            f"Row-count mismatch: parquet has {len(df)} rows but "
            f"predictions.npz has {n_expected}. Re-run the LMC trainer on "
            f"this parquet, or pass the parquet the run was trained on."
        )
    if "regime_code" in df.columns:
        regime = df["regime_code"].to_numpy(dtype=np.int64)
    else:
        LOG.warning(
            "parquet %s has no `regime_code` column; defaulting all rows to "
            "regime=7 (UNKNOWN). Mondrian groups will collapse to a single bucket.",
            parquet_path,
        )
        regime = np.full(len(df), 7, dtype=np.int64)
    return regime


def _recal_task(
    task_label: str,
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    regime: np.ndarray,
    mask: np.ndarray,
    *,
    cal_fraction: float,
    seed: int,
    alphas: tuple[float, ...],
    min_group_n: int,
) -> dict:
    """Run the single-output Mondrian recalibration restricted to ``mask``."""
    obs_idx = np.flatnonzero(mask)
    n_obs = int(obs_idx.size)
    LOG.info(
        "[%s] observed rows: %d / %d (%.1f %%)",
        task_label, n_obs, mask.size, 100.0 * mask.mean(),
    )
    if n_obs < 4:
        raise ValueError(
            f"Task '{task_label}' has only {n_obs} observed rows — too few "
            f"to split into cal/eval halves."
        )
    y = y_true[obs_idx].astype(np.float64)
    mu = pred_mean[obs_idx].astype(np.float64)
    sigma = np.maximum(pred_std[obs_idx].astype(np.float64), 1e-6)
    g = regime[obs_idx].astype(np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_obs)
    n_cal = int(cal_fraction * n_obs)
    n_cal = max(2, min(n_cal, n_obs - 2))  # keep at least 2 in each half
    cal_idx = perm[:n_cal]
    eval_idx = perm[n_cal:]
    LOG.info("[%s] cal=%d eval=%d", task_label, cal_idx.size, eval_idx.size)

    cal = ConformalCalibrator().fit_mondrian(
        y[cal_idx], mu[cal_idx], sigma[cal_idx],
        groups=g[cal_idx], alphas=tuple(alphas),
        min_group_n=min_group_n,
    )

    eval_regimes = g[eval_idx]
    out: dict = {
        "n_observed": n_obs,
        "n_cal": int(cal_idx.size),
        "n_eval": int(eval_idx.size),
        "per_regime": {},
        "marginal": {},
        "n_cal_per_regime": {
            int(k): int(v) for k, v in (cal.n_cal_per_group or {}).items()
        },
    }
    for a in alphas:
        cov_mond = cal.coverage_mondrian(
            y[eval_idx], mu[eval_idx], sigma[eval_idx],
            groups=eval_regimes, alpha=float(a),
        )
        cov_marg = cal.coverage(
            y[eval_idx], mu[eval_idx], sigma[eval_idx], alpha=float(a),
        )
        per_regime_cov: dict[int, dict] = {}
        for gr in np.unique(eval_regimes):
            mask_g = eval_regimes == gr
            lo, hi = cal.interval_mondrian(
                mu[eval_idx][mask_g], sigma[eval_idx][mask_g],
                groups=eval_regimes[mask_g], alpha=float(a),
            )
            covered = (
                (y[eval_idx][mask_g] >= lo) & (y[eval_idx][mask_g] <= hi)
            ).mean()
            n_cal_g = int((cal.n_cal_per_group or {}).get(int(gr), 0))
            per_regime_cov[int(gr)] = {
                "name": REGIME_NAMES.get(int(gr), f"R{int(gr)}"),
                "n_eval": int(mask_g.sum()),
                "n_cal": n_cal_g,
                "coverage": float(covered),
                "uses_marginal_fallback": n_cal_g < min_group_n,
            }
        a_key = str(float(a))
        out["per_regime"][a_key] = per_regime_cov
        out["marginal"][a_key] = {
            "coverage_marginal_only": float(cov_marg),
            "coverage_mondrian": float(cov_mond),
            "gap_mondrian": float(cov_mond - a),
            "gap_marginal": float(cov_marg - a),
        }
        LOG.info(
            "[%s] α=%.2f  marginal=%.3f  mondrian=%.3f  gap_mond=%+.3f",
            task_label, a, cov_marg, cov_mond, cov_mond - a,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="LMC run directory containing predictions.npz.",
    )
    parser.add_argument(
        "--parquet", type=Path, required=True,
        help="Parquet the LMC trainer iterated to produce predictions.npz. "
             "Used to recover the per-row regime_code (LMC predictions.npz "
             "does not carry regime).",
    )
    parser.add_argument(
        "--cal-fraction", type=float, default=0.5,
        help="Random fraction of observed rows used as calibration set "
             "(remaining fraction is the held-out evaluation set).",
    )
    parser.add_argument(
        "--min-group-n", type=int, default=30,
        help="Minimum rows per regime to fit a group-specific quantile. "
             "Groups with fewer rows fall back to the marginal quantile.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS),
        help="Nominal coverage levels to fit + evaluate.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s"
    )

    npz_path = args.run_dir / "predictions.npz"
    if not npz_path.exists():
        parser.error(f"predictions.npz not found at {npz_path}")
    LOG.info("Loading %s", npz_path)
    data = _load_predictions(npz_path)

    n = int(data["y_n"].size)
    LOG.info("Loaded n=%d predictions (both tasks)", n)
    regime = _load_regime(args.parquet, n_expected=n)

    alphas = tuple(float(a) for a in args.alphas)

    task_n = _recal_task(
        "task_n",
        y_true=data["y_n"],
        pred_mean=data["pred_mean_n"],
        pred_std=data["pred_std_n"],
        regime=regime,
        mask=data["mask_n"].astype(bool),
        cal_fraction=args.cal_fraction,
        seed=args.seed,
        alphas=alphas,
        min_group_n=args.min_group_n,
    )
    task_gw = _recal_task(
        "task_gw",
        y_true=data["y_gw"],
        pred_mean=data["pred_mean_gw"],
        pred_std=data["pred_std_gw"],
        regime=regime,
        mask=data["mask_gw"].astype(bool),
        cal_fraction=args.cal_fraction,
        seed=args.seed,
        alphas=alphas,
        min_group_n=args.min_group_n,
    )

    out = {
        "run_dir": str(args.run_dir.resolve()),
        "parquet": str(args.parquet.resolve()),
        "n_total": n,
        "cal_fraction": float(args.cal_fraction),
        "seed": int(args.seed),
        "alphas": [float(a) for a in alphas],
        "min_group_n": int(args.min_group_n),
        "task_n": task_n,
        "task_gw": task_gw,
    }
    out_path = args.run_dir / "conformal_mondrian_lmc.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    LOG.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
