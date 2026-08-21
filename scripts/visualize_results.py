#!/usr/bin/env python
"""Diagnostic plots for a trained foundation model on Kanto borings.

Builds a 6-panel PNG covering: predicted vs actual scatter, residuals,
calibration, spatial map, per-regime breakdown, per-depth RMSE.
Run after :mod:`scripts.train_kanto_smoke` to inspect a saved model.

Example::

    cd backend
    uv run python -m scripts.visualize_results \
        --run-dir ../data/runs/kanto/kanto_30pct_4k_30ep
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from national.data.boring_dataset import BoringDataset
from national.evaluation.calibration import (
    TemperatureScaler,
    coverage,
    reliability_diagram,
)
from national.evaluation.spatial_kfold import spatial_kfold_split
from national.models.foundation import FoundationModel
from national.tiling.regime_classifier import Regime

LOG = logging.getLogger("scripts.visualize_results")


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=repo / "data/runs/kanto/kanto_30pct_4k_30ep")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=repo / "data/features/borings_kanto_aist.parquet",
    )
    parser.add_argument("--train-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--regime-one-hot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force the dataset's regime_one_hot flag to True/False. When "
            "unset, the script reads ``regime_one_hot`` from "
            "``summary.json`` so the visualisation always matches the "
            "training-time dataset layout."
        ),
    )
    parser.add_argument(
        "--aist-granular-one-hot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Force the dataset's aist_granular_one_hot flag to True/False. "
            "When unset, the script reads ``aist_granular_one_hot`` from "
            "``summary.json`` to mirror the training-time encoder input "
            "layout. Required for Paper B' Pillar 1 multimodal runs."
        ),
    )
    parser.add_argument(
        "--depth-slice-m",
        type=float,
        default=5.0,
        help="Depth (m) for the spatial heatmap (panel D).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--pred-batch",
        type=int,
        default=20_000,
        help="Batch size for posterior prediction (keeps large SVGP from OOM).",
    )
    parser.add_argument(
        "--temperature-scale",
        action="store_true",
        help=(
            "Fit a single-scalar temperature on the predictive std so the "
            "Gaussian NLL is minimized post-hoc. Reports raw vs scaled "
            "reliability and writes ``temperature_scaling.json`` plus "
            "``diagnostics_calibrated.png`` next to the run."
        ),
    )
    parser.add_argument(
        "--ts-folds",
        type=int,
        default=3,
        help="Spatial K-fold count used for the leave-one-fold-out TS estimate.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    if args.device == "mps":
        import gpytorch.settings as gp_settings

        gp_settings._linalg_dtype_cholesky._global_value = torch.float32
        gp_settings._linalg_dtype_symeig._global_value = torch.float32

    artifact = args.run_dir / "foundation_model.pt"
    if not artifact.exists():
        parser.error(f"Model artifact missing: {artifact}")
    LOG.info("Loading model %s", artifact)
    model = FoundationModel.load(artifact).to(args.device)
    model.eval()

    # Read regime_one_hot and target_transform from the run's summary.json
    # so the dataset we reload matches the one the training used exactly.
    summary_path = args.run_dir / "summary.json"
    summary = (
        json.loads(summary_path.read_text()) if summary_path.exists() else {}
    )
    regime_one_hot = (
        args.regime_one_hot
        if args.regime_one_hot is not None
        else bool(summary.get("regime_one_hot", False))
    )
    aist_granular_one_hot = (
        args.aist_granular_one_hot
        if args.aist_granular_one_hot is not None
        else bool(summary.get("aist_granular_one_hot", False))
    )
    extra_one_hot_columns: dict[str, int] | None = None
    if aist_granular_one_hot:
        from national.data.derived.aist_granular import (
            N_ERA_CODES,
            N_LITHO_MACRO_CODES,
        )
        extra_one_hot_columns = {
            "aist_era_code": N_ERA_CODES,
            "aist_litho_macro_code": N_LITHO_MACRO_CODES,
        }
    target_transform = summary.get("target_transform", "none")
    # The training run may have used a subset of derived feature columns
    # (covariate ablation) or none at all. Honour that when reloading so
    # the encoder input dim matches the trained inducing-point dim.
    feature_columns_from_summary = summary.get("feature_columns")
    if feature_columns_from_summary is None:
        feature_columns_to_use = [
            "absolute_elevation", "river_distance_km", "coast_distance_km",
        ]
    else:
        feature_columns_to_use = list(feature_columns_from_summary)
    LOG.info(
        "Reload config from summary: regime_one_hot=%s target_transform=%s "
        "feature_columns=%s",
        regime_one_hot, target_transform, feature_columns_to_use,
    )

    # Recreate the exact training subsample so we can plot the model on the
    # data it actually saw. Use the same seed and train-fraction.
    LOG.info("Loading dataset %s", args.parquet)
    ds = BoringDataset(
        args.parquet,
        feature_columns=feature_columns_to_use,
        regime_one_hot=regime_one_hot,
        extra_one_hot_columns=extra_one_hot_columns,
        target_transform=target_transform,
        standardize_target=True,
    )
    # If the summary recorded an n_smoke (the actual number of rows the
    # training used), prefer that over the CLI default --train-fraction.
    # This keeps the diagnostic in sync with the model regardless of how
    # the training was launched.
    n_total = len(ds)
    summary_n_smoke = int(summary.get("n_smoke", 0))
    if summary_n_smoke > 0:
        n_smoke = min(summary_n_smoke, n_total)
        LOG.info(
            "Using n_smoke=%d from summary (overrides --train-fraction=%s)",
            n_smoke,
            args.train_fraction,
        )
    else:
        n_smoke = int(n_total * args.train_fraction)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(n_total, size=n_smoke, replace=False))
    sub_x = torch.from_numpy(ds._x[idx]).float()
    sub_y = torch.from_numpy(ds._y_raw[idx]).float()
    sub_regime = torch.from_numpy(ds._regime[idx].astype(np.int64))

    LOG.info("Predicting on %d samples (batched)...", n_smoke)
    # Batch the posterior so a 500k × 6k matern kernel does not OOM. 20 000
    # rows keep peak GPU memory under ~10 GiB on a 6k-inducing SVGP.
    pred_batch = int(getattr(args, "pred_batch", 20_000) or 20_000)
    pred_means: list[np.ndarray] = []
    pred_stds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, sub_x.shape[0], pred_batch):
            end = min(start + pred_batch, sub_x.shape[0])
            pred_chunk = model.predict(sub_x[start:end])
            pred_means.append(pred_chunk.mean.cpu().numpy())
            pred_stds.append(pred_chunk.std.cpu().numpy())
    pred_mean_arr = np.concatenate(pred_means, axis=0)
    pred_std_arr = np.concatenate(pred_stds, axis=0)

    # ``target_transform`` was already pulled from the run's summary.json
    # above; just invert per its log-normal moments here.
    from national.data.boring_dataset import invert_target_transform_moments

    y_pred_arr, y_std_arr = invert_target_transform_moments(
        pred_mean_arr.astype(np.float64),
        pred_std_arr.astype(np.float64),
        target_transform,
    )
    y_pred = y_pred_arr.astype(np.float32)
    y_std = y_std_arr.astype(np.float32)
    y_true = sub_y.numpy()
    residual = y_true - y_pred
    abs_resid = np.abs(residual)

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"FoundationModel diagnostics  --  {args.run_dir.name}  "
        f"(n={n_smoke:,}, RMSE={np.sqrt((residual**2).mean()):.2f})",
        fontsize=14,
    )

    _plot_pred_vs_actual(axes[0, 0], y_true, y_pred)
    _plot_residual_by_regime(axes[0, 1], residual, sub_regime.numpy())
    _plot_calibration(axes[0, 2], y_std, abs_resid)
    _plot_spatial_heatmap(
        axes[1, 0],
        sub_x[:, 0].numpy(),
        sub_x[:, 1].numpy(),
        sub_x[:, 2].numpy(),
        y_pred,
        depth_slice_m=args.depth_slice_m,
    )
    _plot_per_depth_rmse(axes[1, 1], sub_x[:, 2].numpy(), y_true, y_pred)
    _plot_spatial_kfold(axes[1, 2], sub_x[:, 0].numpy(), sub_x[:, 1].numpy(), y_true, y_pred, args.seed)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = args.run_dir / "diagnostics.png"
    fig.savefig(out, dpi=120)
    LOG.info("Wrote %s", out)
    print(f"\nDiagnostics saved to: {out}\n")

    summary_path = args.run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print("Spatial K-fold summary:")
        for fold in summary.get("spatial_kfold", []):
            print(
                f"  fold {fold['fold']}: RMSE={fold['rmse']:.2f}  "
                f"MAE={fold['mae']:.2f}  mean_std={fold['std_mean']:.2f}"
            )
        rd = reliability_diagram(y_true, y_pred, y_std, alphas=(0.5, 0.8, 0.95))
        print("\nReliability diagram (empirical coverage @ nominal alpha):")
        for _, row in rd.iterrows():
            print(
                f"  alpha={row['nominal']:.2f}  empirical={row['empirical']:.3f}  "
                f"gap={row['gap']:+.3f}"
            )

    if args.temperature_scale:
        _run_temperature_scaling(
            args=args,
            sub_x=sub_x,
            y_true=y_true,
            y_pred=y_pred,
            y_std=y_std,
            sub_regime=sub_regime,
        )
    return 0


def _run_temperature_scaling(
    *,
    args: argparse.Namespace,
    sub_x: torch.Tensor,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    sub_regime: torch.Tensor,
) -> None:
    """Fit a single-scalar Temperature Scaling on the predictive std.

    Two estimates are reported:
        - ``global``: fit and evaluate on all rows (sanity check, optimistic).
        - ``cv``:     mean of leave-one-fold-out estimates from spatial
                      K-fold, so the T fit on training rows is evaluated on
                      held-out rows.

    Writes:
        - ``<run-dir>/temperature_scaling.json``
        - ``<run-dir>/diagnostics_calibrated.png``
    """
    alphas: tuple[float, float, float] = (0.5, 0.8, 0.95)
    rd_raw = reliability_diagram(y_true, y_pred, y_std, alphas=alphas)

    # Global fit (uses all rows for both fit and eval).
    global_scaler = TemperatureScaler().fit(y_true, y_pred, y_std)
    T_global = float(global_scaler.T)
    rd_global = reliability_diagram(
        y_true, y_pred, global_scaler.apply(y_std), alphas=alphas
    )

    # Leave-one-fold-out CV estimate.
    sub_df = pd.DataFrame(
        {
            "latitude_deg": sub_x[:, 0].numpy(),
            "longitude_deg": sub_x[:, 1].numpy(),
            "n_value": y_true,
        }
    )
    folds = spatial_kfold_split(
        sub_df, n_folds=int(args.ts_folds), mesh_level=2, seed=args.seed
    )
    per_fold: list[dict[str, float]] = []
    cv_scaled_std = np.empty_like(y_std, dtype=np.float64)
    for fi, (train_idx, test_idx) in enumerate(folds):
        # Fit T on the held-OUT-of-this-fold rows (i.e. the other folds' test
        # rows), apply to this fold's test rows. ``train_idx`` are the
        # held-out rows that were predicted by a model that did not see them
        # any more than the test rows; we keep them as the fit set.
        T_fold = float(np.sqrt(
            (((y_true[train_idx] - y_pred[train_idx]) / y_std[train_idx]) ** 2).mean()
        ))
        cv_scaled_std[test_idx] = y_std[test_idx] * T_fold
        per_fold.append({"fold": fi, "T": T_fold, "n_fit": int(len(train_idx)), "n_eval": int(len(test_idx))})
    rd_cv = reliability_diagram(y_true, y_pred, cv_scaled_std, alphas=alphas)
    T_cv_mean = float(np.mean([row["T"] for row in per_fold]))

    print("\n=== Temperature Scaling ===")
    print(f"T (global fit):       {T_global:.4f}")
    print(f"T (CV mean over {len(per_fold)} folds): {T_cv_mean:.4f}  "
          f"[per-fold: {', '.join(f'{r['T']:.3f}' for r in per_fold)}]")
    print("\nReliability raw vs scaled:")
    print(f"  {'alpha':<8}{'raw_emp':<10}{'raw_gap':<10}{'cv_emp':<10}{'cv_gap':<10}{'glob_emp':<10}{'glob_gap':<10}")
    for a, raw, cv, gl in zip(alphas, rd_raw.itertuples(), rd_cv.itertuples(), rd_global.itertuples(), strict=True):
        print(
            f"  {a:<8.2f}"
            f"{raw.empirical:<10.3f}{raw.gap:<+10.3f}"
            f"{cv.empirical:<10.3f}{cv.gap:<+10.3f}"
            f"{gl.empirical:<10.3f}{gl.gap:<+10.3f}"
        )

    # Z-distribution diagnostics: a single scalar TS can only fix the scale of
    # the predictive std (mean Z^2 -> 1). If the residuals are heavy-tailed,
    # the coverage gap reflects a *shape* mismatch that TS cannot touch.
    from scipy.stats import norm as _norm
    z_raw = (y_true - y_pred) / y_std
    abs_z = np.abs(z_raw)
    z_stats = {
        "mean_z2": float((z_raw * z_raw).mean()),
        "mean_z": float(z_raw.mean()),
        "kurtosis": float(((z_raw - z_raw.mean()) ** 4).mean() / ((z_raw - z_raw.mean()) ** 2).mean() ** 2),
        "median_abs_z": float(np.median(abs_z)),
        "per_alpha_T": [
            {
                "alpha": float(a),
                "nominal_z": float(_norm.ppf((1 + a) / 2)),
                "empirical_abs_z_quantile": float(np.quantile(abs_z, a)),
                "per_alpha_T": float(np.quantile(abs_z, a) / float(_norm.ppf((1 + a) / 2))),
            }
            for a in alphas
        ],
    }
    print("\nZ-residual diagnostics (Gaussian reference values in parentheses):")
    print(f"  mean(Z²)   = {z_stats['mean_z2']:.4f}  (1.000)")
    print(f"  mean(Z)    = {z_stats['mean_z']:.4f}  (0.000)")
    print(f"  median|Z|  = {z_stats['median_abs_z']:.4f}  (0.6745)")
    print(f"  kurtosis   = {z_stats['kurtosis']:.3f}  (3.000)   — >3 means heavier-tailed than Gaussian")
    print("  Per-alpha T (= empirical |Z|@alpha / nominal):")
    for row in z_stats["per_alpha_T"]:
        print(
            f"    alpha={row['alpha']:.2f}  nominal_z={row['nominal_z']:.3f}  "
            f"emp|Z|={row['empirical_abs_z_quantile']:.3f}  T_alpha={row['per_alpha_T']:.3f}"
        )

    payload = {
        "T_global": T_global,
        "T_cv_mean": T_cv_mean,
        "per_fold": per_fold,
        "z_diagnostics": z_stats,
        "raw_reliability": rd_raw.to_dict(orient="records"),
        "cv_reliability": rd_cv.to_dict(orient="records"),
        "global_reliability": rd_global.to_dict(orient="records"),
    }
    ts_path = args.run_dir / "temperature_scaling.json"
    ts_path.write_text(json.dumps(payload, indent=2))
    LOG.info("Wrote %s", ts_path)

    # ---- companion calibration plot --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: reliability before/after (CV-based).
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    axes[0].plot(rd_raw["nominal"], rd_raw["empirical"], "o-", color="C3", label="raw")
    axes[0].plot(rd_cv["nominal"], rd_cv["empirical"], "s-", color="C2",
                 label=f"TS (T_cv={T_cv_mean:.3f})")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Nominal coverage")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("Reliability before / after temperature scaling")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="lower right")

    # Panel 2: empirical Z = (y - mu) / sigma histogram, raw vs scaled.
    z_raw = (y_true - y_pred) / y_std
    z_scaled = (y_true - y_pred) / cv_scaled_std
    bins = np.linspace(-5, 5, 81)
    axes[1].hist(z_raw, bins=bins, alpha=0.45, color="C3", density=True, label=f"raw (mean Z²={ (z_raw**2).mean():.2f})")
    axes[1].hist(z_scaled, bins=bins, alpha=0.45, color="C2", density=True, label=f"TS  (mean Z²={ (z_scaled**2).mean():.2f})")
    xs = np.linspace(-5, 5, 200)
    axes[1].plot(xs, np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi), "k--", lw=1, label="N(0,1)")
    axes[1].set_xlim(-5, 5)
    axes[1].set_xlabel("Z = (y_true - y_pred) / y_std")
    axes[1].set_ylabel("density")
    axes[1].set_title("Z-residual distribution")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(
        f"Temperature Scaling -- {args.run_dir.name}  "
        f"(T_cv={T_cv_mean:.3f}, T_global={T_global:.3f})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = args.run_dir / "diagnostics_calibrated.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    LOG.info("Wrote %s", plot_path)
    # Avoid an unused-variable warning when ``coverage`` is only referenced
    # transitively through ``reliability_diagram``.
    _ = coverage


# --------------------------------------------------------------------------- #
# Panel plotters
# --------------------------------------------------------------------------- #
def _plot_pred_vs_actual(ax, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """A: scatter with parity line, capped at 99th percentile so a few extreme
    boring rows don't compress the bulk of the points into a corner."""
    cap = float(np.percentile(np.concatenate([y_true, y_pred]), 99.0))
    ax.scatter(y_true, y_pred, s=4, alpha=0.05, color="C0", rasterized=True)
    lim = (0, max(cap, 1.0))
    ax.plot(lim, lim, color="black", linestyle="--", linewidth=1, label="parity")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual N-value")
    ax.set_ylabel("Predicted N-value")
    ax.set_title("A. Predicted vs. actual (99th-percentile clip)")
    ax.legend(loc="upper left")


def _plot_residual_by_regime(ax, residual: np.ndarray, regime: np.ndarray) -> None:
    """B: residual distribution per regime. Wider spread for harder regimes."""
    codes_present = sorted(np.unique(regime))
    data = []
    labels = []
    for code in codes_present:
        mask = regime == code
        if mask.sum() < 30:
            continue
        data.append(residual[mask])
        labels.append(f"{Regime(int(code)).name}\n(n={int(mask.sum()):,})")
    ax.boxplot(data, labels=labels, showfliers=False, widths=0.6)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Residual  (actual − predicted)")
    ax.set_title("B. Residual distribution by regime")
    ax.tick_params(axis="x", rotation=20)


def _plot_calibration(ax, y_std: np.ndarray, abs_resid: np.ndarray) -> None:
    """C: predicted std vs |residual| binned. A well-calibrated model lies on y=x."""
    bins = np.linspace(y_std.min(), np.percentile(y_std, 99), 11)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bin_idx = np.digitize(y_std, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(centers) - 1)
    mean_abs_resid = np.zeros(len(centers))
    for k in range(len(centers)):
        mask = bin_idx == k
        if mask.sum() > 0:
            mean_abs_resid[k] = abs_resid[mask].mean()
    ax.plot(centers, centers * np.sqrt(2 / np.pi), color="black", linestyle="--", label="ideal (Gaussian)")
    ax.plot(centers, mean_abs_resid, marker="o", color="C2", label="empirical")
    ax.set_xlabel("Predicted std")
    ax.set_ylabel("Mean |residual| in bin")
    ax.set_title("C. Calibration (binned)")
    ax.legend()


def _plot_spatial_heatmap(
    ax,
    lat: np.ndarray,
    lon: np.ndarray,
    depth: np.ndarray,
    y_pred: np.ndarray,
    *,
    depth_slice_m: float,
) -> None:
    """D: 2-D map of predictions at a chosen depth slice."""
    mask = (depth > depth_slice_m - 1.0) & (depth < depth_slice_m + 1.0)
    if mask.sum() < 10:
        ax.text(0.5, 0.5, "no samples in depth slice", ha="center", va="center")
        ax.set_title(f"D. Predicted N at depth {depth_slice_m} m")
        return
    sc = ax.scatter(
        lon[mask], lat[mask], c=y_pred[mask], s=8, alpha=0.6, cmap="viridis",
        vmin=0, vmax=float(np.percentile(y_pred[mask], 99))
    )
    plt.colorbar(sc, ax=ax, label="Predicted N")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"D. Predicted N at depth ≈{depth_slice_m} m (n={int(mask.sum()):,})")


def _plot_per_depth_rmse(
    ax,
    depth: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """E: RMSE binned by depth."""
    bin_edges = np.array([0, 2, 5, 10, 15, 20, 30, 50, 100])
    centers, rmse, n_per_bin = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = (depth >= lo) & (depth < hi)
        if mask.sum() < 50:
            continue
        residual = y_true[mask] - y_pred[mask]
        rmse.append(float(np.sqrt((residual ** 2).mean())))
        centers.append(f"{lo}-{hi} m")
        n_per_bin.append(int(mask.sum()))
    bars = ax.bar(centers, rmse, color="C1")
    for bar, n in zip(bars, n_per_bin, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"n={n:,}",
            ha="center",
            fontsize=8,
        )
    ax.set_ylabel("RMSE in bin")
    ax.set_xlabel("Depth bin")
    ax.set_title("E. Per-depth RMSE")
    ax.tick_params(axis="x", rotation=20)


def _plot_spatial_kfold(
    ax,
    lat: np.ndarray,
    lon: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seed: int,
) -> None:
    """F: per-fold RMSE map (colored points by fold), with mean RMSE in title."""
    df = pd.DataFrame({"latitude_deg": lat, "longitude_deg": lon})
    folds = spatial_kfold_split(df, n_folds=3, mesh_level=2, seed=seed)
    rmse_per_fold = []
    fold_labels = np.zeros(len(lat), dtype=int)
    for fi, (_, test_idx) in enumerate(folds):
        residual = y_true[test_idx] - y_pred[test_idx]
        rmse_per_fold.append(float(np.sqrt((residual**2).mean())))
        fold_labels[test_idx] = fi
    sc = ax.scatter(lon, lat, c=fold_labels, s=3, alpha=0.5, cmap="tab10", rasterized=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = "F. Spatial 3-fold partition  --  RMSE per fold: " + ", ".join(
        f"{r:.2f}" for r in rmse_per_fold
    )
    ax.set_title(title)


if __name__ == "__main__":
    sys.exit(main())
