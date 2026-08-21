#!/usr/bin/env python
"""End-to-end Phase B smoke train on real Kanto borings.

This is the first script in the project that trains the foundation
model on actual KuniJiban data instead of synthetic samples. The
runtime is short (~5 minutes on a laptop CPU) but it exercises every
production code path: ``BoringDataset`` -> ``FoundationModel`` ->
``FoundationTrainer`` -> ``FoundationModel.predict`` -> spatial K-fold
RMSE.

Output: a JSON summary at ``data/runs/kanto/<run_name>/summary.json`` plus a
saved foundation artifact. Re-runs the same hyperparameters
deterministically given the same ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from national.data.boring_dataset import BoringDataset
from national.evaluation.spatial_kfold import spatial_kfold_split
from national.models.foundation import (
    EncoderSpec,
    FoundationModel,
    FoundationSpec,
    SVGPSpec,
    init_inducing_points,
)
from national.training.trainer import FoundationTrainer

LOG = logging.getLogger("scripts.train_kanto_smoke")

# The DKL encoder's Fourier parametrization, hoisted to module constants so
# the MS-SGE per-band gate machinery (which derives each band's wavelength
# lambda_k FROM this parametrization -- prereg risk (iii): read the actual
# implementation, never assume) can never diverge from the EncoderSpec built
# in main(). NOTE: 12 bands, not the EncoderSpec default 16 -- the pilot
# encoder has always been built with fourier_bands=12 here.
ENCODER_FOURIER_BANDS = 12
ENCODER_FOURIER_SCALE = 4.0


def parse_snapshot_epochs(raw: str) -> list[int]:
    """Parse ``--save-snapshots "5,8,11"`` into a sorted, de-duplicated
    ``[5, 8, 11]`` (1-indexed epochs; "after N epochs of training").

    Empty string -> ``[]`` (the feature stays off). Non-integer tokens and
    epochs < 1 fail loud — a silently-dropped snapshot epoch would only be
    discovered after a full cluster run.
    """
    epochs: set[int] = set()
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            epoch = int(tok)
        except ValueError as exc:
            raise ValueError(
                f"--save-snapshots: {tok!r} is not an integer (in {raw!r})"
            ) from exc
        if epoch < 1:
            raise ValueError(
                f"--save-snapshots: epochs are 1-indexed and must be >= 1; "
                f"got {epoch} (in {raw!r})"
            )
        epochs.add(epoch)
    return sorted(epochs)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Factored out of :func:`main` so tests can exercise flag wiring (e.g.
    ``--no-residual-geo``) without running the full data-loading + training
    pipeline.
    """
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=repo / "data/features/borings_kanto.parquet")
    parser.add_argument("--output-dir", type=Path, default=repo / "data/runs/kanto/kanto_smoke")
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-inducing", type=int, default=2000)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.05,
        help="Random subsample fraction for the smoke run (default: 5%%).",
    )
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--indep-lambda",
        type=float,
        default=0.0,
        help=(
            "Weight on the HSIC(encoder_output, lat/lon) independence "
            "penalty (national.training.losses.hsic_regularizer). 0.0 "
            "(default) disables it. Ramped in linearly over the first "
            "few epochs by the trainer (see indep_warmup_epochs)."
        ),
    )
    parser.add_argument(
        "--log-indep-diagnostic",
        action="store_true",
        help=(
            "Record one {epoch, hsic_raw_mean, task_loss_mean} entry per "
            "training epoch (national.training.trainer.FoundationTrainer, "
            "cfg.training.log_indep_diagnostic), where hsic_raw_mean is the "
            "UNWEIGHTED HSIC(encoder_output, lat/lon) scale -- useful even "
            "when --indep-lambda=0.0, since that is exactly how you read "
            "off the natural HSIC scale before picking a lambda for a "
            "sweep. Purely diagnostic: does not add an encoder forward and "
            "cannot change the training trajectory. Persisted to "
            "summary.json under 'indep_diagnostic'."
        ),
    )
    parser.add_argument(
        "--adv-lambda",
        type=float,
        default=0.0,
        help=(
            "Weight on the adversarial coordinate-critic term "
            "(national.training.adversarial): a small MLP critic is "
            "trained every batch to regress standardized (lat, lon) from "
            "the encoder output, and the encoder is trained to defeat it "
            "(detach-alternating min-max, see adversarial.py's module "
            "docstring for the exact per-batch ordering). This is the GRL "
            "arm of the removal-mechanism bake-off: unlike --indep-lambda "
            "(which penalizes an RBF-kernel-scale STATISTICAL dependence), "
            "this attacks coordinate DECODABILITY directly. 0.0 (default) "
            "disables it -- the critic is never constructed. Ramped in "
            "linearly over the first few epochs on the encoder-facing term "
            "only (see cfg.training.adv_warmup_epochs); the critic itself "
            "trains from epoch 0. Per-epoch diagnostics "
            "({critic_mse_mean, critic_r2, task_loss_mean}) are always "
            "recorded when this is > 0.0 and persisted to summary.json "
            "under 'adv_diagnostic'."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | cpu | mps | cuda. 'auto' picks cuda > mps > cpu.",
    )
    parser.add_argument(
        "--regime-one-hot",
        action="store_true",
        help=(
            "Append 8-way regime one-hot to encoder input (in addition to the "
            "FiLM block). Significantly increases the regime signal at the "
            "cost of 8 extra encoder input columns."
        ),
    )
    parser.add_argument(
        "--aist-granular-one-hot",
        action="store_true",
        help=(
            "Append the AIST granular categorical one-hots "
            "(aist_era_code 11-way + aist_litho_macro_code 15-way) to "
            "the encoder input. Requires a parquet on the v3 schema "
            "(see scripts/add_aist_granular_to_parquet.py). Mutually "
            "compatible with --regime-one-hot: this flag adds 26 extra "
            "dimensions on top of the regime one-hot, for a total of "
            "+34 categorical encoder inputs. Paper B' Pillar 1 starter."
        ),
    )
    parser.add_argument(
        "--target-transform",
        choices=("none", "log1p"),
        default="none",
        help=(
            "Forward transform applied to the regression target before "
            "standardization. ``log1p`` matches the SPT N-value's "
            "heavy-right-tail distribution to a Gaussian likelihood much "
            "more cleanly than the raw N. Inverse transform is applied "
            "automatically for the K-fold RMSE / MAE reporting."
        ),
    )
    parser.add_argument(
        "--target-stats",
        choices=("legacy", "train_only"),
        default="legacy",
        help=(
            "WHICH rows fit the target-standardization mean/sd. 'legacy' "
            "(default) is the historical behaviour: the scaler is computed "
            "at BoringDataset construction from the FULL parquet — i.e. "
            "BEFORE the LRO exclusion / subsample / in-region carve, so a "
            "held-out region's label statistics leak into the scaler that "
            "standardizes the training target and destandardizes every "
            "prediction (the codex R4 review's verified confound: early-"
            "training predictions sit near the full-data mean, which lies "
            "between the train-only and held-out means, artificially "
            "favouring early snapshots out-of-region). 'train_only' refits "
            "the scaler on EXACTLY the final training rows — post-LRO-"
            "exclusion, post-subsample, POST-in-region-carve: only rows the "
            "model actually trains on (the carve rows are evaluation rows, "
            "so they must not touch the scaler either). Every historical "
            "result depends on 'legacy'; do NOT change the default. "
            "Provenance (mode, fitted stats, full-parquet stats, mean gap) "
            "is recorded in summary.json."
        ),
    )
    parser.add_argument(
        "--heteroscedastic-noise",
        action="store_true",
        help=(
            "Enable the heteroscedastic NoiseHead -- a small MLP that "
            "predicts log_variance from (depth_norm, regime_one_hot). "
            "Switches the model's likelihood from GaussianLikelihood "
            "(homoscedastic) to FixedNoiseGaussianLikelihood. Addresses "
            "the alpha=0.50 calibration over-cautiousness and the per-depth "
            "RMSE inflation."
        ),
    )
    parser.add_argument(
        "--kernel-type",
        choices=["matern52", "matern32", "matern12", "rbf"],
        default="matern52",
        help=(
            "GP kernel family on the encoder output. matern52 (default) is "
            "twice-differentiable; matern32 is once-differentiable; matern12 "
            "is non-differentiable (rough); rbf is infinitely smooth. The "
            "right choice depends on how rough the regression target is in "
            "the encoded feature space."
        ),
    )
    parser.add_argument(
        "--mean-type",
        choices=["constant", "linear"],
        default="constant",
        help=(
            "Prior mean function. 'constant' learns a single bias; 'linear' "
            "learns a linear combination of the encoder output dimensions "
            "(gpytorch.means.LinearMean). Useful when the target has a "
            "monotonic trend in the encoded representation."
        ),
    )
    parser.add_argument(
        "--inducing-init",
        choices=["random", "kmeans_pp", "kmeans_pp_stratified"],
        default="random",
        help=(
            "Inducing point initialization strategy. random is the historical "
            "default; kmeans_pp gives more uniformly-spread inducing; "
            "kmeans_pp_stratified balances across AIST regimes."
        ),
    )
    parser.add_argument(
        "--likelihood-type",
        choices=["gaussian", "studentt", "censored"],
        default="gaussian",
        help=(
            "Observation likelihood. gaussian (default) is homoscedastic "
            "Gaussian. studentt switches to a Student-t likelihood that "
            "natively handles the heavy-tail residual structure we observed "
            "(kurtosis ≈ 9.3). censored uses a right-censored Gaussian for "
            "the N≤100 cap (see --censored-cap). Ignored if "
            "--heteroscedastic-noise is also set "
            "(that path forces FixedNoiseGaussianLikelihood)."
        ),
    )
    parser.add_argument(
        "--censored-cap",
        type=float,
        default=100.0,
        help=(
            "Right-censoring threshold in raw N units. Used only when "
            "--likelihood-type=censored. Defaults to 100 to match the "
            "SPT N cap used throughout the paper."
        ),
    )
    parser.add_argument(
        "--regime-balanced-sampler",
        action="store_true",
        help=(
            "Up-weight rare regimes during minibatching (RegimeBalancedSampler). "
            "Recommended at national scale where volcanic-ash / limestone are "
            "orders of magnitude rarer than alluvial."
        ),
    )
    parser.add_argument(
        "--regime-balance-alpha",
        type=float,
        default=0.5,
        help=(
            "Temper for --regime-balanced-sampler: 0 = uniform, 1 = full "
            "inverse-frequency, 0.5 (default) = sqrt temper."
        ),
    )
    parser.add_argument(
        "--buffer-meshes",
        type=int,
        default=0,
        help=(
            "If > 0, use spatial_kfold_split_buffered with the given ring "
            "size (in secondary-mesh cells) to exclude train rows within "
            "that distance from any test mesh. 0 (default) means use the "
            "plain spatial_kfold_split."
        ),
    )
    parser.add_argument(
        "--leave-prefecture",
        type=str,
        default="",
        help=(
            "If non-empty, switch from K-fold to leave-prefecture-out "
            "evaluation; the named Kanto prefecture is the held-out test "
            "set, the rest is train. One of {tokyo, kanagawa, saitama, "
            "chiba, ibaraki, tochigi, gunma}."
        ),
    )
    parser.add_argument(
        "--leave-region",
        type=str,
        default="",
        help=(
            "If non-empty, switch from K-fold to leave-region-out evaluation "
            "for one of Japan's 8 standard geographic regions. The named "
            "region is the held-out test set, the rest is train. One of "
            "{hokkaido, tohoku, kanto, chubu, kansai, chugoku, shikoku, "
            "kyushu_okinawa}. Mutually exclusive with --leave-prefecture."
        ),
    )
    parser.add_argument(
        "--region-column",
        type=str,
        default="",
        help=(
            "If non-empty (together with --leave-region), switch the "
            "leave-region-out split from the JP bbox taxonomy "
            "(national.evaluation.leave_region_out.DEFAULT_REGIONS) to a "
            "categorical parquet column: rows whose value in this column "
            "equals --leave-region become the held-out test set. This is "
            "how non-JP archives whose folds are defined by a label column "
            "rather than bounding boxes (e.g. the UK BGS parquet's 'region' "
            "column with its 5 macro-regions, the fold definitions of "
            "scripts.uk_transfer_test) plug into the same DKL trainer. "
            "Ignored when --leave-region is empty."
        ),
    )
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=None,
        help=(
            "Explicit list of derived feature column names to use (in "
            "addition to the mandatory lat/lon/depth). Default is "
            "['absolute_elevation', 'river_distance_km', 'coast_distance_km']. "
            "Pass an empty list to ablate all derived features."
        ),
    )
    parser.add_argument(
        "--zero-fourier",
        action="store_true",
        help=(
            "Zero out the random-Fourier features over (lat, lon) in the "
            "encoder. Counter-test for 'is the encoder just memorising "
            "spatial coordinates?'"
        ),
    )
    parser.add_argument(
        "--save-snapshots",
        type=str,
        default="",
        help=(
            "EES snapshot battery (P-R3c..f, docs/research/"
            "2026-07-13_r3_preregistration.md): comma-separated 1-indexed "
            "epoch list (e.g. '5,8,11,14,17,20,25'). At the END of each "
            "listed epoch the full model state is saved in the "
            "FoundationModel ARTIFACT format to "
            "<output-dir>/checkpoints/ep{N}.pt (loadable directly with "
            "FoundationModel.load / scripts.nmi_ees_eval). Saving consumes "
            "no RNG, so the training trajectory is bit-identical with or "
            "without this flag. Empty (default) disables it."
        ),
    )
    parser.add_argument(
        "--sge-gate",
        choices=("off", "raw", "fourier", "test_only",
                 "nyquist", "lowpass", "nyquist_rev", "nyquist2"),
        default="off",
        help=(
            "Support-Gated Encodings (P-SGE, docs/research/"
            "2026-07-12_sge_preregistration.md): multiply the encoder's "
            "Fourier coordinate block by a FIXED per-row support gate "
            "g = sigmoid((t - DI)/s) precomputed at data load from the "
            "training coordinates (national.data.sge_gate). 'raw' computes "
            "the dissimilarity index DI in raw train-standardized lat/lon "
            "space (the invention); 'fourier' computes it in the 16-band "
            "Fourier feature space instead (the mechanism placebo, "
            "predicted to fail via near-aliasing); 'test_only' trains "
            "exactly like base and applies the raw-space gate only at "
            "eval/predict. 'off' (default) disables gating. MS-SGE per-band "
            "modes (P-MS, docs/research/2026-07-13_msge_preregistration.md; "
            "one gate column per Fourier band, g_k = sigmoid((0.5*lambda_k "
            "- d_NN)/(0.25*lambda_k)) with lambda_k read off the encoder's "
            "actual _FourierFeatures parametrization and d_NN the raw-space "
            "NN distance to the fixed 10k reference): 'nyquist' (arm G, the "
            "invention), 'lowpass' (arm H, STATIC mask keeping only bands "
            "with lambda >= the standardized equivalent of 100 km; no "
            "adaptivity, no dropout), 'nyquist_rev' (arm I, the reversed "
            "placebo: closes long-lambda bands, keeps short). Round-3 G2 "
            "rule (P-R3a/b, docs/research/2026-07-13_r3_preregistration.md): "
            "'nyquist2' (arm msge2) replaces the per-band sigmoid with "
            "g_k = exp(-(d_NN/(0.5*lambda_k))^2) — exactly 1 at d=0 (the "
            "handicap-free gate; the sigmoid rule caps in-support at "
            "sigma(2)~0.88), same lambda ladder, no new constants."
        ),
    )
    parser.add_argument(
        "--gate-dropout",
        type=float,
        default=0.0,
        help=(
            "TRAINING-ONLY per-row gate dropout probability p: with prob p "
            "a row's gate is set to 0 for that step, teaching the "
            "covariate-fallback pathway (in-support rows almost never see "
            "low gates otherwise). Meaningful with --sge-gate raw/fourier "
            "(the pre-registered p=0.2), or with --sge-gate off as the "
            "'drop' control arm (no gate ever, but the Fourier block is "
            "zeroed per-row with prob p during training). For the MS-SGE "
            "band modes nyquist/nyquist_rev the dropout is INDEPENDENT per "
            "band per row (pre-registered p=0.2). Refused with --sge-gate "
            "test_only (that arm's train path must be bit-identical to "
            "base) and with --sge-gate lowpass (arm H is static by "
            "pre-registration)."
        ),
    )
    parser.add_argument(
        "--add-residual-geo",
        dest="add_residual_geo",
        action="store_true",
        help=(
            "Add the short-scale Matern-3/2 residual kernel over raw "
            "(lat, lon) degrees, in addition to the DKL kernel over encoder "
            "features (SVGPSpec.add_residual_geo). Default: on (current "
            "behaviour)."
        ),
    )
    parser.add_argument(
        "--no-residual-geo",
        dest="add_residual_geo",
        action="store_false",
        help=(
            "Disable the raw-(lat, lon) residual kernel, leaving the "
            "encoder as the only coordinate pathway into the GP. Needed "
            "for the lambda-frontier (HSIC independence) experiments, "
            "which must isolate the encoder as the sole spatial channel."
        ),
    )
    parser.set_defaults(add_residual_geo=True)
    parser.add_argument(
        "--kfold-test-fold",
        type=int,
        default=-1,
        help=(
            "If >= 0, switch from train-on-all + report-K-fold-metrics "
            "to a single proper hold-out run: train on all folds EXCEPT "
            "the given one, and evaluate on that one. Enables proper "
            "buffered / leave-prefecture / nested spatial cross-validation "
            "by launching one job per held-out fold. Default -1 preserves "
            "the historical train-on-all + report-K-fold behaviour."
        ),
    )
    parser.add_argument(
        "--fold-assignment",
        choices=["random", "contiguous"],
        default="random",
        help=(
            "Spatial K-fold mesh assignment. 'random' (default) is the "
            "load-balanced random shuffle of mesh codes across folds, "
            "which interleaves folds spatially; 'contiguous' uses "
            "k-means on mesh centroids to produce geographic-block folds. "
            "Random shuffle is biased toward optimistic K-fold metrics "
            "(test rows surrounded by training neighbours); contiguous "
            "is the stricter reviewer-defensible variant."
        ),
    )
    parser.add_argument(
        "--encoder-dim",
        type=int,
        default=24,
        help=(
            "Encoder output dimensionality (the latent feeding the GP). "
            "Default 24 is what every published ablation used. Wider (32, 48, 64) "
            "tests whether LinearMean's win was an encoder-capacity gap."
        ),
    )
    parser.add_argument(
        "--studentt-df",
        type=float,
        default=4.0,
        help=(
            "Initial degrees of freedom ν for Student-t likelihood. The value "
            "is learnable during training, constrained to ν > 2 so the marginal "
            "variance stays finite. Default 4 is a pragmatic init for moderate "
            "heavy-tail (kurtosis 6 at ν=4). Ignored unless "
            "--likelihood-type=studentt."
        ),
    )
    parser.add_argument(
        "--pred-batch",
        type=int,
        default=20_000,
        help=(
            "Batch size for the K-fold posterior prediction at the end of "
            "training. 50k fits on a 48 GB GPU; drop to 10–20k on a 12 GB "
            "GPU (RTX 4070 Ti) to avoid OOM in the matern kernel materialisation."
        ),
    )
    parser.add_argument(
        "--baseline-pred-train",
        type=Path,
        default=None,
        help=(
            "Path to .npy of CatBoost/LightGBM mean predictions on the "
            "spatial out-of-bag training rows (one prediction per index in "
            "--baseline-idx-train). When set, switches the SVGP target to "
            "residuals y - baseline_pred for hybrid tree+GP training. The "
            "baseline mean is added back at inference time."
        ),
    )
    parser.add_argument(
        "--baseline-pred-test",
        type=Path,
        default=None,
        help=(
            "Path to .npy of CatBoost/LightGBM mean predictions on the "
            "held-out test rows (one prediction per index in "
            "--baseline-idx-test). Re-added to the SVGP residual prediction "
            "at fold-evaluation time."
        ),
    )
    parser.add_argument(
        "--baseline-idx-train",
        type=Path,
        default=None,
        help="Row indices into the parquet for --baseline-pred-train.",
    )
    parser.add_argument(
        "--baseline-idx-test",
        type=Path,
        default=None,
        help="Row indices into the parquet for --baseline-pred-test.",
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="catboost",
        help=(
            "Identifier of the teacher baseline (catboost / lightgbm). Used "
            "for provenance only; the actual predictions are loaded from "
            "--baseline-pred-{train,test}."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases run logging (requires WANDB_API_KEY env).",
    )
    parser.add_argument(
        "--wandb-project",
        default="geo-estimation-national",
        help="W&B project name. Used only when --wandb is set.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="W&B run name. Defaults to output-dir basename.",
    )
    return parser


def _categorical_region_holdout(
    parquet_path: Path,
    region_column: str,
    region_value: str,
    smoke_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-region-out split keyed by a categorical parquet column.

    Counterpart of ``national.evaluation.leave_region_out.leave_region_out_split``
    for archives whose folds are defined by a label column instead of JP
    bounding boxes (e.g. the UK BGS parquet's ``region`` column). The column
    is re-read from the parquet (BoringDataset does not retain it) and sliced
    by ``smoke_idx`` so it aligns row-for-row with the subsampled training
    tensors — BoringDataset preserves parquet row order, and ``smoke_idx``
    indexes into that order.

    Returns ``(train_keep, holdout_test_idx)`` as positional indices into the
    subsample (i.e. into ``smoke_idx``'s row order), matching the contract of
    the JP bbox path in :func:`main`.
    """
    import pandas as pd  # local import mirrors BoringDataset's lazy style

    values = pd.read_parquet(parquet_path, columns=[region_column])[region_column]
    known = sorted({str(v) for v in values.dropna().unique()})
    if str(region_value) not in known:
        raise ValueError(
            f"Unknown --leave-region {region_value!r} for --region-column "
            f"{region_column!r}; parquet has: {known}"
        )
    sub_values = values.to_numpy()[smoke_idx]
    holdout_mask = np.asarray([str(v) == str(region_value) for v in sub_values])
    if not holdout_mask.any():
        raise RuntimeError(
            f"--leave-region {region_value!r} matched no rows in the "
            f"subsample (train_fraction too small?)."
        )
    if holdout_mask.all():
        raise RuntimeError(
            f"--leave-region {region_value!r} matched EVERY subsampled row; "
            "nothing left to train on."
        )
    all_pos = np.arange(len(smoke_idx), dtype=np.int64)
    return all_pos[~holdout_mask], all_pos[holdout_mask]


def _predict_batched(
    model,
    eval_x: torch.Tensor,
    pred_batch: int,
    eval_gates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Chunked posterior prediction (transformed/standardised-target units
    are already inverted inside ``model.predict``'s denormalisation, but the
    target-transform inversion is the caller's job).

    ``eval_gates`` (test_only arm): per-row support gates aligned with
    ``eval_x``; installed chunk-by-chunk via
    ``model.encoder.set_eval_gate`` and always cleared afterwards.
    """
    pred_means: list[np.ndarray] = []
    pred_stds: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(0, eval_x.shape[0], pred_batch):
                end = min(start + pred_batch, eval_x.shape[0])
                if eval_gates is not None:
                    model.encoder.set_eval_gate(
                        torch.from_numpy(
                            np.asarray(eval_gates[start:end], dtype=np.float32)
                        )
                    )
                pred_chunk = model.predict(eval_x[start:end])
                pred_means.append(pred_chunk.mean.cpu().numpy())
                pred_stds.append(pred_chunk.std.cpu().numpy())
    finally:
        if eval_gates is not None:
            model.encoder.set_eval_gate(None)
    return np.concatenate(pred_means, axis=0), np.concatenate(pred_stds, axis=0)


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    # Support-Gated Encodings (P-SGE): resolve the arm plan first so flag
    # combinations the pre-registration does not define fail before any
    # data is loaded.
    from national.data.sge_gate import (
        INREGION_FRACTION,
        INREGION_SEED,
        draw_inregion_holdout,
        fit_msge_gate,
        fit_sge_gate,
        resolve_sge_plan,
    )

    sge_plan = resolve_sge_plan(args.sge_gate, args.gate_dropout)
    # EES snapshot battery (P-R3c..f): parse + range-check BEFORE any data
    # is loaded so a mis-specified epoch list fails in seconds, not after a
    # full training run. The trainer re-validates against its own n_epochs.
    snapshot_epochs = parse_snapshot_epochs(args.save_snapshots)
    if snapshot_epochs and snapshot_epochs[-1] > args.n_epochs:
        raise ValueError(
            f"--save-snapshots {args.save_snapshots!r} requests epoch "
            f"{snapshot_epochs[-1]} but --n-epochs is {args.n_epochs}: the "
            "snapshot would silently never be written."
        )
    if args.zero_fourier and args.sge_gate != "off":
        raise ValueError(
            "--zero-fourier with --sge-gate is contradictory: gating a "
            "zeroed Fourier block tests nothing. Pick one arm."
        )
    if args.heteroscedastic_noise and sge_plan.append_gate_column:
        raise ValueError(
            "--heteroscedastic-noise extracts the trailing regime one-hot "
            "columns from x, which the appended SGE gate column would "
            "corrupt. Not supported together."
        )

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    LOG.info("Using device=%s", args.device)

    # MPS does not support float64 (Apple Silicon GPUs are float32 only). GPyTorch's
    # variational strategy defaults its Cholesky factorization to float64 for
    # numerical stability; we force float32 globally before any model is built so
    # the SVGP forward pass works on MPS. Note: this can in theory hurt stability
    # on very ill-conditioned kernel matrices -- on a healthy SVGP (well-spaced
    # inducing points, reasonable lengthscales) the difference is negligible.
    if args.device == "mps":
        import gpytorch.settings as gp_settings
        gp_settings._linalg_dtype_cholesky._global_value = torch.float32
        gp_settings._linalg_dtype_symeig._global_value = torch.float32

    # kmeans_pp_stratified inducing init can produce near-degenerate K_zz
    # in some regimes (high-density clusters → close inducing points). The
    # gpytorch default jitter 1e-8 / max_tries=3 then fails with NotPSDError.
    # Bump both unconditionally — extra jitter is cheap and only matters
    # near the singular regime.
    import gpytorch.settings as gp_settings  # noqa: E402
    gp_settings.cholesky_jitter._global_float_value = 1e-4
    gp_settings.cholesky_max_tries._global_value = 10

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load dataset --------------------------------------------------------
    if args.feature_cols is None:
        feature_cols = ["absolute_elevation", "river_distance_km", "coast_distance_km"]
    elif len(args.feature_cols) == 1 and args.feature_cols[0].upper() == "NONE":
        # Sentinel for "no derived features" — POSIX argparse cannot
        # accept zero-argument nargs="+", so we use NONE.
        feature_cols = []
    else:
        feature_cols = list(args.feature_cols)
    LOG.info("Using feature_cols=%s", feature_cols)
    # Hybrid mode: the BoringDataset target is residual y - baseline_pred,
    # not raw y. The baseline predictions are required for both the training
    # rows (residual target) and the eventual held-out test rows
    # (re-addition at inference). Both --baseline-pred-{train,test} +
    # --baseline-idx-{train,test} must be supplied together.
    hybrid_mode = args.baseline_pred_train is not None
    if hybrid_mode:
        if args.baseline_idx_train is None:
            raise ValueError(
                "--baseline-pred-train requires --baseline-idx-train"
            )
        if args.baseline_pred_test is None or args.baseline_idx_test is None:
            raise ValueError(
                "Hybrid mode also requires --baseline-pred-test and "
                "--baseline-idx-test for inference-time mean re-addition"
            )
        # log1p is non-monotonic on signed residuals (they can be negative).
        # Force the residual target to the no-transform path.
        if args.target_transform != "none":
            LOG.warning(
                "Hybrid mode forces target_transform='none' "
                "(requested %r overridden)", args.target_transform,
            )
            args.target_transform = "none"
        LOG.info(
            "Hybrid mode ON: baseline=%s, residual target switched on",
            args.baseline_name,
        )
    # Paper B' Pillar 1: optional granular AIST one-hots (era 11-way +
    # lithology macro 15-way). Requires a v3 parquet with the two extra
    # columns; see backend/scripts/add_aist_granular_to_parquet.py.
    extra_one_hot_columns: dict[str, int] | None = None
    if args.aist_granular_one_hot:
        # Resolve at import time so we don't take a hard dep on the module
        # unless the user opts in.
        from national.data.derived.aist_granular import (
            N_ERA_CODES,
            N_LITHO_MACRO_CODES,
        )
        extra_one_hot_columns = {
            "aist_era_code": N_ERA_CODES,
            "aist_litho_macro_code": N_LITHO_MACRO_CODES,
        }
        LOG.info(
            "AIST granular one-hot ON: %d + %d = %d extra encoder dims",
            N_ERA_CODES,
            N_LITHO_MACRO_CODES,
            N_ERA_CODES + N_LITHO_MACRO_CODES,
        )
    dataset = BoringDataset(
        args.parquet,
        feature_columns=feature_cols,
        depth_scale_m=30.0,
        standardize_target=True,
        regime_one_hot=args.regime_one_hot,
        extra_one_hot_columns=extra_one_hot_columns,
        target_transform=args.target_transform,
        baseline_pred_npy=args.baseline_pred_train if hybrid_mode else None,
        baseline_idx_npy=args.baseline_idx_train if hybrid_mode else None,
        baseline_pred_test_npy=args.baseline_pred_test if hybrid_mode else None,
        baseline_idx_test_npy=args.baseline_idx_test if hybrid_mode else None,
    )
    LOG.info(
        "Loaded BoringDataset: %d rows, %d features (mean=%.2f std=%.2f)",
        len(dataset),
        dataset.n_features,
        dataset.target_mean,
        dataset.target_std,
    )

    # 2. Subsample for the smoke run ----------------------------------------
    n_total = len(dataset)
    n_smoke = int(n_total * args.train_fraction)
    rng = np.random.default_rng(args.seed)
    smoke_idx = rng.choice(n_total, size=n_smoke, replace=False)
    smoke_idx.sort()
    sub_x = torch.from_numpy(dataset._x[smoke_idx]).float()
    # ``dataset._y_raw`` is the *original-unit* target, preserved for
    # K-fold metric reporting (RMSE / MAE are computed in N-value units).
    # ``dataset._y`` is post-transform AND post-standardization, the
    # quantity the GP actually fits against.
    sub_y_raw = torch.from_numpy(dataset._y_raw[smoke_idx]).float()
    sub_y_std = torch.from_numpy(dataset._y[smoke_idx]).float()
    sub_regime = torch.from_numpy(dataset._regime[smoke_idx].astype(np.int64))
    # Hybrid mode: baseline_pred_per_row is in raw-N units (same as _y_raw).
    # We slice to the smoke subsample so the inference path can add it back.
    if hybrid_mode:
        sub_baseline_pred = dataset.baseline_pred_per_row[smoke_idx].astype(np.float32)
    else:
        sub_baseline_pred = None
    # Full-subsample copy retained for the in-region-holdout evaluation
    # (the working `sub_baseline_pred` gets sliced to train / holdout rows).
    sub_baseline_pred_full = sub_baseline_pred
    LOG.info("Subsampled %d / %d rows for smoke training", n_smoke, n_total)

    # Optional proper hold-out: train on rows OUTSIDE the held-out fold
    # so we can measure spatial-generalisation under buffered CV,
    # leave-prefecture-out, or nested-spatial conformal.
    if args.leave_prefecture and args.leave_region:
        raise ValueError(
            "--leave-prefecture and --leave-region are mutually exclusive."
        )
    holdout_active = (
        args.kfold_test_fold >= 0
        or bool(args.leave_prefecture)
        or bool(args.leave_region)
    )
    holdout_test_idx: np.ndarray | None = None
    if holdout_active:
        sub_df_for_split = pd.DataFrame(
            {
                "latitude_deg": sub_x[:, 0].numpy(),
                "longitude_deg": sub_x[:, 1].numpy(),
                "n_value": sub_y_raw.numpy(),
            }
        )
        if args.leave_prefecture:
            from national.evaluation.prefecture_regions import leave_prefecture_out_split

            split_iter = list(
                leave_prefecture_out_split(
                    sub_df_for_split, prefectures=[args.leave_prefecture]
                )
            )
            if not split_iter:
                raise RuntimeError(
                    f"leave_prefecture_out_split produced no fold for "
                    f"prefecture {args.leave_prefecture!r}; check bbox."
                )
            _, train_keep, holdout_test_idx = split_iter[0]
            LOG.info(
                "Hold-out (leave-prefecture=%s): train_keep=%d, test=%d",
                args.leave_prefecture, len(train_keep), len(holdout_test_idx),
            )
        elif args.leave_region and args.region_column:
            # Categorical leave-region-out (non-JP archives, e.g. UK BGS):
            # fold membership comes from a parquet label column, not a bbox.
            train_keep, holdout_test_idx = _categorical_region_holdout(
                args.parquet, args.region_column, args.leave_region, smoke_idx
            )
            LOG.info(
                "Hold-out (leave-region=%s via column %r): train_keep=%d, test=%d",
                args.leave_region, args.region_column,
                len(train_keep), len(holdout_test_idx),
            )
        elif args.leave_region:
            from national.evaluation.leave_region_out import (
                DEFAULT_REGIONS,
                leave_region_out_split,
            )

            if args.leave_region not in DEFAULT_REGIONS:
                raise ValueError(
                    f"Unknown --leave-region {args.leave_region!r}; "
                    f"choices: {sorted(DEFAULT_REGIONS)}"
                )
            chosen = {args.leave_region: DEFAULT_REGIONS[args.leave_region]}
            split_iter = list(leave_region_out_split(sub_df_for_split, regions=chosen))
            if not split_iter:
                raise RuntimeError(
                    f"leave_region_out_split produced no fold for region "
                    f"{args.leave_region!r}; check the bbox vs the data extent."
                )
            _, train_keep, holdout_test_idx = split_iter[0]
            LOG.info(
                "Hold-out (leave-region=%s): train_keep=%d, test=%d",
                args.leave_region, len(train_keep), len(holdout_test_idx),
            )
        else:
            if args.buffer_meshes > 0:
                from national.evaluation.spatial_kfold import (
                    spatial_kfold_split_buffered,
                )

                fold_splits = spatial_kfold_split_buffered(
                    sub_df_for_split, n_folds=3, mesh_level=2,
                    buffer_meshes=args.buffer_meshes, seed=args.seed,
                    base_split=args.fold_assignment,
                )
            elif args.fold_assignment == "contiguous":
                from national.evaluation.spatial_kfold import (
                    spatial_kfold_split_contiguous,
                )

                fold_splits = spatial_kfold_split_contiguous(
                    sub_df_for_split, n_folds=3, mesh_level=2, seed=args.seed,
                )
            else:
                fold_splits = spatial_kfold_split(
                    sub_df_for_split, n_folds=3, mesh_level=2, seed=args.seed,
                )
            if args.kfold_test_fold >= len(fold_splits):
                raise ValueError(
                    f"--kfold-test-fold={args.kfold_test_fold} out of range "
                    f"for n_folds={len(fold_splits)}"
                )
            train_keep, holdout_test_idx = fold_splits[args.kfold_test_fold]
            LOG.info(
                "Hold-out (fold=%d, buffer=%d): train_keep=%d, test=%d",
                args.kfold_test_fold, args.buffer_meshes,
                len(train_keep), len(holdout_test_idx),
            )

        # Full-subsample aliases; the actual train-side slicing is deferred
        # until after the P-SGE in-region carve-out + gate column below.
        sub_x_full = sub_x
        sub_y_raw_full = sub_y_raw
        sub_y_std_full = sub_y_std
        sub_regime_full = sub_regime
        train_keep = np.asarray(train_keep, dtype=np.int64)
        LOG.info("After hold-out, training side is %d rows", len(train_keep))
    else:
        sub_x_full = sub_x
        sub_y_raw_full = sub_y_raw
        sub_y_std_full = sub_y_std
        sub_regime_full = sub_regime
        train_keep = np.arange(n_smoke, dtype=np.int64)

    # ---- P-SGE: fixed in-region holdout (EVERY run) -------------------------
    # A 5% random draw (numpy seed 777, isolated generator) from the
    # TRAINING-region rows, removed from training and evaluated at the end
    # as `inregion_rmse` alongside the OOR metric (spatial_kfold[0]).
    train_keep, inregion_idx = draw_inregion_holdout(train_keep)
    LOG.info(
        "In-region holdout: %d rows (seed %d, fraction %.2f); training rows: %d",
        len(inregion_idx), INREGION_SEED, INREGION_FRACTION, len(train_keep),
    )

    # ---- target-standardization scaler (--target-stats) ---------------------
    # Leak fix (codex R4 review, docs/research/2026-07-13_codex_review_r4.md):
    # under 'legacy' the scaler was fitted at BoringDataset construction from
    # the FULL parquet — including the held-out region's labels. 'train_only'
    # refits it here, on EXACTLY the final training rows (train_keep is now
    # post-LRO-exclusion, post-subsample, POST-in-region-carve), then
    # re-materializes the standardized target for the whole subsample so the
    # training tensors below carry the clean scaler. The split itself is
    # untouched (the refit consumes no RNG and happens after all row
    # selection). Full-parquet stats are kept for the summary's mean-gap
    # provenance.
    target_stats_full_mean = float(dataset.target_mean)
    target_stats_full_std = float(dataset.target_std)
    if args.target_stats == "train_only":
        train_rows_global = smoke_idx[train_keep]
        dataset.refit_target_stats(train_rows_global)
        sub_y_std = torch.from_numpy(dataset._y[smoke_idx]).float()
        sub_y_std_full = sub_y_std
        LOG.info(
            "target-stats=train_only: scaler refitted on %d training rows "
            "(mean=%.4f std=%.4f); full-parquet scaler was mean=%.4f "
            "std=%.4f; mean gap (train_only - full) = %+.4f",
            len(train_rows_global), dataset.target_mean, dataset.target_std,
            target_stats_full_mean, target_stats_full_std,
            dataset.target_mean - target_stats_full_mean,
        )

    # ---- P-SGE: support gates ------------------------------------------------
    # Static per row (a fixed geometric function of the FINAL training
    # coordinates), so they are precomputed once here. `gates_full` is
    # aligned with the FULL subsample row order.
    lat_np = sub_x_full[:, 0].numpy().astype(np.float64)
    lon_np = sub_x_full[:, 1].numpy().astype(np.float64)
    gates_full: np.ndarray | None = None  # (N,) scalar or (N, n_bands)
    gate_model = None
    msge_model = None
    if sge_plan.band_mode is not None:
        # MS-SGE (P-MS): per-band gates, one column per encoder Fourier
        # band. lambda_k is derived from the SAME (bands, scale) pair the
        # EncoderSpec below is built with (module constants).
        msge_model = fit_msge_gate(
            lat_np[train_keep], lon_np[train_keep],
            mode=sge_plan.band_mode,
            n_bands=ENCODER_FOURIER_BANDS,
            fourier_scale=ENCODER_FOURIER_SCALE,
        )
        gates_full = msge_model.gates(lat_np, lon_np).astype(np.float32)
        LOG.info(
            "MS-SGE gates fitted (mode=%s, %d bands): lambda_km=[%s], "
            "deg_to_std=%.4f km_to_std=%.6f lowpass_keep=%s "
            "train per-band gate means=[%s]",
            msge_model.mode, msge_model.n_bands,
            ", ".join(f"{v:.1f}" for v in msge_model.lambda_km),
            msge_model.deg_to_std, msge_model.km_to_std,
            "".join("1" if v else "0" for v in msge_model.lowpass_keep),
            ", ".join(f"{v:.3f}" for v in gates_full[train_keep].mean(axis=0)),
        )
    elif sge_plan.needs_gates:
        gate_model = fit_sge_gate(
            lat_np[train_keep], lon_np[train_keep], space=sge_plan.gate_space
        )
        gates_full = gate_model.gate(lat_np, lon_np).astype(np.float32)
        LOG.info(
            "SGE gate fitted (space=%s): t=%.4f s=%.4f method=%s n_ref=%d "
            "train-gate mean=%.4f",
            gate_model.space, gate_model.threshold, gate_model.softness,
            gate_model.threshold_method, gate_model.n_ref,
            float(gates_full[train_keep].mean()),
        )
    elif sge_plan.append_gate_column:
        # 'drop' control arm: constant-1 gates (no gate ever at test);
        # only the training-time dropout ever zeroes the Fourier block.
        gates_full = np.ones(sub_x_full.shape[0], dtype=np.float32)

    if sge_plan.append_gate_column:
        gate_cols = torch.from_numpy(
            gates_full if gates_full.ndim == 2 else gates_full[:, None]
        )
        sub_x_full = torch.cat([sub_x_full, gate_cols], dim=1)

    sub_x = sub_x_full[train_keep]
    sub_y_raw = sub_y_raw_full[train_keep]
    sub_y_std = sub_y_std_full[train_keep]
    sub_regime = sub_regime_full[train_keep]
    n_smoke = len(sub_x)
    if hybrid_mode and sub_baseline_pred_full is not None and not holdout_active:
        # Holdout runs slice the baseline to the held-out rows later; the
        # K-fold path evaluates on the (post-carve) training rows instead.
        sub_baseline_pred = sub_baseline_pred_full[train_keep]
    LOG.info("Final training set: %d rows", n_smoke)

    class _SubDataset(torch.utils.data.Dataset):
        # Exposed so FoundationTrainer can build a RegimeBalancedSampler when
        # --regime-balanced-sampler is set (mirrors BoringDataset.regimes).
        regimes = sub_regime.numpy()

        def __len__(self) -> int:
            return n_smoke

        def __getitem__(self, idx: int) -> dict:
            return {
                "x": sub_x[idx],
                "y": sub_y_std[idx],
                "regime": sub_regime[idx],
            }

    sub_dataset = _SubDataset()

    # 3. Build the foundation model -----------------------------------------
    n_input = dataset.n_features
    # P-SGE / P-MS: the appended gate column(s) are extra encoder INPUT
    # columns but not extra features (ResMLPEncoder strips them and
    # multiplies the Fourier block) -- see EncoderSpec.sge_gate_input /
    # EncoderSpec.sge_gate_bands.
    if sge_plan.band_mode is not None:
        n_gate_cols = ENCODER_FOURIER_BANDS
    elif sge_plan.append_gate_column:
        n_gate_cols = 1
    else:
        n_gate_cols = 0
    n_encoder_input = n_input + n_gate_cols
    encoder = EncoderSpec(
        n_input=n_encoder_input,
        n_output=args.encoder_dim,
        n_layers=4,
        hidden=128,
        batchnorm=True,
        dropout=0.0,
        fourier_bands=ENCODER_FOURIER_BANDS,
        fourier_scale=ENCODER_FOURIER_SCALE,
        zero_fourier=args.zero_fourier,
        sge_gate_input=(
            sge_plan.append_gate_column and sge_plan.band_mode is None
        ),
        sge_gate_bands=(
            ENCODER_FOURIER_BANDS if sge_plan.band_mode is not None else 0
        ),
    )
    svgp = SVGPSpec(
        n_inducing=min(args.n_inducing, n_smoke),
        learn_inducing=True,
        whitened=True,
        inducing_init=args.inducing_init,
        kernel_type=args.kernel_type,
        mean_type=args.mean_type,
        likelihood_type=args.likelihood_type,
        studentt_deg_free=args.studentt_df,
        censored_cap=args.censored_cap,
        add_residual_geo=args.add_residual_geo,
    )
    from national.models.foundation import NoiseHeadSpec

    spec = FoundationSpec(
        encoder=encoder,
        svgp=svgp,
        regime_dim=8,
        depth_scale_m=30.0,
        noise_head=NoiseHeadSpec(enabled=args.heteroscedastic_noise),
    )
    # kmeans_pp_stratified requires regime codes so it can budget inducing
    # points per AIST regime. The other methods ignore the kwarg.
    inducing = init_inducing_points(
        sub_x,
        n_inducing=svgp.n_inducing,
        method=args.inducing_init,
        regime_codes=sub_regime if args.inducing_init == "kmeans_pp_stratified" else None,
    )
    model = FoundationModel(spec, inducing_points=inducing)
    model.set_target_stats(dataset.target_mean, dataset.target_std)

    # 4. Trainer config -----------------------------------------------------
    run_name = args.wandb_run_name or args.output_dir.name
    io_cfg: dict = {
        "checkpoint_root": str(args.output_dir / "checkpoints"),
        "run_root": str(args.output_dir),
    }
    if args.wandb:
        io_cfg["wandb"] = {
            "project": args.wandb_project,
            "mode": "online",
        }
    cfg = OmegaConf.create(
        {
            "training": {
                "lr": args.lr,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "warmup_steps": 20,
                "weight_decay": 1e-5,
                "num_workers": 0,
                "checkpoint_every_min": 9999,
                "mmd_weight": 0.0,
                "indep_lambda": args.indep_lambda,
                "log_indep_diagnostic": bool(args.log_indep_diagnostic),
                "adv_lambda": args.adv_lambda,
                # P-SGE training-only gate dropout (0.0 for base/zerofourier/
                # test_only; the resolved plan is authoritative, not the raw
                # CLI value).
                "gate_dropout": float(sge_plan.train_dropout),
                # EES snapshot battery (P-R3c..f): 1-indexed epochs at
                # whose end the trainer saves a FoundationModel-format
                # artifact to <checkpoint_root>/ep{N}.pt. [] disables.
                "snapshot_epochs": list(snapshot_epochs),
                "beta1": 0.9,
                "beta2": 0.999,
                "regime_balanced_sampler": args.regime_balanced_sampler,
                "regime_balance_alpha": args.regime_balance_alpha,
            },
            "run": {"seed": args.seed, "name": run_name},
            "io": io_cfg,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = FoundationTrainer(model=model, dataset=sub_dataset, cfg=cfg, device=args.device)
    LOG.info("Training start...")
    t_start = time.perf_counter()
    output = trainer.fit()
    t_train = time.perf_counter() - t_start
    LOG.info(
        "Trained %d epochs in %.1fs; final_loss=%.4f",
        args.n_epochs,
        t_train,
        output.final_loss,
    )

    # 5. Spatial K-fold RMSE on the subsample -------------------------------
    if holdout_active:
        # Proper hold-out: model was trained on (sub_x, sub_y_*) only;
        # evaluate predictions on the held-out rows from the original
        # full subsample (sub_x_full etc.) and report a single fold.
        eval_x = sub_x_full[holdout_test_idx]
        eval_y_raw = sub_y_raw_full[holdout_test_idx]
        # Align the per-row baseline prediction with the held-out test rows
        # so the hybrid inference path can re-add the CatBoost mean at
        # exactly the right rows. Without this slice the assertion in step
        # 6 below trips when sub_baseline_pred has length len(sub_x_full)
        # while pred_mean has length len(holdout_test_idx).
        if hybrid_mode and sub_baseline_pred is not None:
            sub_baseline_pred = sub_baseline_pred[holdout_test_idx]
        if args.leave_prefecture:
            eval_label = f"prefecture={args.leave_prefecture}"
        elif args.leave_region:
            eval_label = f"region={args.leave_region}"
        else:
            eval_label = (
                f"fold={args.kfold_test_fold}"
                + (f" buffered={args.buffer_meshes}" if args.buffer_meshes > 0 else "")
            )
        LOG.info("Hold-out evaluation set: %d rows (%s)", len(eval_x), eval_label)
        folds = [(np.arange(len(sub_x)), np.arange(len(eval_x)))]
    else:
        sub_df = pd.DataFrame(
            {
                "latitude_deg": sub_x[:, 0].numpy(),
                "longitude_deg": sub_x[:, 1].numpy(),
                "n_value": sub_y_raw.numpy(),
            }
        )
        if args.buffer_meshes > 0:
            from national.evaluation.spatial_kfold import spatial_kfold_split_buffered

            folds = spatial_kfold_split_buffered(
                sub_df, n_folds=3, mesh_level=2,
                buffer_meshes=args.buffer_meshes, seed=args.seed,
                base_split=args.fold_assignment,
            )
            LOG.info(
                "Buffered spatial K-fold (assignment=%s, buffer_meshes=%d): "
                "fold sizes %s",
                args.fold_assignment, args.buffer_meshes,
                [(len(tr), len(te)) for tr, te in folds],
            )
        elif args.fold_assignment == "contiguous":
            from national.evaluation.spatial_kfold import spatial_kfold_split_contiguous

            folds = spatial_kfold_split_contiguous(
                sub_df, n_folds=3, mesh_level=2, seed=args.seed,
            )
        else:
            folds = spatial_kfold_split(sub_df, n_folds=3, mesh_level=2, seed=args.seed)
        eval_x = sub_x
        eval_y_raw = sub_y_raw
    LOG.info("Computing posterior predictions for K-fold RMSE (batched)...")
    # Predict in batches so a 495k × 6k SVGP kernel matrix does not OOM
    # on CUDA. A 50k batch keeps peak memory < ~3 GB on the matern path.
    pred_batch = int(args.pred_batch)
    # P-SGE test_only arm: the model trained exactly like base; the
    # raw-space gate is applied ONLY here, at eval/predict time.
    eval_gates: np.ndarray | None = None
    if sge_plan.eval_gate_only and gates_full is not None:
        eval_gates = (
            gates_full[holdout_test_idx] if holdout_active else gates_full[train_keep]
        )
    pred_mean_trans, pred_std_trans = _predict_batched(
        model, eval_x, pred_batch, eval_gates
    )
    # Map the prediction back into the original N-value units so the
    # K-fold metrics are comparable across runs regardless of which
    # target transform was applied during training.
    from national.data.boring_dataset import invert_target_transform_moments

    pred_mean, pred_std = invert_target_transform_moments(
        pred_mean_trans.astype(np.float64),
        pred_std_trans.astype(np.float64),
        args.target_transform,
    )
    pred_mean = pred_mean.astype(np.float32)
    pred_std = pred_std.astype(np.float32)
    # Hybrid mode: pred_mean is the residual prediction in raw-N units
    # (because we forced target_transform='none'). Re-add the baseline
    # mean to recover the total prediction. pred_std stays as the GP
    # residual sigma — CatBoost contributes no variance.
    if hybrid_mode:
        if sub_baseline_pred is None or sub_baseline_pred.shape[0] != pred_mean.shape[0]:
            raise RuntimeError(
                "Hybrid inference: baseline-prediction slice does not align "
                f"with pred_mean (got {None if sub_baseline_pred is None else sub_baseline_pred.shape}"
                f" vs {pred_mean.shape})"
            )
        pred_mean_residual = pred_mean.copy()
        pred_mean = (pred_mean_residual + sub_baseline_pred).astype(np.float32)
        LOG.info(
            "Hybrid inference: residual prediction range [%.2f, %.2f]; "
            "baseline mean range [%.2f, %.2f]; combined mean range [%.2f, %.2f]",
            float(pred_mean_residual.min()), float(pred_mean_residual.max()),
            float(sub_baseline_pred.min()), float(sub_baseline_pred.max()),
            float(pred_mean.min()), float(pred_mean.max()),
        )
    y_true = eval_y_raw.numpy()
    fold_metrics = []
    for fi, (train_idx, test_idx) in enumerate(folds):
        rmse = float(np.sqrt(((pred_mean[test_idx] - y_true[test_idx]) ** 2).mean()))
        mae = float(np.abs(pred_mean[test_idx] - y_true[test_idx]).mean())
        std_mean = float(pred_std[test_idx].mean())
        fold_metrics.append(
            {"fold": fi, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)), "rmse": rmse, "mae": mae, "std_mean": std_mean}
        )
        LOG.info("fold %d: RMSE=%.2f MAE=%.2f mean_std=%.2f", fi, rmse, mae, std_mean)

    # ---- P-SGE: in-region metric (fixed 5% training-region holdout) --------
    # Evaluated for EVERY run alongside spatial_kfold[0] (the OOR metric),
    # under the same inverse-transform / hybrid re-add pipeline as the main
    # evaluation. test_only applies the raw-space gate here too (the gate is
    # part of the predictor at deployment, in- and out-of-region alike).
    inregion_rmse: float | None = None
    inregion_mae: float | None = None
    if len(inregion_idx) > 0:
        ir_x = sub_x_full[inregion_idx]
        ir_gates = (
            gates_full[inregion_idx]
            if (sge_plan.eval_gate_only and gates_full is not None)
            else None
        )
        ir_mean_trans, ir_std_trans = _predict_batched(model, ir_x, pred_batch, ir_gates)
        ir_mean, _ir_std = invert_target_transform_moments(
            ir_mean_trans.astype(np.float64),
            ir_std_trans.astype(np.float64),
            args.target_transform,
        )
        ir_mean = ir_mean.astype(np.float32)
        if hybrid_mode and sub_baseline_pred_full is not None:
            ir_mean = (ir_mean + sub_baseline_pred_full[inregion_idx]).astype(np.float32)
        ir_true = sub_y_raw_full[inregion_idx].numpy()
        inregion_rmse = float(np.sqrt(((ir_mean - ir_true) ** 2).mean()))
        inregion_mae = float(np.abs(ir_mean - ir_true).mean())
        LOG.info(
            "inregion holdout (n=%d): RMSE=%.2f MAE=%.2f",
            len(ir_true), inregion_rmse, inregion_mae,
        )

    # Per-regime breakdown of the smoke subset -- highlights whether the
    # AIST cache is actually resolving the regime column to anything other
    # than UNKNOWN. The summary stores both the absolute counts and the
    # fraction so it survives configuration changes.
    regime_codes = sub_regime.numpy()
    regime_counts: dict[str, int] = {}
    for code in regime_codes:
        regime_counts[str(int(code))] = regime_counts.get(str(int(code)), 0) + 1

    # P-SGE provenance: the fitted gate's pre-registered statistics plus the
    # realised gate distribution on train / in-region / held-out rows.
    sge_gate_stats = None
    if msge_model is not None and gates_full is not None:
        # MS-SGE (P-MS): per-band lambda list + per-band realised gate means
        # on train / in-region / held-out rows (the *_per_band keys), plus
        # the all-band aggregates under the round-1 key names for
        # cross-family harvest continuity.
        sge_gate_stats = msge_model.stats()
        sge_gate_stats.update(
            {
                "gate_train_mean_per_band": [
                    float(v) for v in gates_full[train_keep].mean(axis=0)
                ],
                "gate_inregion_mean_per_band": (
                    [float(v) for v in gates_full[inregion_idx].mean(axis=0)]
                    if len(inregion_idx) else None
                ),
                "gate_heldout_mean_per_band": (
                    [float(v) for v in gates_full[holdout_test_idx].mean(axis=0)]
                    if holdout_active else None
                ),
                "gate_train_mean": float(gates_full[train_keep].mean()),
                "gate_inregion_mean": (
                    float(gates_full[inregion_idx].mean())
                    if len(inregion_idx) else None
                ),
                "gate_heldout_mean": (
                    float(gates_full[holdout_test_idx].mean())
                    if holdout_active else None
                ),
                "gate_min": float(gates_full.min()),
                "gate_max": float(gates_full.max()),
                "applied": "train+eval per-band columns",
            }
        )
    elif gate_model is not None and gates_full is not None:
        sge_gate_stats = gate_model.stats()
        sge_gate_stats.update(
            {
                "gate_train_mean": float(gates_full[train_keep].mean()),
                "gate_train_median": float(np.median(gates_full[train_keep])),
                "gate_inregion_mean": (
                    float(gates_full[inregion_idx].mean())
                    if len(inregion_idx) else None
                ),
                "gate_heldout_mean": (
                    float(gates_full[holdout_test_idx].mean())
                    if holdout_active else None
                ),
                "gate_min": float(gates_full.min()),
                "gate_max": float(gates_full.max()),
                "applied": (
                    "train+eval column" if sge_plan.append_gate_column else "eval_only"
                ),
            }
        )

    summary = {
        "run_name": "kanto_smoke",
        "n_smoke": n_smoke,
        "n_features": n_input,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "n_inducing": svgp.n_inducing,
        "final_loss": output.final_loss,
        "training_time_seconds": t_train,
        "target_mean": dataset.target_mean,
        "target_std": dataset.target_std,
        "target_transform": args.target_transform,
        # ---- scaler provenance (--target-stats; keys absent pre 2026-07-13,
        # harvest code must .get() them). target_mean/target_std above are
        # always the FITTED scaler (== full-parquet under legacy). The mean
        # gap (fitted - full-parquet, in transformed-target space) is the
        # size of the leak this fold: 0.0 under legacy by construction.
        "target_stats_mode": args.target_stats,
        "target_stats_full_parquet_mean": target_stats_full_mean,
        "target_stats_full_parquet_std": target_stats_full_std,
        "target_stats_mean_gap": float(dataset.target_mean - target_stats_full_mean),
        "target_stats_n_fit_rows": (
            int(len(train_keep)) if args.target_stats == "train_only"
            else int(n_total)
        ),
        "regime_one_hot": args.regime_one_hot,
        "aist_granular_one_hot": bool(args.aist_granular_one_hot),
        "heteroscedastic_noise": args.heteroscedastic_noise,
        "kernel_type": args.kernel_type,
        "mean_type": args.mean_type,
        "inducing_init": args.inducing_init,
        "likelihood_type": args.likelihood_type,
        "studentt_df_init": args.studentt_df,
        "regime_balanced_sampler": args.regime_balanced_sampler,
        "regime_balance_alpha": args.regime_balance_alpha,
        "encoder_dim": args.encoder_dim,
        "spatial_kfold": fold_metrics,
        "regime_distribution": regime_counts,
        "device": args.device,
        "hybrid_mode": hybrid_mode,
        "baseline_name": args.baseline_name if hybrid_mode else None,
        "baseline_pred_train": str(args.baseline_pred_train) if hybrid_mode else None,
        "baseline_pred_test": str(args.baseline_pred_test) if hybrid_mode else None,
        "feature_columns": list(feature_cols),
        "zero_fourier": bool(args.zero_fourier),
        "add_residual_geo": bool(args.add_residual_geo),
        "fold_assignment": args.fold_assignment,
        # Hold-out provenance: which region was held out and (for categorical
        # non-JP archives) which parquet column defined the fold. None for
        # plain K-fold runs. Historic summaries (pre 2026-07-11) lack these
        # keys — harvest code must .get() them.
        "leave_region": args.leave_region or None,
        "region_column": args.region_column or None,
        "indep_lambda": args.indep_lambda,
        "log_indep_diagnostic": bool(args.log_indep_diagnostic),
        # Per-epoch [{epoch, hsic_raw_mean, task_loss_mean}, ...] -- empty
        # unless --log-indep-diagnostic was passed. See
        # FoundationTrainer.fit()'s log_indep_diagnostic block.
        "indep_diagnostic": list(output.state.indep_diagnostic),
        "adv_lambda": args.adv_lambda,
        # ---- P-SGE (docs/research/2026-07-12_sge_preregistration.md) ----
        # sge_gate: off | raw | fourier | test_only. gate_dropout: the raw
        # CLI value (mode off + p>0 == the 'drop' control arm).
        # sge_gate_stats: t, s, train-DI quantiles, dbar, threshold method,
        # and the realised mean gate on train / in-region / held-out rows
        # (None for ungated arms). inregion_*: the fixed 5% training-region
        # holdout metric (seed 777, disjoint from training), in EVERY
        # summary regardless of arm.
        "sge_gate": args.sge_gate,
        "gate_dropout": float(args.gate_dropout),
        "sge_gate_stats": sge_gate_stats,
        "inregion_rmse": inregion_rmse,
        "inregion_mae": inregion_mae,
        "inregion_n": int(len(inregion_idx)),
        "inregion_holdout_seed": INREGION_SEED,
        "inregion_fraction": INREGION_FRACTION,
        # EES snapshot battery (P-R3c..f): which 1-indexed epochs were
        # snapshotted to checkpoints/ep{N}.pt (None when the feature was
        # off). Historic summaries (pre round-3) lack this key — harvest
        # code must .get() it.
        "snapshot_epochs": snapshot_epochs or None,
    }
    # Per-epoch [{epoch, critic_mse_mean, critic_r2, task_loss_mean}, ...]
    # -- only written when non-empty (i.e. --adv-lambda > 0.0), mirroring
    # how --baseline-name etc. are only written in hybrid mode above. See
    # national.training.adversarial and FoundationTrainer.fit()'s
    # adversarial block.
    if output.state.adv_diagnostic:
        summary["adv_diagnostic"] = list(output.state.adv_diagnostic)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOG.info("Wrote %s", summary_path)

    # Persist per-row predictions for downstream Phase-2 analyses
    # (hybrid conformal recalibration, Mondrian conformal subgroup tables,
    # locally-weighted conformal, threshold-classifier comparison). The
    # smoke trainer's K-fold loop already populated `pred_mean` and
    # `pred_std` over `eval_x` rows.
    np.savez(
        args.output_dir / "predictions.npz",
        pred_mean=pred_mean,
        pred_std=pred_std,
        y_true=y_true,
        regime=sub_regime.numpy(),
        baseline_pred=(sub_baseline_pred if hybrid_mode else np.zeros_like(pred_mean)),
        hybrid_mode=np.array([1 if hybrid_mode else 0], dtype=np.int32),
    )
    LOG.info("Wrote %s", args.output_dir / "predictions.npz")

    # 6. Save the foundation artifact ---------------------------------------
    artifact_path = args.output_dir / "foundation_model.pt"
    model.save(artifact_path)
    LOG.info("Saved foundation artifact to %s", artifact_path)

    # 7. Auto-generate the diagnostic plot so every training run leaves a
    # visualisable trail. Done in a subprocess to keep the heavy
    # matplotlib import out of the training memory footprint.
    # P-SGE: gate-column arms are skipped -- the visualizer rebuilds its own
    # dataset without the appended gate column, so the reload would fail on
    # the input-width mismatch anyway (summary.json + predictions.npz carry
    # the run's full metric trail).
    if sge_plan.append_gate_column:
        LOG.info("Auto-visualisation skipped (SGE gate column changes the input width).")
        return 0
    try:
        import subprocess

        cmd = [
            sys.executable,
            "-m",
            "scripts.visualize_results",
            "--run-dir",
            str(args.output_dir),
            # Must mirror the training parquet, otherwise the auto-viz reads
            # its own Kanto default and the diagnostics show Kanto-bbox eval
            # of a model trained on a larger dataset (seen on the first
            # national runs: diagnostics RMSE 6.20 vs national K-fold 7.54).
            "--parquet",
            str(args.parquet),
            "--train-fraction",
            str(args.train_fraction),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        # The visualizer needs to know if the dataset was loaded with
        # regime one-hot so its own subsample matches the training one.
        if args.regime_one_hot:
            cmd.append("--regime-one-hot")
        if args.aist_granular_one_hot:
            cmd.append("--aist-granular-one-hot")
        subprocess.run(cmd, check=False)
    except Exception as exc:  # noqa: BLE001 -- viz failure shouldn't break training
        LOG.warning("Auto-visualisation failed (non-fatal): %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
