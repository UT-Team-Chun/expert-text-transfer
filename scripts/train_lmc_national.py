#!/usr/bin/env python
"""Train an LMC multi-task SVGP on the national v4 parquet (Paper B' Pillar 2).

Joint prediction of SPT N-value + groundwater depth via the
:class:`national.models.lmc.LMCModel` head sharing the existing DKL
ResMLP encoder. Mirrors :mod:`scripts.train_kanto_smoke` at a smaller
scope -- single CLI, single run, NaN-masked log-likelihood, posterior
saved alongside the model. Compatible with the existing utens / NFS
output layout (``/mnt/nas/runs/<label>``).

Per-row data layout (from the v4 parquet):

  features = [lat, lon, depth, abs_elev, river_km, coast_km] + regime one-hot
  task_0   = standardized n_value                                  (always observed)
  task_1   = standardized groundwater_depth_m  (NaN ~19% -> mask off)

Targets are standardized per-task independently using observed-only
statistics; the saved JSON carries the means / stds so inverse transform
at inference is deterministic.

Example::

    cd backend
    uv run python -m scripts.train_lmc_national \\
        --parquet ../data/features/borings_japan_v4.parquet \\
        --output-dir ../data/runs/lmc_v4_trial \\
        --n-inducing 8000 --n-epochs 50 --batch-size 4096 \\
        --device cuda --kernel-type rbf --mean-type linear
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

LOG = logging.getLogger("scripts.train_lmc_national")


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    likelihood: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    encoder_spec_dict: dict,
    lmc_spec_dict: dict,
    epoch: int,
    step: int,
    train_keep_idx: np.ndarray,
    holdout_test_idx: np.ndarray | None,
    mu0: float,
    s0: float,
    mu1: float,
    s1: float,
    seed: int,
    n_input: int,
    history: list[dict],
) -> Path:
    """Atomically save a per-epoch checkpoint.

    Writes to ``<path>.tmp`` first then ``os.replace`` to the final name so
    a partial write (NFS preemption, OOM) never leaves a corrupt checkpoint
    file in place. Mirrors the structure of
    :func:`national.training.trainer.SVGPTrainer._save_checkpoint` but
    extended with the K-fold split indices + target-standardization stats
    so the LMC trainer can resume mid-training without recomputing the
    spatial split (which depends on the seed and parquet content).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload: dict = {
        "version": 1,
        "model": model.state_dict(),
        "likelihood": likelihood.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "encoder_spec": dict(encoder_spec_dict),
        "lmc_spec": dict(lmc_spec_dict),
        "epoch": int(epoch),
        "step": int(step),
        "train_keep_idx": np.asarray(train_keep_idx, dtype=np.int64),
        "holdout_test_idx": (
            None if holdout_test_idx is None
            else np.asarray(holdout_test_idx, dtype=np.int64)
        ),
        "mu0": float(mu0),
        "s0": float(s0),
        "mu1": float(mu1),
        "s1": float(s1),
        "seed": int(seed),
        "n_input": int(n_input),
        "history": list(history),
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        payload["rng_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(payload, tmp)
    os.replace(tmp, path)
    LOG.info("Saved checkpoint to %s (epoch=%d step=%d)", path, epoch, step)
    return path


def _prune_old_checkpoints(directory: Path, keep_last: int = 3) -> list[Path]:
    """Rolling deletion of per-epoch checkpoints in ``directory``.

    The pod-local SSD that backs ``/tmp`` on Azure spot VMs is small
    (typically 32-128 GB) and an LMC checkpoint can be hundreds of MB,
    so we keep only the last ``keep_last`` ``epoch_<N>.pt`` files. The
    final-flush step copies whichever is latest to durable NFS, so
    losing the older /tmp copies is safe by design.

    Returns the list of files that were deleted (empty when nothing
    needed pruning). Silent on PermissionError / FileNotFoundError --
    the next epoch's write will overwrite the stale entry anyway.
    """
    if not directory.exists():
        return []
    candidates = sorted(
        directory.glob("epoch_*.pt"),
        key=lambda p: int(p.stem.split("_", 1)[1]),
    )
    if len(candidates) <= keep_last:
        return []
    deleted: list[Path] = []
    for p in candidates[:-keep_last]:
        try:
            p.unlink()
            deleted.append(p)
        except (PermissionError, FileNotFoundError):
            continue
    return deleted


def _flush_to_durable(src: Path, dst_dir: Path) -> Path:
    """Copy the latest ``/tmp`` checkpoint to the durable NFS directory.

    Cross-filesystem ``os.rename`` is not portable on POSIX (raises
    ``OSError: [Errno 18] Invalid cross-device link``); we use
    ``shutil.copy2`` which preserves mtimes and works across mount
    points. The destination directory is created if missing so the
    NFS path can be lazy / mounted-on-demand.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    LOG.info("Flushed checkpoint %s -> %s", src, dst)
    return dst


def _load_checkpoint(path: Path) -> dict:
    """Load a checkpoint emitted by :func:`_save_checkpoint`.

    Raises ``FileNotFoundError`` with a clear message if the file is
    missing, and ``RuntimeError`` with a clear message if torch.load
    fails (e.g. truncated checkpoint from a mid-write crash). The
    SystemExit codepath in :func:`main` catches both and routes them to
    ``parser.error``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"--resume-from-checkpoint: file does not exist: {path}"
        )
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"--resume-from-checkpoint: failed to load checkpoint {path}: {exc}. "
            "The file may be truncated (mid-write crash) or from an "
            "incompatible torch version."
        ) from exc


def _build_features(df: pd.DataFrame, regime_dim: int = 8) -> np.ndarray:
    """Assemble the per-row encoder input tensor in the same column order
    as the single-task path: ``[lat, lon, depth, abs_elev, river_km,
    coast_km, regime_one_hot_0..regime_dim-1, (embed_0..embed_K-1, has_text)]``.

    The text embedding columns are optional. When the parquet contains an
    ``embed_*`` block (the v5 schema, output of
    :mod:`scripts.join_soil_text_to_parquet`) the columns are appended
    after the regime one-hot together with a ``has_text`` indicator
    (``1.0`` for rows that matched a layer, ``0.0`` for the ~42% of v5
    rows where the boring carries no observation_text). Unmatched rows
    have ``embed_*`` filled with zero so the encoder sees a deterministic
    "no signal" input rather than NaN; the ``has_text`` flag lets the
    model learn to discount the zero embedding for those rows.

    Returns a (N, 6 + regime_dim [+ K + 1]) float32 array.
    """
    cols_cont = [
        "latitude_deg",
        "longitude_deg",
        "depth_from_surface",
        "absolute_elevation",
        "river_distance_km",
        "coast_distance_km",
    ]
    missing = [c for c in cols_cont if c not in df.columns]
    if missing:
        raise KeyError(
            f"parquet missing required columns for LMC training: {missing}"
        )
    cont = np.stack([df[c].to_numpy(dtype=np.float32) for c in cols_cont], axis=1)
    if "regime_code" in df.columns:
        regime = df["regime_code"].to_numpy(dtype=np.int64)
    else:
        regime = np.full((len(df),), 7, dtype=np.int64)  # UNKNOWN
    clipped = np.clip(regime, 0, regime_dim - 1)
    one_hot = np.eye(regime_dim, dtype=np.float32)[clipped]

    embed_cols = sorted(
        (c for c in df.columns if c.startswith("embed_")),
        key=lambda c: int(c.split("_", 1)[1]),
    )
    if not embed_cols:
        return np.concatenate([cont, one_hot], axis=1)

    LOG.info(
        "v5 schema detected: appending %d text-embedding columns + has_text flag",
        len(embed_cols),
    )
    emb = np.stack(
        [df[c].to_numpy(dtype=np.float32) for c in embed_cols], axis=1
    )
    has_text = (~np.isnan(emb[:, 0])).astype(np.float32)
    emb = np.nan_to_num(emb, nan=0.0)
    LOG.info(
        "  text-embedding coverage: %d / %d rows (%.1f%%)",
        int(has_text.sum()),
        len(has_text),
        100.0 * has_text.mean(),
    )
    return np.concatenate([cont, one_hot, emb, has_text[:, None]], axis=1)


def _standardize(
    y: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Zero-mean unit-std using observed entries only. Returns
    (standardized values, mean, std). Masked entries are zeroed in the
    output (an arbitrary filler -- the mask drops them from the ELBO)."""
    obs = y[mask]
    mean = float(obs.mean())
    std = float(obs.std()) + 1e-6
    out = np.zeros_like(y, dtype=np.float32)
    out[mask] = ((obs - mean) / std).astype(np.float32)
    return out, mean, std


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-inducing", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--kernel-type",
        choices=("rbf", "matern52", "matern32", "matern12"),
        default="rbf",
    )
    parser.add_argument(
        "--mean-type", choices=("constant", "linear"), default="linear"
    )
    parser.add_argument(
        "--num-latents",
        type=int,
        default=2,
        help="LMC latent-GP count. <= num_tasks; lower = stronger coupling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for inducing-point + batch shuffling + torch RNG.",
    )
    parser.add_argument(
        "--kfold-test-fold",
        type=int,
        default=-1,
        help="Spatial K-fold (n_folds=3, secondary-mesh-keyed) test fold. "
             "When >= 0, the matching fold's rows are held out of training "
             "and a honest held-out RMSE / MAE is reported in summary.json "
             "next to the in-sample train-fit numbers. Use 0 / 1 / 2 to pick "
             "one of three disjoint test folds. Default -1 = train on all "
             "rows (the legacy in-sample-only path, kept for backwards "
             "compatibility with the v2 / v3 / v4 baselines reported before "
             "this flag landed).",
    )
    parser.add_argument(
        "--kfold-mesh-level",
        type=int,
        default=2,
        help="Secondary-mesh level for the K-fold key assignment. Default 2 "
             "(~10 km^2 cells) matches the Paper 1 / Paper B convention.",
    )
    parser.add_argument(
        "--leave-region",
        type=str,
        default="",
        help="If non-empty, switch from spatial K-fold to leave-region-out "
             "(LRO) evaluation: hold out ALL rows whose (lat, lon) fall inside "
             "the named region's bounding box and train on the rest. One of: "
             "hokkaido, tohoku, kanto, chubu, kansai, chugoku, shikoku, "
             "kyushu_okinawa "
             "(national.evaluation.leave_region_out.DEFAULT_REGIONS). "
             "Mutually exclusive with --kfold-test-fold. Used for the "
             "cross-region transferability test of the per-layer text channel.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for the *durable* (NFS) flush of the final "
             "checkpoint. Default is ``<output-dir>/checkpoints``. "
             "Only the latest ``epoch_<N>.pt`` is copied here at the "
             "end of training (plus the final model state). Per-epoch "
             "hot-path writes go to --checkpoint-tmp-dir instead so "
             "the NFS link does not bottleneck training.",
    )
    parser.add_argument(
        "--checkpoint-tmp-dir",
        type=Path,
        default=Path("/tmp/dkl_checkpoints"),
        help="Pod-local SSD directory for per-epoch checkpoint writes. "
             "Default ``/tmp/dkl_checkpoints``. The trainer writes "
             "``<tmp>/<run_name>/epoch_<N>.pt`` on each epoch boundary "
             "(rolling-window of the most recent 3 files) and copies "
             "the latest to ``--checkpoint-dir`` (NFS) at the end of "
             "training. This swaps ~3 GB/s local-SSD writes for the "
             "~300 MB/s NFS link, giving ~10x save speedup on Azure "
             "tailscale spot VMs without losing the durable resume "
             "contract.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Path to a checkpoint emitted by a previous run. When set, "
             "the model / optimizer / scheduler / RNG / K-fold split / "
             "epoch + step counters are restored and training continues "
             "from epoch+1. Errors out if the file does not exist or has "
             "incompatible metadata (n_input, num_tasks, num_latents, seed).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases run logging (requires WANDB_API_KEY).",
    )
    parser.add_argument(
        "--wandb-project",
        default="geo-paperB-national",
        help="W&B project. Used only when --wandb is set.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="W&B run name. Defaults to output-dir basename.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.leave_region and args.kfold_test_fold >= 0:
        parser.error(
            "--leave-region and --kfold-test-fold are mutually exclusive: "
            "pick spatial K-fold OR leave-region-out, not both."
        )

    # ---- Resume bookkeeping (fail-fast before any heavy lifting) -----------
    # The K-fold indices, target-standardization stats, and the epoch / step
    # counters live in the checkpoint -- we need them in scope before the
    # data preparation block below picks up where the previous run left off.
    checkpoint_payload: dict | None = None
    resume_epoch: int = 0
    resume_step: int = 0
    if args.resume_from_checkpoint is not None:
        try:
            checkpoint_payload = _load_checkpoint(args.resume_from_checkpoint)
        except FileNotFoundError as exc:
            parser.error(str(exc))
        except RuntimeError as exc:
            parser.error(str(exc))
        # The checkpoint stores ``epoch`` as the index of the epoch that
        # finished writing the file; resume picks up at epoch+1.
        resume_epoch = int(checkpoint_payload.get("epoch", -1)) + 1
        resume_step = int(checkpoint_payload.get("step", 0))
        LOG.info(
            "Resuming from checkpoint %s: starting epoch=%d step=%d",
            args.resume_from_checkpoint, resume_epoch, resume_step,
        )
        # Soft check on seed: warn rather than error since the seed
        # affects only the K-fold split (which we restore directly from
        # the checkpoint anyway). The hard checks on n_input / num_tasks
        # / num_latents land further down once we know the parquet shape.
        ckpt_seed = int(checkpoint_payload.get("seed", args.seed))
        if ckpt_seed != args.seed:
            LOG.warning(
                "Checkpoint seed=%d differs from CLI seed=%d; trusting "
                "checkpoint for K-fold split / RNG state.",
                ckpt_seed, args.seed,
            )

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.output_dir.name,
            config={
                "stage": "pillar2_lmc_joint",
                "n_inducing": args.n_inducing,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "kernel_type": args.kernel_type,
                "mean_type": args.mean_type,
                "num_latents": args.num_latents,
                "device": args.device,
                "parquet": str(args.parquet),
                "output_dir": str(args.output_dir),
                "seed": args.seed,
            },
        )
        LOG.info("W&B run: %s", wandb_run.url)

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    LOG.info("Using device=%s", args.device)
    if args.device == "mps":
        # Same MPS dtype workaround as the single-task path.
        gpytorch.settings._linalg_dtype_cholesky._global_value = torch.float32
        gpytorch.settings._linalg_dtype_symeig._global_value = torch.float32

    # NotPSDError defence-in-depth for LMC. The per-latent variational
    # batch creates a stack of (M, M) inducing covariance matrices whose
    # collective conditioning is significantly worse than the single-task
    # case (each latent has independent kernel hyperparameters). The
    # combination of lengthscale=3.0 init in lmc.py + the larger jitter
    # + more max_tries below gives the cholesky decomposition a wider
    # numerical margin without changing the trained-model semantics.
    # First failed cluster run (cb1: m8k, m12k_l1, both NotPSDError after
    # default 1e-6 jitter exhausted) drove the choice of these values.
    gpytorch.settings.cholesky_jitter._set_value(
        float_value=1e-4, double_value=1e-6, half_value=1e-3
    )
    gpytorch.settings.cholesky_max_tries._global_value = 6

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Data --------------------------------------------------------------
    LOG.info("Reading %s", args.parquet)
    df = pd.read_parquet(args.parquet)
    LOG.info("Input rows=%d, cols=%s", len(df), list(df.columns))
    if "groundwater_depth_m" not in df.columns:
        parser.error(
            "Parquet has no `groundwater_depth_m` column. Run "
            "scripts.add_aist_granular_to_parquet with --groundwater-csv first."
        )

    x = _build_features(df)
    # Task 0: SPT N (always observed in the v4 schema after enrich filter).
    y0 = df["n_value"].to_numpy(dtype=np.float32)
    m0 = np.isfinite(y0)
    # Task 1: groundwater (NaN for ~19% of rows -> mask out).
    y1 = df["groundwater_depth_m"].to_numpy(dtype=np.float32)
    m1 = np.isfinite(y1)

    y0_std, mu0, s0 = _standardize(y0, m0)
    y1_std, mu1, s1 = _standardize(y1, m1)
    if checkpoint_payload is not None:
        # The standardization stats are pinned to the parquet content at
        # the time the checkpoint was written. Override the fresh values
        # with the saved ones (and re-standardize y0 / y1 against them)
        # so the model's predictions un-standardize back to the exact
        # same target distribution -- the model itself was trained
        # against the saved (mu, s), and the optimizer / scheduler state
        # we are about to restore is calibrated to the same scale.
        mu0 = float(checkpoint_payload.get("mu0", mu0))
        s0 = float(checkpoint_payload.get("s0", s0))
        mu1 = float(checkpoint_payload.get("mu1", mu1))
        s1 = float(checkpoint_payload.get("s1", s1))
        y0_std = np.zeros_like(y0, dtype=np.float32)
        y0_std[m0] = ((y0[m0] - mu0) / max(s0, 1e-6)).astype(np.float32)
        y1_std = np.zeros_like(y1, dtype=np.float32)
        y1_std[m1] = ((y1[m1] - mu1) / max(s1, 1e-6)).astype(np.float32)
    LOG.info(
        "Task 0 (N): mean=%.3f std=%.3f, %d / %d observed (%.1f%%)",
        mu0, s0, m0.sum(), len(m0), 100.0 * m0.mean(),
    )
    LOG.info(
        "Task 1 (gw): mean=%.3f std=%.3f, %d / %d observed (%.1f%%)",
        mu1, s1, m1.sum(), len(m1), 100.0 * m1.mean(),
    )

    y_stack = np.stack([y0_std, y1_std], axis=1)
    mask_stack = np.stack([m0, m1], axis=1)
    n, n_input = x.shape
    LOG.info("Feature matrix: (%d, %d)", n, n_input)

    # ---- Optional spatial K-fold held-out split ---------------------------
    # The default (--kfold-test-fold=-1) trains on every row and reports
    # only an in-sample train-fit RMSE -- useful for the v3 / v4 / early-v5
    # baselines but not honest. With --kfold-test-fold in {0, 1, 2} we
    # carve off the chosen secondary-mesh-keyed fold from training, train
    # on the remaining ~2/3, and evaluate on the held-out fold to land a
    # genuinely cross-mesh-validated RMSE alongside the train fit.
    holdout_test_idx: np.ndarray | None = None
    train_keep_idx: np.ndarray = np.arange(n)
    if checkpoint_payload is not None and "train_keep_idx" in checkpoint_payload:
        # Restore the K-fold split from the checkpoint rather than
        # recomputing it. This makes the resume bit-deterministic even if
        # ``spatial_kfold_split`` later changes its tie-breaking rule, and
        # it lets the trainer resume on a parquet that was filtered /
        # re-ordered between runs (we trust the checkpoint's view of the
        # split).
        train_keep_idx = np.asarray(
            checkpoint_payload["train_keep_idx"], dtype=np.int64
        )
        if checkpoint_payload.get("holdout_test_idx") is not None:
            holdout_test_idx = np.asarray(
                checkpoint_payload["holdout_test_idx"], dtype=np.int64
            )
        LOG.info(
            "Restored K-fold split from checkpoint: train=%d held-out=%d",
            len(train_keep_idx),
            0 if holdout_test_idx is None else len(holdout_test_idx),
        )
    elif args.kfold_test_fold >= 0:
        from national.evaluation.spatial_kfold import spatial_kfold_split

        split_df = pd.DataFrame({
            "latitude_deg": df["latitude_deg"].to_numpy(),
            "longitude_deg": df["longitude_deg"].to_numpy(),
            "n_value": y0,
        })
        fold_splits = spatial_kfold_split(
            split_df, n_folds=3, mesh_level=args.kfold_mesh_level, seed=args.seed,
        )
        if args.kfold_test_fold >= len(fold_splits):
            parser.error(
                f"--kfold-test-fold={args.kfold_test_fold} out of range "
                f"for n_folds={len(fold_splits)}"
            )
        train_keep_idx, holdout_test_idx = fold_splits[args.kfold_test_fold]
        LOG.info(
            "Spatial K-fold: train_keep=%d (%.1f%%), held-out=%d (%.1f%%) "
            "rows; n_folds=3 mesh_level=%d seed=%d",
            len(train_keep_idx),
            100.0 * len(train_keep_idx) / max(1, n),
            len(holdout_test_idx),
            100.0 * len(holdout_test_idx) / max(1, n),
            args.kfold_mesh_level,
            args.seed,
        )
    elif args.leave_region:
        from national.evaluation.leave_region_out import (
            DEFAULT_REGIONS,
            leave_region_out_split,
        )

        if args.leave_region not in DEFAULT_REGIONS:
            parser.error(
                f"Unknown --leave-region {args.leave_region!r}; "
                f"choices: {sorted(DEFAULT_REGIONS)}"
            )
        # leave_region_out_split reads latitude_deg / longitude_deg and yields
        # POSITIONAL indices into the frame; a thin lat/lon frame keeps the
        # indices aligned with the x / y_stack rows built from ``df`` order.
        split_df = pd.DataFrame({
            "latitude_deg": df["latitude_deg"].to_numpy(),
            "longitude_deg": df["longitude_deg"].to_numpy(),
        })
        chosen = {args.leave_region: DEFAULT_REGIONS[args.leave_region]}
        split_iter = list(leave_region_out_split(split_df, regions=chosen))
        if not split_iter:
            parser.error(
                f"leave_region_out_split produced no fold for region "
                f"{args.leave_region!r}; check the bbox vs the data extent."
            )
        _, train_keep_idx, holdout_test_idx = split_iter[0]
        LOG.info(
            "Leave-region-out (held-out=%s): train_keep=%d (%.1f%%), "
            "held-out=%d (%.1f%%) rows",
            args.leave_region,
            len(train_keep_idx),
            100.0 * len(train_keep_idx) / max(1, n),
            len(holdout_test_idx),
            100.0 * len(holdout_test_idx) / max(1, n),
        )

    # Restrict the training side. Prediction below still runs on all n
    # rows so the held-out evaluation lives at the same scale as the
    # train fit and predictions.npz keeps a complete row index for
    # any downstream conformal / Mondrian recalibration that wants it.
    if holdout_test_idx is not None:
        x_train = x[train_keep_idx]
        y_stack_train = y_stack[train_keep_idx]
        mask_stack_train = mask_stack[train_keep_idx]
    else:
        x_train = x
        y_stack_train = y_stack
        mask_stack_train = mask_stack
    n_train = len(x_train)

    # ---- Model -------------------------------------------------------------
    # Late import to avoid pulling the heavy FoundationModel deps unless we
    # actually train.
    from national.models.foundation import EncoderSpec, ResMLPEncoder
    from national.models.lmc import LMCModel, LMCSpec, masked_multitask_log_prob

    encoder_spec = EncoderSpec(
        n_input=n_input,
        n_output=24,  # match the single-task hero config
        n_layers=4,
        hidden=128,
        fourier_bands=12,
    )
    encoder = ResMLPEncoder(encoder_spec)

    rng = np.random.default_rng(args.seed)
    # Inducing points sampled ONLY from the training side of the K-fold,
    # otherwise the held-out fold's spatial bins would leak into the
    # inducing-point set and the held-out RMSE would be optimistic.
    inducing_idx = rng.choice(
        n_train, size=min(args.n_inducing, n_train), replace=False
    )
    inducing = torch.from_numpy(x_train[inducing_idx]).float()

    lmc_spec = LMCSpec(
        num_tasks=2,
        num_latents=args.num_latents,
        kernel_type=args.kernel_type,
        mean_type=args.mean_type,
    )
    model = LMCModel(inducing, encoder, lmc_spec).to(args.device)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=2,
    ).to(args.device)

    optim = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=args.lr,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.n_epochs * max(1, n // args.batch_size)
    )

    ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_stack_train),
        torch.from_numpy(mask_stack_train.astype(np.bool_)),
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # ---- Checkpoint plumbing ----------------------------------------------
    # Two-tier layout (NFS write bandwidth is the bottleneck on Azure spot
    # VMs with tailscale-mounted NAS):
    #
    # * ``checkpoint_tmp_dir`` -- pod-local SSD (e.g. ``/tmp/dkl_checkpoints``).
    #   Per-epoch saves go here. Fast (~3 GB/s). Rolling window of the most
    #   recent 3 files keeps the ephemeral disk from filling up.
    # * ``checkpoint_dir`` -- durable NFS (e.g. ``<output-dir>/checkpoints``).
    #   We copy the *latest* per-epoch checkpoint here once at the end of
    #   training, so spot preemption between the final epoch and the
    #   end-of-training flush still leaves a resumable artifact on the
    #   shared filesystem (along with ``lmc_model.pt`` / ``foundation_model.pt``).
    #
    # The legacy single-tier behaviour (per-epoch writes straight to NFS)
    # can still be approximated by passing ``--checkpoint-tmp-dir`` equal
    # to ``--checkpoint-dir`` -- the trainer still copies the latest file
    # at the end, but the source and destination resolve to the same path
    # and ``shutil.copy2`` is a no-op.
    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else args.output_dir / "checkpoints"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Per-epoch hot-path -- namespace by run so concurrent runs on the
    # same pod do not stomp on each other's epoch_<N>.pt files.
    checkpoint_tmp_dir = args.checkpoint_tmp_dir / args.output_dir.name
    checkpoint_tmp_dir.mkdir(parents=True, exist_ok=True)
    LOG.info(
        "Checkpoint layout: tmp=%s (per-epoch, rolling-3) durable=%s "
        "(final flush)",
        checkpoint_tmp_dir, checkpoint_dir,
    )

    # ---- Restore from checkpoint (if requested) ---------------------------
    # All heavy-state objects are now constructed; load the state dicts
    # and RNG so the next training iteration starts exactly where the
    # previous run left off. Mismatched architectures (n_input, num_tasks,
    # num_latents) bail out here rather than silently producing garbage.
    history: list[dict] = []
    if checkpoint_payload is not None:
        ckpt_n_input = int(
            checkpoint_payload.get("n_input", n_input)
        )
        ckpt_lmc_spec = checkpoint_payload.get("lmc_spec", {})
        if ckpt_n_input != n_input:
            parser.error(
                f"--resume-from-checkpoint: n_input mismatch "
                f"(checkpoint={ckpt_n_input}, current={n_input}). "
                "The parquet schema or regime_dim changed between runs."
            )
        if int(ckpt_lmc_spec.get("num_tasks", lmc_spec.num_tasks)) != lmc_spec.num_tasks:
            parser.error(
                "--resume-from-checkpoint: num_tasks mismatch between "
                "checkpoint and current run; refusing to load."
            )
        if int(ckpt_lmc_spec.get("num_latents", lmc_spec.num_latents)) != lmc_spec.num_latents:
            parser.error(
                "--resume-from-checkpoint: num_latents mismatch "
                f"(checkpoint={ckpt_lmc_spec.get('num_latents')}, "
                f"current={lmc_spec.num_latents})."
            )
        model.load_state_dict(checkpoint_payload["model"])
        likelihood.load_state_dict(checkpoint_payload["likelihood"])
        if "optimizer_state_dict" in checkpoint_payload:
            optim.load_state_dict(checkpoint_payload["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint_payload:
            # Save the freshly-constructed T_max so the loaded scheduler
            # state cannot drag the LR schedule horizon back to whatever
            # value the previous run picked. The user's current --n-epochs
            # is the authoritative training length, so the resumed run's
            # cosine annealing must finish at args.n_epochs * steps_per_epoch,
            # not at the original total. ``last_epoch`` (the step counter)
            # from the checkpoint is preserved -- we only override T_max.
            fresh_t_max = cosine.T_max
            cosine.load_state_dict(checkpoint_payload["scheduler_state_dict"])
            cosine.T_max = fresh_t_max
        history = list(checkpoint_payload.get("history", []))
        if "rng_torch" in checkpoint_payload:
            torch.set_rng_state(checkpoint_payload["rng_torch"])
        if "rng_numpy" in checkpoint_payload:
            np.random.set_state(checkpoint_payload["rng_numpy"])
        if "rng_cuda" in checkpoint_payload and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint_payload["rng_cuda"])

    # ---- Train -------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info(
        "Training LMC: num_tasks=%d num_latents=%d kernel=%s mean=%s M=%d "
        "epochs=%d batch=%d lr=%.0e",
        lmc_spec.num_tasks, lmc_spec.num_latents, lmc_spec.kernel_type,
        lmc_spec.mean_type, args.n_inducing, args.n_epochs,
        args.batch_size, args.lr,
    )
    t0 = time.time()
    step = resume_step
    for epoch in range(resume_epoch, args.n_epochs):
        ep_losses: list[float] = []
        for xb, yb, mb in loader:
            xb = xb.to(args.device, non_blocking=True)
            yb = yb.to(args.device, non_blocking=True)
            mb = mb.to(args.device, non_blocking=True)
            optim.zero_grad()
            out = model(xb)
            # Masked log-likelihood (per observed cell) replaces the
            # standard ELBO data-fit term. The KL term we recover from
            # the VariationalELBO with num_data=n_obs_per_task averaged.
            ll = masked_multitask_log_prob(likelihood, out, yb, mb)
            # KL approximation via VariationalELBO is awkward with the
            # masked LL; instead, scale ll by num_data and add the prior
            # KL from the variational strategy directly. ``kl_divergence``
            # is the standard component used inside VariationalELBO.
            kl_div = (
                model.svgp.variational_strategy.kl_divergence().sum()
                / max(1, int(mb.sum()))
            )
            elbo = ll - kl_div
            loss = -elbo
            loss.backward()
            optim.step()
            cosine.step()
            step += 1
            ep_losses.append(float(loss.detach()))
        avg = sum(ep_losses) / max(1, len(ep_losses))
        history.append({"epoch": epoch, "step": step, "loss": avg})
        LOG.info("epoch=%d step=%d loss=%.4f", epoch, step, avg)
        if wandb_run is not None:
            wandb_run.log(
                {"epoch": epoch, "step": step, "loss": avg},
                step=step,
            )
        # Per-epoch checkpoint save (after the W&B log so the step counter
        # is consistent). The atomic ``.tmp -> os.replace`` write in
        # :func:`_save_checkpoint` keeps the file safe against mid-write
        # preemption on the spot-instance cluster. We write to the
        # pod-local SSD (``checkpoint_tmp_dir``) -- the NFS flush happens
        # once at the end of training so per-epoch saves do not block on
        # the ~300 MB/s tailscale link.
        _save_checkpoint(
            checkpoint_tmp_dir / f"epoch_{epoch}.pt",
            model=model,
            likelihood=likelihood,
            optimizer=optim,
            scheduler=cosine,
            encoder_spec_dict=encoder_spec.__dict__,
            lmc_spec_dict=lmc_spec.__dict__,
            epoch=epoch,
            step=step,
            train_keep_idx=train_keep_idx,
            holdout_test_idx=holdout_test_idx,
            mu0=mu0,
            s0=s0,
            mu1=mu1,
            s1=s1,
            seed=args.seed,
            n_input=n_input,
            history=history,
        )
        # Rolling deletion to keep /tmp under control (Azure ephemeral
        # disk is small). Only the most recent 3 epochs survive; the
        # final flush at end-of-training copies whichever is latest.
        pruned = _prune_old_checkpoints(checkpoint_tmp_dir, keep_last=3)
        if pruned:
            LOG.debug(
                "Pruned %d stale tmp checkpoints: %s",
                len(pruned), [str(p) for p in pruned],
            )

    t_train = time.time() - t0
    LOG.info("Training done in %.1f s", t_train)

    # ---- Final NFS flush --------------------------------------------------
    # Copy the latest per-epoch checkpoint from pod-local SSD to durable
    # NFS so spot preemption between the final epoch and the end-of-
    # training artifact write still leaves a resumable file at the
    # shared output path. The fan-out (`epoch_<latest>.pt` + the final
    # ``lmc_model.pt`` / ``foundation_model.pt`` below) is the durable
    # artifact set for downstream evaluators.
    tmp_ckpts = sorted(
        checkpoint_tmp_dir.glob("epoch_*.pt"),
        key=lambda p: int(p.stem.split("_", 1)[1]),
    )
    if tmp_ckpts:
        latest_tmp = tmp_ckpts[-1]
        _flush_to_durable(latest_tmp, checkpoint_dir)
    else:
        LOG.warning(
            "No per-epoch checkpoints in %s to flush -- training may "
            "have exited before completing a single epoch.",
            checkpoint_tmp_dir,
        )

    # ---- Eval --------------------------------------------------------------
    model.eval()
    likelihood.eval()
    pred_means_t0: list[np.ndarray] = []
    pred_means_t1: list[np.ndarray] = []
    pred_stds_t0: list[np.ndarray] = []
    pred_stds_t1: list[np.ndarray] = []
    pred_loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=max(1024, args.batch_size),
        shuffle=False,
    )
    with torch.no_grad():
        for (xb,) in pred_loader:
            xb = xb.to(args.device, non_blocking=True)
            out = model(xb)
            post = likelihood(out)
            mean = post.mean.cpu().numpy()  # (B, 2)
            std = post.variance.clamp_min(1e-12).sqrt().cpu().numpy()
            pred_means_t0.append(mean[:, 0])
            pred_means_t1.append(mean[:, 1])
            pred_stds_t0.append(std[:, 0])
            pred_stds_t1.append(std[:, 1])
    pred_mean_n = np.concatenate(pred_means_t0) * s0 + mu0  # un-standardize
    pred_mean_gw = np.concatenate(pred_means_t1) * s1 + mu1
    pred_std_n = np.concatenate(pred_stds_t0) * s0
    pred_std_gw = np.concatenate(pred_stds_t1) * s1

    # Train-fit metrics: in-sample, computed on whatever rows the trainer
    # actually saw. When a K-fold is in effect we restrict to train_keep_idx
    # so the in-sample number is honest about what it represents (it's not
    # "all rows" anymore).
    eval_train_idx = train_keep_idx if holdout_test_idx is not None else np.arange(n)
    m0_tr = m0[eval_train_idx]
    m1_tr = m1[eval_train_idx]
    rmse_n = float(np.sqrt(
        ((pred_mean_n[eval_train_idx][m0_tr] - y0[eval_train_idx][m0_tr]) ** 2).mean()
    ))
    mae_n = float(np.abs(pred_mean_n[eval_train_idx][m0_tr] - y0[eval_train_idx][m0_tr]).mean())
    rmse_gw = float(np.sqrt(
        ((pred_mean_gw[eval_train_idx][m1_tr] - y1[eval_train_idx][m1_tr]) ** 2).mean()
    ))
    mae_gw = float(np.abs(pred_mean_gw[eval_train_idx][m1_tr] - y1[eval_train_idx][m1_tr]).mean())
    LOG.info(
        "TRAIN-FIT N : RMSE=%.3f MAE=%.3f std_mean=%.3f",
        rmse_n, mae_n, float(pred_std_n[eval_train_idx][m0_tr].mean()),
    )
    LOG.info(
        "TRAIN-FIT gw: RMSE=%.3f MAE=%.3f std_mean=%.3f",
        rmse_gw, mae_gw, float(pred_std_gw[eval_train_idx][m1_tr].mean()),
    )

    # ---- Held-out (spatial K-fold) evaluation -----------------------------
    holdout_rmse_n: float | None = None
    holdout_mae_n: float | None = None
    holdout_rmse_gw: float | None = None
    holdout_mae_gw: float | None = None
    if holdout_test_idx is not None:
        m0_ho = m0[holdout_test_idx]
        m1_ho = m1[holdout_test_idx]
        diff_n = pred_mean_n[holdout_test_idx][m0_ho] - y0[holdout_test_idx][m0_ho]
        diff_gw = pred_mean_gw[holdout_test_idx][m1_ho] - y1[holdout_test_idx][m1_ho]
        holdout_rmse_n = float(np.sqrt((diff_n ** 2).mean()))
        holdout_mae_n = float(np.abs(diff_n).mean())
        holdout_rmse_gw = float(np.sqrt((diff_gw ** 2).mean()))
        holdout_mae_gw = float(np.abs(diff_gw).mean())
        LOG.info(
            "HOLDOUT  N : RMSE=%.3f MAE=%.3f std_mean=%.3f (fold=%d, n_eval=%d)",
            holdout_rmse_n, holdout_mae_n,
            float(pred_std_n[holdout_test_idx][m0_ho].mean()),
            args.kfold_test_fold, int(m0_ho.sum()),
        )
        LOG.info(
            "HOLDOUT  gw: RMSE=%.3f MAE=%.3f std_mean=%.3f (fold=%d, n_eval=%d)",
            holdout_rmse_gw, holdout_mae_gw,
            float(pred_std_gw[holdout_test_idx][m1_ho].mean()),
            args.kfold_test_fold, int(m1_ho.sum()),
        )

    # ---- Save --------------------------------------------------------------
    summary = {
        "run_name": "lmc_v4_trial",
        "parquet": str(args.parquet),
        "n_rows": int(n),
        "n_input": int(n_input),
        "n_inducing": int(args.n_inducing),
        "n_epochs": int(args.n_epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "device": args.device,
        "kernel_type": args.kernel_type,
        "mean_type": args.mean_type,
        "num_tasks": 2,
        "num_latents": int(args.num_latents),
        "task_0_name": "n_value",
        "task_0_mean": mu0,
        "task_0_std": s0,
        "task_0_observed": int(m0.sum()),
        "task_1_name": "groundwater_depth_m",
        "task_1_mean": mu1,
        "task_1_std": s1,
        "task_1_observed": int(m1.sum()),
        "training_time_seconds": t_train,
        "final_loss": history[-1]["loss"] if history else None,
        "train_fit_rmse_n": rmse_n,
        "train_fit_mae_n": mae_n,
        "train_fit_rmse_gw": rmse_gw,
        "train_fit_mae_gw": mae_gw,
        "kfold_test_fold": int(args.kfold_test_fold),
        "leave_region": args.leave_region or None,
        "split_mode": (
            "leave_region_out" if args.leave_region
            else "spatial_kfold" if args.kfold_test_fold >= 0
            else "in_sample"
        ),
        "kfold_n_train": int(len(train_keep_idx)),
        "kfold_n_holdout": (
            int(len(holdout_test_idx)) if holdout_test_idx is not None else 0
        ),
        "holdout_rmse_n": holdout_rmse_n,
        "holdout_mae_n": holdout_mae_n,
        "holdout_rmse_gw": holdout_rmse_gw,
        "holdout_mae_gw": holdout_mae_gw,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("Wrote %s/summary.json", args.output_dir)
    # Persist the K-fold index split alongside predictions.npz so any
    # downstream conformal / coverage analysis can recompute honest
    # held-out statistics without re-running the trainer. The boolean
    # ``is_holdout`` mask is the easiest representation to consume.
    is_holdout = np.zeros(n, dtype=np.bool_)
    if holdout_test_idx is not None:
        is_holdout[holdout_test_idx] = True
    np.savez(
        args.output_dir / "predictions.npz",
        pred_mean_n=pred_mean_n.astype(np.float32),
        pred_mean_gw=pred_mean_gw.astype(np.float32),
        pred_std_n=pred_std_n.astype(np.float32),
        pred_std_gw=pred_std_gw.astype(np.float32),
        y_n=y0.astype(np.float32),
        y_gw=y1.astype(np.float32),
        mask_n=m0,
        mask_gw=m1,
        is_holdout=is_holdout,
    )
    LOG.info("Wrote %s/predictions.npz", args.output_dir)
    final_model_payload = {
        "model": model.state_dict(),
        "likelihood": likelihood.state_dict(),
        "encoder_spec": encoder_spec.__dict__,
        "lmc_spec": lmc_spec.__dict__,
    }
    torch.save(final_model_payload, args.output_dir / "lmc_model.pt")
    LOG.info("Saved %s/lmc_model.pt", args.output_dir)
    # Mirror the final weights to ``foundation_model.pt`` for the LRO
    # evaluator + map driver, which both load via that canonical name.
    # We use ``shutil.copy2`` rather than a second ``torch.save`` so the
    # two artifacts are byte-identical (no risk of state_dict ordering
    # drift between the two writes).
    shutil.copy2(
        args.output_dir / "lmc_model.pt",
        args.output_dir / "foundation_model.pt",
    )
    LOG.info("Mirrored %s/foundation_model.pt", args.output_dir)
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "train_fit_rmse_n": rmse_n,
                "train_fit_mae_n": mae_n,
                "train_fit_rmse_gw": rmse_gw,
                "train_fit_mae_gw": mae_gw,
                "training_time_seconds": t_train,
                "final_loss": history[-1]["loss"] if history else None,
                "task_0_observed": int(m0.sum()),
                "task_1_observed": int(m1.sum()),
                "kfold_test_fold": int(args.kfold_test_fold),
                "kfold_n_train": int(len(train_keep_idx)),
                "kfold_n_holdout": (
                    int(len(holdout_test_idx))
                    if holdout_test_idx is not None
                    else 0
                ),
                "holdout_rmse_n": holdout_rmse_n,
                "holdout_mae_n": holdout_mae_n,
                "holdout_rmse_gw": holdout_rmse_gw,
                "holdout_mae_gw": holdout_mae_gw,
            }
        )
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
