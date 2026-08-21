"""Distributed training loop for the foundation model.

The trainer is structured so the *same* code path works for:

- Single-CPU / single-MPS local smoke runs (no ``torch.distributed`` at all).
- Single-node multi-CPU (DDP over Gloo).
- Multi-node Miyabi-G runs launched under ``torchrun`` (DDP over NCCL).

Distributed setup is detected from the ``WORLD_SIZE``/``RANK``/``LOCAL_RANK``
environment variables (set by ``torchrun``). When those are absent we run in
single-process mode and skip the ``init_process_group`` call so that local
tests don't need a distributed launcher.

Checkpoints are atomic (write to ``.tmp`` then ``os.replace``) and contain
model weights, optimizer state, scheduler state, RNG state, and the trainer
``TrainerState`` (epoch, step, best metric). Resume support is opt-in via
``--resume-if-exists`` in the Hydra driver -- the trainer reads the newest
``latest.pt`` if present.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import gpytorch
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from national.models.foundation import FoundationModel
from national.training.adversarial import (
    CoordinateCritic,
    critic_step,
    encoder_adversarial_term,
    standardize_coords,
)
from national.training.losses import elbo_with_regime_weights, hsic_regularizer, mmd_regularizer
from national.training.schedulers import cosine_warmup

LOG = logging.getLogger("national.training.trainer")

# Linear warmup length (epochs) for the HSIC independence penalty. A fresh
# encoder + a not-yet-converged SVGP Cholesky is fragile; hitting it with the
# full-strength coordinate-independence gradient from step 0 has caused
# instability in smoke runs, so `indep_lambda` is ramped in linearly over
# this many epochs (see the `eff_lambda` computation in `fit()`).
_INDEP_WARMUP_EPOCHS = 5

# Linear warmup length (epochs) for the adversarial coordinate-critic term
# (see `national.training.adversarial`). Same rationale as
# `_INDEP_WARMUP_EPOCHS`: the encoder shouldn't be hit with a full-strength
# adversarial gradient before the SVGP/critic have had a chance to settle.
# Unlike `indep_lambda`, the CRITIC itself is not warmed up -- it starts
# training from epoch 0 whenever `adv_lambda > 0` (see the `adv_lambda`
# block in `fit()`) so it isn't cold the moment the encoder term switches
# on; only `eff_adv_lambda` (the encoder-facing term) ramps.
_ADV_WARMUP_EPOCHS = 5


@dataclass
class TrainerState:
    """Mutable training progress tracked across checkpoints."""

    epoch: int = 0
    step: int = 0
    best_metric: float | None = None
    history: list[dict[str, float]] = field(default_factory=list)
    # Populated once per epoch when `cfg.training.log_indep_diagnostic` is
    # set -- see the `log_indep_diagnostic` block in `fit()`. Empty list when
    # the flag is off (the default), so existing checkpoints/callers that
    # never look at this field are unaffected.
    indep_diagnostic: list[dict[str, float]] = field(default_factory=list)
    # Populated once per epoch whenever `cfg.training.adv_lambda > 0` -- see
    # the adversarial-critic block in `fit()`. Each entry is
    # {epoch, critic_mse_mean, critic_r2, task_loss_mean}. Empty list when
    # adv_lambda is 0 (the default), so existing checkpoints/callers that
    # never look at this field are unaffected.
    adv_diagnostic: list[dict[str, float]] = field(default_factory=list)


@dataclass
class TrainerOutput:
    """Returned by :meth:`FoundationTrainer.fit`."""

    final_loss: float
    state: TrainerState
    last_checkpoint: Path | None


def _is_distributed_environment() -> bool:
    """``True`` if torchrun-style env vars suggest we're in DDP."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


class FoundationTrainer:
    """SVGP training loop with DDP, mini-batches, checkpointing, logging."""

    def __init__(
        self,
        model: FoundationModel,
        dataset: Dataset,
        cfg: Any,
        *,
        device: torch.device | str = "cpu",
        sample_weight_fn=None,  # noqa: ANN001
        log_every: int = 50,
    ) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.log_every = int(log_every)
        self.sample_weight_fn = sample_weight_fn
        self.state = TrainerState()
        self._wandb_run = None  # populated by _maybe_init_wandb on the master rank
        # Adversarial coordinate critic (national.training.adversarial).
        # Created LAZILY on the first training batch once cfg.training.
        # adv_lambda > 0 (z_dim -- the encoder output width -- is only known
        # then). Deliberately kept as plain trainer attributes, NOT
        # submodules of `self.model_module` and NOT part of `self.optimizer`:
        # the critic is a training-time "sparring partner" that shapes the
        # encoder's gradient, not a piece of the deployed model, so it must
        # never leak into `FoundationModel.state_dict()` / `.save()` (that
        # artifact format is a published contract) nor into
        # `self.model.parameters()` (which would make DistributedDataParallel
        # try to wrap and gradient-sync it as if it were part of the model).
        # Because `_save_checkpoint` below only ever touches
        # `self.model_module.state_dict()` and `self.optimizer.state_dict()`,
        # keeping the critic off of both means checkpoints are byte-for-byte
        # identical in shape whether or not adv_lambda was used -- no
        # checkpoint-format version bump needed for this feature.
        self._adv_critic: CoordinateCritic | None = None
        self._adv_critic_optimizer: torch.optim.Optimizer | None = None

        self.rank, self.local_rank, self.world_size = self._setup_distributed()
        self.is_master = self.rank == 0
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)

        self.model_module = model.to(self.device)
        if self.world_size > 1:
            self.model = DistributedDataParallel(
                self.model_module,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        else:
            self.model = self.model_module

        # FoundationModel.parameters() already traverses encoder, GP, likelihood,
        # and the regime FiLM submodule, so a single sweep covers everything.
        params = list(self.model_module.parameters())
        self.optimizer = torch.optim.Adam(
            params,
            lr=float(getattr(cfg.training, "lr", 5e-3)),
            betas=(
                float(getattr(cfg.training, "beta1", 0.9)),
                float(getattr(cfg.training, "beta2", 0.999)),
            ),
            weight_decay=float(getattr(cfg.training, "weight_decay", 1e-5)),
        )

        total_steps = max(
            1,
            int(getattr(cfg.training, "n_epochs", 100))
            * max(1, len(self.dataset) // max(1, int(getattr(cfg.training, "batch_size", 1024)))),
        )
        self.scheduler = cosine_warmup(
            self.optimizer,
            warmup_steps=int(getattr(cfg.training, "warmup_steps", 100)),
            total_steps=total_steps,
            min_lr_ratio=0.01,
        )

        self._mll_num_data = len(self.dataset)

    # ------------------------------------------------------------------ fit
    def fit(self) -> TrainerOutput:
        """Run training. Returns final loss and the (saved) checkpoint path."""
        seed = int(getattr(self.cfg.run, "seed", 42))
        self._set_deterministic(seed)

        batch_size = int(getattr(self.cfg.training, "batch_size", 1024))
        n_epochs = int(getattr(self.cfg.training, "n_epochs", 100))
        mmd_weight = float(getattr(self.cfg.training, "mmd_weight", 0.0))
        indep_lambda = float(getattr(self.cfg.training, "indep_lambda", 0.0))
        indep_warmup_epochs = int(
            getattr(self.cfg.training, "indep_warmup_epochs", _INDEP_WARMUP_EPOCHS)
        )
        # Diagnostic-only: measures the UNWEIGHTED (lambda-free) HSIC scale
        # once per epoch, for calibrating `--indep-lambda` before committing
        # to a sweep. Must never influence training -- see the no_grad block
        # around `hsic_raw` below, which reuses the cached encoder output
        # (`last_batch_encoded`) so it costs no extra encoder forward and
        # cannot change the optimized `loss` or the RNG stream (no random
        # draws happen in `hsic_regularizer`), so turning it on is a no-op
        # for the training trajectory.
        log_indep_diagnostic = bool(
            getattr(self.cfg.training, "log_indep_diagnostic", False)
        )
        # Adversarial coordinate critic (national.training.adversarial).
        # `adv_lambda` gates the critic's existence entirely: 0.0 (default)
        # means the critic is never constructed and this whole block is a
        # no-op, mirroring `indep_lambda`'s gating. Unlike `indep_lambda`,
        # there is no separate "diagnostic-only" flag -- the per-epoch
        # {critic_mse_mean, critic_r2, task_loss_mean} diagnostic is
        # recorded whenever adv_lambda > 0, since (a) the critic already
        # exists in that case (no extra cost to read its MSE) and (b) the
        # un-warmed-up critic MSE during the warmup epochs IS the
        # calibration signal (how decodable are coordinates before the
        # encoder starts fighting back).
        adv_lambda = float(getattr(self.cfg.training, "adv_lambda", 0.0))
        adv_warmup_epochs = int(
            getattr(self.cfg.training, "adv_warmup_epochs", _ADV_WARMUP_EPOCHS)
        )
        # k critic-only steps per encoder step (all reusing this batch's
        # cached phi -- see `critic_step`'s docstring). 1 is the standard
        # single-step alternation.
        adv_critic_steps = int(getattr(self.cfg.training, "adv_critic_steps", 1))
        adv_critic_lr = float(getattr(self.cfg.training, "adv_critic_lr", 1e-3))
        # SGE gate dropout (P-SGE, docs/research/2026-07-12_sge_preregistration.md):
        # TRAINING-ONLY, per-row with prob p the batch's trailing gate column
        # is zeroed, teaching the covariate-fallback pathway that a low gate
        # will route through at test time. Applied HERE (on a clone of the
        # batch tensor, before the forward) rather than inside the encoder so
        # that (a) it can never fire at predict time, (b) it never touches
        # the learned inducing points' gate coordinate, and (c) the dataset
        # tensors are never mutated. Requires a model whose encoder actually
        # carries the trailing gate column -- fail loud otherwise instead of
        # silently training the wrong arm.
        gate_dropout = float(getattr(self.cfg.training, "gate_dropout", 0.0))
        # EES snapshot battery (P-R3c..f, docs/research/
        # 2026-07-13_r3_preregistration.md): save the full model state (in
        # the FoundationModel ARTIFACT format, so `FoundationModel.load`
        # reads it directly) at the END of each listed 1-indexed epoch, to
        # `<checkpoint_root>/ep{N}.pt`. torch.save consumes no RNG, so
        # snapshotting can never perturb the training trajectory (pinned by
        # tests/national/test_ees_snapshots.py). Empty list (the default)
        # disables the feature entirely.
        snapshot_epochs = sorted(
            {int(e) for e in (getattr(self.cfg.training, "snapshot_epochs", None) or [])}
        )
        if snapshot_epochs:
            if snapshot_epochs[0] < 1 or snapshot_epochs[-1] > n_epochs:
                raise ValueError(
                    f"cfg.training.snapshot_epochs {snapshot_epochs} out of "
                    f"range for n_epochs={n_epochs}: snapshot epochs are "
                    "1-indexed and a snapshot beyond the last epoch would "
                    "silently never be written."
                )
            for attr in ("spec", "gp"):
                if not hasattr(self.model_module, attr):
                    raise ValueError(
                        "cfg.training.snapshot_epochs requires a "
                        "FoundationModel-shaped model (missing attribute "
                        f"{attr!r}) — snapshots are saved in the "
                        "FoundationModel artifact format."
                    )
        _enc_spec = getattr(
            getattr(self.model_module, "spec", None), "encoder", None
        )
        model_has_gate_input = bool(getattr(_enc_spec, "sge_gate_input", False))
        # MS-SGE (P-MS): number of trailing per-band gate columns (0 when the
        # model carries the scalar gate or no gate at all). Per-band dropout
        # is INDEPENDENT per band per row -- see the per-batch block below.
        model_gate_bands = int(getattr(_enc_spec, "sge_gate_bands", 0) or 0)
        if gate_dropout > 0.0 and not (model_has_gate_input or model_gate_bands > 0):
            raise ValueError(
                "cfg.training.gate_dropout > 0 but the model's encoder has no "
                "trailing SGE gate column (EncoderSpec.sge_gate_input=False "
                "and sge_gate_bands=0) -- the dropout would silently be a "
                "no-op on the wrong column."
            )
        checkpoint_dir = Path(getattr(self.cfg.io, "checkpoint_root", "./checkpoints"))
        checkpoint_every_min = float(getattr(self.cfg.training, "checkpoint_every_min", 30))
        if self.is_master:
            self._maybe_init_wandb()

        sampler = None
        shuffle = True
        if self.world_size > 1:
            sampler = DistributedSampler(
                self.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
            )
            shuffle = False
        elif (
            bool(getattr(self.cfg.training, "regime_balanced_sampler", False))
            and hasattr(self.dataset, "regimes")
        ):
            # Up-weight rare regimes (Risk #2). Single-GPU only; DDP keeps the
            # DistributedSampler so ranks see disjoint shards.
            from national.tiling.regime_sampler import RegimeBalancedSampler

            sampler = RegimeBalancedSampler(
                self.dataset.regimes,
                alpha=float(getattr(self.cfg.training, "regime_balance_alpha", 0.5)),
                seed=int(getattr(self.cfg.run, "seed", 42)),
            )
            shuffle = False
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=int(getattr(self.cfg.training, "num_workers", 0)),
            pin_memory=self.device.type == "cuda",
            drop_last=False,
            collate_fn=_collate,
        )

        last_ckpt: Path | None = None
        last_ckpt_at = time.monotonic()
        final_loss = float("nan")

        self.model.train()
        self.model_module.likelihood.train()
        for epoch in range(self.state.epoch, n_epochs):
            self.state.epoch = epoch
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
            # Linear warmup of the HSIC independence penalty over the first
            # `indep_warmup_epochs` epochs. At epoch 0 the encoder is
            # untrained and the SVGP Cholesky is still settling; a
            # full-strength penalty gradient from step 0 has caused
            # instability in smoke runs, so we ramp eff_indep_lambda from 0
            # up to indep_lambda linearly before holding it constant.
            eff_indep_lambda = indep_lambda * min(1.0, epoch / max(1, indep_warmup_epochs))
            # Same linear-warmup shape as eff_indep_lambda, applied to the
            # ENCODER-facing adversarial term only -- the critic itself
            # trains at full strength from epoch 0 whenever adv_lambda > 0
            # (see the per-batch block below), only the `-eff_adv_lambda *
            # MSE(...)` term added to the encoder's loss is ramped.
            eff_adv_lambda = adv_lambda * min(1.0, epoch / max(1, adv_warmup_epochs))
            epoch_loss = 0.0
            n_batches = 0
            epoch_hsic_raw_sum = 0.0
            epoch_task_loss_sum = 0.0
            epoch_diag_batches = 0
            epoch_critic_mse_sum = 0.0
            epoch_adv_task_loss_sum = 0.0
            epoch_adv_diag_batches = 0
            for batch in loader:
                x = batch["x"].to(self.device, non_blocking=True)
                # SGE gate dropout (training only; see the block above the
                # epoch loop). Clone before writing so the underlying
                # dataset tensor is never mutated when the DataLoader hands
                # us a view (num_workers=0 path).
                if gate_dropout > 0.0:
                    if model_gate_bands > 0:
                        # MS-SGE: independent per band per row -- each of the
                        # trailing gate-band columns is zeroed with prob p
                        # separately, so the fallback pathway sees every
                        # partial band subset, not just all-or-none.
                        drop_mask = (
                            torch.rand(
                                x.shape[0], model_gate_bands, device=x.device
                            )
                            < gate_dropout
                        )
                        if bool(drop_mask.any()):
                            x = x.clone()
                            x[:, -model_gate_bands:][drop_mask] = 0.0
                    else:
                        drop_mask = (
                            torch.rand(x.shape[0], device=x.device) < gate_dropout
                        )
                        if bool(drop_mask.any()):
                            x = x.clone()
                            x[drop_mask, -1] = 0.0
                y = batch["y"].to(self.device, non_blocking=True)
                regime = batch.get("regime")
                if regime is not None:
                    regime = regime.to(self.device, non_blocking=True)
                sample_weights = (
                    self.sample_weight_fn(regime) if self.sample_weight_fn is not None else None
                )

                self.optimizer.zero_grad(set_to_none=True)
                with gpytorch.settings.num_likelihood_samples(8):
                    pred_dist = self.model(x)
                    # Heteroscedastic path: the noise head produces a
                    # per-point variance that FixedNoiseGaussianLikelihood
                    # consumes. Homoscedastic path keeps noise=None and
                    # uses the likelihood's own learned noise.
                    noise = (
                        self.model_module.predict_noise_variance(x)
                        if hasattr(self.model_module, "predict_noise_variance")
                        else None
                    )
                    loss = elbo_with_regime_weights(
                        pred_dist,
                        self.model_module.likelihood,
                        y,
                        num_data=self._mll_num_data,
                        model=self.model_module.gp,
                        sample_weights=sample_weights,
                        noise=noise,
                    )
                    # Snapshot BEFORE any mmd/hsic/adversarial term is added
                    # -- this is the "task loss" half of both the HSIC and
                    # the adversarial per-epoch diagnostics. A plain float,
                    # not part of the backward graph, so it cannot itself
                    # perturb training. Needed whenever EITHER diagnostic
                    # consumer is active.
                    task_loss_value = (
                        float(loss.detach().cpu().item())
                        if (log_indep_diagnostic or adv_lambda > 0.0)
                        else None
                    )
                    # `x.shape[0] >= 4` guards the trailing mini-batch: the
                    # DataLoader is built with drop_last=False, so the final
                    # batch of an epoch can be smaller than
                    # hsic_regularizer's minimum batch size of 4 -- skip both
                    # the penalty AND the diagnostic for that one tiny batch
                    # rather than crashing.
                    # Shared by the HSIC/MMD independence penalty AND the
                    # adversarial critic block below -- both need the same
                    # "is this batch big enough to form the relevant
                    # statistic" guard (HSIC's Gram-matrix pair needs >=4
                    # rows; the critic's per-batch coordinate
                    # standardization is degenerate below that too).
                    batch_supports_hsic = x.shape[0] >= 4
                    if (
                        mmd_weight > 0.0
                        or eff_indep_lambda > 0.0
                        or (log_indep_diagnostic and batch_supports_hsic)
                        or (adv_lambda > 0.0 and batch_supports_hsic)
                    ):
                        # Reuse the encoder output the GP loss above already
                        # produced, instead of calling `encoder(x)` a second
                        # time. `pred_dist = self.model(x)` internally ran
                        # `encoder(cat([inducing_points, x]))` exactly once;
                        # re-running `encoder(x)` here would be a *second*
                        # forward through the same BatchNorm layers, so
                        # lambda>0 batches would update BN running stats
                        # twice per step while lambda=0 batches update once
                        # -- a confound across the very lambda-frontier this
                        # penalty is used to sweep. `last_batch_encoded`
                        # slices out the rows of the cached tensor that
                        # correspond to `x` (the trailing rows, since the
                        # variational strategy prepends the inducing
                        # points), so `phi` is the SAME graph node the GP
                        # loss used -- gradients are consistent and the
                        # encoder runs exactly once per step either way. This
                        # holds for the diagnostic branch too: it is a pure
                        # read of an already-cached tensor, not a forward
                        # pass, so enabling `log_indep_diagnostic` never adds
                        # an encoder call.
                        phi = self.model_module.gp.last_batch_encoded(x.shape[0])
                    if mmd_weight > 0.0:
                        phi_ref = torch.randn_like(phi)
                        loss = loss + mmd_weight * mmd_regularizer(phi, phi_ref)
                    if eff_indep_lambda > 0.0 and batch_supports_hsic:
                        # lat/lon are always the first two input columns
                        # (see national/data/boring_dataset.py: x =
                        # [lat, lon, depth, ...]).
                        coords = x[:, :2]
                        loss = loss + eff_indep_lambda * hsic_regularizer(phi, coords)
                    if log_indep_diagnostic and batch_supports_hsic:
                        # UNWEIGHTED HSIC, purely for logging the natural
                        # scale of the penalty at indep_lambda's current
                        # (possibly zero) value -- this is the whole point:
                        # calibrate lambda by first observing the
                        # un-penalized HSIC scale. Computed under no_grad so
                        # it is disconnected from the backward graph and
                        # cannot affect `loss` or gradients, whether or not
                        # the real penalty above also fired this step.
                        with torch.no_grad():
                            hsic_raw = hsic_regularizer(phi, x[:, :2])
                        epoch_hsic_raw_sum += float(hsic_raw.item())
                        epoch_task_loss_sum += task_loss_value
                        epoch_diag_batches += 1

                    # Adversarial coordinate critic (national.training.
                    # adversarial). Gated on the RAW adv_lambda (not
                    # eff_adv_lambda) so the critic starts training from
                    # epoch 0 even during the encoder-term warmup -- see
                    # `_ADV_WARMUP_EPOCHS`'s docstring above. `phi` here is
                    # the SAME cached tensor the HSIC/MMD block above reused
                    # (guarded into existence by the same `or (adv_lambda >
                    # 0.0 and batch_supports_hsic)` clause), so this adds no
                    # additional encoder forward.
                    adv_term_added_this_step = False
                    if adv_lambda > 0.0 and batch_supports_hsic:
                        coords_std = standardize_coords(x[:, :2], dtype=phi.dtype)
                        if self._adv_critic is None:
                            self._adv_critic = CoordinateCritic(
                                z_dim=phi.shape[-1]
                            ).to(device=self.device, dtype=phi.dtype)
                            self._adv_critic_optimizer = torch.optim.Adam(
                                self._adv_critic.parameters(), lr=adv_critic_lr
                            )
                        # Step 1 (see encoder_adversarial_term's ordering
                        # contract): critic-only update(s) on this batch's
                        # DETACHED phi. Runs every batch adv_lambda>0,
                        # independent of the encoder-term warmup.
                        critic_mse = critic_step(
                            self._adv_critic,
                            self._adv_critic_optimizer,
                            phi,
                            coords_std,
                            k=adv_critic_steps,
                        )
                        if eff_adv_lambda > 0.0:
                            # Step 2: freeze critic params, compute the
                            # encoder-facing term with `phi` (NOT detached)
                            # so gradient flows phi -> encoder. Step 3: add
                            # to the main loss (still inside this `with`
                            # block, before the single `loss.backward()`
                            # below covers ELBO + HSIC/MMD + this term in
                            # one graph -- step 4 of the ordering contract).
                            adv_term = encoder_adversarial_term(
                                self._adv_critic, phi, coords_std, eff_adv_lambda
                            )
                            loss = loss + adv_term
                            adv_term_added_this_step = True
                        # Diagnostic: recorded every batch adv_lambda>0,
                        # regardless of whether the encoder term fired this
                        # step (mirrors indep_diagnostic's calibration
                        # rationale -- the un-warmed-up critic MSE is itself
                        # informative).
                        epoch_critic_mse_sum += critic_mse
                        epoch_adv_task_loss_sum += task_loss_value
                        epoch_adv_diag_batches += 1
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if adv_term_added_this_step:
                    # Step 6 of the ordering contract: re-enable the
                    # critic's gradient so the NEXT batch's `critic_step`
                    # (step 1) can update it normally. Safe to do here
                    # (after the main optimizer.step()) rather than
                    # immediately after `loss.backward()`, since nothing
                    # between the freeze (step 2, above) and here reads or
                    # writes `critic.parameters()`'s `.grad` -- and
                    # `critic_step`'s own `critic_optimizer.zero_grad()`
                    # call at the start of the next batch clears whatever
                    # (unused, since requires_grad was False) gradient state
                    # might otherwise have accumulated.
                    for p in self._adv_critic.parameters():
                        p.requires_grad_(True)

                self.state.step += 1
                epoch_loss += float(loss.detach().cpu().item())
                n_batches += 1

                if self.is_master and self.state.step % self.log_every == 0:
                    mean_loss = epoch_loss / max(1, n_batches)
                    lr_now = self.scheduler.get_last_lr()[0]
                    LOG.info(
                        "epoch=%d step=%d loss=%.4f lr=%.3e",
                        epoch,
                        self.state.step,
                        mean_loss,
                        lr_now,
                    )
                    self._wandb_log(
                        {"epoch": epoch, "loss": mean_loss, "lr": lr_now},
                        step=self.state.step,
                    )

                if self.is_master and (time.monotonic() - last_ckpt_at) >= checkpoint_every_min * 60:
                    last_ckpt = self._save_checkpoint(checkpoint_dir, name="latest.pt")
                    last_ckpt_at = time.monotonic()

            final_loss = epoch_loss / max(1, n_batches)
            self.state.history.append({"epoch": epoch, "loss": final_loss})
            # EES snapshots: `epoch` is the 0-indexed loop variable, the
            # pre-registered snapshot list is 1-indexed ("after N epochs of
            # training"), hence epoch + 1. Master rank only (same as the
            # regular checkpoints).
            if snapshot_epochs and self.is_master and (epoch + 1) in snapshot_epochs:
                self._save_model_snapshot(checkpoint_dir, epoch=epoch + 1)
            if log_indep_diagnostic:
                hsic_raw_mean = epoch_hsic_raw_sum / max(1, epoch_diag_batches)
                task_loss_mean = epoch_task_loss_sum / max(1, epoch_diag_batches)
                self.state.indep_diagnostic.append(
                    {
                        "epoch": epoch,
                        "hsic_raw_mean": hsic_raw_mean,
                        "task_loss_mean": task_loss_mean,
                    }
                )
                if self.is_master:
                    LOG.info(
                        "epoch %d indep diagnostic: hsic_raw_mean=%.6f task_loss_mean=%.4f",
                        epoch,
                        hsic_raw_mean,
                        task_loss_mean,
                    )
            if adv_lambda > 0.0:
                critic_mse_mean = epoch_critic_mse_sum / max(1, epoch_adv_diag_batches)
                # coords_std is per-batch standardized to unit variance, so
                # the trivial "predict the mean" baseline has MSE == 1.0 and
                # R^2 = 1 - MSE / Var = 1 - MSE follows directly.
                critic_r2 = 1.0 - critic_mse_mean
                adv_task_loss_mean = epoch_adv_task_loss_sum / max(1, epoch_adv_diag_batches)
                self.state.adv_diagnostic.append(
                    {
                        "epoch": epoch,
                        "critic_mse_mean": critic_mse_mean,
                        "critic_r2": critic_r2,
                        "task_loss_mean": adv_task_loss_mean,
                    }
                )
                if self.is_master:
                    LOG.info(
                        "epoch %d adv diagnostic: critic_mse_mean=%.6f critic_r2=%.4f "
                        "task_loss_mean=%.4f",
                        epoch,
                        critic_mse_mean,
                        critic_r2,
                        adv_task_loss_mean,
                    )
            if self.is_master:
                LOG.info("epoch %d done; mean loss=%.4f", epoch, final_loss)
                self._wandb_log({"epoch_loss": final_loss}, step=self.state.step)

        # Always end with a final checkpoint.
        if self.is_master:
            last_ckpt = self._save_checkpoint(checkpoint_dir, name="final.pt")
            self._wandb_finish()

        return TrainerOutput(final_loss=final_loss, state=self.state, last_checkpoint=last_ckpt)

    # ------------------------------------------------------------ checkpoints
    def save_checkpoint(self, path: Path, *, step: int) -> Path:
        """Public wrapper kept for backward compat with the Phase A interface."""
        self.state.step = int(step)
        return self._save_checkpoint(Path(path).parent, name=Path(path).name)

    def _save_checkpoint(self, directory: Path, *, name: str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "version": 1,
            "state": asdict(self.state),
            "model_state_dict": self.model_module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_numpy": np.random.get_state(),
        }
        if torch.cuda.is_available():
            payload["rng_cuda"] = torch.cuda.get_rng_state_all()
        torch.save(payload, tmp)
        os.replace(tmp, path)
        LOG.info("Saved checkpoint to %s", path)
        return path

    def _save_model_snapshot(self, directory: Path, *, epoch: int) -> Path:
        """EES snapshot (P-R3c..f): persist the CURRENT model state in the
        FoundationModel ARTIFACT format (version / spec / state_dict /
        inducing_shape — the exact payload of ``FoundationModel.save``) so
        ``scripts.nmi_ees_eval`` reloads it with ``FoundationModel.load``.

        Deliberately NOT the trainer-checkpoint format: no optimizer /
        scheduler / RNG state (a snapshot is a frozen predictor, not a
        resume point) — this also keeps each file at model size instead of
        ~3x with Adam moments. Atomic (tmp + os.replace), like
        ``_save_checkpoint``."""
        m = self.model_module
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"ep{epoch}.pt"
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "version": getattr(m, "ARTIFACT_VERSION", 1),
            "spec": asdict(m.spec),
            "state_dict": m.state_dict(),
            "inducing_shape": tuple(m.gp.variational_strategy.inducing_points.shape),
        }
        torch.save(payload, tmp)
        os.replace(tmp, path)
        LOG.info("Saved EES snapshot (epoch %d) to %s", epoch, path)
        return path

    def maybe_load_checkpoint(self, path: Path) -> int:
        """Resume model/optimizer/scheduler/RNG/TrainerState from `path`.

        NOTE on the adversarial critic (national.training.adversarial): the
        critic and its optimizer are intentionally NOT part of `payload`
        (see the comment on `self._adv_critic` in `__init__` for why), so
        resuming a checkpoint with `adv_lambda > 0` restarts the critic from
        a fresh random init at whatever epoch training resumes from, even
        though the encoder/GP/likelihood resume exactly where they left
        off. This is a deliberate scope trade-off for the pilot: the critic
        re-warms quickly (it is a small MLP with a cheap, well-posed
        regression target), and `adv_diagnostic` above IS restored, so nothing
        about resumability of the published model artifact or the reported
        metrics is affected -- only a training-time detail (the critic
        briefly starts weaker than at the point of interruption) that a
        resumed run's own `adv_diagnostic` trace will show transparently
        (a small critic_mse blip right at the resume epoch) rather than
        hide.
        """
        path = Path(path)
        if not path.exists():
            return 0
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.model_module.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "scheduler_state_dict" in payload:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        state = payload.get("state", {})
        self.state = TrainerState(
            epoch=int(state.get("epoch", 0)),
            step=int(state.get("step", 0)),
            best_metric=state.get("best_metric"),
            history=list(state.get("history", [])),
            indep_diagnostic=list(state.get("indep_diagnostic", [])),
            adv_diagnostic=list(state.get("adv_diagnostic", [])),
        )
        if "rng_torch" in payload:
            torch.set_rng_state(payload["rng_torch"])
        if "rng_numpy" in payload:
            np.random.set_state(payload["rng_numpy"])
        if "rng_cuda" in payload and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
        LOG.info("Resumed from checkpoint %s (step=%d)", path, self.state.step)
        return self.state.step

    # ------------------------------------------------------------- internals
    def _setup_distributed(self) -> tuple[int, int, int]:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if _is_distributed_environment() and torch.distributed.is_available():
            if not torch.distributed.is_initialized():
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                torch.distributed.init_process_group(
                    backend=backend, rank=rank, world_size=world_size
                )
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size

    # ----------------------------------------------------------------- wandb
    def _maybe_init_wandb(self) -> None:
        """Initialize a Weights & Biases run if cfg.io.wandb is configured.

        Honors offline mode so Miyabi-G compute nodes (no outbound HTTPS) can
        write run logs to ``cfg.io.run_root`` for post-job ``wandb sync``.
        """
        wb_cfg = getattr(getattr(self.cfg, "io", None), "wandb", None)
        if wb_cfg is None:
            return
        mode = str(getattr(wb_cfg, "mode", "online")).lower()
        if mode == "disabled":
            return
        try:
            import wandb  # heavy; only imported when actually used
        except ImportError:
            LOG.warning("wandb not installed; skipping W&B logging.")
            return
        run_name = str(getattr(self.cfg.run, "name", "default"))
        project = str(getattr(wb_cfg, "project", "geo-estimation-national"))
        self._wandb_run = wandb.init(
            project=project,
            name=run_name,
            mode=mode,
            dir=str(getattr(self.cfg.io, "run_root", ".")),
            config=_omegaconf_to_dict(self.cfg),
            reinit=True,
        )

    def _wandb_log(self, metrics: dict, step: int) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.log(metrics, step=int(step))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("wandb log failed: %s", exc)

    def _wandb_finish(self) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.finish()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("wandb finish failed: %s", exc)
        finally:
            self._wandb_run = None

    def _set_deterministic(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # cudnn.deterministic is intentionally NOT forced -- it costs ~2x on
        # Hopper and the SVGP loss is already mostly non-deterministic from
        # mini-batch shuffling. Use ``torch.use_deterministic_algorithms(True)``
        # in tests if you need bit-reproducibility.


def _omegaconf_to_dict(cfg: Any) -> dict:
    """Best-effort conversion of an OmegaConf or dict into a plain dict."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except ImportError:
        pass
    if isinstance(cfg, dict):
        return cfg
    return {"cfg_repr": repr(cfg)}


def _collate(samples: Iterable[dict]) -> dict:
    """Stack a list of ``BoringSample``-style dicts into a batched dict.

    Each sample must expose at least an ``x`` (shape ``(D,)``) and a ``y``
    (scalar). Optionally a ``regime`` (int scalar) is stacked. Extra string
    metadata is dropped.
    """
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    regimes: list[torch.Tensor] = []
    for s in samples:
        xs.append(s["x"])
        ys.append(s["y"].reshape(()))
        if "regime" in s:
            regimes.append(s["regime"].reshape(()))
    out: dict[str, torch.Tensor] = {
        "x": torch.stack(xs, dim=0),
        "y": torch.stack(ys, dim=0),
    }
    if regimes:
        out["regime"] = torch.stack(regimes, dim=0).long()
    return out


__all__ = ["TrainerState", "TrainerOutput", "FoundationTrainer"]
