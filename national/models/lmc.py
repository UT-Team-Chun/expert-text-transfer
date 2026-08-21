"""Linear Model of Coregionalization (LMC) head for joint multi-task GP.

Paper B' Pillar 2: extends the single-output DKL+SVGP foundation model
(:mod:`national.models.foundation`) to a 2-task joint Gaussian Process
predicting SPT N-value and groundwater depth simultaneously. The two
outputs are coupled through a small set of latent GPs combined by a
learned task-mixing matrix (Bonilla, Chai, Williams 2007), so cross-task
information flow improves both targets relative to independent SVGPs:

* SPT N: noisier, more rows (100% of training data).
* Groundwater: cleaner, fewer rows (81% of training data; 19% are NaN).

When a row's groundwater target is NaN the corresponding task slot is
masked in the per-row likelihood term, so the model trains on the
incomplete-pair dataset without imputing or dropping rows. The shared
encoder still sees every row.

API mirrors :class:`national.models.foundation._DKLApproximateGP`:
constructor takes inducing points + an encoder + a config dataclass;
``forward(x)`` returns a
``gpytorch.distributions.MultitaskMultivariateNormal`` over ``num_tasks``.
Companion ``MultitaskGaussianLikelihood`` lives outside the module and is
configured by the trainer.

Reference: GPyTorch's SVGP_Multitask_GP_Regression tutorial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import gpytorch
import torch
from torch import nn


# ----------------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------------


@dataclass
class LMCSpec:
    """Configuration of the LMC multitask SVGP head.

    Attributes:
        num_tasks: Number of correlated output dimensions. For Paper B'
            Pillar 2 this is 2 (SPT N + groundwater depth). The 3-task
            extension (+ Fc proxy, e.g.) keeps the same API.
        num_latents: Number of latent GPs whose outputs are mixed by the
            learned ``num_tasks × num_latents`` matrix. The "rank" of
            the implied task covariance. ``num_latents <= num_tasks`` is
            typical; ``num_latents == num_tasks`` recovers an independent
            multitask GP, ``num_latents == 1`` is the maximally-coupled
            extreme. We default to ``num_tasks`` because cross-task
            transfer is the whole point of going LMC.
        kernel_type: Per-latent kernel family. Same options as
            :class:`SVGPSpec` -- ν≥1 was found agnostic on Paper 1 / B.
        mean_type: Per-latent mean function. ``constant`` is the
            multitask default; LinearMean over the encoded feature
            space is also supported.
        whitened: Whitened variational parameterisation (Hensman et al.
            2015). Default True, matches the single-task path.
        learn_inducing: Whether the inducing locations are trainable.
            Default True.
    """

    num_tasks: int = 2
    num_latents: int = 2
    kernel_type: Literal["matern52", "matern32", "matern12", "rbf"] = "rbf"
    mean_type: Literal["constant", "linear"] = "constant"
    whitened: bool = True
    learn_inducing: bool = True


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


class _LMCApproximateGP(gpytorch.models.ApproximateGP):
    """SVGP head whose latent GPs are mixed into ``num_tasks`` outputs.

    The variational strategy stack is:

    * ``CholeskyVariationalDistribution`` with batch_shape ``[num_latents]``
      so each latent GP has its own ``M × M`` Cholesky factor.
    * Inner ``VariationalStrategy`` (whitened) that does the standard SVGP
      operations per-latent.
    * Outer ``LMCVariationalStrategy`` that combines the latent means /
      covariances through a learned ``num_tasks × num_latents`` matrix
      and returns a ``MultitaskMultivariateNormal``.

    The kernel and mean are batched along the latent dim so each latent
    GP has its own hyperparameters. Inducing locations are shared across
    latents -- one fewer source of degeneracy and lower memory cost.
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        encoder: nn.Module,
        spec: LMCSpec,
    ) -> None:
        if spec.num_latents < 1:
            raise ValueError(
                f"LMCSpec.num_latents must be >= 1, got {spec.num_latents}"
            )
        if spec.num_tasks < 2:
            raise ValueError(
                f"LMCSpec.num_tasks must be >= 2 (use _DKLApproximateGP for "
                f"single-task), got {spec.num_tasks}"
            )

        # Broadcast inducing points across latents: shape (num_latents, M, D).
        # Sharing the location values is fine; the batched variational
        # distribution gives each latent its own mean / covariance.
        if inducing_points.dim() == 2:
            inducing_points = inducing_points.unsqueeze(0).expand(
                spec.num_latents, -1, -1
            ).clone()
        elif inducing_points.dim() != 3:
            raise ValueError(
                f"inducing_points must be 2- or 3-D, got shape {inducing_points.shape}"
            )

        n_inducing = inducing_points.size(-2)
        latent_batch = torch.Size([spec.num_latents])

        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=n_inducing,
            batch_shape=latent_batch,
        )
        inner_strategy_cls = (
            gpytorch.variational.VariationalStrategy
            if spec.whitened
            else gpytorch.variational.UnwhitenedVariationalStrategy
        )
        inner_strategy = inner_strategy_cls(
            self,
            inducing_points,
            variational_dist,
            learn_inducing_locations=spec.learn_inducing,
        )
        # The LMC wrapper. ``latent_dim`` selects which input batch axis is
        # the latent-GP index after the encoder runs (it lives at -1 because
        # our encoder returns a (B, n_output) tensor that gets broadcast to
        # (latents, B, n_output) by the inner strategy).
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            inner_strategy,
            num_tasks=spec.num_tasks,
            num_latents=spec.num_latents,
            latent_dim=-1,
        )
        super().__init__(variational_strategy)

        self.encoder = encoder
        # ``ResMLPEncoder.spec.n_output`` is the encoded feature width. We
        # fall back to inducing_points.size(-1) for unit tests that pass a
        # bare nn.Module.
        n_enc = getattr(getattr(encoder, "spec", None), "n_output", None)
        if n_enc is None:
            n_enc = inducing_points.size(-1)

        # Per-latent mean + kernel via batch_shape=latent.
        if spec.mean_type == "constant":
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=latent_batch)
        elif spec.mean_type == "linear":
            self.mean_module = gpytorch.means.LinearMean(
                input_size=n_enc, batch_shape=latent_batch
            )
            # Same zero-init trick as the single-task path: keep the
            # weights at 0 so training starts equivalent to ConstantMean
            # bias-only. Without this, LinearMean's default N(0, 1) weights
            # multiplied by encoded features ~5 push the SVGP variational
            # strategy into non-PSD Cholesky on the first step.
            with torch.no_grad():
                self.mean_module.weights.zero_()
        else:
            raise ValueError(f"Unknown mean_type: {spec.mean_type}")

        if spec.kernel_type == "rbf":
            base = gpytorch.kernels.RBFKernel(
                ard_num_dims=n_enc, batch_shape=latent_batch
            )
        else:
            nu = {"matern52": 2.5, "matern32": 1.5, "matern12": 0.5}[spec.kernel_type]
            base = gpytorch.kernels.MaternKernel(
                nu=nu, ard_num_dims=n_enc, batch_shape=latent_batch
            )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base, batch_shape=latent_batch
        )
        # Initialise the per-latent kernel lengthscale to a much larger
        # value (~3) than GPyTorch's default (~0.6). At training start
        # the encoded features have scale ~5-10 (encoder output, before
        # the kernel sees them) and a 0.6 lengthscale produces
        # near-degenerate Gram matrices on a 8 k-inducing-point set -- the
        # cholesky factor goes non-PSD even after adding 1e-6 jitter. This
        # is the LMC counterpart of the LinearMean zero-init fix that
        # protects the single-task SVGP from the same failure.
        with torch.no_grad():
            base.lengthscale = torch.full(
                base.lengthscale.shape,
                3.0,
                device=base.lengthscale.device,
                dtype=base.lengthscale.dtype,
            )

        self.spec = spec

    def forward(  # type: ignore[override]
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        """Per-latent prior. The LMC wrapper turns this into a
        ``MultitaskMultivariateNormal`` over ``num_tasks``.

        Input shape handling: GPyTorch's LMC stack broadcasts the input
        to ``(num_latents, B, D)`` before this forward is called, but the
        DKL encoder (``ResMLPEncoder`` + BatchNorm + Fourier features)
        only handles 2-D ``(B, D)`` input. We flatten across the latent
        batch axis, encode once (the encoder is shared across tasks --
        that is the LMC convention -- so encoding per-latent would just
        recompute identical features), then reshape back so the
        per-latent kernel + mean operate on the correct ``(num_latents,
        B, n_encoded)`` shape.
        """
        if x.dim() > 2:
            orig_leading = x.shape[:-1]
            z_flat = self.encoder(x.reshape(-1, x.shape[-1]))
            z = z_flat.reshape(*orig_leading, z_flat.shape[-1])
        else:
            z = self.encoder(x)
        mean = self.mean_module(z)
        covar = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ----------------------------------------------------------------------------
# Masked log-likelihood
# ----------------------------------------------------------------------------


def masked_multitask_log_prob(
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    posterior: gpytorch.distributions.MultitaskMultivariateNormal,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-row, per-task masked log-likelihood.

    Standard ``MultitaskGaussianLikelihood.expected_log_prob`` requires
    every (row, task) pair to have an observation. Real KuniJiban data
    has 19% missing groundwater readings. To train on the incomplete
    pairs without imputation, we instead compute the log-likelihood
    element-wise from the predictive mean and variance and zero out the
    contribution of masked entries.

    Args:
        likelihood: configured with ``num_tasks=T`` and per-task
            (or shared) noise.
        posterior: ``MultitaskMultivariateNormal`` from the LMC head.
            Shape: batch (B, T) mean + (T, B, B) or (B, T) variances
            depending on the cov structure.
        targets: (B, T) observed values. Where ``target_mask[i, t]``
            is False this entry is ignored; pass any finite filler
            (e.g. 0.0) to keep autograd happy.
        target_mask: (B, T) bool tensor; True = observed.

    Returns:
        Scalar log-likelihood summed over observed entries and divided
        by the number of observed entries (so the magnitude does not
        scale with the fraction of valid rows). Compatible with
        ``VariationalELBO`` data fit term.
    """
    if targets.shape != target_mask.shape:
        raise ValueError(
            f"targets {tuple(targets.shape)} and target_mask "
            f"{tuple(target_mask.shape)} must have the same shape"
        )
    # ``MultitaskGaussianLikelihood(predictive)`` returns a Gaussian whose
    # mean / variance are (B, T) per-task. Add noise here so the
    # likelihood term matches the standard ELBO data term.
    noisy = likelihood(posterior)
    mean = noisy.mean  # (B, T)
    var = noisy.variance.clamp_min(1e-8)  # (B, T)
    log_prob = -0.5 * (
        torch.log(2 * torch.pi * var) + (targets - mean) ** 2 / var
    )  # (B, T)
    masked = log_prob * target_mask.to(log_prob.dtype)
    n_obs = target_mask.sum().clamp_min(1)
    return masked.sum() / n_obs


# ----------------------------------------------------------------------------
# Convenience: end-to-end model wrapper
# ----------------------------------------------------------------------------


class LMCModel(nn.Module):
    """Encoder + LMC SVGP head packaged for training / saving.

    Mirrors the layout of :class:`national.models.foundation.FoundationModel`
    at a smaller scope: no FiLM, no NoiseHead, no censored likelihood --
    Paper B' Pillar 2 first iteration targets the joint point-estimate
    + calibrated uncertainty headline. Those extensions can graft on
    later (the ``LMCVariationalStrategy`` outputs a fully differentiable
    MVN, so FiLM-style task biasing is a 5-line addition).
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        encoder: nn.Module,
        spec: LMCSpec,
    ) -> None:
        super().__init__()
        self.svgp = _LMCApproximateGP(inducing_points, encoder, spec)
        self.spec = spec

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultitaskMultivariateNormal:
        return self.svgp(x)

    @property
    def num_tasks(self) -> int:
        return int(self.spec.num_tasks)


__all__ = [
    "LMCSpec",
    "LMCModel",
    "masked_multitask_log_prob",
]
