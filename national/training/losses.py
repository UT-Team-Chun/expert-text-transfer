"""SVGP losses and regularizers.

Two objectives are exposed:

- :func:`elbo_with_regime_weights` -- SVGP variational lower bound, unbiased
  on shuffled mini-batches, with optional per-sample weights used to up-weight
  rare geological regimes. The KL term is divided by the full dataset size
  ``num_data`` so the per-step gradient magnitude is invariant to batch size
  (matching GPyTorch's ``VariationalELBO`` convention).

- :func:`mmd_regularizer` -- Maximum Mean Discrepancy between the encoder
  output distribution on a batch and a reference distribution (typically a
  fixed unit-Gaussian draw). It penalizes encoder collapse without forcing a
  specific shape on the learned features.

- :func:`hsic_regularizer` -- Hilbert-Schmidt Independence Criterion between
  the encoder output and the raw (lat, lon) coordinates. Minimizing it
  discourages the encoder from re-deriving an absolute-position "lookup
  table" (which would blow up out-of-region generalization) while still
  allowing it to use coordinates indirectly through covariates.
"""

from __future__ import annotations

import gpytorch
import torch


def elbo_with_regime_weights(
    predictive_dist: gpytorch.distributions.MultivariateNormal,
    likelihood: gpytorch.likelihoods.Likelihood,
    y: torch.Tensor,
    *,
    num_data: int,
    model: gpytorch.models.ApproximateGP,
    sample_weights: torch.Tensor | None = None,
    beta: float = 1.0,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted SVGP loss (negative ELBO) suitable for an Adam minimizer.

    Mathematically the unbiased mini-batch ELBO is::

        ELBO = (N / B) * sum_b w_b * E_q[ log p(y_b | f_b) ] - beta * KL(q(u) || p(u))

    where ``B`` is the batch size, ``N`` is ``num_data``, ``w_b`` is a per-
    sample weight (default 1), and ``beta`` is the KL temperature. The loss
    returned is ``-ELBO / N`` so the magnitude is comparable across dataset
    sizes.

    Args:
        predictive_dist: ``model(x_batch)``.
        likelihood: Gaussian likelihood used to integrate out ``f``.
        y: target tensor of shape ``(B,)``.
        num_data: total dataset size (NOT the batch size).
        model: the SVGP module -- used to read the variational KL.
        sample_weights: optional ``(B,)`` non-negative weights.
        beta: KL temperature.
    """
    # Gaussian + FixedNoiseGaussian have closed-form expected_log_prob;
    # StudentTLikelihood (and other _OneDimensionalLikelihood subclasses)
    # fall back to Gauss-Hermite quadrature internally, which is a drop-in
    # replacement at this call site. CensoredGaussianLikelihood inherits
    # GaussianLikelihood and overrides expected_log_prob to right-censor
    # at a configurable cap. We allow all four explicitly and reject
    # anything else to flag accidental new-likelihood wiring.
    if not isinstance(
        likelihood,
        (
            gpytorch.likelihoods.GaussianLikelihood,
            gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
            gpytorch.likelihoods.StudentTLikelihood,
        ),
    ):
        raise TypeError(
            "elbo_with_regime_weights supports Gaussian / FixedNoiseGaussian "
            "/ StudentT / CensoredGaussian likelihoods only; got "
            f"{type(likelihood).__name__}."
        )
    if y.dim() != 1:
        raise ValueError(f"y must be 1-D, got shape {tuple(y.shape)}")
    batch_size = y.shape[0]
    if batch_size == 0:
        raise ValueError("Empty batch.")
    if num_data <= 0:
        raise ValueError(f"num_data must be positive, got {num_data}")

    if sample_weights is None:
        weights = y.new_ones(batch_size)
    else:
        if sample_weights.shape != y.shape:
            raise ValueError(
                f"sample_weights shape {tuple(sample_weights.shape)} != y shape "
                f"{tuple(y.shape)}"
            )
        weights = sample_weights.to(dtype=y.dtype, device=y.device)
        if (weights < 0).any():
            raise ValueError("sample_weights must be non-negative.")

    # FixedNoiseGaussianLikelihood needs the per-point noise tensor passed
    # at every call. GaussianLikelihood ignores the kwarg.
    if noise is not None:
        per_point_ll = likelihood.expected_log_prob(y, predictive_dist, noise=noise)
    else:
        per_point_ll = likelihood.expected_log_prob(y, predictive_dist)
    weighted_mean_ll = (weights * per_point_ll).sum() / weights.sum().clamp_min(1e-12)
    data_term = num_data * weighted_mean_ll

    kl = model.variational_strategy.kl_divergence().sum()

    elbo = data_term - beta * kl
    return -elbo / num_data


def mmd_regularizer(
    phi: torch.Tensor,
    phi_ref: torch.Tensor,
    *,
    sigmas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    """Multi-bandwidth squared MMD between ``phi`` and ``phi_ref``.

    Uses a mixture of RBF kernels with the given bandwidths. The reference
    sample is typically drawn from ``N(0, I)`` of the same dimension as the
    encoder output. The penalty is non-negative and may be minimized directly.

    Args:
        phi: encoder outputs, shape ``(B, D)``.
        phi_ref: reference sample, shape ``(M, D)``.
        sigmas: RBF bandwidths to average over.
    """
    if phi.dim() != 2 or phi_ref.dim() != 2:
        raise ValueError(
            f"phi and phi_ref must be 2-D; got shapes {tuple(phi.shape)} and "
            f"{tuple(phi_ref.shape)}"
        )
    if phi.shape[-1] != phi_ref.shape[-1]:
        raise ValueError(
            f"phi feature dim {phi.shape[-1]} != phi_ref feature dim {phi_ref.shape[-1]}"
        )

    def _pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return (diff * diff).sum(dim=-1)

    dxx = _pairwise_sq_dist(phi, phi)
    dyy = _pairwise_sq_dist(phi_ref, phi_ref)
    dxy = _pairwise_sq_dist(phi, phi_ref)

    out = phi.new_zeros(())
    for sigma in sigmas:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        gamma = 1.0 / (2.0 * sigma * sigma)
        out = out + (
            torch.exp(-gamma * dxx).mean()
            + torch.exp(-gamma * dyy).mean()
            - 2.0 * torch.exp(-gamma * dxy).mean()
        )
    return out / len(sigmas)


def hsic_regularizer(
    z: torch.Tensor,
    coords: torch.Tensor,
    *,
    max_batch: int = 4096,
) -> torch.Tensor:
    """Biased V-statistic HSIC between encoder features ``z`` and ``coords``.

    Implements the classic Gretton et al. (2005) biased estimator::

        HSIC_b(Z, C) = trace(K H L H) / (B - 1)^2

    where ``K`` is the RBF Gram matrix of ``z``, ``L`` is the RBF Gram
    matrix of (batch-standardized) ``coords``, ``H = I - (1/B) 1 1^T`` is
    the centering matrix, and ``B`` is the (possibly subsampled) batch size.

    We deliberately use the BIASED estimator, not the unbiased U-statistic.
    The biased estimator is provably non-negative (it is an MMD-in-feature-
    space quantity), so minimizing it drives dependence toward zero and
    stops there. The unbiased estimator can go slightly *negative* for
    finite samples; an optimizer minimizing it would happily exploit that
    finite-sample noise and drive it arbitrarily negative instead of
    actually decorrelating the encoder output from the coordinates.

    The trace is NOT computed by forming ``K @ H @ L @ H`` (an O(B^3) matmul
    chain that would dominate every training step once ``max_batch`` is in
    the thousands). Instead we use the standard algebraic identity::

        trace(K H L H) = trace(K L) - (2 / B) * (K @ 1) . (L @ 1)
                         + (1 / B^2) * sum(K) * sum(L)

    which only needs an elementwise product and two matrix-vector products,
    i.e. O(B^2) work -- the same asymptotic cost as materializing ``K`` and
    ``L`` in the first place.

    Args:
        z: encoder outputs, shape ``(B, D)``. Gradients flow through this
            argument -- that is the whole point, since ``z`` is what
            training adjusts to reduce coordinate leakage.
        coords: raw spatial columns (e.g. ``[lat, lon]``), shape ``(B, C)``.
            Gradients are not assumed to flow through this argument; in the
            trainer it is normally a detached slice of the input batch.
        max_batch: if the batch exceeds this many rows, a deterministic
            contiguous prefix of ``max_batch`` rows is used instead. Both
            ``K`` and ``L`` are dense ``B x B`` matrices, so an unbounded
            batch size risks an OOM on MPS (same rationale as the
            n_inducing cap used on MPS). The prefix is RNG-free
            (not a random draw) so the penalty is reproducible given the
            same batch tensor; it is not a biased spatial sample because
            the caller's DataLoader has already shuffled row order.

    Returns:
        A non-negative 0-dim tensor.
    """
    if z.dim() != 2 or coords.dim() != 2:
        raise ValueError(
            f"z and coords must be 2-D; got shapes {tuple(z.shape)} and {tuple(coords.shape)}"
        )
    if z.shape[0] != coords.shape[0]:
        raise ValueError(f"z batch size {z.shape[0]} != coords batch size {coords.shape[0]}")

    batch_size = z.shape[0]
    if batch_size < 4:
        raise ValueError(
            "hsic_regularizer needs a batch size >= 4 to form a centered "
            f"Gram matrix pair; got {batch_size}"
        )

    if batch_size > max_batch:
        z = z[:max_batch]
        coords = coords[:max_batch]
        batch_size = z.shape[0]

    # Standardize coords within the batch: L's median-heuristic bandwidth
    # (like K's) is scale-sensitive, and raw lat/lon degrees live on a very
    # different numeric scale than typical encoder outputs. Standardizing
    # is a per-column affine map, so it cannot itself introduce or remove
    # statistical dependence between coords and z -- it only puts the two
    # kernels on comparable footing.
    coords = coords.to(dtype=z.dtype)
    coords_mean = coords.mean(dim=0, keepdim=True)
    coords_std = coords.std(dim=0, keepdim=True).clamp_min(1e-8)
    coords_n = (coords - coords_mean) / coords_std

    def _pairwise_sq_dist(a: torch.Tensor) -> torch.Tensor:
        diff = a.unsqueeze(1) - a.unsqueeze(0)
        return (diff * diff).sum(dim=-1)

    dz = _pairwise_sq_dist(z)
    dc = _pairwise_sq_dist(coords_n)

    def _median_heuristic_gamma(sq_dists: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(sq_dists.shape[0], dtype=torch.bool, device=sq_dists.device)
        median_sq = torch.median(sq_dists[~eye]).clamp_min(1e-12)
        # .detach(): the bandwidth is treated as a fixed data statistic, not
        # a learnable quantity. If gradients flowed through the median here,
        # the encoder could cheat by shrinking z's pairwise spread so
        # gamma_z grows and K saturates toward a near-constant matrix
        # (HSIC -> 0) without any real reduction in statistical dependence.
        # Detaching does NOT make the forward value scale-invariant -- that
        # property (for the coords side) comes from the per-batch
        # standardization applied to `coords` above, which holds regardless
        # of whether the bandwidth is detached. What detaching buys us is a
        # stable *backward* pass: it keeps median() -- a non-smooth,
        # subgradient-only op -- out of the gradient graph, so the bandwidth
        # behaves as a fixed hyperparameter-like scale reference rather than
        # a quantity the optimizer can (mis)train through.
        return (1.0 / median_sq).detach()

    gamma_z = _median_heuristic_gamma(dz)
    gamma_c = _median_heuristic_gamma(dc)

    K = torch.exp(-gamma_z * dz)
    L = torch.exp(-gamma_c * dc)

    b = float(batch_size)
    k_row_sums = K.sum(dim=1)
    l_row_sums = L.sum(dim=1)
    trace_kl = (K * L).sum()
    cross_term = torch.dot(k_row_sums, l_row_sums)

    trace_khlh = trace_kl - (2.0 / b) * cross_term + (1.0 / (b * b)) * k_row_sums.sum() * l_row_sums.sum()
    hsic = trace_khlh / ((b - 1.0) ** 2)
    # The identity above is exact, but floating-point cancellation can push
    # a near-zero (genuinely independent) estimate a hair below 0. Clamp so
    # the documented non-negativity contract always holds numerically.
    return hsic.clamp_min(0.0)


__all__ = ["elbo_with_regime_weights", "mmd_regularizer", "hsic_regularizer"]
