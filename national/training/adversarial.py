"""Adversarial coordinate critic (GRL-style) for encoder anti-decodability.

Context (NMI methods paper removal-mechanism bake-off): the soft-HSIC
deconfounding arm (``national.training.losses.hsic_regularizer``) achieved
statistical independence between the encoder output and (lat, lon) at the
kernel's resolvable scale, but out-of-region RMSE was unmoved -- the
hypothesis is that coordinate *information* remained decodable even though
the *dependence* HSIC measures (an RBF-kernel-scale statistic) had been
driven to ~0. Gradient-reversal / adversarial training (Ganin & Lempitsky
2015; Elazar & Goldberg 2018 for the NLP precedent that adversarial removal
is often *partial*) attacks decodability directly: a critic network learns
to regress raw (lat, lon) from the encoder representation, and the encoder
is trained to defeat it.

Implementation note -- DETACH-ALTERNATING, not a literal
``torch.autograd.Function`` gradient-reversal layer. The two are
mathematically equivalent (both implement the min-max game
``min_encoder max_critic -MSE(critic(phi), coords)``, i.e.
``min_encoder min_critic [MSE(critic(phi), coords) - MSE via encoder]``)
but the alternating-optimizer formulation is more numerically stable in
practice (no need to tune a single combined backward pass's gradient-scale
interaction between two objectives with opposite sign) and is easier to
audit step-by-step, which matters for a result this paper leans on. Three
building blocks:

1. :class:`CoordinateCritic` -- a plain MLP regressor ``z -> (lat, lon)``.
2. :func:`critic_step` -- one (or ``k``) critic-only optimizer step(s)
   minimizing ``MSE(critic(phi.detach()), coords_std)``. ``phi`` is
   detached, so no gradient reaches the encoder from this step.
3. :func:`encoder_adversarial_term` -- computes
   ``-eff_adv_lambda * MSE(critic(phi), coords_std)`` with the critic's
   parameters temporarily frozen (``requires_grad_(False)``), so that when
   the caller adds this term to the main loss and calls a single
   ``loss.backward()``, gradient flows ``phi -> encoder`` (maximizing the
   critic's error, i.e. minimizing coordinate decodability) WITHOUT also
   accumulating a (redundant, sign-conflicting) gradient on the critic's own
   parameters. The caller is responsible for re-enabling
   ``requires_grad_(True)`` on the critic after the main
   ``loss.backward()`` completes, so the next batch's :func:`critic_step`
   trains normally -- see ``FoundationTrainer.fit()``'s adversarial block
   for the exact per-batch ordering contract.

The critic is intentionally NOT part of ``FoundationModel``/``FoundationTrainer``'s
checkpointed module tree: it is an auxiliary "sparring partner" used only to
shape the encoder's *training signal*, not a piece of the deployed model.
See ``FoundationTrainer``'s docstring/comments at the adversarial block for
why this also sidesteps any checkpoint-format changes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Linear warmup length (epochs) for the adversarial encoder term, mirroring
# trainer.py's `_INDEP_WARMUP_EPOCHS` for the HSIC penalty. Exposed here so
# trainer.py can import a single shared default instead of hard-coding "5"
# in two places; `cfg.training.adv_warmup_epochs` can still override it.
_ADV_WARMUP_EPOCHS = 5


class CoordinateCritic(nn.Module):
    """MLP regressor ``z -> (lat_std, lon_std)``.

    Mirrors the hidden width/depth of
    ``national.evaluation.memorization_metric.MINEStatisticsNet`` (two
    hidden layers of width ``hidden``, SiLU activations, 3 Linear layers
    total) so the critic's capacity is comparable to the audit's MI
    statistics network -- but this is a plain regressor
    (``z -> 2`` standardized coordinates), not a MINE statistics net
    (``(z, coord) -> scalar score``). A regressor is the right shape for
    the detach-alternating min-max game: its loss (MSE against the true,
    per-batch-standardized coordinates) is directly interpretable as
    "how well can coordinates be decoded from z right now", with
    ``R^2 = 1 - MSE`` since the regression target is unit-variance.

    Weights use PyTorch's default ``nn.Linear`` initialization (Kaiming
    uniform), which draws from the GLOBAL ``torch`` RNG. Callers must NOT
    call ``torch.manual_seed()`` around critic construction to force a
    particular init -- doing so mid-training would reset the global RNG
    stream that the training ``DataLoader``'s shuffling and any dropout
    layers also depend on, silently perturbing the *encoder's* training
    trajectory as a side effect of an unrelated auxiliary network's init
    (this is the exact trap flagged in
    ``national.evaluation.memorization_audit.py``'s MINE estimator, which
    deliberately calls ``torch.manual_seed`` at an audit-time boundary
    where that is safe -- mid-training it is not). Accepting whatever the
    already-seeded global stream provides at construction time keeps the
    whole run reproducible given the top-level training seed, with no
    special-casing needed.
    """

    def __init__(self, z_dim: int, hidden: int = 128, coord_dim: int = 2) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.coord_dim = int(coord_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, coord_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def standardize_coords(
    coords: torch.Tensor, *, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Per-batch standardization of raw coordinates.

    Mirrors ``hsic_regularizer``'s ``coords_n`` computation in
    ``national.training.losses`` verbatim (same rationale: raw lat/lon
    degrees live on a very different numeric scale than an MSE loss
    against encoder-scale features, and a per-column affine map cannot
    itself introduce or remove predictability -- it only puts the
    regression target on a scale where ``R^2 = 1 - MSE`` holds cleanly).

    Args:
        coords: raw coordinate columns, shape ``(B, C)``.
        dtype: if given, cast ``coords`` to this dtype before computing
            the per-column mean/std (so the result matches the encoder
            output's dtype -- relevant on MPS, where the trainer may force
            float32 globally for MPS Cholesky compatibility).

    Returns:
        Standardized coordinates, shape ``(B, C)``, mean ~0 std ~1 per
        column within THIS batch.
    """
    if dtype is not None:
        coords = coords.to(dtype=dtype)
    mean = coords.mean(dim=0, keepdim=True)
    std = coords.std(dim=0, keepdim=True).clamp_min(1e-8)
    return (coords - mean) / std


def critic_step(
    critic: CoordinateCritic,
    critic_optimizer: torch.optim.Optimizer,
    phi: torch.Tensor,
    coords_std: torch.Tensor,
    *,
    k: int = 1,
) -> float:
    """Run ``k`` critic-only optimizer step(s) on this batch.

    Each step minimizes ``MSE(critic(phi.detach()), coords_std)``.
    ``phi`` is detached BEFORE being fed to the critic, so no gradient
    reaches the encoder from this function -- only ``critic``'s own
    parameters are updated (``critic_optimizer`` must be constructed with
    ``critic.parameters()`` only; see ``FoundationTrainer.__init__``'s
    lazy critic-optimizer construction).

    With ``k > 1`` all steps reuse the SAME (already detached) ``phi`` /
    ``coords_std`` pair -- there is no additional encoder forward pass to
    draw a fresh mini-batch from, since the whole point of caching ``phi``
    (see ``_DKLApproximateGP.last_batch_encoded``) is that the encoder runs
    exactly once per optimizer step. This is an approximation of "train the
    critic near-optimality per encoder step" (the WGAN-style prescription)
    using repeated passes over one batch rather than several distinct
    batches; ``k=1`` (the default) is the standard single-step alternation.

    Args:
        critic: the coordinate critic.
        critic_optimizer: optimizer over ``critic.parameters()`` ONLY.
        phi: cached encoder output for this batch, WITH its autograd graph
            (detached internally -- callers do not need to detach first).
        coords_std: per-batch-standardized target coordinates, shape
            ``(B, coord_dim)``.
        k: number of critic-only steps to run on this batch. Default 1.

    Returns:
        The LAST step's critic MSE as a plain Python float (no grad) --
        used for the per-epoch ``critic_mse_mean`` / ``critic_r2``
        diagnostic. When ``k == 1`` this is the only value computed.
    """
    phi_detached = phi.detach()
    last_mse = float("nan")
    for _ in range(max(1, int(k))):
        critic_optimizer.zero_grad(set_to_none=True)
        pred = critic(phi_detached)
        loss = F.mse_loss(pred, coords_std)
        loss.backward()
        critic_optimizer.step()
        last_mse = float(loss.detach().item())
    return last_mse


def encoder_adversarial_term(
    critic: CoordinateCritic,
    phi: torch.Tensor,
    coords_std: torch.Tensor,
    eff_adv_lambda: float,
) -> torch.Tensor:
    """``-eff_adv_lambda * MSE(critic(phi), coords_std)``, critic FROZEN.

    ``phi`` is NOT detached here -- gradient must flow ``phi -> encoder``
    so that adding this (negative) term to the encoder's task loss and
    calling ``loss.backward()`` adversarially pushes the encoder to
    INCREASE the critic's error (i.e. decrease coordinate decodability).

    The critic's own parameters are frozen (``requires_grad_(False)``) for
    the duration of this call so that the caller's single combined
    ``loss.backward()`` does not also write a gradient into
    ``critic.parameters()`` -- that gradient would have the OPPOSITE sign
    intent from :func:`critic_step`'s update (which wants the critic to
    get BETTER at decoding, not worse), so accumulating it here would be
    at best redundant and at worst directly fight the critic's own
    training step for this batch.

    Ordering contract (see ``FoundationTrainer.fit()`` for the concrete
    per-batch sequence this is embedded in):

    1. :func:`critic_step` has already run on this batch (critic trained,
       ``critic_optimizer.step()`` already called).
    2. This function is called: freezes critic params, computes the term.
    3. Caller adds the returned term to the main ``loss``.
    4. Caller calls ``loss.backward()`` ONCE (covers the ELBO + any
       HSIC/MMD penalty + this adversarial term in a single graph).
    5. Caller calls the MAIN optimizer's ``optimizer.step()`` (encoder +
       GP + likelihood params -- NOT the critic; the critic is not a
       member of that optimizer's parameter group).
    6. Caller re-enables ``critic.requires_grad_(True)`` so step 1 works
       again on the NEXT batch.

    Freezing happens here (step 2) rather than immediately after step 1's
    ``critic_optimizer.step()`` only because this function IS step 2 --
    the two are the same moment. Unfreezing must happen no earlier than
    after step 4 (the backward pass that this freeze protects) but MAY
    happen either right after step 4 or after step 5; this implementation
    documents doing it after step 5 purely so a single "did we add an adv
    term this batch" flag governs both the loss-add and the unfreeze,
    rather than needing two independent conditionals.

    Args:
        critic: the coordinate critic (already critic-step-updated this
            batch).
        phi: cached encoder output for this batch, autograd-attached.
        coords_std: per-batch-standardized target coordinates.
        eff_adv_lambda: the WARMED-UP lambda (see ``_ADV_WARMUP_EPOCHS`` /
            trainer.py's ``eff_adv_lambda`` computation) -- already ramped,
            not the raw ``adv_lambda`` CLI value.

    Returns:
        A scalar tensor (part of the encoder's autograd graph via ``phi``).
    """
    for p in critic.parameters():
        p.requires_grad_(False)
    pred = critic(phi)
    mse = F.mse_loss(pred, coords_std)
    return -float(eff_adv_lambda) * mse


__all__ = [
    "CoordinateCritic",
    "standardize_coords",
    "critic_step",
    "encoder_adversarial_term",
]
