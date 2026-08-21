"""Regime-balanced minibatch sampler.

Addresses Risk #2 (SVGP underfit on rare regimes): alluvial dominates the
KuniJiban corpus while volcanic-ash / limestone are 2-3 orders of magnitude
rarer (in Kanto: alluvial ~332k rows vs limestone 3, volcanic-ash 98), so
uniform minibatching almost never shows the FiLM-modulated head a rare-regime
point. This sampler up-weights rare regimes with a *tempered* inverse-frequency
weight so balance does not collapse into repeating a 3-row regime thousands of
times per epoch (which would overfit those few points).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


class RegimeBalancedSampler(Sampler[int]):
    """Weighted (with-replacement) sampler keyed by per-row regime code.

    Per-row weight is ``(1 / count[regime])**alpha`` normalised over all rows:

    - ``alpha = 0`` -> uniform (equivalent to shuffling with replacement);
    - ``alpha = 1`` -> full inverse-frequency (every regime equally likely);
    - ``alpha = 0.5`` (default) -> sqrt temper, a middle ground that lifts rare
      regimes substantially without pathological oversampling.

    Args:
        regimes: per-row integer regime codes (length = dataset size).
        alpha: temper exponent in ``[0, 1]``.
        num_samples: samples drawn per epoch (defaults to the dataset size, so
            ELBO ``num_data`` accounting is unchanged).
        seed: RNG seed; each ``__iter__`` (epoch) draws a fresh permutation from
            the same generator, so epochs differ but the run is reproducible.
    """

    def __init__(
        self,
        regimes: Sequence[int] | np.ndarray,
        *,
        alpha: float = 0.5,
        num_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        regimes = np.asarray(regimes).reshape(-1)
        if regimes.size == 0:
            raise ValueError("regimes must be non-empty.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        _, inverse, counts = np.unique(
            regimes, return_inverse=True, return_counts=True
        )
        per_row_count = counts[inverse].astype(np.float64)
        weights = (1.0 / per_row_count) ** float(alpha)
        weights /= weights.sum()

        self._weights = torch.as_tensor(weights, dtype=torch.double)
        self._num_samples = int(num_samples) if num_samples is not None else int(regimes.size)
        self._generator = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        idx = torch.multinomial(
            self._weights, self._num_samples, replacement=True, generator=self._generator
        )
        yield from idx.tolist()


__all__ = ["RegimeBalancedSampler"]
