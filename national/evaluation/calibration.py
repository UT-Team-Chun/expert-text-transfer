"""Calibration metrics for predictive uncertainty.

For a probabilistic regressor producing Gaussian marginals ``N(y_pred, y_std)``,
calibration measures whether empirical coverage matches nominal coverage. A
well-calibrated model with ``alpha=0.95`` predictive intervals will contain
the true value ~95% of the time. Mis-calibration shows up as systematic
gap between empirical and nominal coverage at multiple alpha levels.

``TemperatureScaler`` provides the standard post-hoc fix: rescale the
predictive std by a single positive scalar ``T`` so the average squared
Z-residual equals 1. This is the closed-form Gaussian NLL minimizer over
all rescalings ``sigma -> T * sigma``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


def coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    alpha: float,
) -> float:
    """Empirical coverage of a symmetric central interval at level ``alpha``.

    Returns the fraction of points whose true value falls within
    ``y_pred ± z * y_std`` where ``z = Φ^{-1}((1+alpha)/2)``.

    Args:
        y_true / y_pred / y_std: 1-D arrays of equal length.
        alpha: nominal coverage in ``(0, 1)``.

    Returns:
        Fraction in ``[0, 1]``.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_std = np.asarray(y_std)
    if not (y_true.shape == y_pred.shape == y_std.shape):
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}, "
            f"y_std={y_std.shape}"
        )
    if (y_std <= 0).any():
        raise ValueError("y_std must be strictly positive.")

    z = float(norm.ppf(0.5 * (1.0 + alpha)))
    lo = y_pred - z * y_std
    hi = y_pred + z * y_std
    return float(((y_true >= lo) & (y_true <= hi)).mean())


def reliability_diagram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    alphas: Sequence[float] = (0.5, 0.8, 0.95),
) -> pd.DataFrame:
    """Reliability table over multiple nominal coverage levels.

    Columns: ``alpha``, ``nominal``, ``empirical``, ``gap``
    (= empirical - nominal). A positive gap means the model is over-cautious
    (intervals are too wide); negative means over-confident.
    """
    rows = []
    for a in alphas:
        emp = coverage(y_true, y_pred, y_std, a)
        rows.append({"alpha": float(a), "nominal": float(a), "empirical": emp, "gap": emp - a})
    return pd.DataFrame(rows)


@dataclass
class TemperatureScaler:
    """Single-scalar post-hoc rescaling of Gaussian predictive std.

    Given pointwise marginals ``N(y_pred, y_std)`` that systematically over-
    or under-cover the truth, find a positive scalar ``T`` so that
    ``N(y_pred, T * y_std)`` is empirically calibrated. The maximum-
    likelihood solution under the Gaussian NLL is closed-form:

        T = sqrt(mean(((y_true - y_pred) / y_std) ** 2))

    Interpretation:
        - ``T == 1.0`` -> already calibrated, no change needed.
        - ``T < 1.0``  -> the raw std is too wide (over-cautious); tighten.
        - ``T > 1.0``  -> the raw std is too narrow (over-confident); widen.

    All ``alpha`` levels share the same ``T``. Per-alpha temperatures are
    avoidable here because Gaussian quantiles are linear in std.
    """

    T: float | None = None

    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
    ) -> "TemperatureScaler":
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_std = np.asarray(y_std, dtype=np.float64)
        if not (y_true.shape == y_pred.shape == y_std.shape):
            raise ValueError(
                f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}, "
                f"y_std={y_std.shape}"
            )
        if (y_std <= 0).any():
            raise ValueError("y_std must be strictly positive.")
        z = (y_true - y_pred) / y_std
        self.T = float(np.sqrt(np.mean(z * z)))
        return self

    def apply(self, y_std: np.ndarray) -> np.ndarray:
        if self.T is None:
            raise RuntimeError("TemperatureScaler is not fitted; call fit() first.")
        return np.asarray(y_std, dtype=np.float64) * self.T

    def fit_apply(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Convenience: fit on the same inputs, return (T, scaled_std)."""
        self.fit(y_true, y_pred, y_std)
        return float(self.T), self.apply(y_std)


@dataclass
class ConformalCalibrator:
    """Split conformal prediction with locally-adaptive (normalized) scores.

    For each calibration point compute the absolute normalized residual
    ``s_i = |y_true_i - y_pred_i| / y_std_i``. For a desired nominal level
    ``alpha``, the conformal radius is the
    ``ceil((n_cal + 1) * alpha) / n_cal``-quantile of ``s``. The
    test-time interval at the same alpha is then
    ``y_pred ± q_alpha * y_std``.

    This gives a distribution-free coverage guarantee *under exchangeability
    of the residuals*. In particular it survives the kurtosis-9 heavy-tail
    failure mode that broke :class:`TemperatureScaler` (which assumed a
    Gaussian z-distribution).

    Use cases here:
        - α=0.50 over-cautious: ``q_0.5`` will be much smaller than the
          Gaussian z=0.674, tightening the interval.
        - α=0.95 already calibrated: ``q_0.95`` ≈ z=1.96, so the interval
          stays similar.

    Notes:
        - ``alphas`` must be specified at ``fit`` time (one quantile per α).
        - Test-time evaluation uses the *same* α; calling with an unfitted
          α raises.
        - Per-point ``y_std`` provides "local adaptivity": the interval
          widens where the model itself is uncertain.
    """

    quantiles: dict[float, float] | None = None
    n_cal: int | None = None
    # Mondrian (group-conditional) extension. ``quantiles_per_group`` maps a
    # hashable group label -> {alpha: quantile}. ``n_cal_per_group`` keeps the
    # per-group calibration-set size so we can decide when to fall back to the
    # marginal quantile (Romano et al. 2020, Mondrian conformal prediction).
    quantiles_per_group: dict | None = None
    n_cal_per_group: dict | None = None
    min_group_n: int = 30
    # Weighted (covariate-shift) extension -- see fit_weighted / interval_weighted.
    # ``cal_scores_weighted`` are the |y-mu|/sigma nonconformity scores of the
    # calibration set; ``cal_weights_weighted`` are the caller-supplied
    # density ratios w(x_i) = dP_test/dP_cal at those same calibration points.
    cal_scores_weighted: np.ndarray | None = None
    cal_weights_weighted: np.ndarray | None = None

    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        alphas: Sequence[float] = (0.5, 0.8, 0.95),
    ) -> "ConformalCalibrator":
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_std = np.asarray(y_std, dtype=np.float64)
        if not (y_true.shape == y_pred.shape == y_std.shape):
            raise ValueError(
                f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}, "
                f"y_std={y_std.shape}"
            )
        if (y_std <= 0).any():
            raise ValueError("y_std must be strictly positive.")
        s = np.abs(y_true - y_pred) / y_std
        s_sorted = np.sort(s)
        n = len(s_sorted)
        quantiles: dict[float, float] = {}
        for a in alphas:
            if not 0 < a < 1:
                raise ValueError(f"alpha must be in (0, 1); got {a}")
            # Finite-sample correction (Lei et al. 2018):
            # k = ceil((n+1) * alpha) gives a valid (1-α)... wait, here we
            # use alpha as the *nominal coverage* (e.g. 0.5 means 50% CI),
            # so we need the alpha-quantile of |s|. Standard split conformal
            # gives valid coverage at level alpha by taking
            # ceil((n+1) * alpha)-th order statistic.
            k = int(np.ceil((n + 1) * a))
            k = min(max(k, 1), n)
            quantiles[float(a)] = float(s_sorted[k - 1])
        self.quantiles = quantiles
        self.n_cal = int(n)
        return self

    def interval(
        self,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.quantiles is None:
            raise RuntimeError("ConformalCalibrator is not fitted; call fit() first.")
        if float(alpha) not in self.quantiles:
            raise KeyError(
                f"alpha={alpha} not fitted (available: {sorted(self.quantiles)})"
            )
        q = self.quantiles[float(alpha)]
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_std = np.asarray(y_std, dtype=np.float64)
        return y_pred - q * y_std, y_pred + q * y_std

    def coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        alpha: float,
    ) -> float:
        lo, hi = self.interval(y_pred, y_std, alpha)
        y_true = np.asarray(y_true, dtype=np.float64)
        return float(((y_true >= lo) & (y_true <= hi)).mean())

    # ---- Mondrian (group-conditional) -----------------------------------
    def fit_mondrian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        groups: np.ndarray,
        alphas: Sequence[float] = (0.5, 0.8, 0.95),
        min_group_n: int = 30,
    ) -> "ConformalCalibrator":
        """Fit per-group quantiles of the normalised residuals.

        ``groups[i]`` is the discrete subgroup label of calibration row
        ``i`` (e.g. AIST regime code, depth bin, predicted-μ quintile).
        Groups with fewer than ``min_group_n`` calibration points use the
        marginal quantile at evaluation time. The marginal quantile is
        fitted alongside so callers can use ``interval_mondrian`` without
        first calling :meth:`fit`.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_std = np.asarray(y_std, dtype=np.float64)
        groups = np.asarray(groups)
        if not (y_true.shape == y_pred.shape == y_std.shape == groups.shape):
            raise ValueError(
                "Shape mismatch in fit_mondrian: "
                f"y_true={y_true.shape}, y_pred={y_pred.shape}, "
                f"y_std={y_std.shape}, groups={groups.shape}"
            )
        if (y_std <= 0).any():
            raise ValueError("y_std must be strictly positive.")
        # Fit marginal as fallback.
        self.fit(y_true, y_pred, y_std, alphas=alphas)
        s = np.abs(y_true - y_pred) / y_std
        unique_groups = np.unique(groups)
        per_group: dict = {}
        per_group_n: dict = {}
        for g in unique_groups:
            mask = groups == g
            n_g = int(mask.sum())
            per_group_n[g.item() if hasattr(g, "item") else g] = n_g
            if n_g < min_group_n:
                continue  # fall back to marginal at eval time
            s_g = np.sort(s[mask])
            q_g: dict[float, float] = {}
            for a in alphas:
                if not 0 < a < 1:
                    raise ValueError(f"alpha must be in (0, 1); got {a}")
                k = int(np.ceil((n_g + 1) * a))
                k = min(max(k, 1), n_g)
                q_g[float(a)] = float(s_g[k - 1])
            per_group[g.item() if hasattr(g, "item") else g] = q_g
        self.quantiles_per_group = per_group
        self.n_cal_per_group = per_group_n
        self.min_group_n = int(min_group_n)
        return self

    def interval_mondrian(
        self,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        groups: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Group-conditional symmetric interval ``y_pred ± q_group * y_std``.

        Test-time groups whose calibration count was below ``min_group_n``
        fall back to the marginal quantile.
        """
        if self.quantiles_per_group is None:
            raise RuntimeError(
                "ConformalCalibrator was not fit with Mondrian groups; "
                "call fit_mondrian() first."
            )
        if self.quantiles is None or float(alpha) not in self.quantiles:
            raise KeyError(
                f"alpha={alpha} not fitted at marginal level either "
                f"(available: {sorted(self.quantiles or {})})"
            )
        groups = np.asarray(groups)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_std = np.asarray(y_std, dtype=np.float64)
        q_marginal = float(self.quantiles[float(alpha)])
        q_per_row = np.full(y_pred.shape, q_marginal, dtype=np.float64)
        for i, g in enumerate(groups):
            key = g.item() if hasattr(g, "item") else g
            qg_dict = self.quantiles_per_group.get(key)
            if qg_dict is not None and float(alpha) in qg_dict:
                q_per_row[i] = float(qg_dict[float(alpha)])
        return y_pred - q_per_row * y_std, y_pred + q_per_row * y_std

    def coverage_mondrian(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        groups: np.ndarray,
        alpha: float,
    ) -> float:
        lo, hi = self.interval_mondrian(y_pred, y_std, groups, alpha)
        y_true = np.asarray(y_true, dtype=np.float64)
        return float(((y_true >= lo) & (y_true <= hi)).mean())

    # ---- Locally-weighted -----------------------------------------------
    @staticmethod
    def _weighted_quantile(values: np.ndarray, weights: np.ndarray,
                           alpha: float) -> float:
        """Weighted ``alpha``-quantile via cumulative sum.

        Uses the conformal finite-sample correction ``q = (n+1)*alpha`` only
        when weights are uniform; for general weights we take the smallest
        ``v`` such that the cumulative weight up to ``v`` is ``>= alpha``.
        """
        order = np.argsort(values)
        v_sorted = values[order]
        w_sorted = weights[order]
        w_total = w_sorted.sum()
        if w_total <= 0:
            return float(np.max(values))
        cumw = np.cumsum(w_sorted) / w_total
        idx = int(np.searchsorted(cumw, alpha, side="left"))
        idx = min(idx, len(v_sorted) - 1)
        return float(v_sorted[idx])

    def interval_locally_weighted(
        self,
        y_pred_test: np.ndarray,
        y_std_test: np.ndarray,
        cal_features: np.ndarray,
        cal_scores: np.ndarray,
        test_features: np.ndarray,
        alpha: float,
        bandwidth: float = 0.1,
        kernel: str = "gaussian",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Locally-weighted conformal interval (Lei et al. 2018 §5.2).

        For each test row ``j`` we compute a Gaussian-kernel weight over the
        calibration set, ``w_ij = exp(-||x*_j - x_i||^2 / (2 h^2))``, and
        return the weighted ``alpha``-quantile of the calibration scores
        ``s_i = |y_i - mu_i| / sigma_i`` as the per-row conformal radius.

        ``cal_features`` and ``test_features`` must live in the same
        (already-standardised) feature space; both the simple
        ``(lat, lon, depth, regime one-hot)`` space and the encoder-latent
        space ``encoder(x)`` are valid choices and yield different
        adaptivity behaviour --- which is the point of the comparison.
        """
        y_pred_test = np.asarray(y_pred_test, dtype=np.float64)
        y_std_test = np.asarray(y_std_test, dtype=np.float64)
        cal_features = np.asarray(cal_features, dtype=np.float64)
        cal_scores = np.asarray(cal_scores, dtype=np.float64)
        test_features = np.asarray(test_features, dtype=np.float64)
        if cal_features.shape[1] != test_features.shape[1]:
            raise ValueError(
                f"feature-dim mismatch: cal={cal_features.shape[1]} "
                f"vs test={test_features.shape[1]}"
            )
        if kernel != "gaussian":
            raise NotImplementedError(
                f"locally-weighted kernel={kernel!r} not implemented yet"
            )
        h2 = 2.0 * (float(bandwidth) ** 2)
        n_test = test_features.shape[0]
        q_per_row = np.empty(n_test, dtype=np.float64)
        # Stream over test rows so a 100k cal × 100k test outer product
        # does not blow memory. Each iteration is O(n_cal).
        for j in range(n_test):
            d2 = np.sum((cal_features - test_features[j]) ** 2, axis=1)
            w = np.exp(-d2 / h2)
            q_per_row[j] = self._weighted_quantile(cal_scores, w, float(alpha))
        return (
            y_pred_test - q_per_row * y_std_test,
            y_pred_test + q_per_row * y_std_test,
        )

    # ---- Weighted split-conformal under covariate shift ------------------
    #
    # Tibshirani, Foygel Barber, Candès & Ramdas (2019), "Conformal
    # Prediction Under Covariate Shift" (NeurIPS). Standard split-conformal
    # assumes the calibration and test points are exchangeable; under
    # covariate shift (dP_test/dP_cal = w(x) != 1) that fails, and the fix is
    # to re-weight each calibration score by the density ratio at that point
    # AND to add a point mass at +inf carrying the *test* point's own
    # (normalized) weight. That +inf atom is not optional bookkeeping -- it
    # is the finite-sample correction that makes the interval valid; if you
    # drop it (as :meth:`_weighted_quantile` above does, which is why that
    # helper is NOT reused here) the interval under-covers.
    #
    # Note on the ``alpha`` convention: matching :meth:`fit` throughout this
    # class, ``alpha`` is the *target coverage* (e.g. 0.95 for a 95% CI), so
    # we take the ``alpha``-quantile of the (weighted, augmented) score
    # distribution -- equivalently the ``(1 - significance)``-quantile with
    # ``significance = 1 - alpha`` in the original paper's notation.

    def fit_weighted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        weights: np.ndarray,
    ) -> "ConformalCalibrator":
        """Store calibration nonconformity scores + calibration weights.

        ``weights[i]`` is the density ratio ``w(x_i) = dP_test/dP_cal``
        evaluated at calibration point ``i``. The caller supplies these
        (e.g. from a fitted logistic-regression discriminator between
        calibration and target-region covariates, clipped to avoid
        blow-ups) -- this method does not estimate the density ratio
        itself.

        Unlike :meth:`fit`, no per-alpha quantile is pre-computed here: the
        test-point weight enters the +inf atom's mass, so the quantile is
        necessarily test-point-dependent and must be recomputed inside
        :meth:`interval_weighted`.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_std = np.asarray(y_std, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if not (y_true.shape == y_pred.shape == y_std.shape == weights.shape):
            raise ValueError(
                "Shape mismatch in fit_weighted: "
                f"y_true={y_true.shape}, y_pred={y_pred.shape}, "
                f"y_std={y_std.shape}, weights={weights.shape}"
            )
        if (y_std <= 0).any():
            raise ValueError("y_std must be strictly positive.")
        if (weights < 0).any():
            raise ValueError("weights (density ratios) must be non-negative.")
        if weights.sum() <= 0:
            raise ValueError("sum of calibration weights must be positive.")
        self.cal_scores_weighted = np.abs(y_true - y_pred) / y_std
        self.cal_weights_weighted = weights
        return self

    @staticmethod
    def _weighted_quantile_with_test_atom(
        cal_scores: np.ndarray,
        cal_weights: np.ndarray,
        weight_test: float,
        alpha: float,
    ) -> float:
        """Weighted ``alpha``-quantile of calibration scores AUGMENTED with a
        point mass at ``+inf`` carrying the test point's own weight.

        Normalize by the *combined* calibration + test weight mass::

            p_i   = w(x_i)      / (sum_j w(x_j) + w(x_test))   (per cal point)
            p_inf = w(x_test)   / (sum_j w(x_j) + w(x_test))

        and take the level-``alpha`` quantile of
        ``sum_i p_i * delta_{s_i}  +  p_inf * delta_{+inf}``.

        Sanity check (reduction to standard split-conformal): if every
        weight (calibration AND test) equals 1, ``p_i = p_inf = 1/(n+1)``
        and the smallest index ``i`` (0-indexed) with cumulative mass
        ``(i+1)/(n+1) >= alpha`` is exactly ``i = ceil((n+1)*alpha) - 1``,
        matching :meth:`fit`'s ``k = ceil((n_cal + 1) * alpha)`` order
        statistic -- the classic "+1" in the split-conformal finite-sample
        correction *is* the test point's own unit weight.

        Returns ``float('inf')`` when the ``+inf`` atom alone accounts for
        enough mass that the ``alpha``-quantile falls on it, i.e. when
        ``weight_test > (1 - alpha) * (sum(cal_weights) + weight_test)``.
        Under that extreme covariate shift the only honest interval is all
        of R.
        """
        cal_scores = np.asarray(cal_scores, dtype=np.float64)
        cal_weights = np.asarray(cal_weights, dtype=np.float64)
        if weight_test < 0:
            raise ValueError("weight_test must be non-negative.")
        total = cal_weights.sum() + weight_test
        if total <= 0:
            raise ValueError(
                "sum of calibration + test weights must be positive."
            )
        order = np.argsort(cal_scores)
        s_sorted = cal_scores[order]
        p_sorted = cal_weights[order] / total
        cumw = np.cumsum(p_sorted)
        # cumw[-1] == 1 - p_inf (finite mass). If the requested quantile
        # level exceeds the finite mass, it falls in the +inf atom.
        if cumw[-1] < alpha - 1e-12:
            return float("inf")
        idx = int(np.searchsorted(cumw, alpha, side="left"))
        idx = min(idx, len(s_sorted) - 1)
        return float(s_sorted[idx])

    def interval_weighted(
        self,
        y_pred_test: np.ndarray,
        y_std_test: np.ndarray,
        alpha: float,
        test_weights: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Weighted split-conformal interval under covariate shift.

        Requires :meth:`fit_weighted` to have been called first.

        Args:
            y_pred_test / y_std_test: model mean/std at the test points.
            alpha: target coverage (same convention as :meth:`fit`).
            test_weights: ``w(x_test) = dP_test/dP_cal`` at each test point
                (a density ratio the caller supplies, e.g. from the same
                discriminator used for the calibration weights). May be a
                scalar (broadcasts to every test row) or a per-row array.

        Returns:
            ``(lower, upper)`` arrays matching ``y_pred_test``'s shape.
            Rows where the test point's own weight is large enough that the
            ``+inf`` atom dominates the ``alpha``-quantile get
            ``lower=-inf, upper=+inf`` -- the honest interval under extreme
            covariate shift is the whole real line, not a silently-too-
            narrow band.
        """
        if self.cal_scores_weighted is None or self.cal_weights_weighted is None:
            raise RuntimeError(
                "ConformalCalibrator has no weighted calibration set; "
                "call fit_weighted() first."
            )
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")

        y_pred_test = np.asarray(y_pred_test, dtype=np.float64)
        y_std_test = np.asarray(y_std_test, dtype=np.float64)
        if y_pred_test.shape != y_std_test.shape:
            raise ValueError(
                f"Shape mismatch: y_pred_test={y_pred_test.shape}, "
                f"y_std_test={y_std_test.shape}"
            )
        scalar_input = y_pred_test.ndim == 0
        y_pred_flat = np.atleast_1d(y_pred_test)
        y_std_flat = np.atleast_1d(y_std_test)
        w_test_flat = np.broadcast_to(
            np.atleast_1d(np.asarray(test_weights, dtype=np.float64)),
            y_pred_flat.shape,
        )
        if (w_test_flat < 0).any():
            raise ValueError("test_weights must be non-negative.")

        lo = np.empty_like(y_pred_flat)
        hi = np.empty_like(y_pred_flat)
        for j in range(y_pred_flat.shape[0]):
            q = self._weighted_quantile_with_test_atom(
                self.cal_scores_weighted,
                self.cal_weights_weighted,
                float(w_test_flat[j]),
                float(alpha),
            )
            if np.isinf(q):
                lo[j] = -np.inf
                hi[j] = np.inf
            else:
                lo[j] = y_pred_flat[j] - q * y_std_flat[j]
                hi[j] = y_pred_flat[j] + q * y_std_flat[j]

        if scalar_input:
            return float(lo[0]), float(hi[0])
        return lo, hi

    def coverage_weighted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        alpha: float,
        test_weights: np.ndarray | float,
    ) -> float:
        """Empirical coverage of :meth:`interval_weighted` (``+inf`` bounds
        trivially cover any finite ``y_true``)."""
        lo, hi = self.interval_weighted(y_pred, y_std, alpha, test_weights)
        y_true = np.asarray(y_true, dtype=np.float64)
        lo = np.atleast_1d(np.asarray(lo, dtype=np.float64))
        hi = np.atleast_1d(np.asarray(hi, dtype=np.float64))
        return float(((y_true >= lo) & (y_true <= hi)).mean())


@dataclass
class IsotonicCalibrator:
    """Isotonic recalibration of predictive CDFs (Kuleshov et al. 2018).

    For each calibration point compute the predicted CDF value
    ``p_i = Phi((y_true_i - y_pred_i) / y_std_i)``. If the predictive
    distribution is calibrated these ``{p_i}`` follow Uniform(0, 1).
    We fit an isotonic regression ``R: [0, 1] -> [0, 1]`` from the
    *sorted* ``p_i`` to their empirical ranks so that ``R(p)`` is the
    *true* probability mass to the left of the nominal quantile ``p``.

    Test-time symmetric interval at nominal coverage ``alpha`` is built by
    inverting ``R`` at the two cutoff probabilities ``(1 - alpha) / 2`` and
    ``(1 + alpha) / 2`` to get the *nominal* quantile probabilities, then
    mapping them back through ``Phi^{-1}`` to z-scores and finally to
    ``y_pred ± z * y_std``.

    Unlike :class:`TemperatureScaler` (single scalar) this can correct any
    monotonic miscalibration shape — including the heavy-tail / narrow-bulk
    pattern we observed (kurtosis ≈ 9, per-α T spanning 0.22–1.17).
    """

    # Sorted nominal probabilities (n_cal,) and the empirical CDF values at
    # those points (n_cal,). Together they define a non-decreasing step
    # function p_nominal -> p_empirical.
    p_grid: np.ndarray | None = None
    p_empirical: np.ndarray | None = None

    def fit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
    ) -> "IsotonicCalibrator":
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_std = np.asarray(y_std, dtype=np.float64)
        if not (y_true.shape == y_pred.shape == y_std.shape):
            raise ValueError(
                f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}, "
                f"y_std={y_std.shape}"
            )
        if (y_std <= 0).any():
            raise ValueError("y_std must be strictly positive.")
        z = (y_true - y_pred) / y_std
        # Predictive CDF value of each y_true under the model.
        p = norm.cdf(z)
        # Fit a *non-decreasing* map p -> empirical-rank-of-p. Use sklearn's
        # PAVA-based isotonic regression which is the standard tool.
        from sklearn.isotonic import IsotonicRegression

        order = np.argsort(p)
        p_sorted = p[order]
        empirical_rank = (np.arange(1, len(p_sorted) + 1) - 0.5) / len(p_sorted)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_sorted, empirical_rank)
        # Densify on a fine grid so interval() is cheap and JSON-saveable.
        grid = np.linspace(1e-4, 1.0 - 1e-4, 2001)
        self.p_grid = grid
        self.p_empirical = np.asarray(iso.transform(grid), dtype=np.float64)
        # Numerical hygiene: ensure monotone non-decreasing on the dense grid.
        self.p_empirical = np.maximum.accumulate(self.p_empirical)
        return self

    def _invert(self, target: float) -> float:
        """Find nominal p such that empirical rank equals target."""
        if self.p_grid is None or self.p_empirical is None:
            raise RuntimeError("IsotonicCalibrator is not fitted; call fit() first.")
        idx = np.searchsorted(self.p_empirical, target)
        idx = int(np.clip(idx, 0, len(self.p_grid) - 1))
        return float(self.p_grid[idx])

    def interval(
        self,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1); got {alpha}")
        p_lo_target = 0.5 - 0.5 * alpha
        p_hi_target = 0.5 + 0.5 * alpha
        p_lo = self._invert(p_lo_target)
        p_hi = self._invert(p_hi_target)
        z_lo = float(norm.ppf(p_lo))
        z_hi = float(norm.ppf(p_hi))
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_std = np.asarray(y_std, dtype=np.float64)
        return y_pred + z_lo * y_std, y_pred + z_hi * y_std

    def coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        alpha: float,
    ) -> float:
        lo, hi = self.interval(y_pred, y_std, alpha)
        y_true = np.asarray(y_true, dtype=np.float64)
        return float(((y_true >= lo) & (y_true <= hi)).mean())


__all__ = [
    "coverage",
    "reliability_diagram",
    "TemperatureScaler",
    "ConformalCalibrator",
    "IsotonicCalibrator",
]
