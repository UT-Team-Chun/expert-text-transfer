"""Unit tests for ``national.evaluation.calibration.TemperatureScaler``."""

from __future__ import annotations

import numpy as np
import pytest

from national.evaluation.calibration import (
    ConformalCalibrator,
    IsotonicCalibrator,
    TemperatureScaler,
    coverage,
    reliability_diagram,
)


@pytest.fixture
def well_calibrated_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """y_true drawn from N(y_pred, y_std) so the empirical mean Z^2 ~ 1."""
    rng = np.random.default_rng(0)
    n = 50_000
    y_pred = rng.normal(0.0, 5.0, size=n)
    y_std = rng.uniform(0.5, 3.0, size=n)
    y_true = y_pred + y_std * rng.standard_normal(size=n)
    return y_true, y_pred, y_std


def test_temperature_one_when_already_calibrated(well_calibrated_sample) -> None:
    """T should land near 1.0 on a perfectly calibrated synthetic sample."""
    y_true, y_pred, y_std = well_calibrated_sample
    scaler = TemperatureScaler().fit(y_true, y_pred, y_std)
    assert scaler.T is not None
    assert abs(scaler.T - 1.0) < 0.02  # within 2 % of the truth at 50k samples


def test_temperature_lt1_for_over_cautious(well_calibrated_sample) -> None:
    """If raw y_std is inflated 2x, T should land near 0.5 (= 1/2)."""
    y_true, y_pred, y_std = well_calibrated_sample
    scaler = TemperatureScaler().fit(y_true, y_pred, y_std * 2.0)
    assert scaler.T is not None
    assert abs(scaler.T - 0.5) < 0.01


def test_temperature_gt1_for_over_confident(well_calibrated_sample) -> None:
    """If raw y_std is halved (too narrow), T should land near 2.0."""
    y_true, y_pred, y_std = well_calibrated_sample
    scaler = TemperatureScaler().fit(y_true, y_pred, y_std * 0.5)
    assert scaler.T is not None
    assert abs(scaler.T - 2.0) < 0.04


def test_apply_scales_std(well_calibrated_sample) -> None:
    y_true, y_pred, y_std = well_calibrated_sample
    scaler = TemperatureScaler().fit(y_true, y_pred, y_std * 2.0)
    scaled = scaler.apply(y_std * 2.0)
    # After applying the fitted T to the same inflated std, mean Z^2 should be 1.
    z2 = ((y_true - y_pred) / scaled) ** 2
    assert abs(z2.mean() - 1.0) < 0.02


def test_fit_apply_returns_tuple(well_calibrated_sample) -> None:
    y_true, y_pred, y_std = well_calibrated_sample
    T, scaled = TemperatureScaler().fit_apply(y_true, y_pred, y_std)
    assert isinstance(T, float)
    assert scaled.shape == y_std.shape
    np.testing.assert_allclose(scaled, y_std * T)


def test_apply_without_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        TemperatureScaler().apply(np.array([1.0, 2.0]))


def test_fit_rejects_nonpositive_std() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        TemperatureScaler().fit(
            np.zeros(3), np.zeros(3), np.array([1.0, 0.0, 1.0])
        )


def test_fit_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        TemperatureScaler().fit(np.zeros(3), np.zeros(4), np.ones(3))


def test_scaling_fixes_coverage_gap(well_calibrated_sample) -> None:
    """End-to-end: an over-cautious model has alpha=0.50 empirical >> 0.50;
    applying TS brings it back near 0.50."""
    y_true, y_pred, y_std = well_calibrated_sample
    inflated = y_std * 2.0
    raw_cov = coverage(y_true, y_pred, inflated, alpha=0.50)
    assert raw_cov - 0.50 > 0.15  # raw is over-cautious

    T, scaled = TemperatureScaler().fit_apply(y_true, y_pred, inflated)
    new_cov = coverage(y_true, y_pred, scaled, alpha=0.50)
    assert abs(new_cov - 0.50) < 0.02


def test_reliability_diagram_with_scaled_std(well_calibrated_sample) -> None:
    """The downstream reliability table accepts the scaled std unchanged."""
    y_true, y_pred, y_std = well_calibrated_sample
    inflated = y_std * 2.0
    _, scaled = TemperatureScaler().fit_apply(y_true, y_pred, inflated)
    table = reliability_diagram(y_true, y_pred, scaled)
    # Every nominal alpha should be within 2 pp of the truth after TS.
    for _, row in table.iterrows():
        assert abs(row["gap"]) < 0.02


# ---------- ConformalCalibrator -----------------------------------------


@pytest.fixture
def heavy_tail_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """y_true drawn from Student-t (kurtosis ≈ 6) so a single τ cannot fix
    every α. Mimics the SPT N value miscalibration pattern (kurtosis 9.3
    on real data).
    """
    rng = np.random.default_rng(0)
    n = 50_000
    y_pred = rng.normal(0.0, 5.0, size=n)
    y_std = rng.uniform(0.5, 3.0, size=n)
    eps = rng.standard_t(df=5, size=n)  # heavy tail
    y_true = y_pred + y_std * eps
    return y_true, y_pred, y_std


def test_conformal_calibrated_on_heavy_tail(heavy_tail_sample) -> None:
    """Conformal coverage matches nominal α within finite-sample error."""
    y_true, y_pred, y_std = heavy_tail_sample
    cal = ConformalCalibrator().fit(y_true[:25_000], y_pred[:25_000], y_std[:25_000])
    for alpha in (0.5, 0.8, 0.95):
        emp = cal.coverage(y_true[25_000:], y_pred[25_000:], y_std[25_000:], alpha)
        # Conformal gives valid coverage at α; finite-sample slack ~0.01
        assert abs(emp - alpha) < 0.015


def test_conformal_outperforms_ts_on_heavy_tail(heavy_tail_sample) -> None:
    """On the heavy-tail synthetic data TS fails at α=0.50 but conformal
    succeeds — same pattern observed on real kanto data."""
    y_true, y_pred, y_std = heavy_tail_sample
    n_cal = 25_000
    # Held-out test set
    yt_test = y_true[n_cal:]
    yp_test = y_pred[n_cal:]
    ys_test = y_std[n_cal:]

    ts = TemperatureScaler().fit(y_true[:n_cal], y_pred[:n_cal], y_std[:n_cal])
    ts_cov_50 = coverage(yt_test, yp_test, ts.apply(ys_test), 0.5)

    cal = ConformalCalibrator().fit(y_true[:n_cal], y_pred[:n_cal], y_std[:n_cal])
    cal_cov_50 = cal.coverage(yt_test, yp_test, ys_test, 0.5)

    # TS may leave residual gap because Student-t bulk is narrower than Gaussian;
    # conformal should be tighter.
    assert abs(cal_cov_50 - 0.5) < abs(ts_cov_50 - 0.5)


def test_conformal_interval_without_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        ConformalCalibrator().interval(np.zeros(3), np.ones(3), 0.5)


def test_conformal_unfitted_alpha_raises(heavy_tail_sample) -> None:
    y_true, y_pred, y_std = heavy_tail_sample
    cal = ConformalCalibrator().fit(
        y_true[:1000], y_pred[:1000], y_std[:1000], alphas=(0.5, 0.95)
    )
    with pytest.raises(KeyError):
        cal.interval(y_pred[:5], y_std[:5], alpha=0.80)


# ---------- IsotonicCalibrator ------------------------------------------


def test_isotonic_calibrated_on_heavy_tail(heavy_tail_sample) -> None:
    """Isotonic CDF remap closes coverage gaps on heavy-tail data."""
    y_true, y_pred, y_std = heavy_tail_sample
    cal = IsotonicCalibrator().fit(y_true[:25_000], y_pred[:25_000], y_std[:25_000])
    for alpha in (0.5, 0.8, 0.95):
        emp = cal.coverage(y_true[25_000:], y_pred[25_000:], y_std[25_000:], alpha)
        assert abs(emp - alpha) < 0.03


def test_isotonic_identity_on_already_calibrated(well_calibrated_sample) -> None:
    """If predictions are already calibrated, isotonic should be ~identity
    (within finite-sample noise)."""
    y_true, y_pred, y_std = well_calibrated_sample
    cal = IsotonicCalibrator().fit(y_true, y_pred, y_std)
    for alpha in (0.5, 0.8, 0.95):
        emp = cal.coverage(y_true, y_pred, y_std, alpha)
        assert abs(emp - alpha) < 0.02


def test_isotonic_interval_without_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        IsotonicCalibrator().interval(np.zeros(3), np.ones(3), 0.5)
