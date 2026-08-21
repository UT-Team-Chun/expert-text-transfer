"""Smoke + behaviour tests for scripts/run_mondrian_recal_lmc.py.

Builds a synthetic LMC predictions.npz with the 8 required keys plus a
matching synthetic parquet carrying ``regime_code``. Verifies the driver
runs end-to-end, the JSON exposes per-task structure, per-regime coverage
is close to nominal on calibration-rich data, rare regimes fall back to
the marginal quantile, and a missing predictions.npz raises SystemExit.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_lmc_run(tmp_path, *, n: int = 6_000, seed: int = 0):
    """Build a (run_dir, parquet_path) pair with consistent row order.

    Both tasks live in standardized-ish units. We inject a rare regime
    (code 6, ~2 rows) so the marginal-fallback branch is exercised. Task
    "gw" has ~80 % observed mask to mirror the real LMC schema.
    """
    rng = np.random.default_rng(seed)
    regimes = rng.integers(0, 6, size=n).astype(np.int16)
    regimes[:2] = 6  # 2 rows of rare regime (below min_group_n=30)

    # Task N: always observed, mild Gaussian residuals.
    y_n = rng.normal(5.0, 2.0, size=n).astype(np.float32)
    pred_mean_n = (y_n + rng.normal(0.0, 1.0, size=n)).astype(np.float32)
    pred_std_n = np.full(n, 1.5, dtype=np.float32)
    mask_n = np.ones(n, dtype=bool)

    # Task groundwater: ~80 % observed; the masked entries carry filler.
    y_gw_true = rng.normal(2.0, 1.0, size=n).astype(np.float32)
    pred_mean_gw = (y_gw_true + rng.normal(0.0, 0.6, size=n)).astype(np.float32)
    pred_std_gw = np.full(n, 0.8, dtype=np.float32)
    mask_gw = rng.uniform(size=n) < 0.8
    y_gw = y_gw_true.copy()
    y_gw[~mask_gw] = np.nan  # NaN filler in unobserved rows

    run_dir = tmp_path / "lmc_synth_run"
    run_dir.mkdir()
    np.savez(
        run_dir / "predictions.npz",
        pred_mean_n=pred_mean_n,
        pred_mean_gw=pred_mean_gw,
        pred_std_n=pred_std_n,
        pred_std_gw=pred_std_gw,
        y_n=y_n,
        y_gw=y_gw,
        mask_n=mask_n,
        mask_gw=mask_gw,
    )

    # Parquet must have len == n with regime_code in the same row order.
    parquet_path = tmp_path / "synth_borings.parquet"
    df = pd.DataFrame({"regime_code": regimes.astype(np.int64)})
    df.to_parquet(parquet_path)
    return run_dir, parquet_path


@pytest.fixture
def synthetic_lmc_run(tmp_path):
    return _make_synthetic_lmc_run(tmp_path)


def test_recal_lmc_runs_end_to_end(synthetic_lmc_run):
    from scripts.run_mondrian_recal_lmc import main

    run_dir, parquet_path = synthetic_lmc_run
    rc = main([
        "--run-dir", str(run_dir),
        "--parquet", str(parquet_path),
        "--seed", "0",
    ])
    assert rc == 0
    out_path = run_dir / "conformal_mondrian_lmc.json"
    assert out_path.exists()
    out = json.loads(out_path.read_text())
    assert "task_n" in out
    assert "task_gw" in out
    # Sanity: structural keys present per task.
    for task_key in ("task_n", "task_gw"):
        block = out[task_key]
        assert "marginal" in block
        assert "per_regime" in block
        assert "n_observed" in block
        assert "n_cal" in block
        assert "n_eval" in block


def test_recal_lmc_marginal_coverage_near_nominal(synthetic_lmc_run):
    from scripts.run_mondrian_recal_lmc import main

    run_dir, parquet_path = synthetic_lmc_run
    rc = main([
        "--run-dir", str(run_dir),
        "--parquet", str(parquet_path),
        "--seed", "0",
    ])
    assert rc == 0
    out = json.loads((run_dir / "conformal_mondrian_lmc.json").read_text())
    # Split conformal gives a finite-sample guarantee, so empirical
    # marginal coverage should be within ~5% of nominal on this 6k synth.
    for task_key in ("task_n", "task_gw"):
        for a in ("0.5", "0.8", "0.95"):
            marg = out[task_key]["marginal"][a]["coverage_marginal_only"]
            mond = out[task_key]["marginal"][a]["coverage_mondrian"]
            assert abs(marg - float(a)) < 0.05, (
                f"{task_key} α={a}: marginal={marg:.3f}"
            )
            assert abs(mond - float(a)) < 0.05, (
                f"{task_key} α={a}: mondrian={mond:.3f}"
            )


def test_recal_lmc_rare_regime_falls_back(synthetic_lmc_run):
    from scripts.run_mondrian_recal_lmc import main

    run_dir, parquet_path = synthetic_lmc_run
    rc = main([
        "--run-dir", str(run_dir),
        "--parquet", str(parquet_path),
        "--seed", "0",
        "--min-group-n", "30",
    ])
    assert rc == 0
    out = json.loads((run_dir / "conformal_mondrian_lmc.json").read_text())
    # Regime 6 has only 2 rows total; both halves of the cal/eval split
    # are below min_group_n. If the rare regime survived sampling into
    # the eval split, it must use the marginal fallback.
    for task_key in ("task_n", "task_gw"):
        per_regime_05 = out[task_key]["per_regime"]["0.5"]
        if "6" in per_regime_05:
            r6 = per_regime_05["6"]
            assert r6["n_cal"] < 30
            assert r6["uses_marginal_fallback"] is True


def test_recal_lmc_rejects_missing_predictions(tmp_path):
    from scripts.run_mondrian_recal_lmc import main

    empty_dir = tmp_path / "empty_run"
    empty_dir.mkdir()
    parquet_path = tmp_path / "irrelevant.parquet"
    pd.DataFrame({"regime_code": [0, 1, 2]}).to_parquet(parquet_path)
    with pytest.raises(SystemExit):
        main(["--run-dir", str(empty_dir), "--parquet", str(parquet_path)])


def test_recal_lmc_rejects_parquet_row_mismatch(tmp_path):
    """If the parquet length doesn't match the npz, abort with a clear error."""
    from scripts.run_mondrian_recal_lmc import main

    run_dir, _ = _make_synthetic_lmc_run(tmp_path, n=500, seed=1)
    # Build a *wrong-length* parquet (regime would be misaligned).
    wrong_parquet = tmp_path / "wrong_len.parquet"
    pd.DataFrame({"regime_code": np.zeros(123, dtype=np.int64)}).to_parquet(
        wrong_parquet
    )
    with pytest.raises(ValueError, match="Row-count mismatch"):
        main([
            "--run-dir", str(run_dir),
            "--parquet", str(wrong_parquet),
            "--seed", "0",
        ])
