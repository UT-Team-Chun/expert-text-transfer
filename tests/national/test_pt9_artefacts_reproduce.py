"""P-T9 (coordinate-free arm): the collation JSON must reproduce from artefacts.

The eight P-T9 runs were trained on the utens cluster and for a while only
their collation JSON was in the repo, which would have left the manuscript's
coordinate-free result resting on a number no one could recheck. The runs were
retrieved on 2026-08-19; this module holds the JSON to the artefacts so a later
edit to either side cannot drift unnoticed.

Skipped where ``data/runs/`` is absent (it is gitignored, ~190 MB for these
eight runs); it runs on the machine that builds the Zenodo bundle, which is
the machine whose numbers ship.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_COLLATION = _REPO / "docs/research/2026-08-18_pt9_coordfree.json"
_RUNS = _REPO / "data/runs"

_ARMS = ("coordfree", "zfonly")


def _per_region() -> dict:
    return json.loads(_COLLATION.read_text(encoding="utf-8"))["per_region"]


def _run_dirs() -> list[Path]:
    return [_RUNS / rec[arm]["run_dir"]
            for rec in _per_region().values() for arm in _ARMS]


def _have_runs() -> bool:
    return all((d / "predictions.npz").exists() for d in _run_dirs())


pytestmark = pytest.mark.skipif(
    not _COLLATION.exists() or not _have_runs(),
    reason="P-T9 run artefacts are not present locally (data/runs is gitignored)")


@pytest.mark.parametrize("arm", _ARMS)
def test_holdout_metrics_recompute_from_predictions(arm: str):
    """RMSE, MAE and n_eval must come back out of predictions.npz."""
    for region, rec in _per_region().items():
        z = np.load(_RUNS / rec[arm]["run_dir"] / "predictions.npz")
        y = np.asarray(z["y_true"], float).ravel()
        p = np.asarray(z["pred_mean"], float).ravel()
        where = f"{region}/{arm}"
        assert len(y) == rec[arm]["n_eval"], where
        assert np.sqrt(np.mean((y - p) ** 2)) == pytest.approx(
            rec[arm]["holdout_rmse"], abs=5e-3), where
        assert np.mean(np.abs(y - p)) == pytest.approx(
            rec[arm]["holdout_mae"], abs=5e-3), where


def test_the_two_arms_are_genuinely_different_runs():
    """Both arms hold out the same rows, so equal file size proves nothing."""
    import hashlib

    for region, rec in _per_region().items():
        digests = [
            hashlib.sha256(
                (_RUNS / rec[arm]["run_dir"] / "predictions.npz").read_bytes()
            ).hexdigest() for arm in _ARMS]
        assert digests[0] != digests[1], (
            f"{region}: coordfree and zfonly predictions are byte-identical, "
            "so one arm did not run with the flag it claims")
        # Read the arm configuration off each RUN's own summary.json, not off
        # the collation JSON. Asserting the JSON against itself passed happily
        # when the two arms' payloads were swapped -- which flips the reported
        # mean effect from -0.19 % to +0.199 %, a sign reversal of the number
        # in the manuscript's coordinate-free table.
        for arm, want_geo in (("coordfree", False), ("zfonly", True)):
            summ = json.loads(
                (_RUNS / rec[arm]["run_dir"] / "summary.json").read_text())
            where = f"{region}/{arm}"
            assert summ["leave_region"] == region, where
            assert summ["zero_fourier"] is True, where
            assert summ["add_residual_geo"] is want_geo, where
            # the two arms must be identical apart from that one flag
            assert summ["n_epochs"] == rec[arm]["n_epochs"], where
            assert summ["n_inducing"] == rec[arm]["n_inducing"], where
            assert rec[arm]["add_residual_geo"] is want_geo, (
                f"{where}: collation JSON disagrees with the run artefact")


def test_reported_effect_recomputes_from_the_per_region_deltas():
    """The -0.19 % the manuscript quotes, rebuilt from the arm RMSEs."""
    per = _per_region()
    deltas = []
    for rec in per.values():
        # A is the zfonly baseline (Fourier off, residual-geo kernel still on);
        # B is the arm that is genuinely coordinate-free. A negative delta
        # therefore means dropping the last coordinate path did not cost
        # accuracy -- which is the point of the test.
        a, b = rec["zfonly"]["holdout_rmse"], rec["coordfree"]["holdout_rmse"]
        assert (b - a) / a * 100 == pytest.approx(rec["pct_B_minus_A"], abs=5e-3)
        deltas.append(rec["pct_B_minus_A"])
    summary = json.loads(_COLLATION.read_text(encoding="utf-8"))["summary"]
    assert float(np.mean(deltas)) == pytest.approx(
        summary["mean_pct_B_minus_A"], abs=5e-3)
    assert max(abs(d) for d in deltas) == pytest.approx(
        summary["worst_abs_pct"], abs=5e-3)
    assert len(deltas) == summary["n_regions"]
