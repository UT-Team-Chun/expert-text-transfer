"""P-T7 (identity-join 3-fold): the reported effect must reproduce from artefacts.

These three cells were recorded as having no ``predictions.npz`` -- a
conclusion drawn by looking only at the local ``data/runs/``. The files were on
the cluster NAS the whole time, and were retrieved on 2026-08-19. This pins the
manuscript's in-distribution numbers to them, so neither side can drift.

Skipped where ``data/runs/`` is absent (gitignored, ~206 MB for these three).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_COLLATION = _REPO / "docs/research/2026-08-14_pt7_identity_join_3fold.json"
_RUNS = _REPO / "data/runs"
_CELLS = [f"dkl_national_lmc_v5id_sarashina_l2_kfold{i}" for i in (0, 1, 2)]


def _have() -> bool:
    return _COLLATION.exists() and all(
        (_RUNS / c / "predictions.npz").exists() for c in _CELLS)


pytestmark = pytest.mark.skipif(
    not _have(), reason="P-T7 predictions are not present locally")


def _heldout(cell: str) -> tuple[np.ndarray, np.ndarray]:
    """The held-out fold only.

    ``predictions.npz`` covers all 2,663,955 parquet rows; the reported figure
    is over ``is_holdout & mask_n``. Averaging every row instead gives 8.117
    rather than 8.505 -- a plausible-looking number that is not the one the
    manuscript reports.
    """
    z = np.load(_RUNS / cell / "predictions.npz")
    sel = np.asarray(z["is_holdout"], bool) & np.asarray(z["mask_n"], bool)
    return (np.asarray(z["y_n"], float)[sel],
            np.asarray(z["pred_mean_n"], float)[sel])


def test_per_fold_rmse_and_mae_recompute():
    per_fold = json.loads(_COLLATION.read_text())["text_arm"]["per_fold"]
    for i, cell in enumerate(_CELLS):
        rec = per_fold[f"kfold{i}"]
        y, p = _heldout(cell)
        assert len(y) == rec["kfold_n_holdout"], cell
        assert np.sqrt(np.mean((y - p) ** 2)) == pytest.approx(
            rec["holdout_rmse_n"], abs=5e-3), cell
        assert np.mean(np.abs(y - p)) == pytest.approx(
            rec["holdout_mae_n"], abs=5e-3), cell


def test_headline_mean_matches_the_manuscript_macro():
    """8.505 +/- 0.162 is what \\idTextRMSE prints in the paper."""
    rmses = [float(np.sqrt(np.mean((y - p) ** 2)))
             for y, p in (_heldout(c) for c in _CELLS)]
    assert float(np.mean(rmses)) == pytest.approx(8.505, abs=5e-3)
    assert float(np.std(rmses, ddof=1)) == pytest.approx(0.162, abs=5e-3)

    macros = (_REPO / "docs/paper/paper_2_national/headline_numbers.tex").read_text()
    assert r"\newcommand{\idTextRMSE}{8.505 \pm 0.162}" in macros
