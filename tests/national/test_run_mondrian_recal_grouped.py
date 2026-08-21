"""Tests for the grouped-conformal re-evaluation runner (P-T8).

The delicate part of this script is ``_align_identity``: it attaches borehole
identity to a ``predictions.npz`` that carries no identity of its own, so a
silent misalignment would quietly evaluate the wrong boreholes. Two real
hazards are covered here:

* leave-region-out runs are a NON-contiguous subset of the v4id parquet (rows
  are selected by a geographic bounding box, not by position), and
* the LRO runner wrote a TRAIN-fold ``regime`` array into ``predictions.npz``
  (2,168,230 rows against 495,725 predictions for kanto) -- the quirk already
  documented at ``scripts/run_tta_lro.py:143``. The runner must therefore take
  the regime from the parquet for LRO runs, and must still refuse a same-length
  regime array that disagrees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from national.evaluation.leave_region_out import DEFAULT_REGIONS
from scripts.run_mondrian_recal_grouped import (
    _align_identity, _split_masks, _winkler,
)

# A tiny synthetic corpus: some rows inside the kanto bbox, some outside it.
KANTO = DEFAULT_REGIONS["kanto"]


def _mid(lo: float, hi: float) -> float:
    return 0.5 * (lo + hi)


@pytest.fixture()
def v4(tmp_path):
    """Synthetic v4id parquet with 6 kanto rows interleaved among 6 others."""
    la0, la1, lo0, lo1 = KANTO
    lat_in, lon_in = _mid(la0, la1), _mid(lo0, lo1)
    # somewhere in the far north-west, outside every bbox we care about here
    lat_out, lon_out = 44.5, 141.0
    # 6 inside, 7 outside -- deliberately unequal, so the train-sized regime
    # array of an LRO run cannot accidentally match len(y) and take the
    # same-length branch.
    inside = [True, False, True, True, False, False, True, False, True, True,
              False, False, False]
    df = pd.DataFrame({
        "n_value": np.arange(len(inside), dtype="float32") + 1.0,
        "regime_code": np.arange(len(inside), dtype="int64") % 4,
        "boring_file": [f"b{i // 3}" for i in range(len(inside))],
        "latitude_deg": [lat_in if i else lat_out for i in inside],
        "longitude_deg": [lon_in if i else lon_out for i in inside],
    })
    path = tmp_path / "v4id.parquet"
    df.to_parquet(path)
    return path, df, np.array(inside)


def test_full_corpus_aligns_positionally(v4):
    path, df, _ = v4
    y = df["n_value"].to_numpy(np.float64)
    regime = df["regime_code"].to_numpy(np.int64)
    bf, lat, lon, reg_used, how = _align_identity(y, regime, v4_path=path)
    assert how == "positional (full corpus)"
    assert np.array_equal(reg_used, regime)
    assert list(bf) == list(df["boring_file"])


def test_full_corpus_regime_disagreement_aborts(v4):
    """A same-length regime that disagrees is a real misalignment, not a quirk."""
    path, df, _ = v4
    y = df["n_value"].to_numpy(np.float64)
    bad = df["regime_code"].to_numpy(np.int64).copy()
    bad[0] = (bad[0] + 1) % 4
    with pytest.raises(SystemExit, match="regime disagrees"):
        _align_identity(y, bad, v4_path=path)


def test_lro_subset_matches_bbox_and_is_not_contiguous(v4):
    """The held-out rows are scattered, so a contiguous-window search would fail."""
    path, df, inside = v4
    idx = np.flatnonzero(inside)
    assert np.any(np.diff(idx) > 1), "fixture must be non-contiguous to be a real test"
    y = df["n_value"].to_numpy(np.float64)[inside]
    train_sized_regime = np.zeros(int((~inside).sum()), dtype="int64")

    bf, lat, lon, reg_used, how = _align_identity(
        y, train_sized_regime, v4_path=path)

    assert "held-out region bbox 'kanto'" in how
    assert "regime from parquet" in how
    assert np.array_equal(reg_used, df["regime_code"].to_numpy(np.int64)[inside])
    assert list(bf) == list(df["boring_file"][inside])
    assert len(reg_used) == len(y)


def test_lro_accepts_a_correctly_sized_matching_regime(v4):
    path, df, inside = v4
    y = df["n_value"].to_numpy(np.float64)[inside]
    good = df["regime_code"].to_numpy(np.int64)[inside]
    _, _, _, reg_used, how = _align_identity(y, good, v4_path=path)
    assert "held-out region bbox 'kanto'" in how
    assert "regime from parquet" not in how
    assert np.array_equal(reg_used, good)


def test_lro_rejects_a_correctly_sized_mismatching_regime(v4):
    path, df, inside = v4
    y = df["n_value"].to_numpy(np.float64)[inside]
    bad = df["regime_code"].to_numpy(np.int64)[inside].copy()
    bad[0] = (bad[0] + 1) % 4
    with pytest.raises(SystemExit, match="regime disagrees"):
        _align_identity(y, bad, v4_path=path)


def test_unalignable_rows_abort_rather_than_guess(v4):
    path, df, inside = v4
    y = df["n_value"].to_numpy(np.float64)[inside] + 100.0  # values in no region
    with pytest.raises(SystemExit, match="do not align"):
        _align_identity(y, np.zeros(3, dtype="int64"), v4_path=path)


def test_borehole_split_keeps_every_borehole_on_one_side():
    n = 300
    boring = np.array([f"b{i % 30}" for i in range(n)])
    lat = np.full(n, 35.0)
    lon = np.full(n, 139.0)
    cal, ev = _split_masks("borehole", n, boring, lat, lon, seed=42)
    assert not (cal & ev).any() and (cal | ev).all()
    for b in np.unique(boring):
        m = boring == b
        assert cal[m].all() or ev[m].all(), f"borehole {b} straddles the split"


def test_row_split_does_straddle_boreholes():
    """Contrast case: the legacy row split is expected to straddle boreholes."""
    n = 300
    boring = np.array([f"b{i % 30}" for i in range(n)])
    cal, _ = _split_masks("row", n, boring, np.zeros(n), np.zeros(n), seed=42)
    straddled = sum(0 < cal[boring == b].sum() < (boring == b).sum()
                    for b in np.unique(boring))
    assert straddled > 0


def test_winkler_reduces_to_width_when_all_covered():
    y = np.array([1.0, 2.0, 3.0])
    lo, hi = y - 1.0, y + 1.0
    assert _winkler(y, lo, hi, 0.8) == pytest.approx(2.0)


def test_winkler_penalises_misses_by_two_over_alpha():
    y = np.array([5.0])
    lo, hi = np.array([0.0]), np.array([2.0])
    # width 2, miss above by 3, penalty 2/(1-0.8) * 3 = 30
    assert _winkler(y, lo, hi, 0.8) == pytest.approx(32.0)
