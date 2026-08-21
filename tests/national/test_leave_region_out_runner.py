"""Leave-region-out / leave-block-out runner + the geological-block split.

The runner is exercised with the fast HistGradientBoosting model (no GPBoost
dependency) on a synthetic multi-region DataFrame; the gpboost path is smoke-
tested only when the optional dependency is installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from national.evaluation.leave_region_out import (
    GEOLOGICAL_BLOCKS,
    leave_block_out_split,
)

# (lat_min, lat_max, lon_min, lon_max) centres for three DEFAULT_REGIONS.
_REGION_CENTRES = {
    "kanto": (36.0, 139.5),
    "tohoku": (39.0, 140.5),
    "kansai": (34.5, 135.5),
}


def _synthetic_multiregion_df(n_per_region: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for ri, (region, (clat, clon)) in enumerate(_REGION_CENTRES.items()):
        lat = clat + rng.uniform(-0.4, 0.4, n_per_region)
        lon = clon + rng.uniform(-0.4, 0.4, n_per_region)
        depth = rng.uniform(0.5, 30.0, n_per_region)
        elev = rng.uniform(0.0, 200.0, n_per_region)
        river = rng.uniform(0.0, 10.0, n_per_region)
        coast = rng.uniform(0.0, 50.0, n_per_region)
        regime = rng.integers(0, 3, n_per_region)  # a few regimes
        n_value = (
            3.0 + 0.4 * depth + 0.01 * elev + 2.0 * ri + rng.normal(0, 2.0, n_per_region)
        )
        frames.append(
            pd.DataFrame(
                {
                    "latitude_deg": lat,
                    "longitude_deg": lon,
                    "depth_from_surface": depth,
                    "absolute_elevation": elev,
                    "river_distance_km": river,
                    "coast_distance_km": coast,
                    "regime_code": regime,
                    "n_value": n_value,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_leave_block_out_split_yields_disjoint_blocks():
    df = _synthetic_multiregion_df()
    folds = list(leave_block_out_split(df))
    names = [name for name, _, _ in folds]
    # Only blocks with data present should be yielded.
    assert "central_east" in names  # contains kanto
    assert "north" in names  # contains tohoku
    assert "southwest" in names  # contains kansai
    for name, train_idx, test_idx in folds:
        assert len(test_idx) > 0
        assert len(train_idx) > 0
        # train and test are a partition of all rows.
        assert set(train_idx).isdisjoint(set(test_idx))
        assert len(train_idx) + len(test_idx) == len(df)


def test_leave_block_out_split_rejects_unknown_region():
    df = _synthetic_multiregion_df(n_per_region=10)
    with pytest.raises(ValueError, match="unknown region"):
        list(leave_block_out_split(df, blocks={"bad": ["atlantis"]}))


def test_geological_blocks_members_are_known_regions():
    from national.evaluation.leave_region_out import DEFAULT_REGIONS

    members = {m for ms in GEOLOGICAL_BLOCKS.values() for m in ms}
    assert members <= set(DEFAULT_REGIONS)


def test_run_leave_region_out_structure_and_gate():
    from national.evaluation.leave_region_out_runner import run_leave_region_out

    df = _synthetic_multiregion_df(n_per_region=300, seed=1)
    result = run_leave_region_out(df, partition="region", model="hgb", seed=1)

    assert set(result) == {"config", "per_fold", "aggregate", "gate"}
    assert len(result["per_fold"]) >= 3
    for fold in result["per_fold"]:
        assert fold["n_fit"] > 0 and fold["n_cal"] > 0 and fold["n_test"] > 0
        assert np.isfinite(fold["rmse"]) and fold["rmse"] >= 0
        assert "0.95" in fold["per_alpha"]
        cov = fold["per_alpha"]["0.95"]["coverage_marginal"]
        assert 0.0 <= cov <= 1.0
        assert len(fold["per_regime"]) >= 1
        assert "rmse" in fold["per_regime"][0]

    gate = result["gate"]
    for key in ("pass", "pass_rmse", "pass_coverage", "mean_test_rmse", "mean_coverage_95"):
        assert key in gate
    assert isinstance(gate["pass"], bool)


def test_run_leave_region_out_block_partition():
    from national.evaluation.leave_region_out_runner import run_leave_region_out

    df = _synthetic_multiregion_df(n_per_region=300, seed=2)
    result = run_leave_region_out(df, partition="block", model="hgb", seed=2)
    assert result["config"]["partition"] == "block"
    assert len(result["per_fold"]) >= 3


def test_run_leave_region_out_rejects_missing_columns():
    from national.evaluation.leave_region_out_runner import run_leave_region_out

    df = _synthetic_multiregion_df(n_per_region=20).drop(columns=["coast_distance_km"])
    with pytest.raises(ValueError, match="missing columns"):
        run_leave_region_out(df, model="hgb")


def test_run_leave_region_out_gpboost_smoke():
    pytest.importorskip("gpboost")
    from national.evaluation.leave_region_out_runner import run_leave_region_out

    df = _synthetic_multiregion_df(n_per_region=200, seed=3)
    result = run_leave_region_out(
        df,
        partition="region",
        model="gpboost",
        seed=3,
        gpboost_kwargs={"num_neighbors": 8, "n_boost_iter": 30},
    )
    assert len(result["per_fold"]) >= 3
    for fold in result["per_fold"]:
        assert np.isfinite(fold["rmse"])


# ---------------------------------------------------------------------------
# Disjoint administrative-polygon Kanto-prefecture partition (review-response).
# ---------------------------------------------------------------------------

def _synthetic_kanto_df(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "latitude_deg": rng.uniform(34.9, 37.1, n),
        "longitude_deg": rng.uniform(138.4, 141.0, n),
    })


def test_assignment_labels_are_in_range():
    from national.evaluation.prefecture_regions import (
        KANTO_PREFECTURES,
        assign_nearest_prefecture,
        assign_prefecture,
    )
    df = _synthetic_kanto_df()
    lats, lons = df.latitude_deg.to_numpy(), df.longitude_deg.to_numpy()
    # Nearest-centre fallback: every row -> exactly one known prefecture.
    a = assign_nearest_prefecture(lats, lons)
    assert len(a) == len(df)
    assert set(a.tolist()) <= set(KANTO_PREFECTURES)
    # Containment assignment: synthetic points are not in the contained-location
    # lookup, so they are labelled "other" (train-only) -- never a Kanto fold,
    # never forced into a prefecture.
    b = assign_prefecture(lats, lons)
    assert len(b) == len(df)
    assert set(b.tolist()) <= set(KANTO_PREFECTURES) | {"other"}


def test_assign_prefecture_roundtrips_the_polygon_lookup():
    """On the lookup's own locations, polygon assignment reproduces it exactly."""
    from national.evaluation.prefecture_regions import (
        KANTO_PREFECTURES,
        _POLYGON_LOOKUP_PATH,
        assign_prefecture,
    )
    if not _POLYGON_LOOKUP_PATH.exists():
        pytest.skip("administrative-polygon lookup asset not present")
    lut = pd.read_parquet(_POLYGON_LOOKUP_PATH)
    got = assign_prefecture(lut.latitude_deg.to_numpy(), lut.longitude_deg.to_numpy())
    assert (got == lut.pref.to_numpy()).all()
    # The lookup holds only CONTAINED boreholes -> all real Kanto prefectures.
    assert set(lut.pref.tolist()) <= set(KANTO_PREFECTURES)


def test_polygon_folds_are_genuine_disjoint_prefectures():
    """The 7 held-out folds are genuine administrative prefectures: pairwise
    disjoint, covering the 435,732 in-prefecture boreholes (the ~12% in
    neighbouring prefectures within the study bbox are train-only)."""
    from pathlib import Path

    from national.evaluation.prefecture_regions import (
        _POLYGON_LOOKUP_PATH,
        leave_prefecture_out_split,
    )
    parquet = (Path(__file__).resolve().parents[3]
               / "data/features/borings_kanto_aist.parquet")
    if not (_POLYGON_LOOKUP_PATH.exists() and parquet.exists()):
        pytest.skip("Kanto corpus / polygon lookup not present")
    df = pd.read_parquet(parquet, columns=["latitude_deg", "longitude_deg"])
    test_sets, total = [], 0
    for _pref, train_idx, test_idx in leave_prefecture_out_split(df):
        assert set(train_idx).isdisjoint(set(test_idx))      # no leakage
        assert len(train_idx) + len(test_idx) == len(df)     # train = complement
        test_sets.append(set(test_idx.tolist()))
        total += len(test_idx)
    assert len(test_sets) == 7                  # all seven prefectures non-empty
    assert total == 435_732                     # in-prefecture measurement rows
    assert total < len(df) == 495_725           # neighbouring prefs are train-only
    for i in range(len(test_sets)):
        for j in range(i + 1, len(test_sets)):
            assert test_sets[i].isdisjoint(test_sets[j])


def test_nearest_centre_fallback_partitions_the_corpus(monkeypatch):
    """Degraded mode (lookup asset absent): nearest-centre assignment yields a
    disjoint partition that sums to the full corpus."""
    import national.evaluation.prefecture_regions as pr
    monkeypatch.setattr(pr, "_load_polygon_lookup", lambda: None)
    df = _synthetic_kanto_df()
    n = len(df)
    test_sets, total = [], 0
    for _pref, train_idx, test_idx in pr.leave_prefecture_out_split(df):
        assert set(train_idx).isdisjoint(set(test_idx))
        assert len(train_idx) + len(test_idx) == n
        test_sets.append(set(test_idx.tolist()))
        total += len(test_idx)
    assert total == n                           # fallback partitions the corpus
    for i in range(len(test_sets)):
        for j in range(i + 1, len(test_sets)):
            assert test_sets[i].isdisjoint(test_sets[j])


def test_leave_prefecture_out_split_subset_matches_full():
    """A prefecture SUBSET yields exactly those folds, each bit-identical to the
    full-partition fold --- the property the parallel per-prefecture LPO sweep
    (run via the ``--prefectures`` flag across nodes) relies on to stay
    condition-preserving."""
    from pathlib import Path

    from national.evaluation.prefecture_regions import (
        _POLYGON_LOOKUP_PATH,
        leave_prefecture_out_split,
    )
    parquet = (Path(__file__).resolve().parents[3]
               / "data/features/borings_kanto_aist.parquet")
    if not (_POLYGON_LOOKUP_PATH.exists() and parquet.exists()):
        pytest.skip("Kanto corpus / polygon lookup not present")
    df = pd.read_parquet(parquet, columns=["latitude_deg", "longitude_deg"])
    full = {p: set(test.tolist()) for p, _, test in leave_prefecture_out_split(df)}
    subset = {p: set(test.tolist()) for p, _, test
              in leave_prefecture_out_split(df, prefectures=["tokyo", "gunma"])}
    assert set(subset) == {"tokyo", "gunma"}             # only the requested folds
    assert subset["tokyo"] == full["tokyo"]              # identical to full partition
    assert subset["gunma"] == full["gunma"]
