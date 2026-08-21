"""Unit tests for the NC-revision stratified-null harness (scripts.nc_null_controls)."""
from __future__ import annotations

import numpy as np

from scripts.nc_null_controls import _coarse_class, _strat_perm


def test_strat_perm_stays_within_stratum():
    # every permuted index must come from the SAME stratum as its destination
    strata = np.array(["a", "a", "a", "b", "b", "c", "c", "c", "c"])
    rng = np.random.default_rng(0)
    p = _strat_perm(strata, rng)
    assert sorted(p.tolist()) == list(range(len(strata)))  # a valid permutation
    for i, j in enumerate(p):
        assert strata[i] == strata[j], f"index {j} (stratum {strata[j]}) leaked into stratum {strata[i]}"


def test_strat_perm_singleton_fixed():
    # a stratum of size 1 cannot move
    strata = np.array(["solo", "x", "x", "x"])
    rng = np.random.default_rng(1)
    p = _strat_perm(strata, rng)
    assert p[0] == 0  # singleton stays put


def test_strat_perm_actually_shuffles_large_stratum():
    strata = np.array(["a"] * 200)
    rng = np.random.default_rng(2)
    p = _strat_perm(strata, rng)
    assert (p != np.arange(200)).mean() > 0.5  # most rows moved


def test_coarse_class_uk_takes_principal_caps_noun():
    # BS5930 capitalises the principal soil/rock; we take the last CAP lithology noun
    texts = ["Soft brown slightly gravelly silty CLAY.",
             "Medium dense grey sandy fine to coarse GRAVEL of sandstone.",
             "weak grey thinly bedded MUDSTONE.",
             "no lithology word here"]
    cls = _coarse_class(texts, "uk")
    assert cls.tolist() == ["CLAY", "GRAVEL", "MUDSTONE", "other"]


def test_coarse_class_japan_keyword_priority():
    texts = ["砂質シルト", "礫混じり砂", "粘土", "凝灰岩", "ローム層", "判別不能"]
    cls = _coarse_class(texts, "japan")
    # rock>gravel>sand>silt>clay>loam priority; "砂質シルト" hits 砂(sand) before シルト
    assert cls[2] == "clay"
    assert cls[3] == "rock"
    assert cls[4] == "loam"
    assert cls[5] == "other"
    assert set(cls.tolist()) <= {"rock", "organic", "gravel", "sand", "silt", "clay", "loam", "mud", "other"}
