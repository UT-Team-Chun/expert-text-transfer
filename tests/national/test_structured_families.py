"""Tests for the family-structured parser (P-T10 groundwork)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.text_leakage_controls import (
    _structured_litho_features,
    structured_families,
)

TEXTS_JP = ["φ2〜5mmの亜円礫まじり細砂。褐色、淘汰良い。", "風化した粘土、含水多い 30%"]
DF_JP = pd.DataFrame({"regime_code": [0, 3], "aist_litho_macro_code": [1, 5]})


def test_text_only_mode_excludes_archive_codes() -> None:
    with_codes = structured_families(TEXTS_JP, DF_JP, "japan",
                                     include_archive_codes=True)
    text_only = structured_families(TEXTS_JP, DF_JP, "japan",
                                    include_archive_codes=False)
    # the dictionary block shrinks to the 10 keyword flags without the
    # regime/litho one-hots
    assert text_only["dictionary"].shape[1] == 10
    assert with_codes["dictionary"].shape[1] > 10


def test_sorting_family_detects_vocabulary() -> None:
    fams = structured_families(TEXTS_JP, DF_JP, "japan",
                               include_archive_codes=False)
    assert fams["sorting"][0].sum() >= 1  # 淘汰 in text 0
    assert fams["sorting"][1].sum() == 0
    uk = structured_families(["Well graded SAND"], pd.DataFrame(index=[0]),
                             "uk", include_archive_codes=False)
    assert uk["sorting"][0].sum() >= 1


def test_legacy_concat_excludes_sorting_for_bit_compat() -> None:
    fams = structured_families(TEXTS_JP, DF_JP, "japan",
                               include_archive_codes=True)
    legacy = _structured_litho_features(TEXTS_JP, DF_JP, "japan")
    expected = sum(m.shape[1] for k, m in fams.items() if k != "sorting")
    assert legacy.shape[1] == expected


def test_family_order_deterministic() -> None:
    f1 = structured_families(TEXTS_JP, DF_JP, "japan")
    f2 = structured_families(TEXTS_JP, DF_JP, "japan")
    assert list(f1.keys()) == list(f2.keys())
    for k in f1:
        np.testing.assert_array_equal(f1[k], f2[k])
