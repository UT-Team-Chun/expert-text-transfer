"""Unit tests for the provenance-folds runner helpers (P-T4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.nc_provenance_folds import MIN_FOLD_ROWS, _top_labels, _year_bin


def test_year_bin_edges() -> None:
    assert _year_bin(1965) == "<=1979"
    assert _year_bin(1979) == "<=1979"
    assert _year_bin(1980) == "1980s"
    assert _year_bin(1999) == "1990s"
    assert _year_bin(2007) == "2000s"
    assert _year_bin(2019) == "2010s"
    assert _year_bin(2020) == ">=2020"
    assert _year_bin(None) is None
    assert _year_bin(float("nan")) is None


def test_top_labels_by_borehole_count_with_row_floor() -> None:
    rows = []
    # contractor A: 5 boreholes x many rows; B: 3 boreholes x many rows;
    # C: 10 boreholes but too few rows to clear MIN_FOLD_ROWS.
    for b in range(5):
        rows += [{"boring_file": f"a{b}", "surveyor_name": "A"}] * (MIN_FOLD_ROWS // 4)
    for b in range(3):
        rows += [{"boring_file": f"b{b}", "surveyor_name": "B"}] * (MIN_FOLD_ROWS // 2)
    for b in range(10):
        rows += [{"boring_file": f"c{b}", "surveyor_name": "C"}] * 2
    df = pd.DataFrame(rows)
    labels = _top_labels(df, "surveyor_name", top_n=3)
    # C has the most boreholes but fails the row floor; A and B qualify.
    assert "C" not in labels
    assert set(labels) == {"A", "B"}


# --------------------------------------------------------- shard / combine


def test_combine_ignores_non_shard_json_and_its_own_output(tmp_path):
    """A shard directory also collects incidental JSON.

    `--list-folds` writes a fold map there, and a previous `--combine` run
    leaves its own output there. Neither is a shard: the fold map has no
    per-fold blocks, and re-ingesting the combined output would trip the
    double-count guard and make `--combine` non-idempotent.
    """
    import json
    from scripts.nc_provenance_folds import combine

    shard = {"families": {"dtd": {
        "fold_col": "dtd_version",
        "per_fold": {"2.10": {"content_pct_block": -8.0,
                              "content_pct_row": -8.2},
                     "4.00": {"content_pct_block": -4.0,
                              "content_pct_row": -4.1}},
        "fold_p_block": [0.01, 0.02], "fold_p_row": [0.01, 0.02]}},
        "config": {"seeds": [42]}}
    (tmp_path / "dtd__a.json").write_text(json.dumps(shard))
    # a --list-folds map: right key, wrong shape underneath
    (tmp_path / "fold_map.json").write_text(json.dumps(
        {"families": {"dtd": ["2.10", "4.00"]}}))

    out = tmp_path / "combined.json"
    first = combine(tmp_path, out)
    assert first["families"]["dtd"]["n_folds"] == 2
    assert first["config"]["combined_from_shards"] == ["dtd__a.json"]

    # idempotent: the output now sits in the shard dir and must be ignored
    second = combine(tmp_path, out)
    assert second["families"]["dtd"] == first["families"]["dtd"]
    assert second["config"]["combined_from_shards"] == ["dtd__a.json"]


def test_combine_refuses_a_fold_present_in_two_shards(tmp_path):
    import json
    from scripts.nc_provenance_folds import combine

    blk = {"fold_col": "dtd_version",
           "per_fold": {"2.10": {"content_pct_block": -8.0}},
           "fold_p_block": [0.01], "fold_p_row": [0.01]}
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps({"families": {"dtd": blk}}))
    with pytest.raises(SystemExit, match="more than one shard"):
        combine(tmp_path, tmp_path / "out.json")


def test_combine_errors_when_nothing_is_a_shard(tmp_path):
    import json
    from scripts.nc_provenance_folds import combine

    (tmp_path / "fold_map.json").write_text(json.dumps(
        {"families": {"dtd": ["2.10"]}}))
    with pytest.raises(SystemExit, match="no --shard outputs"):
        combine(tmp_path, tmp_path / "out.json")
