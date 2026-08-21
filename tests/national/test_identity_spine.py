"""Tests for the borehole-identity spine (NC pre-review response R0-1).

Covers, in order:

1. ``enrich(include_identity=True)`` carries ``boring_file`` (XML basename)
   and ``dtd_version`` through to the parquet, as pandas categoricals, and
   the basename normalisation strips both ``data/...`` and ``../data/...``
   prefixes (the SPT-side and text-side CSVs disagree on the prefix).
2. ``include_identity=False`` (the default) reproduces the legacy schema
   exactly -- regression guard for every v2/v3/v4 consumer.
3. ``include_identity=True`` on a CSV without identity columns fails loudly
   (not silently NaN).
4. ``scripts.attach_identity_to_parquet`` equality gate: accepts a CSV that
   reproduces the v4 rows and attaches identity; REJECTS a CSV whose rows
   drifted (simulating a stale/wrong upstream), instead of writing a
   silently-inconsistent v5.
5. The v5 output preserves every v4 column byte-for-byte even when two
   boreholes share byte-identical coordinates (the case that makes
   (lat, lon) an invalid borehole key in the first place).
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from national.data.enrich import EnrichmentSpec, enrich


# ------------------------------------------------------------------ helpers

_CSV_HEADER = [
    "file_path", "dtd_version", "longitude_deg", "latitude_deg",
    "mouth_elevation", "spt_start_depth", "n_value",
]


def _write_borings_csv(path: Path, rows: list[list]) -> Path:
    with path.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(_CSV_HEADER)
        w.writerows(rows)
    return path


def _rows_two_files_shared_coords() -> list[list]:
    """Two boreholes at byte-identical coordinates + one elsewhere.

    The shared-coordinate pair is exactly the pathology that broke the
    rounded-coordinate join: identity must survive to keep them apart.
    """
    return [
        # file A and file B at the SAME (lat, lon), different depths/N.
        ["data/kunijiban/xml/100001_xml.html", "2.10", 139.5, 35.5, 12.0, 1.0, 5.0],
        ["data/kunijiban/xml/100001_xml.html", "2.10", 139.5, 35.5, 12.0, 2.0, 7.0],
        ["../data/kunijiban/xml/100002_xml.html", "4.00", 139.5, 35.5, 12.0, 1.0, 30.0],
        ["../data/kunijiban/xml/100002_xml.html", "4.00", 139.5, 35.5, 12.0, 2.0, 33.0],
        # a third borehole at its own location
        ["data/kunijiban/xml/100003_xml.html", "3.00", 141.0, 40.0, 8.0, 1.5, 12.0],
    ]


# ------------------------------------------------------------------ enrich()


def test_enrich_carries_identity_columns(tmp_path: Path) -> None:
    borings = _write_borings_csv(tmp_path / "b.csv", _rows_two_files_shared_coords())
    out = tmp_path / "out.parquet"
    enrich(EnrichmentSpec(borings_csv=borings, output_parquet=out,
                          aist_granular=False, include_identity=True))
    df = pd.read_parquet(out)

    assert "boring_file" in df.columns and "dtd_version" in df.columns
    # basename normalisation: both prefix styles collapse to the bare name
    assert set(df["boring_file"].unique()) == {
        "100001_xml.html", "100002_xml.html", "100003_xml.html",
    }
    assert set(df["dtd_version"].astype(str).unique()) == {"2.10", "3.00", "4.00"}
    # identity separates the two boreholes that share coordinates
    shared = df[(df["latitude_deg"] == np.float32(35.5))]
    assert shared["boring_file"].nunique() == 2
    # categorical dtype so the 2.66M-row parquet stays cheap
    assert isinstance(df["boring_file"].dtype, pd.CategoricalDtype)
    assert isinstance(df["dtd_version"].dtype, pd.CategoricalDtype)


def test_enrich_default_schema_unchanged(tmp_path: Path) -> None:
    borings = _write_borings_csv(tmp_path / "b.csv", _rows_two_files_shared_coords())
    out = tmp_path / "out.parquet"
    enrich(EnrichmentSpec(borings_csv=borings, output_parquet=out,
                          aist_granular=False))
    df = pd.read_parquet(out)
    assert "boring_file" not in df.columns
    assert "dtd_version" not in df.columns
    assert list(df.columns) == [
        "latitude_deg", "longitude_deg", "depth_from_surface",
        "absolute_elevation", "n_value", "river_distance_km",
        "coast_distance_km", "regime_code",
    ]


def test_enrich_identity_missing_columns_fails_loudly(tmp_path: Path) -> None:
    # CSV without file_path/dtd_version (the legacy 5-column layout)
    borings = tmp_path / "b.csv"
    with borings.open("w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["longitude_deg", "latitude_deg", "mouth_elevation",
                    "spt_start_depth", "n_value"])
        w.writerow([139.5, 35.5, 12.0, 1.0, 5.0])
    with pytest.raises(KeyError, match="file_path"):
        enrich(EnrichmentSpec(borings_csv=borings,
                              output_parquet=tmp_path / "out.parquet",
                              aist_granular=False, include_identity=True))


# ------------------------------------- scripts.attach_identity_to_parquet


def _make_v4_like(tmp_path: Path, rows: list[list]) -> tuple[Path, Path]:
    """Build a v4-style parquet from `rows` (via the real enrich pipeline,
    identity OFF) plus the identity-bearing upstream CSV."""
    borings = _write_borings_csv(tmp_path / "src.csv", rows)
    v4 = tmp_path / "v4.parquet"
    enrich(EnrichmentSpec(borings_csv=borings, output_parquet=v4,
                          aist_granular=True))
    return v4, borings


def test_attach_identity_builds_v4id_superset(tmp_path: Path) -> None:
    from scripts.attach_identity_to_parquet import build

    rows = _rows_two_files_shared_coords()
    v4_path, csv_path = _make_v4_like(tmp_path, rows)
    out = tmp_path / "v4id.parquet"
    assert build(v4_path, csv_path, out) == 0

    v4 = pd.read_parquet(v4_path)
    v5 = pd.read_parquet(out)
    assert len(v5) == len(v4)
    assert list(v5.columns) == [*v4.columns, "boring_file", "dtd_version"]
    # every v4 column identical as a multiset (sort by all shared columns)
    key = list(v4.columns)
    a = v4.sort_values(key).reset_index(drop=True)
    b = v5[key].sort_values(key).reset_index(drop=True)
    for c in key:
        av, bv = a[c].to_numpy(), b[c].to_numpy()
        if np.issubdtype(av.dtype, np.floating):
            assert np.array_equal(av, bv, equal_nan=True), c
        else:
            assert np.array_equal(av, bv), c
    # the shared-coordinate boreholes stay distinguishable
    shared = v5[v5["latitude_deg"] == np.float32(35.5)]
    assert shared["boring_file"].nunique() == 2


def test_attach_identity_rejects_drifted_csv(tmp_path: Path) -> None:
    from scripts.attach_identity_to_parquet import build

    rows = _rows_two_files_shared_coords()
    v4_path, _ = _make_v4_like(tmp_path, rows)

    # A drifted upstream: same row count, one N value changed.
    bad_rows = [list(r) for r in rows]
    bad_rows[0][6] = 99.0  # n_value 5.0 -> 99.0
    bad_csv = _write_borings_csv(tmp_path / "bad.csv", bad_rows)

    with pytest.raises(SystemExit, match="differs from v4|no identity row"):
        build(v4_path, bad_csv, tmp_path / "v4id.parquet")


def test_attach_identity_rejects_wrong_rowcount(tmp_path: Path) -> None:
    from scripts.attach_identity_to_parquet import build

    rows = _rows_two_files_shared_coords()
    v4_path, _ = _make_v4_like(tmp_path, rows)
    short_csv = _write_borings_csv(tmp_path / "short.csv", rows[:-1])

    with pytest.raises(SystemExit, match="row-count mismatch"):
        build(v4_path, short_csv, tmp_path / "v4id.parquet")
