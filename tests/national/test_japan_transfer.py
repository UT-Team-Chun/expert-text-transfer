"""Tests for the Japan leak-proof transfer driver (scripts.japan_transfer_test).

The Japanese domain is rebuilt locally by (a) assigning each borehole to one of the
eight standard regions from its coordinates, and (b) re-parsing SPT N-values from the
KuniJiban XML and joining each SPT depth to the layer text that contains it. These
tests pin the region assignment, the path resolution, and the XML SPT extraction +
depth-interval join (including the DTD tag-family fallback and the N-range filter).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.japan_transfer_test import _region_of, _resolve_xml, _spt_rows_for_file


def test_region_of_assigns_japanese_blocks():
    assert _region_of(43.06, 141.35) == "hokkaido"   # Sapporo
    assert _region_of(35.68, 139.69) == "kanto"       # Tokyo
    assert _region_of(26.21, 127.68) == "kyushu_okinawa"  # Okinawa
    assert _region_of(0.0, 0.0) is None               # off Japan


def test_resolve_xml_maps_to_repo_data():
    p = _resolve_xml("../data/kunijiban/xml/107283_xml.html")
    assert p.parts[-3:] == ("kunijiban", "xml", "107283_xml.html")
    assert "data" in p.parts


# Minimal KuniJiban-style XML: two SPT measurements (4.00-DTD 100-200/200-300 tags),
# one a refusal-like 0, plus a layer table so the depth-interval join can attach text.
_XML = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<ボーリング情報>\n'
    '  <標準貫入試験><標準貫入試験_開始深度>1.15</標準貫入試験_開始深度>'
    '<標準貫入試験_100_200打撃回数>6</標準貫入試験_100_200打撃回数>'
    '<標準貫入試験_200_300打撃回数>10</標準貫入試験_200_300打撃回数></標準貫入試験>\n'
    '  <標準貫入試験><標準貫入試験_開始深度>3.0</標準貫入試験_開始深度>'
    '<標準貫入試験_100_200打撃回数>0</標準貫入試験_100_200打撃回数>'
    '<標準貫入試験_200_300打撃回数>0</標準貫入試験_200_300打撃回数></標準貫入試験>\n'
    '</ボーリング情報>\n'
)


def _layers_df() -> pd.DataFrame:
    return pd.DataFrame({
        "file_path": ["x", "x"],
        "latitude_deg": [43.30, 43.30], "longitude_deg": [140.60, 140.60],
        "depth_top_m": [0.0, 1.0], "depth_bottom_m": [1.0, 2.85],
        "observation_text": ["topsoil", "gravelly silt with sub-angular clasts"],
    })


def test_spt_rows_extracts_n_and_joins_layer_text(tmp_path: Path):
    xml = tmp_path / "x_xml.html"
    xml.write_text(_XML, encoding="utf-8")
    rows = _spt_rows_for_file(xml, _layers_df())
    # The 1.15 m SPT has N=6+10=16 and falls in the [1.0, 2.85) layer; the 3.0 m
    # measurement has all-zero increments (N=0) and is dropped by the 0<N<=100 filter.
    assert len(rows) == 1
    r = rows[0]
    assert r["n_value"] == 16.0
    assert r["depth_from_surface"] == 1.15
    assert "gravelly silt" in r["text"]
    assert r["latitude_deg"] == 43.30


def test_spt_rows_empty_on_unparseable(tmp_path: Path):
    bad = tmp_path / "bad_xml.html"
    bad.write_text("not xml at all <<<", encoding="utf-8")
    assert _spt_rows_for_file(bad, _layers_df()) == []
