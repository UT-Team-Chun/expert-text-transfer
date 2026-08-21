"""Tests for the Storm Events third-domain transfer test (scripts.storm_transfer_test).

The breadth claim ("the text-content effect holds beyond geotechnics") rests on (a) the
size-descriptor stripper actually removing the colloquial/numeric hail-size cues that
would otherwise let the model read the answer off the text, and (b) the ingest building
the (lat, lon, target, region, text) schema the leak-proof evaluator consumes. These
tests pin both.
"""

from __future__ import annotations

import gzip
from pathlib import Path

from scripts.storm_transfer_test import _strip_size, ingest


def test_strip_size_removes_colloquial_and_numeric_cues():
    assert "golf" not in _strip_size("Golf ball size hail fell").lower()
    assert "ping" not in _strip_size("Ping-pong ball size hail").lower()
    assert "nickel" not in _strip_size("Nickel size hail reported").lower()
    # numeric diameters
    assert "1.5" not in _strip_size("Hail to 1.5 inches in diameter")
    assert "inch" not in _strip_size("2 inch hail observed").lower()
    # quarter/half fraction-size words
    assert "quarter" not in _strip_size("Quarter size hail").lower()


def test_strip_size_preserves_nonsize_content():
    s = _strip_size("Hail caused roof and vehicle damage near the airport")
    for w in ("hail", "roof", "vehicle", "damage", "airport"):
        assert w in s.lower()


def _write_fake_storm(dir_: Path, n_per_state: int = 250) -> Path:
    """Two states with enough hail events to pass min_state=200, plus a sparse state."""
    header = ("STATE,EVENT_TYPE,MAGNITUDE,BEGIN_LAT,BEGIN_LON,EVENT_NARRATIVE")
    rows = [header]
    for st, lat in (("KANSAS", 38.5), ("TEXAS", 31.0)):
        for i in range(n_per_state):
            rows.append(f"{st},Hail,{1.0 + (i % 5) * 0.25},{lat},-97.0,"
                        f"Golf ball size hail damaged roofs near town {i}.")
    # sparse state (dropped by min_state) + a non-hail event (dropped by type)
    rows.append("ALASKA,Hail,1.0,61.0,-149.0,Pea size hail briefly.")
    rows.append("KANSAS,Tornado,3,38.6,-97.1,A tornado touched down.")
    p = dir_ / "StormEvents_details-ftp_v1.0_d2099_c20990101.csv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("\n".join(rows) + "\n")
    return p


def test_ingest_builds_schema_and_filters(tmp_path: Path):
    _write_fake_storm(tmp_path)
    df = ingest(tmp_path, event_type="Hail", min_state=200)
    assert set(["latitude_deg", "longitude_deg", "n_value", "region", "text"]).issubset(df.columns)
    # KANSAS + TEXAS kept; ALASKA dropped (sparse); Tornado dropped (wrong type)
    assert set(df["region"].unique()) == {"KANSAS", "TEXAS"}
    assert (df["n_value"] > 0).all()


def test_ingest_strip_size_changes_text(tmp_path: Path):
    _write_fake_storm(tmp_path)
    raw = ingest(tmp_path, event_type="Hail", min_state=200, strip_size=False)
    stripped = ingest(tmp_path, event_type="Hail", min_state=200, strip_size=True)
    assert any("golf" in t.lower() for t in raw["text"])
    assert not any("golf" in t.lower() for t in stripped["text"])
