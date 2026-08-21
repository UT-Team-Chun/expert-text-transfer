#!/usr/bin/env python
"""National KuniJiban header-metadata extraction (all 191,572 XMLs).

NC pre-review response R0-1 (2026-08-11). The pre-review claimed
"survey year/agency are not in the public schema"; a field-presence audit on
a 4,000-file random sample refuted that: `調査名` (project), `発注機関名称`
(client) and `調査会社_名称` (contractor) are 98-100% populated in EVERY DTD
version, and a survey date is recoverable for ~100% of files once two parser
defects are fixed. This script extracts the per-borehole grouping metadata
that the provenance-transfer folds (leave-project/-client/-contractor/-year/
-DTD-out) and the borehole-grouped nulls need.

Fixes over the Kanto-only precedent (`run_raw_n_defence_phase_n.py`):

1. **v1.10 date form**: v1.10 has no `調査期間_開始年月日`; it splits the date
   into `調査期間_開始年/月/日` (and `_終了年/月/日`). Reassembled here.
2. **End-date fallback**: `調査期間_開始年月日` is empty in ~48% of v2.10
   files while `調査期間_終了年月日` is ~100%; `survey_year` prefers the
   start date and falls back to the end date.
3. **Project fields**: `調査名`, `事業工事名`, `ボーリング名`,
   `ボーリング連番`, `テクリスコード` (v4.00, a registered national project
   ID) are extracted; the Kanto script had none of them.

Output columns (one row per XML file):

    boring_file, dtd_version, survey_name, project_name, boring_name,
    boring_serial, tecris_code, orderer_name, orderer_bucket, surveyor_name,
    survey_start_date, survey_end_date, survey_year,
    latitude_deg, longitude_deg

CLI::

    cd backend
    .venv/bin/python -m scripts.extract_kunijiban_metadata \
        --xml-dir ../data/kunijiban/xml \
        --out ../data/features/derived/kunijiban_metadata.parquet
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("scripts.extract_kunijiban_metadata")

REPO = Path(__file__).resolve().parents[2]

#: Tags shared by every DTD version (presence 98-100% except where noted).
_SIMPLE_TAGS = {
    "調査名": "survey_name",
    "事業工事名": "project_name",       # 23-76% populated
    "ボーリング名": "boring_name",       # empty in ~59% of v2.10
    "ボーリング連番": "boring_serial",
    "テクリスコード": "tecris_code",     # 85% in v4.00, ~0 elsewhere
    "発注機関名称": "orderer_name",
    "調査会社_名称": "surveyor_name",
    "調査期間_開始年月日": "survey_start_date",
    "調査期間_終了年月日": "survey_end_date",
}

#: v1.10 splits dates into year/month/day elements.
_V110_DATE_PARTS = {
    "survey_start_date": ("調査期間_開始年", "調査期間_開始月", "調査期間_開始日"),
    "survey_end_date": ("調査期間_終了年", "調査期間_終了月", "調査期間_終了日"),
}


def bucket_orderer(name: str | None) -> str:
    """Coarse client bucket (mirrors run_raw_n_defence_phase_n)."""
    if not name:
        return "other"
    n = name
    if "国土交通省" in n or "国交省" in n or "国土交通" in n:
        return "mlit"
    if "NEXCO" in n or "高速道路" in n or "首都高" in n:
        return "nexco"
    if "JR" in n or "鉄道" in n or "東日本旅客鉄道" in n:
        return "jr"
    if "都" in n or "府" in n or "県" in n:
        if "市" in n or "町" in n or "村" in n:
            return "munic"
        return "pref"
    if "市" in n or "町" in n or "村" in n or "区" in n:
        return "munic"
    return "other"


def _decode(raw: bytes) -> str | None:
    """KuniJiban XMLs are UTF-8 on disk despite declaring Shift_JIS."""
    for enc in ("utf-8", "shift_jis", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def _tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>([^<]*)</{tag}>", text)
    if m:
        v = m.group(1).strip()
        return v or None
    return None


def parse_file(path: Path) -> dict | None:
    """Extract the metadata row for one XML. Returns None on I/O failure."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    text = _decode(raw)
    if text is None:
        return None

    out: dict[str, object] = {"boring_file": path.name}
    m = re.search(r'DTD_version="([^"]+)"', text)
    out["dtd_version"] = m.group(1) if m else None
    for tag, key in _SIMPLE_TAGS.items():
        out[key] = _tag(text, tag)

    # v1.10 (and any file missing the combined form): reassemble Y/M/D parts.
    for key, parts in _V110_DATE_PARTS.items():
        if out.get(key):
            continue
        y, mo, d = (_tag(text, t) for t in parts)
        if y and y.isdigit():
            mo_i = int(mo) if mo and mo.isdigit() else 1
            d_i = int(d) if d and d.isdigit() else 1
            out[key] = f"{int(y):04d}-{mo_i:02d}-{d_i:02d}"

    # survey_year: start date preferred, end date fallback (48% of v2.10
    # files lack a start date but carry an end date).
    year = None
    for key in ("survey_start_date", "survey_end_date"):
        v = out.get(key)
        if isinstance(v, str):
            m = re.match(r"(\d{4})", v)
            if m:
                candidate = int(m.group(1))
                # Sanity window: the corpus contains typos like year 0 and
                # 2306; treat anything outside the plausible survey era as
                # missing rather than poisoning leave-year-out folds.
                if 1930 <= candidate <= 2030:
                    year = candidate
                    break
    out["survey_year"] = year
    out["orderer_bucket"] = bucket_orderer(out.get("orderer_name"))  # type: ignore[arg-type]
    # Composite project key: `調査名` is ~100% populated corpus-wide but
    # individual files can still be empty (e.g. the curated 3.00 sample,
    # which carries the project only in `事業工事名`). Fall back through the
    # project-identifying fields; None only when all are empty.
    out["project_key"] = (
        out.get("survey_name") or out.get("project_name") or out.get("tecris_code")
    )

    # Coordinates (DMS -> decimal), for sanity joins only -- identity is the
    # filename.
    lat = lon = None
    lat_m = re.search(
        r"<緯度_度>(\d+)</緯度_度>\s*<緯度_分>(\d+)</緯度_分>\s*<緯度_秒>([\d.]+)</緯度_秒>",
        text,
    )
    lon_m = re.search(
        r"<経度_度>(\d+)</経度_度>\s*<経度_分>(\d+)</経度_分>\s*<経度_秒>([\d.]+)</経度_秒>",
        text,
    )
    if lat_m:
        lat = int(lat_m.group(1)) + int(lat_m.group(2)) / 60 + float(lat_m.group(3)) / 3600
    if lon_m:
        lon = int(lon_m.group(1)) + int(lon_m.group(2)) / 60 + float(lon_m.group(3)) / 3600
    out["latitude_deg"] = lat
    out["longitude_deg"] = lon
    return out


def run(xml_dir: Path, out_path: Path, workers: int = 8,
        limit: int | None = None) -> pd.DataFrame:
    files = sorted(xml_dir.glob("*.html")) + sorted(xml_dir.glob("*.xml"))
    if limit:
        files = files[:limit]
    LOG.info("extracting metadata from %d XMLs with %d workers", len(files), workers)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, row in enumerate(ex.map(parse_file, files, chunksize=256)):
            if row is not None:
                rows.append(row)
            if (i + 1) % 20000 == 0:
                LOG.info("  %d / %d", i + 1, len(files))
    df = pd.DataFrame(rows)
    for c in ("dtd_version", "orderer_bucket"):
        df[c] = df[c].astype("category")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    LOG.info("wrote %s: %d rows", out_path, len(df))
    for c in ("survey_name", "project_key", "orderer_name", "surveyor_name",
              "survey_year", "boring_name", "tecris_code"):
        cov = 100.0 * df[c].notna().mean()
        LOG.info("  %-18s %5.1f%% populated", c, cov)
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml-dir", type=Path, default=REPO / "data/kunijiban/xml")
    ap.add_argument("--out", type=Path,
                    default=REPO / "data/features/derived/kunijiban_metadata.parquet")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only parse the first N files (smoke runs).")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    run(args.xml_dir, args.out, workers=args.workers, limit=args.limit)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
