#!/usr/bin/env python
"""Build the per-boring soil-text narrative CSV from raw KuniJiban XML.

Walks ``data/kunijiban/xml/*.html`` (KuniJiban serves XML with .html
extensions, mixed DTD versions 1.10 - 4.00), extracts the per-boring
concatenated ``<観察記事_記事>`` narrative via
:func:`national.data.derived.soil_text_xml.extract_soil_text`, and
writes a single CSV with columns

* ``file_path``         -- source XML absolute path
* ``latitude_deg``      -- WGS84 decimal degrees (north)
* ``longitude_deg``     -- WGS84 decimal degrees (east)
* ``observation_text``  -- concatenated layer narratives, layer-separated by " || "
* ``n_layers``          -- count of non-empty narrative chunks
* ``char_length``       -- byte length of the concatenated narrative

61% of borings carry non-empty narrative (audit on a 200-file random
sample of the live corpus). Sequential walk handles ~1500 files/s on
typical hardware; multiprocess (--workers N) takes the 191 k corpus
from ~2 min to ~30 s.

Example::

    cd backend
    .venv/bin/python -m scripts.extract_soil_text_from_xml \\
        --xml-dir ../data/kunijiban/xml \\
        --output ../data/features/derived/soil_text.csv \\
        --workers 8
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from national.data.derived.soil_text_xml import (
    SoilTextLayerRecord,
    SoilTextRecord,
    extract_soil_text,
    extract_soil_text_layers,
)

LOG = logging.getLogger("scripts.extract_soil_text_from_xml")


def _csv_row(rec: SoilTextRecord) -> list[str]:
    return [
        rec.file_path,
        f"{rec.latitude_deg:.7f}" if rec.latitude_deg == rec.latitude_deg else "",
        f"{rec.longitude_deg:.7f}" if rec.longitude_deg == rec.longitude_deg else "",
        rec.observation_text,
        str(rec.n_layers),
        str(rec.char_length),
    ]


def _layer_csv_rows(recs: list[SoilTextLayerRecord]) -> list[list[str]]:
    """Convert a list of per-layer records into CSV rows. Empty list -> empty list."""
    rows: list[list[str]] = []
    for r in recs:
        rows.append(
            [
                r.file_path,
                f"{r.latitude_deg:.7f}",
                f"{r.longitude_deg:.7f}",
                str(r.layer_idx),
                f"{r.depth_top_m:.3f}",
                f"{r.depth_bottom_m:.3f}",
                r.observation_text,
                str(r.char_length),
            ]
        )
    return rows


def run(
    xml_dir: Path,
    output_csv: Path,
    *,
    workers: int = 1,
    suffix: str = "html",
    limit: int | None = None,
    mode: str = "per_layer",
) -> int:
    """Walk ``xml_dir``, extract soil text, write CSV.

    Args:
        mode: ``per_layer`` (default) writes one CSV row per
            ``<観察記事>`` block (file_path, lat, lon, layer_idx,
            depth_top_m, depth_bottom_m, observation_text, char_length).
            ``per_boring`` writes one CSV row per file with all layers'
            text joined by ` || ` (legacy / depth-agnostic format).

    Returns the number of CSV rows written (= layers for per_layer mode,
    files for per_boring mode).
    """
    if mode not in ("per_layer", "per_boring"):
        raise ValueError(f"mode must be 'per_layer' or 'per_boring'; got {mode!r}")

    pattern = f"*.{suffix.lstrip('.')}"
    paths = sorted(xml_dir.rglob(pattern))
    LOG.info("Found %d XML files in %s (pattern=%s)", len(paths), xml_dir, pattern)
    if limit is not None:
        paths = paths[:limit]
        LOG.info("Limited to first %d files via --limit", limit)
    if not paths:
        raise RuntimeError(f"No XML files matched under {xml_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if mode == "per_layer":
        n_rows, n_files_with_layers = _run_per_layer(paths, output_csv, workers)
        LOG.info(
            "per_layer: wrote %d layer rows across %d files (%.1f%% had >=1 layer)",
            n_rows, len(paths), 100.0 * n_files_with_layers / max(len(paths), 1),
        )
        return n_rows

    return _run_per_boring(paths, output_csv, workers)


def _run_per_layer(
    paths: list[Path], output_csv: Path, workers: int
) -> tuple[int, int]:
    t0 = time.time()
    n_rows = 0
    n_files_with_layers = 0
    with output_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                "file_path",
                "latitude_deg",
                "longitude_deg",
                "layer_idx",
                "depth_top_m",
                "depth_bottom_m",
                "observation_text",
                "char_length",
            ]
        )
        if workers <= 1:
            for i, p in enumerate(paths):
                recs = extract_soil_text_layers(p)
                if recs:
                    n_files_with_layers += 1
                    for row in _layer_csv_rows(recs):
                        writer.writerow(row)
                        n_rows += 1
                if (i + 1) % 5000 == 0:
                    LOG.info(
                        "Progress %d / %d files; %d layer rows so far",
                        i + 1, len(paths), n_rows,
                    )
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(extract_soil_text_layers, p): p for p in paths}
                for i, fut in enumerate(as_completed(futures)):
                    recs = fut.result()
                    if recs:
                        n_files_with_layers += 1
                        for row in _layer_csv_rows(recs):
                            writer.writerow(row)
                            n_rows += 1
                    if (i + 1) % 5000 == 0:
                        LOG.info(
                            "Progress %d / %d files; %d layer rows so far",
                            i + 1, len(paths), n_rows,
                        )
    LOG.info(
        "per_layer: %.1f s wall-clock, mean %.1f layers per file (with-layer files only)",
        time.time() - t0, n_rows / max(n_files_with_layers, 1),
    )
    return n_rows, n_files_with_layers


def _run_per_boring(paths: list[Path], output_csv: Path, workers: int) -> int:
    """Legacy per-boring CSV writer (one row per file, all layers concatenated)."""
    t0 = time.time()
    n_records = 0
    n_with_text = 0
    n_with_coords = 0
    total_chars = 0
    with output_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                "file_path",
                "latitude_deg",
                "longitude_deg",
                "observation_text",
                "n_layers",
                "char_length",
            ]
        )
        if workers <= 1:
            for i, p in enumerate(paths):
                rec = extract_soil_text(p)
                writer.writerow(_csv_row(rec))
                n_records += 1
                if rec.n_layers > 0:
                    n_with_text += 1
                    total_chars += rec.char_length
                if rec.latitude_deg == rec.latitude_deg:
                    n_with_coords += 1
                if (i + 1) % 5000 == 0:
                    LOG.info("Progress %d / %d files", i + 1, len(paths))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(extract_soil_text, p): p for p in paths}
                for i, fut in enumerate(as_completed(futures)):
                    rec = fut.result()
                    writer.writerow(_csv_row(rec))
                    n_records += 1
                    if rec.n_layers > 0:
                        n_with_text += 1
                        total_chars += rec.char_length
                    if rec.latitude_deg == rec.latitude_deg:
                        n_with_coords += 1
                    if (i + 1) % 5000 == 0:
                        LOG.info("Progress %d / %d files", i + 1, len(paths))
    LOG.info(
        "per_boring: %d records in %.1f s (text coverage %.1f%%)",
        n_records, time.time() - t0, 100.0 * n_with_text / max(n_records, 1),
    )
    return n_records


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=repo / "data" / "kunijiban" / "xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "data" / "features" / "derived" / "soil_text.csv",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Multiprocess workers. Default 1 (sequential).",
    )
    parser.add_argument(
        "--suffix",
        default="html",
        help="Filename suffix to match (KuniJiban uses 'html' for served XML).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of files (for smoke tests).",
    )
    parser.add_argument(
        "--mode",
        choices=("per_layer", "per_boring"),
        default="per_layer",
        help="per_layer (default): one CSV row per <観察記事> block with "
             "depth bounds. per_boring (legacy): one row per file with "
             "all layers' text concatenated by ` || `.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(
        args.xml_dir,
        args.output,
        workers=args.workers,
        suffix=args.suffix,
        limit=args.limit,
        mode=args.mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
