#!/usr/bin/env python
"""Depth-interval join: per-layer soil-text embeddings -> per-row boring parquet.

Paper B' Pillar 3 v2 downstream step. The BERT embedding pipeline
(``scripts/embed_soil_text.py``, mode=per_layer) produces a parquet
keyed by (file_path, lat, lon, layer_idx, depth_top_m, depth_bottom_m,
embed_0..N). The training data (v4 ``borings_japan_v4.parquet``) is
keyed by (lat, lon, depth_from_surface, ...). This script joins
the two: for each per-depth boring row, find the matching layer
``L`` such that ``L.depth_top_m <= depth_from_surface < L.depth_bottom_m``
at the same (lat, lon), and attach ``L.embed_*`` as new columns.

Output: a new parquet (v5 schema) with the v4 columns + ``embed_0..N``.
Rows that don't match any layer get ``NaN`` embedding columns;
downstream BoringDataset can mask those out via the missing-feature
path.

Match strategy (``--join-key``):

- ``file`` (default; NC pre-review response R0-1, 2026-08-11): EXACT join on
  the borehole identity -- ``merge_asof(by=['boring_file'], ...)`` where
  ``boring_file`` is the XML basename carried by the v4id parquet
  (``scripts.attach_identity_to_parquet``) and derived from the embedding
  parquet's ``file_path``. A layer can only ever attach to SPT rows of its
  own borehole.
- ``coord`` (legacy): ``merge_asof(by=['lat_r', 'lon_r'], ...)`` on
  4-decimal-rounded coordinates. The join audit showed 9.0% of these 10 m
  keys hold >1 borehole and ~15% of boreholes share byte-identical
  coordinates, so under this mode ~16% of matched rows draw their
  description arbitrarily from a *different* borehole at the same key.
  Retained only so the audit can reproduce and diff the legacy behaviour.

Both modes post-filter ``depth_from_surface < depth_bottom_m`` to enforce
the half-open layer interval.

Example::

    cd backend
    .venv/bin/python -m scripts.join_soil_text_to_parquet \\
        --boring-parquet ../data/features/borings_japan_v4id.parquet \\
        --embedding-parquet ../data/features/derived/soil_text_embed_ruri_full.parquet \\
        --output ../data/features/borings_japan_v5_ruri.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("scripts.join_soil_text_to_parquet")


def _file_key(series: pd.Series) -> pd.Series:
    """Normalise an XML path to its basename (`107283_xml.html`).

    The SPT side stores `data/kunijiban/xml/...` and the text side
    `../data/kunijiban/xml/...`; the basename is the corpus-wide 1:1
    borehole key.
    """
    return (
        series.astype(str).str.replace("\\", "/", regex=False)
        .str.rsplit("/", n=1).str[-1]
    )


def run(
    boring_parquet: Path,
    embedding_parquet: Path,
    output_parquet: Path,
    *,
    join_key: str = "file",
    coord_decimals: int = 4,
) -> int:
    """Depth-interval join + write the text-augmented parquet.

    Returns the number of rows written (= len(boring_parquet)).
    """
    if join_key not in ("file", "coord"):
        raise ValueError(f"join_key must be 'file' or 'coord', got {join_key!r}")
    LOG.info("Reading boring parquet %s", boring_parquet)
    boring = pd.read_parquet(boring_parquet)
    LOG.info("  %d rows × %d cols", len(boring), boring.shape[1])

    LOG.info("Reading embedding parquet %s", embedding_parquet)
    emb = pd.read_parquet(embedding_parquet)
    LOG.info("  %d rows × %d cols", len(emb), emb.shape[1])

    required = {"latitude_deg", "longitude_deg", "depth_top_m", "depth_bottom_m"}
    missing = required - set(emb.columns)
    if missing:
        raise ValueError(
            f"Embedding parquet missing per-layer columns {sorted(missing)}; "
            "did you embed with mode=per_layer?"
        )
    embed_cols = sorted(c for c in emb.columns if c.startswith("embed_"))
    LOG.info("  found %d embedding dimensions", len(embed_cols))
    if not embed_cols:
        raise ValueError("Embedding parquet has no embed_* columns")

    boring = boring.copy()
    emb = emb.copy()
    if join_key == "file":
        if "boring_file" not in boring.columns:
            raise ValueError(
                "join_key='file' needs a `boring_file` column on the boring "
                "parquet -- use borings_japan_v4id.parquet "
                "(scripts.attach_identity_to_parquet) or pass "
                "--join-key coord to reproduce the legacy behaviour."
            )
        if "file_path" not in emb.columns:
            raise ValueError(
                "join_key='file' needs `file_path` on the embedding parquet "
                "(scripts.embed_soil_text has carried it since the per-layer "
                "mode landed); re-embed or pass --join-key coord."
            )
        boring["_grp"] = _file_key(boring["boring_file"])
        emb["_grp"] = _file_key(emb["file_path"])
        by_cols = ["_grp"]
        LOG.info("join mode: EXACT borehole identity (boring_file basename)")
    else:
        # Round coords to 4 decimals (=~10 m) via float64 cast first --
        # mirrors enrich.py's _lookup_aist_legend fix so float32 round-trips
        # don't drop rows. See commit 13ffa36. WARNING: this key merges
        # distinct boreholes (see module docstring); audit-only.
        boring["lat_r"] = boring["latitude_deg"].astype("float64").round(coord_decimals)
        boring["lon_r"] = boring["longitude_deg"].astype("float64").round(coord_decimals)
        emb["lat_r"] = emb["latitude_deg"].astype("float64").round(coord_decimals)
        emb["lon_r"] = emb["longitude_deg"].astype("float64").round(coord_decimals)
        by_cols = ["lat_r", "lon_r"]
        LOG.info("join mode: LEGACY rounded coordinates (%d decimals)", coord_decimals)

    # merge_asof requires both sides globally sorted by the `on` column
    # (it then uses `by` to restrict matches to the same group). Sorting
    # by (group, depth) is NOT enough -- pandas complains
    # "left keys must be sorted" because the global depth axis goes
    # backward between borings. Solution: sort by `on` alone; `by`
    # handles the cross-boring isolation correctly.
    boring = boring.sort_values("depth_from_surface").reset_index(drop=False)
    emb = emb.sort_values("depth_top_m").reset_index(drop=True)

    LOG.info("Running merge_asof on depth (backward fill, half-open interval)")
    merged = pd.merge_asof(
        boring,
        emb[by_cols + ["depth_top_m", "depth_bottom_m"] + embed_cols],
        by=by_cols,
        left_on="depth_from_surface",
        right_on="depth_top_m",
        direction="backward",
        allow_exact_matches=True,
    )

    # Post-filter: enforce the half-open interval. merge_asof gives the
    # most-recent depth_top_m <= depth_from_surface; we need
    # depth_from_surface < depth_bottom_m to be in the same layer.
    bad = merged["depth_from_surface"] >= merged["depth_bottom_m"]
    n_bad = int(bad.fillna(True).sum())
    LOG.info(
        "Out-of-interval matches (depth >= depth_bottom): %d / %d (%.1f%%) -> NaN",
        n_bad, len(merged), 100.0 * n_bad / max(len(merged), 1),
    )
    for col in embed_cols + ["depth_top_m", "depth_bottom_m"]:
        merged.loc[bad, col] = np.nan
    n_matched = int(merged[embed_cols[0]].notna().sum())
    LOG.info(
        "Matched %d / %d boring rows (%.1f%%) to a layer embedding",
        n_matched, len(merged), 100.0 * n_matched / max(len(merged), 1),
    )

    # Drop the helper columns + restore original row order.
    merged = merged.sort_values("index").drop(
        columns=[c for c in ("index", "lat_r", "lon_r", "_grp") if c in merged.columns]
    ).reset_index(drop=True)
    # Drop the depth_top_m/depth_bottom_m columns from the embedding
    # side (they're diagnostic-only; the boring parquet already has
    # depth_from_surface).
    merged = merged.drop(columns=["depth_top_m", "depth_bottom_m"], errors="ignore")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_parquet, index=False)
    LOG.info(
        "Wrote %d rows × %d cols (%d embedding dims) to %s",
        len(merged), merged.shape[1], len(embed_cols), output_parquet,
    )
    return len(merged)


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boring-parquet",
        type=Path,
        default=repo / "data" / "features" / "borings_japan_v4id.parquet",
        help="Identity-bearing boring parquet (v4id). The plain v4 parquet "
             "only works with --join-key coord.",
    )
    parser.add_argument(
        "--embedding-parquet",
        type=Path,
        required=True,
        help="Per-layer embedding parquet (output of "
             "scripts.embed_soil_text in mode=per_layer).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination v5 parquet path.",
    )
    parser.add_argument(
        "--join-key",
        choices=("file", "coord"),
        default="file",
        help="'file' (default): exact borehole-identity join via boring_file "
             "/ file_path basenames. 'coord': legacy 10 m rounded-coordinate "
             "join, retained for the audit only -- it attaches ~16%% of "
             "matched rows to a different borehole's description.",
    )
    parser.add_argument(
        "--coord-decimals",
        type=int,
        default=4,
        help="Decimal places to round (lat, lon) for --join-key coord. "
             "Default 4 (~10 m precision), matching the AIST cache join "
             "convention.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s"
    )
    run(
        args.boring_parquet,
        args.embedding_parquet,
        args.output,
        join_key=args.join_key,
        coord_decimals=args.coord_decimals,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
