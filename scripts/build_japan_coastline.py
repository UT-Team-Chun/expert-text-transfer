#!/usr/bin/env python
"""Build the simplified Japan coastline asset used by the paper basemaps.

Reads the MLIT C23 per-prefecture coastline shapefiles
(``data/raw/mlit/C23-06/C23-06_<PP>_GML.zip``, 74,831 polylines), merges them
into continuous rings, simplifies with Douglas-Peucker, drops islets below a
length threshold, and writes a compact JSON of ``[[ [lon, lat], ... ], ...]``
polylines.

Why an asset rather than reading C23 at figure-build time:

- ``data/`` is git-ignored, and the companion reproduction repository ships
  without the 47 raw zips, so a figure build cannot depend on them.
- Reading + merging C23 costs several seconds and needs geopandas/shapely;
  drawing a basemap should need neither.

The previous basemap was ``_JAPAN_RINGS`` in ``build_paper2_figs.py`` -- 100
hand-digitised vertices described in its own comment as "a per-island
convex-ish hull". At locator-inset scale that hull does not read as Japan.
It is kept as a fallback for environments without the asset.

Tolerance was chosen by rendering 0.005 / 0.01 / 0.02 deg side by side:
0.01 deg keeps the Seto Inland Sea, Noto, Boso, Shimokita and the Hokkaido
peninsulas legible at 109 KB; 0.02 deg starts rounding them off.

CLI:

.. code-block:: bash

    cd backend && .venv/bin/python -m scripts.build_japan_coastline
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

LOG = logging.getLogger("scripts.build_japan_coastline")

REPO = Path(__file__).resolve().parents[2]
DEFAULT_COAST_DIR = REPO / "data/raw/mlit/C23-06"
DEFAULT_OUT = (REPO / "backend/national/data/assets/japan_coastline.json")

#: Douglas-Peucker tolerance and minimum retained polyline length, in degrees.
SIMPLIFY_TOL = 0.01
MIN_LENGTH = 0.08
#: Coordinate rounding: 4 dp is ~11 m, far finer than any figure needs.
COORD_DP = 4


def build(coast_dir: Path, out_path: Path) -> int:
    from shapely.ops import linemerge, unary_union

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from national.data.derived.distances import load_coastlines_from_mlit_dir

    gdf = load_coastlines_from_mlit_dir(coast_dir)
    LOG.info("read %d coastline polylines (crs=%s)", len(gdf), gdf.crs)

    merged = linemerge(unary_union(gdf.geometry.values))
    parts = list(merged.geoms) if merged.geom_type == "MultiLineString" \
        else [merged]
    LOG.info("merged into %d continuous parts", len(parts))

    rings: list[list[list[float]]] = []
    for geom in parts:
        simple = geom.simplify(SIMPLIFY_TOL)
        if simple.length < MIN_LENGTH:
            continue
        rings.append([[round(x, COORD_DP), round(y, COORD_DP)]
                      for x, y in simple.coords])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rings, separators=(",", ":")))
    n_pts = sum(len(r) for r in rings)
    LOG.info("wrote %s: %d parts, %d vertices, %.0f KB",
             out_path, len(rings), n_pts,
             out_path.stat().st_size / 1024)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coast-dir", type=Path, default=DEFAULT_COAST_DIR,
                        help="Directory of C23-06_<PP>_GML.zip files.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    return build(args.coast_dir, args.out)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
