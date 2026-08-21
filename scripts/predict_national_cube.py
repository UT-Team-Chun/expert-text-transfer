#!/usr/bin/env python
"""Generate a national 3D probabilistic SPT-N cube + engineering maps.

Paper B' Pillar 5 headline driver. Takes a trained DKL+SVGP foundation
model artifact (the v2 hero, or any of the Phase D trained cells) and
produces the public artefact: a (statistic, depth, lat, lon) Zarr cube
covering the bbox of mainland Japan at a configurable grid resolution
+ four 2D engineering-application maps derived from the cube:

* **LPI** liquefaction potential (Iwasaki et al. 1984), at a chosen
  scenario PGA. The most engineering-relevant Paper B' deliverable:
  municipal liquefaction zoning is currently done at prefectural scale
  with patchy coverage and decades-old methodology; a model-driven
  national LPI raster is novel for Japan.
* **V_s30** time-averaged shear-wave velocity + NEHRP site class, the
  building-code site-amplification input (ASCE 7-22 / AIJ).
* **Bearing-stratum depth** (first depth at N ≥ 30, Japanese AIJ
  practice for permanent residential / commercial footings).
* **Meyerhof allowable bearing** at a standardised 1 m × 1 m
  residential footing at the bearing stratum (the kPa load a typical
  spread footing can carry without exceeding 25 mm settlement).

All maps are written as Cloud-Optimised GeoTIFF slices alongside the
master Zarr cube. The output directory layout is Zenodo-ready:

    <output_dir>/
      cube.zarr/                      -- (statistic, depth, lat, lon)
      maps/
        lpi_pga30.tif                 -- LPI scenario at PGA = 0.30 g
        vs30.tif                      -- V_s30 (m/s)
        nehrp_site_class.tif          -- A=1, B=2, C=3, D=4, E=5 (uint8)
        bearing_stratum_depth.tif     -- depth (m) to first N >= 30
        allowable_bearing_kpa.tif     -- Meyerhof q_a at the bearing stratum
      manifest.json                   -- per-layer provenance + scenario inputs

Example::

    cd backend
    uv run python -m scripts.predict_national_cube \\
        --model ../data/runs/kanto_full_6k_50ep_linear_rbf/foundation_model.pt \\
        --output-dir ../data/products/national_cube_kanto_pga30 \\
        --bbox 35.0 138.5 37.5 141.0 \\
        --resolution-m 1000 --depths 0 1 2 5 10 15 20 \\
        --scenario-pga 0.30 --groundwater-csv ../data/features/derived/groundwater_depth.csv

Resource notes:
- Output cube size scales as ``(lat × lon × depth) bytes * 2 × float32``.
  Full Japan at 250 m × 30 depths is ~6 GB raw; Zarr-compressed ~1.5 GB.
- Inference cost dominates: ~1 ms / cell on H100 + the model's batch
  serialisation. A 1 km × 30 depth Kanto bbox (~5 M cells) ~5-10 min on
  a single H100; full Japan at 250 m ~3-4 h.
- Maps are computed in a single post-pass scan over the cube (CPU,
  vectorised numpy); typical 250 m national run ~5 min wall-clock for
  all 4 maps.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("scripts.predict_national_cube")


# ============================================================
# Per-tile coord snap (multi-tile combine pre-step)
# ============================================================
#
# Background (Workflow A discovery, 2026-06): the PredictionEngine
# writes one Zarr per tile, each with its own (lat, lon) coord arrays.
# Although the lat step is fixed (CF-style 1 km / 111320 m), the lon
# step is computed at the *tile's mid-latitude*:
#
#     d_lon_tile = resolution_m / (111320 * cos(mid_lat_tile))
#
# so tiles at different mid_lat have *different* lon spacings. When
# ``xr.combine_by_coords`` outer-joins those tiles, each unique lon
# value becomes a new column -> the union cube becomes ~97% NaN
# (one populated column per tile interleaved with NaN columns from
# every other tile's slightly-offset lon axis).
#
# The lat axis has a subtler version of the same problem: tile lat
# origins land on 1/3-degree multiples (24.000, 29.333..., 34.667...,
# 40.000, 45.333...) which are NOT integer multiples of
# d_lat = 1/111320, so tile lat coords are sub-ULP offset from the
# global lat grid and combine_by_coords would duplicate the boundary
# row.
#
# Fix: precompute a single global (lat_axis, lon_axis) finer than any
# tile's native spacing, then snap every tile's coords onto it BEFORE
# combine_by_coords. Each tile becomes a contiguous block of the
# global grid; the union is dense.

_EARTH_R_DEG_M = 111320.0  # CF-style metres per degree latitude


def build_global_lat_lon_axes(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    resolution_m: float = 1000.0,
    margin_cells: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct global lat/lon axes that every per-tile coord can snap to.

    ``d_lat`` is fixed (CF-style); ``d_lon`` is computed at ``lat_min``
    (the lowest latitude in the bbox -> cos() largest -> d_lon
    smallest) so the global lon axis is strictly finer than every
    tile's per-mid-lat lon spacing. This guarantees zero index
    collisions during snap, with max residual ~ 0.5 cell.

    ``margin_cells`` pads both ends of each axis by N global cells.
    Tile generation produces 1-degree-wide cells whose eastern/northern
    edges may sit slightly past the declared bbox (e.g. the easternmost
    Japan tile column ends at ~lon_min+1, so its last lon coord can
    drift to ~147.006 when the bbox declares lon_max=147.0). Without
    margin, those out-of-bbox tile coords are clipped by
    ``np.searchsorted`` onto the last interior cell and may collide
    with the adjacent tile's last legitimate coord. Padding the axis by
    a couple of cells on each side guarantees every realistic tile
    coord has a UNIQUE nearest cell. Default of 2 covers sub-ULP drift
    + a full 1-cell overshoot with one cell of headroom.
    """
    d_lat = resolution_m / _EARTH_R_DEG_M
    d_lon = resolution_m / (
        _EARTH_R_DEG_M * float(np.cos(np.deg2rad(lat_min)))
    )
    # Pad each side by margin_cells * d to tolerate tiles whose
    # extents push slightly past the declared bbox edges.
    if margin_cells < 0:
        raise ValueError(f"margin_cells must be >= 0, got {margin_cells}")
    lat_min_padded = lat_min - margin_cells * d_lat
    lat_max_padded = lat_max + margin_cells * d_lat
    lon_min_padded = lon_min - margin_cells * d_lon
    lon_max_padded = lon_max + margin_cells * d_lon
    # np.linspace is numerically robust for very-large bbox spans
    # (np.arange with float step accumulates rounding error and the
    # +d/2 padding becomes brittle at scale). We compute the cell
    # count from the span and step, then place endpoints exactly.
    n_lat = int(round((lat_max_padded - lat_min_padded) / d_lat)) + 1
    n_lon = int(round((lon_max_padded - lon_min_padded) / d_lon)) + 1
    lat_axis = np.linspace(lat_min_padded, lat_max_padded, num=n_lat, endpoint=True)
    lon_axis = np.linspace(lon_min_padded, lon_max_padded, num=n_lon, endpoint=True)
    return lat_axis, lon_axis


def _snap_to_axis(coords: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Return ``axis`` values nearest to each entry in ``coords``.

    Vectorised O(N log M) via ``np.searchsorted`` + nearest-neighbour
    refinement against ``idx-1``. Equivalent to
    ``np.argmin(|axis - c|)`` per coord but avoids the O(N*M)
    broadcast. ``axis`` must be monotonically increasing
    (``np.arange`` guarantees this).
    """
    axis = np.ascontiguousarray(axis)
    coords = np.ascontiguousarray(coords)
    idx = np.searchsorted(axis, coords)
    idx = np.clip(idx, 1, len(axis) - 1)
    left = axis[idx - 1]
    right = axis[idx]
    pick_left = (coords - left) <= (right - coords)
    snapped_idx = np.where(pick_left, idx - 1, idx)
    return axis[snapped_idx]


def _snap_tile_coords(
    cube: "xr.DataArray",  # noqa: F821 - xarray imported lazily by caller
    lat_axis_global: np.ndarray,
    lon_axis_global: np.ndarray,
) -> "xr.DataArray":  # noqa: F821
    """Snap a tile's lat/lon coords to the shared global axes.

    Raises ``ValueError`` if two source rows/cols collapse to the same
    global index (would mean the global axis is too coarse; recompute
    with a finer ``ref_lat``).
    """
    snapped_lat = _snap_to_axis(cube.coords["lat"].values, lat_axis_global)
    snapped_lon = _snap_to_axis(cube.coords["lon"].values, lon_axis_global)
    if len(np.unique(snapped_lat)) != len(snapped_lat):
        raise ValueError(
            f"lat snap collision in tile (n={len(snapped_lat)}, "
            f"unique={len(np.unique(snapped_lat))}); "
            f"global lat axis too coarse"
        )
    if len(np.unique(snapped_lon)) != len(snapped_lon):
        raise ValueError(
            f"lon snap collision in tile (n={len(snapped_lon)}, "
            f"unique={len(np.unique(snapped_lon))}); "
            f"global lon axis too coarse"
        )
    return cube.assign_coords(lat=snapped_lat, lon=snapped_lon)


def fill_thin_nan_stripes(combined: np.ndarray) -> np.ndarray:
    """Fill the periodic 1-cell NaN stripes left by tile snap-assignment.

    The shared global lon axis (``build_global_lat_lon_axes``) is built at
    ``d_lon = res/(R*cos(lat_min))`` -- the lowest latitude in the bbox --
    so it is strictly FINER than every tile's native lon spacing
    (each tile uses ``cos(mid_lat) <= cos(lat_min)``). When each tile's
    snapped cells are written via ``combined[np.ix_(lat_idx, lon_idx)]``,
    consecutive tile cells land on global indices spaced slightly more than
    1 apart, so interleaved global columns are never assigned and stay NaN.
    This paints periodic 1-cell-wide NaN columns (and the analogous,
    smaller, lat seams) which render as white vertical/horizontal lines in
    the published maps (e.g. ``vs30.nc``) and in fig7/fig8.

    The fill is STRIPE-TARGETED: only NaN cells bounded on BOTH immediate
    sides (along the axis) by finite data are filled, with the previous
    finite neighbour's value (nearest-neighbour, so the cell inherits the
    adjacent borehole-driven value rather than smearing across a seam).
    Genuine no-data regions (offshore / no nearby boreholes / no valid
    groundwater) are wider than 1 cell and so are left NaN. Operates in
    place over the trailing ``(lat, lon)`` axes, looping any leading dims.
    """
    if combined.ndim < 2:
        return combined
    leading = combined.shape[:-2]

    def _fill_axis_1cell(plane: np.ndarray, axis: int) -> None:
        if plane.shape[axis] < 3:
            return
        finite = np.isfinite(plane)
        nan_mask = ~finite
        prev_finite = np.zeros_like(finite)
        next_finite = np.zeros_like(finite)
        src = np.empty_like(plane)
        if axis == 0:
            prev_finite[1:, :] = finite[:-1, :]
            next_finite[:-1, :] = finite[1:, :]
            src[1:, :] = plane[:-1, :]
        else:
            prev_finite[:, 1:] = finite[:, :-1]
            next_finite[:, :-1] = finite[:, 1:]
            src[:, 1:] = plane[:, :-1]
        fillable = nan_mask & prev_finite & next_finite
        plane[fillable] = src[fillable]

    for idx in np.ndindex(*leading) if leading else [()]:
        plane = combined[idx] if leading else combined
        _fill_axis_1cell(plane, axis=1)  # lon sweep (dominant stripe dir)
        _fill_axis_1cell(plane, axis=0)  # lat sweep
        if leading:
            combined[idx] = plane
    return combined


# ============================================================
# Map computation (vectorised over the cube)
# ============================================================


def _lookup_groundwater_grid(
    groundwater_csv: Path | None,
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
) -> np.ndarray:
    """Resolve a per-grid groundwater-depth surface from the CSV.

    Strategy: nearest-neighbour lookup from observed borings. Anything
    further than ``MAX_INTERP_KM`` away from any observed boring gets
    a NaN (the LPI calculation then masks the cell out of the LPI
    map). Could be upgraded to inverse-distance-weighted interpolation
    once we have the J-SHIS groundwater raster integrated.

    Returns a 2-D array shaped ``(lat_axis.size, lon_axis.size)``.
    """
    nan_grid = np.full((lat_axis.size, lon_axis.size), np.nan, dtype=np.float32)
    if groundwater_csv is None or not groundwater_csv.exists():
        LOG.warning(
            "groundwater_csv missing; emitting all-NaN groundwater grid "
            "(LPI map will be empty)."
        )
        return nan_grid
    LOG.info("Loading groundwater observations from %s", groundwater_csv)
    df = pd.read_csv(groundwater_csv)
    df = df[df["groundwater_depth_m"].notna()].reset_index(drop=True)
    if df.empty:
        return nan_grid
    LOG.info(
        "%d observed groundwater points in the input CSV", len(df)
    )

    from scipy.spatial import cKDTree

    obs_xy = np.stack(
        [df["latitude_deg"].to_numpy(), df["longitude_deg"].to_numpy()], axis=-1
    )
    tree = cKDTree(obs_xy)
    LON, LAT = np.meshgrid(lon_axis, lat_axis, indexing="xy")
    query = np.stack([LAT.ravel(), LON.ravel()], axis=-1)
    # Tolerance: ~0.05 degree ≈ 5 km. Beyond that the nearest boring is
    # too far to trust as a local groundwater proxy.
    distances, idx = tree.query(query, distance_upper_bound=0.05)
    depths = df["groundwater_depth_m"].to_numpy()
    valid = np.isfinite(distances)
    out = np.full(LAT.size, np.nan, dtype=np.float32)
    out[valid] = depths[idx[valid]].astype(np.float32)
    return out.reshape(LAT.shape)


def build_regime_grid(
    boring_parquet: Path,
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
    *,
    knn_k: int = 4,
    knn_max_distance_km: float = 20.0,
) -> np.ndarray:
    """Per-cell AIST regime-code grid on the cube's ``(lat, lon)`` axes.

    Deterministically regenerates the same nearest-regime KNN lookup the
    prediction engine uses transiently (a plurality vote over the ``k``
    nearest borings' ``regime_code``; cells beyond ``knn_max_distance_km``
    fall back to UNKNOWN=7). Persisting it as ``maps/regime_code.nc`` lets
    fig8 panel (b) build the Mondrian conformal half-width
    (``sigma * q_group[regime]``) directly, instead of re-deriving the
    grid at figure-build time.

    Returns an ``int16`` array shaped ``(lat_axis.size, lon_axis.size)``.
    """
    import torch

    from national.data.covariate_registry import CovariateSpec
    from national.data.loaders.boring_knn import (
        BoringKnnIndex,
        BoringKnnLoader,
    )

    knn_index = BoringKnnIndex(boring_parquet)
    loader = BoringKnnLoader(
        CovariateSpec(
            name="regime_code", source="boring:knn", path=boring_parquet,
            dtype="int16", normalize="none", category="categorical",
            n_categories=8, fill_value=7,  # UNKNOWN
        ),
        knn_index, column="regime_code", k=knn_k,
        max_distance_km=knn_max_distance_km,
    )
    LON, LAT = np.meshgrid(lon_axis, lat_axis, indexing="xy")
    codes = loader.sample(
        torch.as_tensor(LAT.ravel(), dtype=torch.float64),
        torch.as_tensor(LON.ravel(), dtype=torch.float64),
    )
    return codes.cpu().numpy().astype(np.int16).reshape(LAT.shape)


def build_lpi_map(
    cube_mean: np.ndarray,  # (depth, lat, lon)
    depths: np.ndarray,
    groundwater_grid: np.ndarray,  # (lat, lon)
    scenario_pga: float,
    *,
    fines_content_pct: float = 5.0,
    magnitude: float = 7.5,
) -> np.ndarray:
    """LPI raster from the 3D N cube + per-cell groundwater + scenario."""
    from national.applications.liquefaction import iwasaki_lpi

    n_lat, n_lon = cube_mean.shape[1:]
    lpi_grid = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    for i in range(n_lat):
        for j in range(n_lon):
            gw = float(groundwater_grid[i, j])
            n_profile = cube_mean[:, i, j]
            if not np.all(np.isfinite(n_profile)):
                continue
            try:
                res = iwasaki_lpi(
                    depths,
                    n_profile,
                    water_table_m=gw,
                    pga_g=scenario_pga,
                    fines_content_pct=fines_content_pct,
                    magnitude=magnitude,
                )
                lpi_grid[i, j] = res.lpi
            except (ValueError, RuntimeError):
                # Numerical edge case (e.g. negative N from model
                # extrapolation) -- mask the cell out rather than fail
                # the whole map build.
                continue
    return lpi_grid


def build_vs30_map(
    cube_mean: np.ndarray,
    depths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """V_s30 raster (m/s) + NEHRP site class (A=1..E=5) raster."""
    from national.applications.vs30 import nehrp_class, vs30_from_profile

    n_lat, n_lon = cube_mean.shape[1:]
    vs30_grid = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    class_grid = np.zeros((n_lat, n_lon), dtype=np.uint8)
    class_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    for i in range(n_lat):
        for j in range(n_lon):
            n_profile = cube_mean[:, i, j]
            if not np.all(np.isfinite(n_profile)):
                continue
            try:
                vs30 = vs30_from_profile(depths, n_profile)
                vs30_grid[i, j] = vs30
                class_grid[i, j] = class_map[nehrp_class(vs30).code]
            except (ValueError, RuntimeError):
                continue
    return vs30_grid, class_grid


def build_bearing_maps(
    cube_mean: np.ndarray,
    depths: np.ndarray,
    *,
    n_design: float = 30.0,
    footing_width_m: float = 1.0,
    embedment_depth_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bearing-stratum depth (m) + Meyerhof allowable bearing (kPa) rasters."""
    from national.applications.bearing_capacity import (
        depth_to_bearing_stratum,
        meyerhof_allowable_bearing_kpa,
    )

    n_lat, n_lon = cube_mean.shape[1:]
    depth_grid = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    qa_grid = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    for i in range(n_lat):
        for j in range(n_lon):
            n_profile = cube_mean[:, i, j]
            if not np.all(np.isfinite(n_profile)):
                continue
            d = depth_to_bearing_stratum(depths, n_profile, n_design=n_design)
            if np.isinf(d):
                # No bearing stratum -> mark as "deep foundation required".
                # We leave both grids as NaN at this cell.
                continue
            depth_grid[i, j] = d
            # Use the N at the bearing stratum for the Meyerhof formula.
            stratum_idx = int(np.argmax(depths >= d))
            n_at_stratum = float(n_profile[stratum_idx])
            qa_grid[i, j] = meyerhof_allowable_bearing_kpa(
                n_value=n_at_stratum,
                footing_width_m=footing_width_m,
                embedment_depth_m=embedment_depth_m,
            )
    return depth_grid, qa_grid


# ============================================================
# Cube + maps driver
# ============================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"),
    )
    parser.add_argument(
        "--resolution-m",
        type=float,
        default=1000.0,
        help="Grid resolution in metres. Defaults to 1 km for fast iteration.",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=float,
        default=[0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 16.0, 20.0],
        help="Depths (m) to predict at. Must include 0..20 m for LPI.",
    )
    parser.add_argument(
        "--tile-size-km",
        type=float,
        default=20.0,
        help="Mesh tile size for the cube generation. Default 20 km",
    )
    parser.add_argument(
        "--scenario-pga",
        type=float,
        default=0.30,
        help="Earthquake scenario PGA (g) for the LPI map. Default 0.30 g.",
    )
    parser.add_argument(
        "--fines-content-pct",
        type=float,
        default=5.0,
        help="Assumed fines content for LPI (cleansand worst case = 5%).",
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        default=7.5,
        help="Earthquake scenario M_w for LPI MSF. Default 7.5.",
    )
    parser.add_argument(
        "--groundwater-csv",
        type=Path,
        default=None,
        help="Per-boring groundwater CSV (from extract_groundwater_from_xml). "
             "If absent, the LPI map is all-NaN.",
    )
    parser.add_argument(
        "--n-design",
        type=float,
        default=30.0,
        help="Design SPT N for bearing-stratum depth. Default 30 (AIJ).",
    )
    parser.add_argument(
        "--footing-width-m",
        type=float,
        default=1.0,
        help="Spread-footing width for Meyerhof q_a. Default 1 m (residential).",
    )
    parser.add_argument(
        "--embedment-depth-m",
        type=float,
        default=1.0,
        help="Footing embedment for Meyerhof q_a. Default 1 m.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--boring-parquet",
        type=Path,
        default=Path("/mnt/nas/data/features/borings_japan_v4.parquet"),
        help="Boring parquet to back the KDTree covariate lookup. Used to "
             "sample (absolute_elevation, river_distance_km, "
             "coast_distance_km, regime_code) at each cube grid cell -- the "
             "trained model expects these as input features but no national "
             "raster exists yet, so we IDW-interpolate from the nearest "
             "borings. v4 parquet has all 4 fields populated.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=4,
        help="Number of nearest borings averaged per grid cell in the "
             "BoringKnnLoader. k=1 = strict nearest neighbour. Default 4 "
             "(IDW-weighted mean) smooths the predictor surface without "
             "over-blurring local geology.",
    )
    parser.add_argument(
        "--knn-max-distance-km",
        type=float,
        default=20.0,
        help="Grid cells whose nearest boring is beyond this distance get "
             "the loader's fill_value (0 for continuous, code 7 / UNKNOWN "
             "for regime). 20 km is a safe default at metropolitan-Japan "
             "density; loosen to 50+ km for offshore / mountainous extrapolation.",
    )
    parser.add_argument(
        "--emit-regime-grid",
        action="store_true",
        help="Also write maps/regime_code.nc: the per-cell AIST regime "
             "code (plurality vote over the k nearest borings), on the "
             "cube's lat/lon axes. fig8 panel (b) uses it to build the "
             "Mondrian conformal interval half-width "
             "(sigma * q_group[regime]).",
    )
    parser.add_argument(
        "--no-cube",
        action="store_true",
        help="Skip cube generation; assumes the cube already exists at "
             "<output-dir>/cube/. Useful for re-running just the maps.",
    )
    parser.add_argument(
        "--no-maps",
        action="store_true",
        help="Skip the application maps; only generate the raw cube.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases run logging (requires WANDB_API_KEY).",
    )
    parser.add_argument(
        "--wandb-project",
        default="geo-paperB-national",
        help="W&B project. Used only when --wandb is set.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="W&B run name. Defaults to output-dir basename.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s"
    )

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.output_dir.name,
            config={
                "stage": "pillar5_cube_and_apps",
                "model": str(args.model),
                "bbox": list(args.bbox),
                "resolution_m": args.resolution_m,
                "depths_m": list(args.depths),
                "tile_size_km": args.tile_size_km,
                "scenario_pga_g": args.scenario_pga,
                "magnitude": args.magnitude,
                "fines_content_pct": args.fines_content_pct,
                "n_design": args.n_design,
                "footing_width_m": args.footing_width_m,
                "embedment_depth_m": args.embedment_depth_m,
            },
        )
        LOG.info("W&B run: %s", wandb_run.url)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cube_root = args.output_dir / "cube"
    maps_dir = args.output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    depths_arr = np.asarray(args.depths, dtype=np.float64)
    if depths_arr.max() < 20.0:
        LOG.warning(
            "Max depth %.1f m is shallower than 20 m; LPI integral will "
            "truncate at max depth and may under-estimate.",
            depths_arr.max(),
        )

    # ---- 1. Predict cube --------------------------------------------------
    if not args.no_cube:
        import torch

        if args.device == "auto":
            if torch.cuda.is_available():
                args.device = "cuda"
            elif torch.backends.mps.is_available():
                args.device = "mps"
            else:
                args.device = "cpu"
        LOG.info("Loading foundation model %s on %s", args.model, args.device)
        from national.data.covariate_registry import (
            CovariateRegistry,
            CovariateSpec,
        )
        from national.data.loaders.boring_knn import (
            BoringKnnIndex,
            BoringKnnLoader,
        )
        from national.models.foundation import FoundationModel
        from national.prediction.engine import GridSpec, PredictionEngine
        from national.tiling.tile_manager import TileManager

        model = FoundationModel.load(args.model).to(args.device)
        model.eval()

        # ---- KDTree-backed covariate registry --------------------------
        # The trained model expects rows of
        # [lat, lon, depth, abs_elev, river_km, coast_km, regime_one_hot(8)].
        # Without a rasterized DEM / distance stack we IDW-interpolate
        # the four boring-derived covariates onto the cube grid via a
        # cKDTree on (lat, lon). See national.data.loaders.boring_knn for
        # the rationale + a note on swapping to rasters once they exist.
        LOG.info(
            "Building BoringKnnIndex from %s (k=%d, max %.0f km)",
            args.boring_parquet, args.knn_k, args.knn_max_distance_km,
        )
        knn_index = BoringKnnIndex(args.boring_parquet)

        def _continuous_spec(name: str) -> "CovariateSpec":
            return CovariateSpec(
                name=name, source="boring:knn", path=args.boring_parquet,
                dtype="float32", normalize="none", category="continuous",
                fill_value=0.0,
            )

        loaders = [
            BoringKnnLoader(
                _continuous_spec("absolute_elevation"), knn_index,
                column="absolute_elevation", k=args.knn_k,
                max_distance_km=args.knn_max_distance_km,
            ),
            BoringKnnLoader(
                _continuous_spec("river_distance_km"), knn_index,
                column="river_distance_km", k=args.knn_k,
                max_distance_km=args.knn_max_distance_km,
            ),
            BoringKnnLoader(
                _continuous_spec("coast_distance_km"), knn_index,
                column="coast_distance_km", k=args.knn_k,
                max_distance_km=args.knn_max_distance_km,
            ),
            BoringKnnLoader(
                CovariateSpec(
                    name="regime_code", source="boring:knn",
                    path=args.boring_parquet, dtype="int16",
                    normalize="none", category="categorical",
                    n_categories=8, fill_value=7,  # UNKNOWN
                ),
                knn_index, column="regime_code", k=args.knn_k,
                max_distance_km=args.knn_max_distance_km,
            ),
        ]
        registry = CovariateRegistry(loaders)

        tile_manager = TileManager(
            region_bbox=tuple(args.bbox),
            tile_size_km=args.tile_size_km,
            halo_km=0.0,
        )
        grid = GridSpec(
            resolution_m=args.resolution_m, depths_m=tuple(depths_arr.tolist())
        )
        engine = PredictionEngine(
            model=model,
            registry=registry,
            tile_manager=tile_manager,
            grid=grid,
            device=args.device,
            regime_loader_name="regime_code",
            categorical_one_hot={"regime_code": 8},
        )
        LOG.info(
            "Predicting cube: bbox=%s res=%.0fm depths=%s tiles=%d",
            args.bbox, args.resolution_m, args.depths, len(tile_manager.tiles()),
        )
        t0 = time.time()
        engine.predict_cube(cube_root)
        LOG.info("Cube generated in %.1f s -> %s", time.time() - t0, cube_root)

    if args.no_maps:
        LOG.info("Skipping application maps (--no-maps).")
        return 0

    # ---- 2. Application maps ---------------------------------------------
    import xarray as xr

    # The cube is written as per-tile Zarr groups. Combine all tiles
    # into one global cube by (lat, lon, depth) coords. Tiles are
    # disjoint by design (TileManager), but each tile's lon axis is
    # built at the *tile's mid-latitude* (``d_lon = res /
    # (111320*cos(mid_lat))``), so different tiles have slightly
    # different lon spacings. A bare ``combine_by_coords`` would then
    # treat every tile's lon column as unique and produce a ~97% NaN
    # cube. Workflow A discovery (2026-06) traced this to the per-tile
    # mid_lat-dependent lon_step; the fix is to snap every tile's
    # lat/lon onto a single global axis BEFORE combine_by_coords so
    # each tile lands on a contiguous block of the shared grid.
    tile_dirs = sorted(cube_root.glob("tile_*.zarr"))
    if not tile_dirs:
        LOG.error("No tile cubes found under %s; cannot build maps.", cube_root)
        return 1
    LOG.info("Aggregating %d tile cubes for map build", len(tile_dirs))
    cubes = [xr.open_dataarray(t, engine="zarr") for t in tile_dirs]
    if len(cubes) == 1:
        cube = cubes[0]
    else:
        # The declared --bbox is the *tile origin* bounding box, but
        # TileManager produces 1-degree-wide tiles whose far edges can
        # overshoot lon_max / lat_max by ~1 deg (e.g. easternmost tiles
        # at lon_min=146.0 extend to lon~147.0 even though --bbox
        # declared lon_max=146.0). Building the global axis from the
        # *declared* bbox + a 2-cell margin therefore clips the
        # overshoot tiles' eastern lon coords onto the last interior
        # cell, collapsing ~100 distinct lon entries into ~3 (snap
        # collision -> the maps_only job blows up with ValueError).
        #
        # Fix: derive the outer envelope from the loaded tile coords
        # themselves. This is exact by construction; no bbox-shape
        # assumption is made. We still pass a 2-cell margin so sub-ULP
        # endpoint drift on either side has a unique nearest cell.
        all_lats = np.concatenate([c.coords["lat"].values for c in cubes])
        all_lons = np.concatenate([c.coords["lon"].values for c in cubes])
        lat_min_g = float(all_lats.min())
        lat_max_g = float(all_lats.max())
        lon_min_g = float(all_lons.min())
        lon_max_g = float(all_lons.max())
        LOG.info(
            "Global axis envelope from tile coords: "
            "lat=[%.6f, %.6f], lon=[%.6f, %.6f] "
            "(declared --bbox was lat=[%.4f, %.4f], lon=[%.4f, %.4f])",
            lat_min_g, lat_max_g, lon_min_g, lon_max_g,
            args.bbox[0], args.bbox[2], args.bbox[1], args.bbox[3],
        )
        lat_axis_global, lon_axis_global = build_global_lat_lon_axes(
            lat_min=lat_min_g, lat_max=lat_max_g,
            lon_min=lon_min_g, lon_max=lon_max_g,
            resolution_m=args.resolution_m,
        )
        LOG.info(
            "Snapping %d tile cubes to global axis: lat=%d cells, lon=%d cells",
            len(cubes), len(lat_axis_global), len(lon_axis_global),
        )
        cubes = [
            _snap_tile_coords(c, lat_axis_global, lon_axis_global)
            for c in cubes
        ]
        # Build the combined cube by direct assignment instead of
        # combine_by_coords: with 825+ tiles each spanning a 1-degree
        # block of the global axis, neighbouring tiles in adjacent
        # rows/columns can share boundary rows/columns of snapped
        # coords, breaking combine_by_coords' "Resulting object does
        # not have monotonic global indexes" precondition. The fix is
        # to allocate the global array directly and index every tile's
        # (lat, lon) coords into it -- this naturally handles boundary
        # overlap (last-tile-wins on shared cells, which is what
        # combine_by_coords would have done after a successful merge
        # anyway).
        template = cubes[0]
        non_geo_dims = tuple(d for d in template.dims if d not in ("lat", "lon"))
        non_geo_shape = tuple(template.sizes[d] for d in non_geo_dims)
        combined_shape = non_geo_shape + (
            len(lat_axis_global), len(lon_axis_global),
        )
        combined = np.full(combined_shape, np.nan, dtype=template.dtype)
        # Pre-index global axes for fast np.searchsorted-based lookup.
        for tile in cubes:
            # Ensure dim order matches template (statistic, depth, lat, lon)
            tile_t = tile.transpose(*non_geo_dims, "lat", "lon")
            lat_idx = np.searchsorted(
                lat_axis_global, tile_t.coords["lat"].values
            )
            lon_idx = np.searchsorted(
                lon_axis_global, tile_t.coords["lon"].values
            )
            # combined[..., lat_idx[:, None], lon_idx[None, :]] = vals
            # using np.ix_ to broadcast the spatial index correctly
            # while leaving non-spatial dims as a full-slice prefix.
            non_geo_slices = (slice(None),) * len(non_geo_dims)
            combined[non_geo_slices + np.ix_(lat_idx, lon_idx)] = tile_t.values
        # Fill the periodic 1-cell NaN stripes left by the finer-than-tile
        # global axis snap so the published maps (vs30.nc etc.) do not carry
        # the white vertical/horizontal lines. Width-1 gaps only; genuine
        # wider no-data regions stay NaN.
        combined = fill_thin_nan_stripes(combined)
        coords = {d: template.coords[d].values for d in non_geo_dims}
        coords["lat"] = lat_axis_global
        coords["lon"] = lon_axis_global
        cube = xr.DataArray(
            combined,
            dims=non_geo_dims + ("lat", "lon"),
            coords=coords,
            name=template.name or "prediction",
        )
        LOG.info(
            "Combined cube shape (direct assignment) = %s; "
            "global axis lat=%d lon=%d",
            dict(cube.sizes), len(lat_axis_global), len(lon_axis_global),
        )
    if "statistic" in cube.dims:
        cube_mean = cube.sel(statistic="mean").values  # (depth, lat, lon)
    else:
        cube_mean = cube.values

    lat_axis = cube.coords["lat"].values
    lon_axis = cube.coords["lon"].values
    cube_depths = cube.coords["depth"].values

    LOG.info(
        "Cube shape (depth, lat, lon) = %s; depths %s",
        cube_mean.shape, cube_depths.tolist(),
    )

    LOG.info("Building V_s30 + NEHRP site class maps...")
    vs30_grid, class_grid = build_vs30_map(cube_mean, cube_depths)

    LOG.info("Building bearing-stratum + Meyerhof q_a maps...")
    bearing_depth_grid, qa_grid = build_bearing_maps(
        cube_mean, cube_depths,
        n_design=args.n_design,
        footing_width_m=args.footing_width_m,
        embedment_depth_m=args.embedment_depth_m,
    )

    LOG.info(
        "Building LPI map at PGA=%.2f g, M_w=%.1f, Fc=%.0f%%...",
        args.scenario_pga, args.magnitude, args.fines_content_pct,
    )
    gw_grid = _lookup_groundwater_grid(args.groundwater_csv, lat_axis, lon_axis)
    lpi_grid = build_lpi_map(
        cube_mean, cube_depths, gw_grid,
        scenario_pga=args.scenario_pga,
        fines_content_pct=args.fines_content_pct,
        magnitude=args.magnitude,
    )

    # ---- 3. Write maps + manifest ----------------------------------------
    import xarray as xr

    for grid_arr, name in (
        (lpi_grid, f"lpi_pga{int(args.scenario_pga * 100):02d}"),
        (vs30_grid, "vs30"),
        (class_grid, "nehrp_site_class"),
        (bearing_depth_grid, "bearing_stratum_depth"),
        (qa_grid, "allowable_bearing_kpa"),
    ):
        da = xr.DataArray(
            grid_arr,
            dims=("lat", "lon"),
            coords={"lat": lat_axis, "lon": lon_axis},
            name=name,
        )
        out_path = maps_dir / f"{name}.nc"  # NetCDF for lossless float storage
        da.to_netcdf(out_path)
        LOG.info("Wrote %s (%dx%d)", out_path, grid_arr.shape[0], grid_arr.shape[1])

    if args.emit_regime_grid:
        LOG.info("Building regime-code grid (k=%d, max %.0f km)...",
                 args.knn_k, args.knn_max_distance_km)
        regime_grid = build_regime_grid(
            args.boring_parquet, lat_axis, lon_axis,
            knn_k=args.knn_k, knn_max_distance_km=args.knn_max_distance_km,
        )
        da = xr.DataArray(
            regime_grid, dims=("lat", "lon"),
            coords={"lat": lat_axis, "lon": lon_axis}, name="regime_code",
        )
        out_path = maps_dir / "regime_code.nc"
        da.to_netcdf(out_path)
        LOG.info("Wrote %s (%dx%d)", out_path,
                 regime_grid.shape[0], regime_grid.shape[1])

    manifest = {
        "model": str(args.model),
        "bbox": list(args.bbox),
        "resolution_m": float(args.resolution_m),
        "depths_m": list(map(float, args.depths)),
        "scenario_pga_g": float(args.scenario_pga),
        "magnitude": float(args.magnitude),
        "fines_content_pct": float(args.fines_content_pct),
        "n_design": float(args.n_design),
        "footing_width_m": float(args.footing_width_m),
        "embedment_depth_m": float(args.embedment_depth_m),
        "groundwater_csv": (
            str(args.groundwater_csv) if args.groundwater_csv else None
        ),
        "n_groundwater_cells": int(np.isfinite(gw_grid).sum()),
        "n_total_cells": int(gw_grid.size),
        "layers": [
            {"name": "cube",                       "path": "cube/", "format": "zarr"},
            {"name": f"lpi_pga{int(args.scenario_pga * 100):02d}",
             "path": f"maps/lpi_pga{int(args.scenario_pga * 100):02d}.nc"},
            {"name": "vs30",                       "path": "maps/vs30.nc"},
            {"name": "nehrp_site_class",           "path": "maps/nehrp_site_class.nc"},
            {"name": "bearing_stratum_depth",      "path": "maps/bearing_stratum_depth.nc"},
            {"name": "allowable_bearing_kpa",      "path": "maps/allowable_bearing_kpa.nc"},
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    LOG.info("Wrote %s/manifest.json", args.output_dir)

    if wandb_run is not None:
        # Per-layer summary stats. NaN-mask each before computing.
        def _finite_stats(arr: np.ndarray, name: str) -> None:
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                return
            wandb_run.summary[f"{name}_mean"] = float(finite.mean())
            wandb_run.summary[f"{name}_median"] = float(np.median(finite))
            wandb_run.summary[f"{name}_p10"] = float(np.percentile(finite, 10))
            wandb_run.summary[f"{name}_p90"] = float(np.percentile(finite, 90))
            wandb_run.summary[f"{name}_max"] = float(finite.max())
            wandb_run.summary[f"{name}_min"] = float(finite.min())
            wandb_run.summary[f"{name}_n_finite"] = int(finite.size)
            wandb_run.summary[f"{name}_n_total"] = int(arr.size)

        _finite_stats(lpi_grid, "lpi")
        _finite_stats(vs30_grid, "vs30")
        _finite_stats(bearing_depth_grid, "bearing_stratum_depth_m")
        _finite_stats(qa_grid, "allowable_bearing_kpa")
        _finite_stats(gw_grid, "groundwater_grid")
        wandb_run.summary["n_groundwater_cells_matched"] = int(np.isfinite(gw_grid).sum())
        wandb_run.summary["n_total_cells"] = int(gw_grid.size)
        wandb_run.summary["cube_shape_depth_lat_lon"] = list(cube_mean.shape)
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
