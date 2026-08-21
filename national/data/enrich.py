"""End-to-end enrichment: raw boring CSV -> model-ready Parquet.

Joins the existing 2.7 M-row N-value CSV (per-row depth measurements,
175 k unique locations) with the derived covariate columns the
foundation model expects:

- ``river_distance_km``  -- nearest Class-1 river polyline (MLIT W05).
- ``coast_distance_km``  -- nearest coastline polyline (MLIT C23).
- ``absolute_elevation`` -- ``mouth_elevation - spt_start_depth``.
- ``depth_from_surface`` -- alias for ``spt_start_depth``.
- ``regime_code``        -- categorical regime (AIST geology code mapped
  through :class:`national.tiling.regime_classifier.Regime`). Defaults
  to ``Regime.UNKNOWN`` when the geology cache is empty.

Per-unique-location enrichment is computed once and joined back to the
full row set, which is ~15× faster than per-row geometry queries on a
dataset this size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from national.data.derived.aist_granular import (
    AistEra,
    AistLithoMacro,
    granular_codes_for_aist_cache,
)
from national.data.derived.distances import (
    compute_distance_to_lines,
    load_coastlines_from_mlit_dir,
)
from national.tiling.regime_classifier import Regime

LOG = logging.getLogger("national.data.enrich")


@dataclass
class EnrichmentSpec:
    """Inputs and outputs for one enrichment run."""

    borings_csv: Path
    output_parquet: Path
    river_geojson: Path | None = None
    coast_dir: Path | None = None
    aist_geology_cache: Path | None = None
    target_column: str = "n_value"
    bbox: tuple[float, float, float, float] | None = None  # (lat_min, lon_min, lat_max, lon_max)
    aist_granular: bool = True  # also emit aist_era_code + aist_litho_macro_code
    # Paper B' Pillar 2: optional CSV from scripts/extract_groundwater_from_xml.py
    # carrying per-boring shallowest-observed groundwater depth (m). Joined on
    # rounded (lat, lon) to attach a `groundwater_depth_m` column. NaN where
    # the cache is missing or the boring had no usable <孔内水位> reading.
    groundwater_csv: Path | None = None
    # NC pre-review response R0-1 (2026-08-11): carry the per-row borehole
    # identity (`file_path`, `dtd_version`) from location_n_values.csv into
    # the output parquet. ~15% of boreholes share byte-identical coordinates
    # (dense urban sites), so rounded (lat, lon) is NOT a valid borehole key;
    # every downstream join and grouped resampling unit must use the XML file
    # identity instead. Off by default so legacy schemas reproduce unchanged.
    include_identity: bool = False


def enrich(spec: EnrichmentSpec) -> Path:
    """Run the full enrichment and write a single Parquet file."""
    LOG.info("Reading borings from %s", spec.borings_csv)
    # dtd_version must stay a string: read as float, "2.10" collapses to 2.1
    # and becomes indistinguishable from a hypothetical 2.1. The dtype hint is
    # silently ignored when the column is absent (legacy 5-column CSVs).
    df = pd.read_csv(spec.borings_csv, dtype={"dtd_version": str, "file_path": str})
    df = _normalize_input_columns(df)
    if spec.bbox is not None:
        df = _filter_bbox(df, spec.bbox)
        LOG.info("After bbox filter: %d rows", len(df))

    # 1. Per-row derivations (cheap).
    df["depth_from_surface"] = df["spt_start_depth"]
    df["absolute_elevation"] = df["mouth_elevation"] - df["spt_start_depth"]

    # 2. Per-location derivations (the expensive ones).
    unique_locs = (
        df[["latitude_deg", "longitude_deg"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    LOG.info("%d unique boring locations", len(unique_locs))

    if spec.river_geojson and spec.river_geojson.exists():
        LOG.info("Computing river_distance_km from %s", spec.river_geojson)
        unique_locs["river_distance_km"] = _distance_to_lines_from_geojson(
            unique_locs, spec.river_geojson
        )
    else:
        LOG.warning("river_geojson missing; setting river_distance_km = nan")
        unique_locs["river_distance_km"] = np.float32("nan")

    if spec.coast_dir and spec.coast_dir.exists():
        LOG.info("Computing coast_distance_km from %s", spec.coast_dir)
        unique_locs["coast_distance_km"] = _distance_to_coast(unique_locs, spec.coast_dir)
    else:
        LOG.warning("coast_dir missing; setting coast_distance_km = nan")
        unique_locs["coast_distance_km"] = np.float32("nan")

    unique_locs["regime_code"] = _regime_from_geology(unique_locs, spec.aist_geology_cache)

    if spec.aist_granular:
        granular = _aist_granular_codes(unique_locs, spec.aist_geology_cache)
        if granular is not None:
            unique_locs["aist_era_code"] = granular["aist_era_code"].to_numpy(np.int16)
            unique_locs["aist_litho_macro_code"] = granular[
                "aist_litho_macro_code"
            ].to_numpy(np.int16)
        else:
            unique_locs["aist_era_code"] = np.full(
                (len(unique_locs),), int(AistEra.UNKNOWN), dtype=np.int16
            )
            unique_locs["aist_litho_macro_code"] = np.full(
                (len(unique_locs),), int(AistLithoMacro.UNKNOWN), dtype=np.int16
            )

    include_groundwater = spec.groundwater_csv is not None
    if include_groundwater:
        unique_locs["groundwater_depth_m"] = _groundwater_per_location(
            unique_locs, spec.groundwater_csv
        )

    # 3. Join back to the full row set.
    df = df.merge(unique_locs, on=["latitude_deg", "longitude_deg"], how="left")
    df = _final_schema(
        df,
        target_column=spec.target_column,
        include_aist_granular=spec.aist_granular,
        include_groundwater=include_groundwater,
        include_identity=spec.include_identity,
    )

    spec.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(spec.output_parquet, index=False)
    LOG.info(
        "Wrote %d rows to %s (mean N=%.2f, std=%.2f)",
        len(df),
        spec.output_parquet,
        float(df[spec.target_column].mean()),
        float(df[spec.target_column].std()),
    )
    return spec.output_parquet


# ---------------------------------------------------------------- internals
_REQUIRED = ("longitude_deg", "latitude_deg", "spt_start_depth", "n_value")


def _normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"Boring CSV missing required columns: {missing!r}")
    if "mouth_elevation" not in df.columns:
        df["mouth_elevation"] = 0.0
    df = df.copy()
    # Trim rows with bad coords or missing elevations (KuniJiban has a small
    # number of NaN / -999 / out-of-range entries). Without these filters
    # downstream BoringDataset -> Cholesky decomposition silently produces
    # an all-NaN kernel matrix on any batch that includes the bad rows.
    df = df.dropna(
        subset=[
            "latitude_deg",
            "longitude_deg",
            "n_value",
            "spt_start_depth",
            "mouth_elevation",
        ]
    )
    df = df[(df["latitude_deg"] > 20) & (df["latitude_deg"] < 50)]
    df = df[(df["longitude_deg"] > 120) & (df["longitude_deg"] < 150)]
    df = df[(df["n_value"] >= 0) & (df["n_value"] <= 100)]
    df = df[(df["mouth_elevation"] > -1000) & (df["mouth_elevation"] < 5000)]
    return df.reset_index(drop=True)


def _filter_bbox(
    df: pd.DataFrame, bbox: tuple[float, float, float, float]
) -> pd.DataFrame:
    lat_min, lon_min, lat_max, lon_max = bbox
    mask = (
        (df["latitude_deg"] >= lat_min)
        & (df["latitude_deg"] <= lat_max)
        & (df["longitude_deg"] >= lon_min)
        & (df["longitude_deg"] <= lon_max)
    )
    return df.loc[mask].reset_index(drop=True)


def _distance_to_lines_from_geojson(
    points_df: pd.DataFrame, geojson_path: Path
) -> np.ndarray:
    import geopandas as gpd

    LOG.info("Loading lines GeoJSON: %s", geojson_path)
    lines = gpd.read_file(geojson_path)
    return compute_distance_to_lines(points_df, lines)


def _distance_to_coast(points_df: pd.DataFrame, coast_dir: Path) -> np.ndarray:
    coast = load_coastlines_from_mlit_dir(coast_dir)
    return compute_distance_to_lines(points_df, coast)


def _lookup_aist_legend(
    points_df: pd.DataFrame, cache_path: Path | None
) -> pd.DataFrame | None:
    """Inner helper: join AIST cache against per-location coords.

    Returns a DataFrame aligned 1:1 with ``points_df`` rows, carrying the
    AIST legend fields (``symbol, formation_age_ja, group_ja, lithology_ja``)
    where a match was found and NaN where no match existed. Coordinates are
    rounded to 4 decimals (the cache key precision, ~10 m) before merge.

    Shared by :func:`_regime_from_geology` and :func:`_aist_granular_codes`
    so the cache is read at most once per enrichment run regardless of how
    many derived codes the caller wants.

    Returns ``None`` if the cache is missing or has an invalid schema, so
    the caller can short-circuit to its UNKNOWN-fill path.
    """
    if cache_path is None or not cache_path.exists():
        return None

    LOG.info("Joining AIST geology cache: %s", cache_path)
    cache = pd.read_parquet(cache_path)

    required = {"lat", "lon", "symbol", "formation_age_ja", "group_ja", "lithology_ja"}
    missing = required - set(cache.columns)
    if missing:
        LOG.warning(
            "AIST cache %s is missing columns %s; falling back to UNKNOWN.",
            cache_path,
            sorted(missing),
        )
        return None

    # Match on 4-decimal rounded coordinates (cache key precision).
    # Cast to float64 before rounding: the on-disk parquet stores coords as
    # float32 (saves memory at 2.66M rows) but float32 precision (~7 sig
    # figs) corrupts the 4th decimal -- e.g. lat=31.536016f32 rounds to
    # 31.535999, not 31.5360 -- and silently drops every left-join row.
    # The AIST cache is float64 and round to clean 4-decimal grid, so we
    # promote points_df to float64 before rounding to match.
    lat_r = points_df["latitude_deg"].astype("float64").round(4)
    lon_r = points_df["longitude_deg"].astype("float64").round(4)
    cache_r = cache.copy()
    cache_r["lat"] = cache_r["lat"].astype("float64").round(4)
    cache_r["lon"] = cache_r["lon"].astype("float64").round(4)
    joined = pd.DataFrame({"lat": lat_r, "lon": lon_r}).merge(
        cache_r,
        on=["lat", "lon"],
        how="left",
    )
    return joined


def _regime_from_geology(
    points_df: pd.DataFrame, cache_path: Path | None
) -> np.ndarray:
    """Map per-location AIST legend rows to ``Regime`` ints.

    The cache format produced by
    ``national.data.download.aist_geology.fetch_codes_for_borings`` is a
    Parquet with columns ``lat, lon, symbol, formation_age_ja, group_ja,
    lithology_ja``. We resolve each row via
    :func:`national.data.derived.lithology.regime_from_legend` (rules
    documented at the call site).

    Returns ``Regime.UNKNOWN`` for every boring when the cache is
    missing or has no entry at the location.
    """
    from national.data.derived.lithology import regime_from_legend

    n = len(points_df)
    joined = _lookup_aist_legend(points_df, cache_path)
    if joined is None:
        return np.full((n,), int(Regime.UNKNOWN), dtype=np.int16)

    regime = joined.apply(
        lambda row: int(
            regime_from_legend(
                row.get("symbol"),
                row.get("formation_age_ja"),
                row.get("group_ja"),
                row.get("lithology_ja"),
            )
        ),
        axis=1,
    ).to_numpy(dtype=np.int16)
    n_matched = int((regime != int(Regime.UNKNOWN)).sum())
    LOG.info(
        "Resolved %d / %d boring locations to non-UNKNOWN regimes via AIST.",
        n_matched,
        n,
    )
    return regime


def _aist_granular_codes(
    points_df: pd.DataFrame, cache_path: Path | None
) -> pd.DataFrame | None:
    """Map per-location AIST legend rows to era + lithology macro codes.

    Joins the AIST cache against ``points_df`` coords (sharing the lookup
    with :func:`_regime_from_geology` -- the cache is read once via
    :func:`_lookup_aist_legend`) and applies the granular classifiers in
    :mod:`national.data.derived.aist_granular`.

    Returns:
        DataFrame with columns ``aist_era_code`` and
        ``aist_litho_macro_code`` (both int16), aligned 1:1 with
        ``points_df``. ``None`` if the cache is missing or its schema is
        invalid -- caller should fall back to ``UNKNOWN`` fill.
    """
    joined = _lookup_aist_legend(points_df, cache_path)
    if joined is None:
        return None
    granular = granular_codes_for_aist_cache(joined)
    # Rows that did not match the cache come back with NaN legend strings
    # which collapse to UNKNOWN per the binning contract -- no extra
    # handling needed here.
    n_era_known = int((granular["aist_era_code"] != int(AistEra.UNKNOWN)).sum())
    n_litho_known = int(
        (granular["aist_litho_macro_code"] != int(AistLithoMacro.UNKNOWN)).sum()
    )
    LOG.info(
        "Resolved AIST granular codes: era_known=%d / %d, litho_macro_known=%d / %d",
        n_era_known,
        len(points_df),
        n_litho_known,
        len(points_df),
    )
    return granular


def _groundwater_per_location(
    points_df: pd.DataFrame, csv_path: Path | None
) -> np.ndarray:
    """Resolve per-location shallowest-observed groundwater depth (m).

    The CSV is produced by ``scripts/extract_groundwater_from_xml.py``
    with columns ``file_path, latitude_deg, longitude_deg,
    groundwater_depth_m, n_observations``. Multiple XML files can map
    to the same (lat, lon) (e.g. multi-day or multi-boring at one
    site); we resolve the duplicate by keeping the shallowest non-NaN
    reading (mirrors the per-file selection rule and is the
    engineering-conservative choice for Iwasaki LPI inputs).

    Args:
        points_df: DataFrame with ``latitude_deg``, ``longitude_deg``.
        csv_path: Output of the groundwater extractor, or ``None`` /
            missing for a NaN-fill result.

    Returns:
        ``np.ndarray`` of dtype float32 with shape ``(len(points_df),)``;
        rows without a match come back as NaN.
    """
    n = len(points_df)
    if csv_path is None or not csv_path.exists():
        LOG.warning(
            "groundwater_csv missing or unset; emitting all-NaN column."
        )
        return np.full((n,), np.float32("nan"), dtype=np.float32)

    LOG.info("Joining groundwater CSV: %s", csv_path)
    gw = pd.read_csv(csv_path)
    needed = {"latitude_deg", "longitude_deg", "groundwater_depth_m"}
    missing_cols = needed - set(gw.columns)
    if missing_cols:
        LOG.warning(
            "groundwater_csv %s missing columns %s; emitting NaN.",
            csv_path,
            sorted(missing_cols),
        )
        return np.full((n,), np.float32("nan"), dtype=np.float32)

    # Drop the file_path / n_observations columns and rows with NaN
    # depths to keep the merge cheap, then collapse to the shallowest
    # reading per (rounded lat, lon).
    gw = gw[["latitude_deg", "longitude_deg", "groundwater_depth_m"]].dropna()
    # Match the AIST cache convention: float64 + round(4) so the join
    # survives float32 round-trip of the parquet column when this enrich
    # is called from the migration script path.
    gw["lat_r"] = gw["latitude_deg"].astype("float64").round(4)
    gw["lon_r"] = gw["longitude_deg"].astype("float64").round(4)
    collapsed = (
        gw.groupby(["lat_r", "lon_r"], as_index=False)["groundwater_depth_m"]
        .min()  # shallowest non-NaN per location
    )

    pts = points_df[["latitude_deg", "longitude_deg"]].copy()
    pts["lat_r"] = pts["latitude_deg"].astype("float64").round(4)
    pts["lon_r"] = pts["longitude_deg"].astype("float64").round(4)
    merged = pts.merge(
        collapsed,
        on=["lat_r", "lon_r"],
        how="left",
    )
    depths = merged["groundwater_depth_m"].to_numpy(dtype=np.float32)
    n_matched = int(np.isfinite(depths).sum())
    LOG.info(
        "Resolved groundwater for %d / %d locations (%.1f%%); "
        "depth p50=%.2f m p90=%.2f m max=%.2f m",
        n_matched,
        n,
        100.0 * n_matched / max(n, 1),
        float(np.nanmedian(depths)) if n_matched else float("nan"),
        float(np.nanpercentile(depths, 90.0)) if n_matched else float("nan"),
        float(np.nanmax(depths)) if n_matched else float("nan"),
    )
    return depths


def _final_schema(
    df: pd.DataFrame,
    *,
    target_column: str,
    include_aist_granular: bool = False,
    include_groundwater: bool = False,
    include_identity: bool = False,
) -> pd.DataFrame:
    """Project the enriched DataFrame down to the canonical training schema.

    The v2 schema (Paper B endgame and prior) carries 8 columns: lat/lon,
    depth, elevation, target, river/coast distance, regime code. When
    ``include_aist_granular`` is set (Paper B' multi-modal track) we
    additionally carry the era + lithology macro codes from
    :func:`_aist_granular_codes`. Older parquets without these columns are
    still readable by :class:`national.data.boring_dataset.BoringDataset`
    because the multi-categorical one-hot path is opt-in.
    """
    cols = [
        "latitude_deg",
        "longitude_deg",
        "depth_from_surface",
        "absolute_elevation",
        target_column,
        "river_distance_km",
        "coast_distance_km",
        "regime_code",
    ]
    if include_aist_granular:
        cols += ["aist_era_code", "aist_litho_macro_code"]
    if include_groundwater:
        cols += ["groundwater_depth_m"]
    if include_identity:
        # Borehole identity from the source CSV (see EnrichmentSpec). The
        # basename of the XML file is the corpus-wide 1:1 borehole key
        # (191,572 files <-> 191,572 boreholes); path prefixes vary between
        # extractors (`data/...` vs `../data/...`), so `boring_file` stores
        # the basename only and joins must key on it.
        missing_id = [c for c in ("file_path", "dtd_version") if c not in df.columns]
        if missing_id:
            raise KeyError(
                "include_identity=True but the boring CSV lacks "
                f"{missing_id!r}; regenerate it with the v2 extractor "
                "(scripts in backend/verification/r_okauchi/)."
            )
        df = df.copy()
        df["boring_file"] = (
            df["file_path"].astype(str).str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
        )
        cols += ["boring_file", "dtd_version"]
    out = df[cols].copy()
    out["latitude_deg"] = out["latitude_deg"].astype(np.float32)
    out["longitude_deg"] = out["longitude_deg"].astype(np.float32)
    out["depth_from_surface"] = out["depth_from_surface"].astype(np.float32)
    out["absolute_elevation"] = out["absolute_elevation"].astype(np.float32)
    out[target_column] = out[target_column].astype(np.float32)
    out["river_distance_km"] = out["river_distance_km"].astype(np.float32)
    out["coast_distance_km"] = out["coast_distance_km"].astype(np.float32)
    out["regime_code"] = out["regime_code"].astype(np.int16)
    if include_aist_granular:
        out["aist_era_code"] = out["aist_era_code"].astype(np.int16)
        out["aist_litho_macro_code"] = out["aist_litho_macro_code"].astype(np.int16)
    if include_groundwater:
        # Keep NaN as-is (float32 supports it) -- downstream BoringDataset
        # callers decide whether to treat NaN groundwater as a missing
        # feature (impute / mask) or as a missing target (LMC will mask).
        out["groundwater_depth_m"] = out["groundwater_depth_m"].astype(np.float32)
    if include_identity:
        # Categorical dtype: ~150k distinct strings over 2.66M rows, so the
        # parquet dictionary encoding keeps the identity columns almost free.
        out["boring_file"] = out["boring_file"].astype("category")
        out["dtd_version"] = out["dtd_version"].astype(str).astype("category")
    return out


__all__ = ["EnrichmentSpec", "enrich"]
