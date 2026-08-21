"""KDTree-backed ``CovariateLoader`` for cube prediction.

Paper B' Pillar 5 (3D cube + engineering maps) needs to sample three
continuous covariates (``absolute_elevation``, ``river_distance_km``,
``coast_distance_km``) plus the eight-way ``regime_code`` at every
``(lat, lon)`` grid cell of the prediction cube. The training pipeline
gets these straight from the enriched boring parquet, but the cube
script runs on a regular grid that has no rasterized version of these
features (the AIST raster fetch + MLIT distance computation pipeline
exists but has never been pre-baked to a national-scale raster).

This loader bridges that gap: read the boring parquet once, build a
:class:`scipy.spatial.cKDTree` on ``(lat, lon)``, and at sample time
return the inverse-distance-weighted mean of the ``k`` nearest
borings for continuous covariates, or the plurality regime code for
categorical covariates. The "nearest borings" approximation is a
reasonable interpolant because:

- The KuniJiban corpus is dense in built-up Japan (typically 1-3
  borings per km² in metropolitan areas). The 1 km cube grid will
  almost always find ``k=4`` borings within a few km.
- All three continuous covariates (elevation, river distance, coast
  distance) are slowly-varying geographic quantities; nearest-neighbor
  interpolation is the standard practice for them.
- The covariates are inputs the trained foundation model has already
  conditioned on at boring locations. Predictions in sparsely-sampled
  regions (mountainous interiors, offshore islands) will have higher
  GP posterior variance, which downstream tooling masks via the
  uncertainty raster -- nearest-neighbor extrapolation is honest about
  its limits.

This is the *pragmatic* unblock for Pillar 5. A future pass should
replace it with rasterized DEM + computed distance-to-river/coast
rasters; the cube driver only depends on the :class:`CovariateLoader`
protocol, so the swap is local.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from national.data.covariate_registry import CovariateSpec

LOG = logging.getLogger("national.data.loaders.boring_knn")


def _round_unique(df: pd.DataFrame, decimals: int = 5) -> pd.DataFrame:
    """Collapse the boring parquet to one row per (rounded) location.

    The training parquet stores one row *per depth*, so a single boring
    appears ~20 times. The covariates we want (elev / river / coast /
    regime) are constant per boring -- depth-invariant. Take the first
    observation per rounded location to avoid distorting the KDTree
    density with depth duplicates.
    """
    df = df.copy()
    df["_lat_r"] = df["latitude_deg"].astype("float64").round(decimals)
    df["_lon_r"] = df["longitude_deg"].astype("float64").round(decimals)
    deduped = df.drop_duplicates(subset=["_lat_r", "_lon_r"], keep="first")
    deduped = deduped.drop(columns=["_lat_r", "_lon_r"]).reset_index(drop=True)
    return deduped


class BoringKnnIndex:
    """Shared KDTree + value matrix backing one or more :class:`BoringKnnLoader`s.

    Building the tree is the expensive step (~1 s for 125 k borings).
    The index loads the boring parquet once, deduplicates to one row
    per location, builds the tree, and exposes a ``query`` interface
    that returns the ``k`` nearest borings' indices + great-circle
    distance approximations in km.

    Multiple :class:`BoringKnnLoader` instances can share one index;
    each loader pulls a different column from the shared row index.
    """

    EARTH_RADIUS_KM = 6371.0088

    def __init__(
        self,
        parquet_path: Path,
        *,
        coord_decimals: int = 5,
    ) -> None:
        LOG.info("BoringKnnIndex: loading %s", parquet_path)
        df = pd.read_parquet(parquet_path)
        df = _round_unique(df, decimals=coord_decimals)
        LOG.info(
            "BoringKnnIndex: %d unique boring locations (deduped from raw rows)",
            len(df),
        )
        self.df = df
        lats = df["latitude_deg"].to_numpy(dtype=np.float64)
        lons = df["longitude_deg"].to_numpy(dtype=np.float64)
        # Project (lat, lon) to a local equirectangular xy in km centered
        # on the bbox centroid. Good enough for Japan-scale KDTree
        # neighbor lookups (errors <0.5 % within a 1000 km radius).
        mid_lat = float(lats.mean())
        mid_lon = float(lons.mean())
        self._mid_lat = mid_lat
        self._mid_lon = mid_lon
        self._km_per_deg_lat = np.pi * self.EARTH_RADIUS_KM / 180.0
        self._km_per_deg_lon = self._km_per_deg_lat * np.cos(np.radians(mid_lat))
        xs = (lons - mid_lon) * self._km_per_deg_lon
        ys = (lats - mid_lat) * self._km_per_deg_lat
        self.tree = cKDTree(np.column_stack([xs, ys]))

    def query(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        k: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(distances_km, indices)`` of the ``k`` nearest borings."""
        xs = (lons - self._mid_lon) * self._km_per_deg_lon
        ys = (lats - self._mid_lat) * self._km_per_deg_lat
        pts = np.column_stack([xs, ys])
        return self.tree.query(pts, k=k)


class BoringKnnLoader:
    """Sample one boring-parquet column at arbitrary ``(lat, lon)`` via KDTree.

    The loader implements the :class:`national.data.covariate_registry.CovariateLoader`
    protocol so it slots into the registry that
    :class:`national.prediction.engine.PredictionEngine` walks.

    * For ``spec.category == 'categorical'``: returns the plurality (mode)
      of the ``k`` nearest borings. Equivalent to nearest-neighbor when
      ``k=1``.
    * For continuous specs: returns the inverse-distance-weighted mean of
      the ``k`` nearest borings' column values. ``k=1`` falls back to
      strict nearest-neighbor (no weighting).
    """

    spec: CovariateSpec

    def __init__(
        self,
        spec: CovariateSpec,
        index: BoringKnnIndex,
        column: str,
        *,
        k: int = 4,
        max_distance_km: float | None = None,
    ) -> None:
        if column not in index.df.columns:
            raise KeyError(
                f"BoringKnnLoader: column {column!r} not in parquet "
                f"(have {list(index.df.columns)})"
            )
        self.spec = spec
        self.index = index
        self.column = column
        self.k = int(k)
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.max_distance_km = (
            float(max_distance_km) if max_distance_km is not None else None
        )
        if spec.fill_value is not None:
            self._fill_value: float = float(spec.fill_value)
        elif spec.category == "categorical":
            self._fill_value = -1.0
        else:
            self._fill_value = 0.0

    def sample(
        self,
        lats: torch.Tensor,
        lons: torch.Tensor,
        depths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del depths  # depth-invariant
        lats_np = lats.detach().cpu().numpy().astype(np.float64, copy=False)
        lons_np = lons.detach().cpu().numpy().astype(np.float64, copy=False)
        dists, idxs = self.index.query(lats_np, lons_np, k=self.k)
        # cKDTree.query returns 1-D shapes when k=1; normalise to 2-D so
        # the rest of the math is uniform.
        if self.k == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]

        column_vals = self.index.df[self.column].to_numpy()
        neighbor_vals = column_vals[idxs]  # (N, k)
        out: np.ndarray
        if self.spec.category == "categorical":
            # Plurality vote. For k=1 this is just neighbor_vals[:, 0].
            if self.k == 1:
                out = neighbor_vals[:, 0].astype(np.int64)
            else:
                # Per-row mode via np.unique. Vectorised across rows by
                # using bincount on small integer codes (regime / era /
                # litho are all bounded by tens of categories).
                vmax = int(np.nanmax(neighbor_vals)) + 1
                out = np.empty(len(neighbor_vals), dtype=np.int64)
                for i, row in enumerate(neighbor_vals):
                    out[i] = int(np.bincount(row.astype(np.int64), minlength=vmax).argmax())
        else:
            # Inverse-distance weighting. Add a tiny epsilon so an exact
            # hit (distance 0) doesn't NaN the weight.
            weights = 1.0 / (dists + 1e-9)
            weights = weights / weights.sum(axis=1, keepdims=True)
            out = (neighbor_vals.astype(np.float64) * weights).sum(axis=1)

        # Fill rows whose nearest neighbour exceeded the distance cap.
        if self.max_distance_km is not None:
            too_far = dists[:, 0] > self.max_distance_km
            if too_far.any():
                LOG.info(
                    "BoringKnnLoader(%s): %d / %d query points beyond %.1f km "
                    "from any boring; filling with %s",
                    self.column, int(too_far.sum()), len(too_far),
                    self.max_distance_km, self._fill_value,
                )
                out = out.astype(np.float64, copy=True)
                out[too_far] = self._fill_value

        if self.spec.category == "categorical":
            return torch.as_tensor(out.astype(np.int64))
        return torch.as_tensor(out.astype(np.float32))

    def sample_grid(self, bbox, resolution_m, depth=None):  # noqa: ANN001
        raise NotImplementedError(
            "BoringKnnLoader.sample_grid is not implemented; the cube "
            "engine queries points directly via sample()."
        )


__all__ = ["BoringKnnIndex", "BoringKnnLoader"]
