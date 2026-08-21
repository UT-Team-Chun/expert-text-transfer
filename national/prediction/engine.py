"""Prediction engine: turn a trained FoundationModel into a national Zarr cube.

The engine has three layers:

- :py:meth:`PredictionEngine.predict_points` -- fast online endpoint used by
  the FastAPI route. Takes Tensors / arrays, returns ``(mean, std)``.

- :py:meth:`PredictionEngine.predict_tile` -- batched inference over a
  rectangular tile, returns an ``xarray.DataArray`` with dims
  ``(depth, lat, lon)`` and a ``"statistic"`` axis for mean/std.

- :py:meth:`PredictionEngine.predict_cube` -- iterates over every tile from
  the :class:`TileManager`, calls ``predict_tile``, and writes the result
  into a single Zarr cube with chunked layout. Distributed-aware: when
  ``WORLD_SIZE > 1`` each rank handles a disjoint subset of tiles and
  ranks join via a Zarr open per tile (Zarr supports concurrent writes
  to disjoint chunks).

Memory model: a single tile prediction is ``B = n_lat * n_lon * n_depth``
forward passes. For a 1024x1024 lat-lon tile at 32 depths that's ~33 M
points, which we mini-batch in ``cfg.prediction.batch_size_cells``
chunks (~200 k cells per forward pass on a GH200 is ~5 GB activations).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from national.tiling.tile_manager import TileBounds, TileManager

if TYPE_CHECKING:
    import xarray as xr

    from national.data.covariate_registry import CovariateRegistry
    from national.models.foundation import FoundationModel

LOG = logging.getLogger("national.prediction.engine")


@dataclass(frozen=True)
class GridSpec:
    """Resolution + depth list controlling the predicted cube layout."""

    resolution_m: float
    depths_m: tuple[float, ...]
    batch_size_cells: int = 200_000

    def __post_init__(self) -> None:
        if self.resolution_m <= 0:
            raise ValueError(f"resolution_m must be positive, got {self.resolution_m}")
        if not self.depths_m:
            raise ValueError("depths_m must contain at least one entry.")
        if self.batch_size_cells <= 0:
            raise ValueError(f"batch_size_cells must be positive, got {self.batch_size_cells}")


class PredictionEngine:
    """Run point, tile, and cube inference from a trained foundation model."""

    def __init__(
        self,
        model: "FoundationModel",
        registry: "CovariateRegistry | None",
        tile_manager: TileManager,
        grid: GridSpec,
        *,
        device: torch.device | str = "cpu",
        regime_loader_name: str | None = None,
        categorical_one_hot: dict[str, int] | None = None,
    ) -> None:
        """Build the engine.

        Args:
            model: trained foundation model. The model's expected input
                dim must match the per-row feature shape this engine
                produces: lat/lon/depth + the registry's continuous
                covariates + one-hot expansions of every loader named
                in ``categorical_one_hot``.
            registry: covariate registry. Must contain at least every
                continuous covariate the model was trained on (e.g.
                ``absolute_elevation``, ``river_distance_km``,
                ``coast_distance_km``), and every loader whose name
                appears in ``categorical_one_hot``.
            categorical_one_hot: ``{loader_name: n_categories}`` mapping
                for every categorical covariate the model expects to
                see one-hot-encoded in its input tensor. Order matters:
                the categorical blocks are concatenated in the order
                the dict iterates (insertion order on Python 3.7+),
                so the cube driver MUST use the same ordering the
                training pipeline used (regime, then aist_era,
                then aist_litho_macro, per the BoringDataset
                ``extra_one_hot_columns`` insertion order).
            regime_loader_name: name of the regime loader used for
                post-hoc FiLM adjustment via
                ``FoundationModel.regime_film``. May be ``None`` if the
                trained model has no FiLM head. Distinct from
                ``categorical_one_hot`` because FiLM consumes the raw
                regime codes while the GP input wants one-hot.
        """
        self.model = model
        self.registry = registry
        self.tile_manager = tile_manager
        self.grid = grid
        self.device = torch.device(device) if isinstance(device, str) else device
        self.regime_loader_name = regime_loader_name
        self.categorical_one_hot: dict[str, int] = dict(
            categorical_one_hot if categorical_one_hot is not None else {}
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    # ---- public API --------------------------------------------------------
    @torch.no_grad()
    def predict_points(
        self,
        lats: torch.Tensor | np.ndarray | list[float],
        lons: torch.Tensor | np.ndarray | list[float],
        depths: torch.Tensor | np.ndarray | list[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` at the given coordinates."""
        lat_t = self._as_tensor(lats)
        lon_t = self._as_tensor(lons)
        dep_t = self._as_tensor(depths)
        if not (lat_t.shape == lon_t.shape == dep_t.shape):
            raise ValueError(
                f"Shape mismatch: lats={tuple(lat_t.shape)}, lons={tuple(lon_t.shape)}, "
                f"depths={tuple(dep_t.shape)}"
            )
        x, regime = self._assemble_features(lat_t, lon_t, dep_t)
        pred = self.model.predict(x.to(self.device), regime_codes=regime)
        return pred.mean.cpu(), pred.std.cpu()

    @torch.no_grad()
    def predict_tile(self, tile: TileBounds) -> "xr.DataArray":
        """Predict mean + std over a rectangular tile."""
        import xarray as xr

        lat_axis, lon_axis = self._tile_axes(tile)
        depths = np.asarray(self.grid.depths_m, dtype=np.float64)
        n_lat, n_lon, n_depth = lat_axis.size, lon_axis.size, depths.size

        LON, LAT = np.meshgrid(lon_axis, lat_axis, indexing="xy")
        DEPTHS = depths
        # Flatten to (n_total, 3) row order: (depth_outer, lat, lon).
        flat_lat = np.broadcast_to(LAT[None, :, :], (n_depth, n_lat, n_lon)).reshape(-1)
        flat_lon = np.broadcast_to(LON[None, :, :], (n_depth, n_lat, n_lon)).reshape(-1)
        flat_dep = np.broadcast_to(DEPTHS[:, None, None], (n_depth, n_lat, n_lon)).reshape(-1)

        means = np.empty(flat_lat.size, dtype=np.float32)
        stds = np.empty_like(means)
        batch = int(self.grid.batch_size_cells)
        for start in range(0, flat_lat.size, batch):
            stop = min(start + batch, flat_lat.size)
            mean_b, std_b = self.predict_points(
                flat_lat[start:stop], flat_lon[start:stop], flat_dep[start:stop]
            )
            means[start:stop] = mean_b.numpy().astype(np.float32)
            stds[start:stop] = std_b.numpy().astype(np.float32)

        cube = np.stack(
            [
                means.reshape(n_depth, n_lat, n_lon),
                stds.reshape(n_depth, n_lat, n_lon),
            ],
            axis=0,
        )
        return xr.DataArray(
            cube,
            dims=("statistic", "depth", "lat", "lon"),
            coords={
                "statistic": np.array(["mean", "std"]),
                "depth": depths.astype(np.float32),
                "lat": lat_axis.astype(np.float64),
                "lon": lon_axis.astype(np.float64),
            },
            name="prediction",
            attrs={"tile_id": tile.tile_id},
        )

    def predict_cube(
        self,
        output_path: Path,
        *,
        chunks: dict[str, int] | None = None,
    ) -> Path:
        """Predict the whole region and write to a Zarr cube.

        Distributed-aware: if ``WORLD_SIZE > 1`` each rank handles only the
        tiles it owns (round-robin by tile index). The first rank writes the
        cube skeleton; other ranks write into existing chunks.

        Args:
            output_path: target Zarr directory.
            chunks: optional override for the Zarr chunk layout.
        """
        import xarray as xr
        import zarr

        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        tiles = self.tile_manager.tiles()
        my_tiles = [t for i, t in enumerate(tiles) if i % world == rank]
        LOG.info(
            "predict_cube: rank=%d/%d handling %d of %d tiles",
            rank,
            world,
            len(my_tiles),
            len(tiles),
        )

        # Build the cube skeleton on rank 0 using the first tile's axes
        # to infer per-tile size. We assume tile axes are uniform; if they
        # are not, the writer falls back to one Zarr group per tile.
        if rank == 0:
            self._init_cube_skeleton(output_path, tiles, chunks=chunks)

        # Barrier on file-system: every rank waits until the skeleton exists.
        skeleton = Path(output_path)
        if world > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        elif rank != 0:
            # Single-process simulation -- still wait for skeleton existence.
            while not skeleton.exists():
                pass

        for t in my_tiles:
            tile_da = self.predict_tile(t)
            self._write_tile_into_cube(output_path, t, tile_da)

        if world > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return Path(output_path)

    # ---- internals ---------------------------------------------------------
    def _as_tensor(
        self, x: torch.Tensor | np.ndarray | list[float]
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().to(dtype=torch.float32).reshape(-1)
        return torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(-1)

    def _assemble_features(
        self,
        lats: torch.Tensor,
        lons: torch.Tensor,
        depths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build the per-row feature tensor + regime codes.

        Output order:
          ``[lat, lon, depth, *continuous, *one_hot_cat0, *one_hot_cat1, ...]``

        This must match the column order ``BoringDataset`` produced at
        training time. The continuous block comes from
        :meth:`CovariateRegistry.stack_continuous`; the one-hot blocks
        come from sampling each loader named in
        ``self.categorical_one_hot`` in dict insertion order and
        expanding the returned int codes to ``n_categories``-wide
        one-hot rows.

        The separate ``regime`` return is the *raw* regime code tensor
        used by :meth:`FoundationModel.regime_film` for post-hoc
        per-regime bias / scale correction. It is independent of the
        one-hot block above; both can be present simultaneously (the
        v2 hero pipeline does so) or only one (a model trained without
        FiLM still wants the one-hot, and vice versa).
        """
        cols = [lats.unsqueeze(-1), lons.unsqueeze(-1), depths.unsqueeze(-1)]
        regime: torch.Tensor | None = None
        sampled: dict[str, torch.Tensor] | None = None
        if self.registry is not None:
            if self.registry.continuous_names:
                cov = self.registry.stack_continuous(lats, lons, depths)
                cols.append(cov)
            if (
                self.regime_loader_name is not None
                and self.regime_loader_name in self.registry.names
            ):
                if sampled is None:
                    sampled = self.registry.sample(lats, lons, depths)
                regime_vec = sampled[self.regime_loader_name].long().clamp_min(0)
                regime = regime_vec
            for cat_name, n_cat in self.categorical_one_hot.items():
                if cat_name not in self.registry.names:
                    raise KeyError(
                        f"categorical_one_hot referenced loader {cat_name!r} "
                        f"but the registry only has {self.registry.names!r}"
                    )
                if sampled is None:
                    sampled = self.registry.sample(lats, lons, depths)
                codes = sampled[cat_name].long().clamp(0, n_cat - 1)
                one_hot = torch.nn.functional.one_hot(codes, num_classes=int(n_cat))
                cols.append(one_hot.to(dtype=lats.dtype))
        x = torch.cat(cols, dim=-1)
        return x, regime

    def _tile_axes(self, tile: TileBounds) -> tuple[np.ndarray, np.ndarray]:
        deg_per_m_lat = 1.0 / 111_320.0
        mid_lat = 0.5 * (tile.lat_min + tile.lat_max)
        deg_per_m_lon = deg_per_m_lat / max(0.1, abs(np.cos(np.radians(mid_lat))))
        d_lat = self.grid.resolution_m * deg_per_m_lat
        d_lon = self.grid.resolution_m * deg_per_m_lon
        lat_axis = np.arange(tile.lat_min, tile.lat_max + d_lat / 2.0, d_lat)
        lon_axis = np.arange(tile.lon_min, tile.lon_max + d_lon / 2.0, d_lon)
        return lat_axis, lon_axis

    def _init_cube_skeleton(
        self,
        output_path: Path,
        tiles: list[TileBounds],
        *,
        chunks: dict[str, int] | None,
    ) -> None:
        # For Phase B we lay down one Zarr group per tile. Phase C will switch
        # to a single global cube once the TileManager exposes the global
        # axes (currently it returns a single tile in Phase A/B).
        Path(output_path).mkdir(parents=True, exist_ok=True)
        del chunks  # forward-looking

    def _write_tile_into_cube(
        self,
        output_path: Path,
        tile: TileBounds,
        tile_da: "xr.DataArray",
    ) -> None:
        from shared.io.zarr_writer import write_zarr_cube

        tile_dir = Path(output_path) / f"tile_{tile.tile_id}.zarr"
        chunks = {
            "statistic": 1,
            "depth": min(8, tile_da.sizes["depth"]),
            "lat": min(512, tile_da.sizes["lat"]),
            "lon": min(512, tile_da.sizes["lon"]),
        }
        write_zarr_cube(tile_da, tile_dir, chunks=chunks)


__all__ = ["GridSpec", "PredictionEngine"]
