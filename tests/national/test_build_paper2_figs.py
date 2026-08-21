"""Light smoke tests for ``scripts/build_paper2_figs``.

Each test fabricates the minimum on-disk artefacts each figure needs
(synthetic LRO summary.json, synthetic conformal_mondrian.json, etc.)
in ``tmp_path`` and asserts that the renderer:

- runs without raising,
- writes a non-empty ``.pdf`` to the chosen output path,
- writes a sibling ``.caption.md`` with the figure caption,
- gracefully degrades to a placeholder PDF when the upstream NFS
  artefact is missing (Fig 7 / Fig 8).

These tests deliberately avoid reading the real national parquet /
soil_text_layers.csv to stay fast and hermetic; the placeholder-fallback
branches inside the renderer are what gets exercised here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # noqa: E402  # headless backend before pyplot import

import importlib.util  # noqa: E402

import numpy as np  # noqa: E402
import pytest

# Load build_paper2_figs.py directly from its source location next to
# this test file. We avoid importing through ``scripts.<name>`` because
# pytest may discover the test from a different worktree than the one
# whose ``scripts/__init__.py`` is on the default sys.path, leading to
# a stale-module pickup.
_THIS_BACKEND = Path(__file__).resolve().parents[2]
_BPF_SRC = _THIS_BACKEND / "scripts" / "build_paper2_figs.py"


def _load_bpf():
    spec = importlib.util.spec_from_file_location(
        "build_paper2_figs_under_test", _BPF_SRC)
    assert spec is not None and spec.loader is not None, (
        f"cannot load spec for {_BPF_SRC}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_paper2_figs_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


bpf = _load_bpf()


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    d = tmp_path / "figures"
    d.mkdir()
    return d


def _write_lro_summary(runs_dir: Path, region: str, rmse: float,
                       mae: float) -> None:
    rd = runs_dir / f"dkl_national_lro_{region}"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "summary.json").write_text(json.dumps({
        "spatial_kfold": [{
            "fold": 0, "n_train": 1, "n_test": 1,
            "rmse": rmse, "mae": mae, "std_mean": 7.0,
        }],
    }))


def _write_conformal(runs_dir: Path, run_name: str,
                     coverage_shift: float = 0.0) -> Path:
    rd = runs_dir / run_name
    rd.mkdir(parents=True, exist_ok=True)
    alphas = [0.5, 0.8, 0.95]
    per_regime: dict[str, dict[str, Any]] = {}
    for a in alphas:
        per_regime[str(a)] = {}
        for r in range(8):
            per_regime[str(a)][str(r)] = {
                "name": bpf.REGIME_NAMES[r],
                "n_eval": 1000 + r,
                "n_cal": 1000 + r,
                "coverage": float(a) + coverage_shift,
                "uses_marginal_fallback": False,
            }
    marginal = {str(a): {
        "coverage_marginal_only": float(a) + coverage_shift,
        "coverage_mondrian": float(a) + coverage_shift,
        "gap_mondrian": coverage_shift,
        "gap_marginal": coverage_shift,
    } for a in alphas}
    payload = {
        "run_dir": str(rd),
        "n_total": 8_000,
        "n_cal": 4_000,
        "n_eval": 4_000,
        "cal_fraction": 0.5,
        "seed": 0,
        "alphas": alphas,
        "min_group_n": 100,
        "per_regime": per_regime,
        "marginal": marginal,
        "n_cal_per_regime": {str(r): 1000 + r for r in range(8)},
    }
    cj = rd / "conformal_mondrian.json"
    cj.write_text(json.dumps(payload))
    return cj


@pytest.fixture
def synthetic_runs(tmp_path: Path) -> Path:
    runs = tmp_path / "data" / "runs"
    runs.mkdir(parents=True)
    # All 8 LRO regions with synthetic-but-monotone RMSEs so the figure
    # has a kyushu_okinawa outlier at the tail.
    rmses = [13.0, 13.3, 13.9, 14.0, 14.1, 14.2, 15.0, 18.3]
    for region, rmse in zip(bpf.LRO_REGIONS, rmses):
        _write_lro_summary(runs, region, rmse=rmse, mae=rmse * 0.7)
    # full_v2 + a couple of extra runs so Fig 6 has marginal-gap dots.
    _write_conformal(runs, "dkl_national_full_v2", coverage_shift=0.0005)
    _write_conformal(runs, "dkl_national_rbf_12k_v2", coverage_shift=-0.0009)
    _write_conformal(runs, "dkl_national_lro_kanto", coverage_shift=0.0011)
    # Also drop a summary.json on the primary run so the fig4 national
    # reference line resolves.
    (runs / "dkl_national_full_v2" / "summary.json").write_text(json.dumps({
        "spatial_kfold": [{"fold": 0, "n_train": 1, "n_test": 1,
                            "rmse": 7.40, "mae": 4.50, "std_mean": 7.6}],
    }))
    return runs


# ============================================================
# Sanity helpers
# ============================================================


def _assert_pdf_and_caption(pdf: Path) -> None:
    assert pdf.exists(), f"missing {pdf}"
    assert pdf.stat().st_size > 200, f"{pdf} is suspiciously small"
    cap = pdf.with_suffix(".caption.md")
    assert cap.exists(), f"missing caption {cap}"
    assert cap.read_text().strip(), f"empty caption {cap}"


# ============================================================
# Tests
# ============================================================


def test_fig1_concept_emits_placeholder(out_dir: Path) -> None:
    out = out_dir / "fig1_concept.pdf"
    bpf.fig1_concept(out)
    _assert_pdf_and_caption(out)


def test_fig2_falls_back_when_parquet_missing(out_dir: Path,
                                              tmp_path: Path) -> None:
    """Fig 2 reads the national parquet; on a missing path it emits a
    placeholder rather than raising."""
    missing = tmp_path / "no_such.parquet"
    bpf.fig2_study_area(out_dir / "fig2_study_area.pdf", missing, None)
    _assert_pdf_and_caption(out_dir / "fig2_study_area.pdf")


def test_fig3_runs_without_layers_csv(out_dir: Path) -> None:
    """The hard-coded held-out RMSE table is enough to draw panels (a) +
    (b); panel (c) just becomes a 'not available' note."""
    bpf.fig3_llm_text_gain(out_dir / "fig3_llm_text_gain.pdf",
                           layers_csv=None)
    _assert_pdf_and_caption(out_dir / "fig3_llm_text_gain.pdf")


def test_fig4_lro_gap_renders_from_synthetic_runs(
        out_dir: Path, synthetic_runs: Path) -> None:
    out = out_dir / "fig4_lro_gap.pdf"
    bpf.fig4_lro_gap(out, synthetic_runs)
    _assert_pdf_and_caption(out)


def test_fig4_lro_gap_falls_back_when_no_runs(out_dir: Path,
                                              tmp_path: Path) -> None:
    empty = tmp_path / "empty_runs"
    empty.mkdir()
    out = out_dir / "fig4_lro_gap.pdf"
    bpf.fig4_lro_gap(out, empty)
    _assert_pdf_and_caption(out)


def test_fig5_model_inversion_hardcoded(out_dir: Path) -> None:
    out = out_dir / "fig5_model_inversion.pdf"
    bpf.fig5_model_inversion(out)
    _assert_pdf_and_caption(out)


def test_fig6_conformal_heatmap_from_synthetic(
        out_dir: Path, synthetic_runs: Path) -> None:
    out = out_dir / "fig6_conformal_heatmap.pdf"
    bpf.fig6_conformal_heatmap(out, synthetic_runs,
                               primary_run="dkl_national_full_v2")
    _assert_pdf_and_caption(out)


def test_fig6_conformal_heatmap_falls_back_when_missing(out_dir: Path,
                                                        tmp_path: Path) -> None:
    empty = tmp_path / "no_runs"
    empty.mkdir()
    out = out_dir / "fig6_conformal_heatmap.pdf"
    bpf.fig6_conformal_heatmap(out, empty,
                               primary_run="dkl_national_full_v2")
    _assert_pdf_and_caption(out)


def test_fig7_cube_transect_deferred(out_dir: Path, tmp_path: Path) -> None:
    """Without the NFS-synced cube, Fig 7 (the vertical transect) must
    produce a placeholder PDF rather than raising. Documented contract for
    the deferred slice of the pivot plan."""
    cube = tmp_path / "no_cube"
    out = out_dir / "fig7_cube_slices.pdf"
    bpf.fig7_cube_transect(out, cube)
    _assert_pdf_and_caption(out)


def test_fig8_depth_profiles_deferred_main_only(out_dir: Path,
                                                tmp_path: Path) -> None:
    cube = tmp_path / "no_cube"
    out = out_dir / "fig8_uncertainty.pdf"
    supp = out_dir / "fig8b_lpi_supp.pdf"
    bpf.fig8_depth_profiles_and_lpi(out, cube, main_only=True, supp_out=supp)
    _assert_pdf_and_caption(out)
    _assert_pdf_and_caption(supp)


# ============================================================
# Tile-aggregation tests (real-layout contract)
# ============================================================


def _write_tile_zarrs(cube_root: Path,
                      with_std: bool = True,
                      n_tiles: int = 2) -> None:
    """Fabricate a minimal tiled-zarr cube under ``cube_root/cube/``.

    Mirrors what ``predict_national_cube.py`` writes: a per-tile
    DataArray named ``prediction`` with dims
    ``(statistic, depth, lat, lon)``. Tiles are split along the
    ``lon`` axis so ``combine_by_coords`` has real work to do.
    """
    import xarray as xr

    cube_dir = cube_root / "cube"
    cube_dir.mkdir(parents=True, exist_ok=True)
    depths = np.array([0.0, 3.0, 5.0, 7.0, 13.0, 20.0], dtype=float)
    lat = np.linspace(35.0, 36.0, 4)
    statistics = ["mean", "std"] if with_std else ["mean"]
    # Two longitudinal tiles, disjoint in lon.
    lon_chunks = np.array_split(np.linspace(139.0, 140.0, 6), n_tiles)
    for ti, lon in enumerate(lon_chunks):
        data = np.full(
            (len(statistics), len(depths), len(lat), len(lon)),
            fill_value=float(ti) + 1.0,
            dtype="float32",
        )
        # Give the std slice a non-zero value so p95 reconstruction has signal.
        if with_std:
            data[1, ...] = 0.5
        da = xr.DataArray(
            data,
            dims=("statistic", "depth", "lat", "lon"),
            coords={
                "statistic": statistics,
                "depth": depths,
                "lat": lat,
                "lon": lon,
            },
            name="prediction",
        )
        da.to_zarr(cube_dir / f"tile_{ti:04d}.zarr", mode="w")


def _write_unsnapped_tile_grid(cube_root: Path,
                               with_std: bool = True) -> None:
    """Fabricate a 2x2 tile grid that reproduces the real on-disk
    layout: each tile's ``lon`` axis is built at its *own* mid-latitude
    (``d_lon = res / (111320 * cos(lat))``), so two tiles in the same
    lon column but different lat-bands carry DIFFERENT lon vectors and
    share NO lon coordinate.

    This is exactly what ``engine.predict_cube()`` writes BEFORE any
    coordinate snapping. A bare ``xr.combine_by_coords(..., NaN)`` over
    these tiles is degenerate (either raises 'not monotonic' or
    outer-joins into a ~mostly-NaN grid), which is the fig7/fig8 blank
    bug. ``_load_cube_dataarray`` must snap-then-assign and return a
    dense, regular, monotonic grid.

    Distinct, non-constant per-tile fills let downstream assertions
    verify the data survives assembly (not just the geometry).
    """
    import xarray as xr

    cube_dir = cube_root / "cube"
    cube_dir.mkdir(parents=True, exist_ok=True)
    depths = np.array([0.0, 3.0, 5.0, 7.0, 13.0, 20.0], dtype=float)
    statistics = ["mean", "std"] if with_std else ["mean"]
    res_m = 1000.0
    earth_r = 111320.0
    d_lat = res_m / earth_r
    n_lat, n_lon = 12, 12
    # Two lat-bands and two lon-columns, CONTIGUOUS (band 2 begins right
    # where band 1 ends; col 2 begins right where col 1 ends) -- exactly
    # how adjacent national tiles abut. Each band still computes its own
    # cos(lat)-dependent d_lon, so the two bands in a shared lon column
    # carry divergent lon vectors (the bug precondition) WHILE together
    # tiling a gap-free footprint (so the snapped grid is genuinely
    # dense, not artificially sparse from a fixture lat gap).
    lat_band_starts = [24.0, 24.0 + n_lat * d_lat]
    tile_idx = 0
    for lat0 in lat_band_starts:
        # Each band's lon spacing uses cos at the band's own latitude.
        d_lon = res_m / (earth_r * float(np.cos(np.deg2rad(lat0))))
        lat = lat0 + np.arange(n_lat) * d_lat
        lon_col_starts = [139.0, 139.0 + n_lon * d_lon]
        for lon0 in lon_col_starts:
            lon = lon0 + np.arange(n_lon) * d_lon
            mean_fill = 5.0 + 3.0 * tile_idx  # distinct per tile
            data = np.empty(
                (len(statistics), len(depths), n_lat, n_lon),
                dtype="float32")
            data[0, ...] = mean_fill
            if with_std:
                data[1, ...] = 7.58  # healthy near-flat sigma (per diag)
            da = xr.DataArray(
                data,
                dims=("statistic", "depth", "lat", "lon"),
                coords={
                    "statistic": statistics,
                    "depth": depths,
                    "lat": lat,
                    "lon": lon,
                },
                name="prediction",
            )
            da.to_zarr(cube_dir / f"tile_{tile_idx:04d}.zarr", mode="w")
            tile_idx += 1


def _write_lpi_nc(cube_root: Path) -> None:
    import xarray as xr

    maps_dir = cube_root / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    lat = np.linspace(35.0, 36.0, 4)
    lon = np.linspace(139.0, 140.0, 6)
    da = xr.DataArray(
        np.full((len(lat), len(lon)), 2.5, dtype="float32"),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="lpi_pga30",
    )
    da.to_netcdf(maps_dir / "lpi_pga30.nc")


def test_fig7_cube_transect_loads_tile_zarrs(out_dir: Path,
                                             tmp_path: Path) -> None:
    """Fig 7 must aggregate ``cube/tile_*.zarr`` tiles, select
    ``statistic='mean'``, and render a non-placeholder transect PDF."""
    cube_root = tmp_path / "cube_dir"
    _write_tile_zarrs(cube_root, with_std=True, n_tiles=2)
    out = out_dir / "fig7_cube_slices.pdf"
    # Sample a transect inside the synthetic cube's lon span (139-140 E).
    bpf.fig7_cube_transect(out, cube_root, lat=35.5, lon0=139.05,
                           lon1=139.95, n_samples=12)
    _assert_pdf_and_caption(out)
    # PDF should be materially larger than the placeholder (which is
    # ~2-3 kB of text-only PDF).
    assert out.stat().st_size > 4_000, (
        f"{out} looks like a placeholder ({out.stat().st_size} bytes); "
        "expected a real transect render")


def test_fig7_resolves_when_cube_dir_points_at_cube_subdir(
        out_dir: Path, tmp_path: Path) -> None:
    """Back-compat: --cube-dir pointed directly at the cube/ subdirectory
    (the tile root) must still resolve via the bare tile_*.zarr glob."""
    cube_root = tmp_path / "cube_dir"
    _write_tile_zarrs(cube_root, with_std=True, n_tiles=2)
    out = out_dir / "fig7_cube_slices.pdf"
    # Pass the cube/ subdirectory directly.
    bpf.fig7_cube_transect(out, cube_root / "cube", lat=35.5, lon0=139.05,
                           lon1=139.95, n_samples=12)
    _assert_pdf_and_caption(out)
    assert out.stat().st_size > 4_000


def test_fig8_depth_profiles_render_with_lpi(
        out_dir: Path, tmp_path: Path) -> None:
    """Fig 8 (depth profiles + LPI) renders the 1-D per-site columns and
    the clean LPI map into a non-placeholder PDF."""
    cube_root = tmp_path / "cube_dir"
    _write_tile_zarrs(cube_root, with_std=True, n_tiles=2)
    _write_lpi_nc(cube_root)
    out = out_dir / "fig8_uncertainty.pdf"
    bpf.fig8_depth_profiles_and_lpi(out, cube_root)
    _assert_pdf_and_caption(out)
    assert out.stat().st_size > 4_000


# ============================================================
# New 1-D extraction helpers (transect + depth profile)
# ============================================================


def test_edges_from_centres_midpoints_and_extrapolated_ends() -> None:
    """``_edges_from_centres`` must return n+1 edges: interior edges at the
    midpoints, outer edges extrapolated a half-step so each value is
    cell-centred under pcolormesh."""
    c = np.array([0.0, 1.0, 2.0, 4.0])
    e = bpf._edges_from_centres(c)
    assert e.shape == (5,)
    # Interior edges are the midpoints.
    assert np.allclose(e[1:-1], [0.5, 1.5, 3.0])
    # Outer edges extrapolate the first/last half-step.
    assert np.isclose(e[0], -0.5)
    assert np.isclose(e[-1], 5.0)
    # Monotone, so pcolormesh accepts it.
    assert np.all(np.diff(e) > 0)


def test_edges_from_centres_single_and_empty() -> None:
    assert bpf._edges_from_centres(np.array([3.0])).tolist() == [2.5, 3.5]
    assert bpf._edges_from_centres(np.array([])).tolist() == [0.0, 1.0]


def test_extract_const_lat_transect_is_1d_path_over_depths(
        tmp_path: Path) -> None:
    """The const-lat transect must return a (n_depth, n_lon) section sampled
    along a SINGLE lat row -- a 1-D spatial path, not a 2-D (lat,lon) tile."""
    cube_root = tmp_path / "cube_dir"
    _write_tile_zarrs(cube_root, with_std=True, n_tiles=2)
    cube = bpf._load_cube_dataarray(cube_root)
    mean_da = cube.sel(statistic="mean")
    lon_samples, depths, section, used_lat = bpf._extract_const_lat_transect(
        mean_da, lat=35.5, lon0=139.05, lon1=139.95, n=10)
    assert lon_samples.shape == (10,)
    assert section.shape == (depths.size, 10)
    # Depths ascending so the imshow y-axis is monotone before inversion.
    assert np.all(np.diff(depths) > 0)
    # Sampled on one lat row near the request.
    assert abs(used_lat - 35.5) < 1.0
    # All sampled columns finite (the synthetic cube is dense here).
    assert np.isfinite(section).all()


def test_extract_depth_column_returns_per_depth_mean_and_std(
        tmp_path: Path) -> None:
    """A single (lat,lon) column must return per-depth mean + std with the
    depth axis ascending; the std channel survives so the band can be drawn."""
    cube_root = tmp_path / "cube_dir"
    _write_tile_zarrs(cube_root, with_std=True, n_tiles=2)
    cube = bpf._load_cube_dataarray(cube_root)
    mean_da = cube.sel(statistic="mean")
    std_da = cube.sel(statistic="std")
    depths, mean_col, std_col, ulat, ulon = bpf._extract_depth_column(
        mean_da, std_da, lat=35.5, lon=139.5)
    assert depths.size == mean_col.size == std_col.size
    assert np.all(np.diff(depths) > 0)
    assert np.isfinite(mean_col).any()
    # The synthetic std slice is a constant 0.5 (depth-flat, mirroring the
    # real depth-flat sigma the probe found).
    assert np.allclose(std_col[np.isfinite(std_col)], 0.5)
    assert np.isfinite(ulat) and np.isfinite(ulon)


def test_extract_depth_column_without_std_returns_nan_band(
        tmp_path: Path) -> None:
    """When the cube has no std slice, the column std is all-NaN so the
    profile renderer draws the mean line with no band (no crash)."""
    cube_root = tmp_path / "cube_dir"
    _write_tile_zarrs(cube_root, with_std=False, n_tiles=2)
    cube = bpf._load_cube_dataarray(cube_root)
    mean_da = cube.sel(statistic="mean")
    depths, mean_col, std_col, _, _ = bpf._extract_depth_column(
        mean_da, None, lat=35.5, lon=139.5)
    assert depths.size == mean_col.size
    assert np.isnan(std_col).all()


def test_unsnapped_tiles_have_no_shared_lon_coords(tmp_path: Path) -> None:
    """Sanity guard on the fixture itself: the per-mid-lat lon vectors of
    two lat-bands in the same lon column must NOT share any coordinate
    (this is the precondition that breaks a bare combine_by_coords)."""
    import xarray as xr

    cube_root = tmp_path / "cube_dir"
    _write_unsnapped_tile_grid(cube_root, with_std=True)
    tile_dirs = sorted((cube_root / "cube").glob("tile_*.zarr"))
    lons = [xr.open_dataarray(t, engine="zarr").coords["lon"].values
            for t in tile_dirs]
    # tile 0 (lat band 24.0, lon col 139) vs tile 2 (lat band 24.667,
    # lon col 139): same lon column start, different mid-lat spacing.
    shared = np.intersect1d(lons[0], lons[2])
    assert shared.size <= 1, (
        f"fixture lon vectors unexpectedly share {shared.size} coords; "
        "the per-mid-lat divergence is what the loader fix must handle")
    # And the spacings genuinely differ.
    assert not np.isclose(lons[0][1] - lons[0][0],
                          lons[2][1] - lons[2][0])


def test_load_cube_dataarray_snaps_unsnapped_tiles_to_dense_grid(
        tmp_path: Path) -> None:
    """Regression for the fig7/fig8 blank-panel bug.

    The on-disk tiles carry per-mid-lat lon vectors that share no
    coordinate across lat-bands. ``_load_cube_dataarray`` must snap them
    onto a single global axis and assemble a DENSE, REGULAR, MONOTONIC
    grid -- NOT the ~97%-NaN outer-join that a bare combine_by_coords
    produces (which rendered the panels near-blank)."""
    cube_root = tmp_path / "cube_dir"
    _write_unsnapped_tile_grid(cube_root, with_std=True)
    cube = bpf._load_cube_dataarray(cube_root)

    # Monotonic, regular axes.
    lat = cube.coords["lat"].values
    lon = cube.coords["lon"].values
    assert np.all(np.diff(lat) > 0), "lat axis not strictly increasing"
    assert np.all(np.diff(lon) > 0), "lon axis not strictly increasing"
    assert np.allclose(np.diff(lon), np.diff(lon)[0], rtol=1e-6), (
        "global lon axis is not regular (snap failed)")

    # Density: a 2x2 tile grid of 12x12 cells should fill most of the
    # assembled grid. The degenerate outer-join was ~3% finite; the
    # snapped grid must be the opposite regime (>50% finite).
    mean = cube.sel(statistic="mean").isel(depth=2).values  # depth=5m
    finite_frac = np.isfinite(mean).mean()
    assert finite_frac > 0.5, (
        f"assembled grid only {finite_frac:.1%} finite at depth=5m; "
        "loader still producing the degenerate outer-join")

    # Data survived: the distinct per-tile fills (5, 8, 11, 14) must all
    # appear in the assembled mean slice.
    present = set(np.unique(mean[np.isfinite(mean)]).round(3).tolist())
    for expected in (5.0, 8.0, 11.0, 14.0):
        assert any(abs(v - expected) < 1e-3 for v in present), (
            f"tile fill {expected} missing from assembled grid; "
            f"present values were {sorted(present)}")

    # Sigma channel survives and is the healthy ~7.58 (not collapsed).
    std = cube.sel(statistic="std").isel(depth=2).values
    finite_std = std[np.isfinite(std)]
    assert finite_std.size > 0 and np.allclose(finite_std, 7.58, atol=1e-2)


def test_fig7_renders_populated_from_unsnapped_tiles(
        out_dir: Path, tmp_path: Path) -> None:
    """End-to-end: fig7 transect over the realistic per-mid-lat tile grid
    must produce a populated (non-placeholder) PDF (nearest-grid sampling
    snaps the requested line onto the dense assembled cube)."""
    cube_root = tmp_path / "cube_dir"
    _write_unsnapped_tile_grid(cube_root, with_std=True)
    out = out_dir / "fig7_cube_slices.pdf"
    # The unsnapped fixture lives near lat ~24 N, lon ~139 E; sample inside.
    bpf.fig7_cube_transect(out, cube_root, lat=24.05, lon0=139.02,
                           lon1=139.18, n_samples=12)
    _assert_pdf_and_caption(out)
    assert out.stat().st_size > 4_000


def test_fig8_renders_populated_from_unsnapped_tiles(
        out_dir: Path, tmp_path: Path) -> None:
    """End-to-end: fig8 (depth profiles + LPI) over the realistic
    per-mid-lat tile grid must render populated (1-D columns + LPI map)."""
    cube_root = tmp_path / "cube_dir"
    _write_unsnapped_tile_grid(cube_root, with_std=True)
    _write_lpi_nc(cube_root)
    out = out_dir / "fig8_uncertainty.pdf"
    bpf.fig8_depth_profiles_and_lpi(out, cube_root)
    _assert_pdf_and_caption(out)
    assert out.stat().st_size > 4_000


def test_fig8_profiles_render_when_std_slice_missing(
        out_dir: Path, tmp_path: Path) -> None:
    """If the cube was written with only ``statistic='mean'`` (no std),
    Fig 8 must still render: the profiles draw the mean line with no
    uncertainty band rather than crashing, and the main_only supp file is
    still produced."""
    cube_root = tmp_path / "cube_dir"
    _write_tile_zarrs(cube_root, with_std=False, n_tiles=2)
    out = out_dir / "fig8_uncertainty.pdf"
    bpf.fig8_depth_profiles_and_lpi(out, cube_root, main_only=True,
                                    supp_out=out_dir / "fig8b_lpi_supp.pdf")
    _assert_pdf_and_caption(out)
    # main_only contract: callers expect the supp file to be present.
    _assert_pdf_and_caption(out_dir / "fig8b_lpi_supp.pdf")


def test_main_driver_all_figures_smoke(out_dir: Path, tmp_path: Path,
                                       synthetic_runs: Path) -> None:
    """End-to-end driver call: with synthetic runs + missing parquet/cube,
    the driver must still emit all 8 main-figure PDFs (placeholders where
    upstream data are missing) without raising."""
    rc = bpf.main([
        "--runs-dir", str(synthetic_runs),
        "--parquet", str(tmp_path / "no_parquet.parquet"),
        "--layers-csv", str(tmp_path / "no_layers.csv"),
        "--cube-dir", str(tmp_path / "no_cube"),
        "--out-dir", str(out_dir),
        "--figures", "all",
        "--main-only",
    ])
    assert rc == 0
    for stem in ("fig1_concept", "fig2_study_area", "fig3_llm_text_gain",
                 "fig4_lro_gap", "fig5_model_inversion",
                 "fig6_conformal_heatmap", "fig7_cube_slices",
                 "fig8_uncertainty"):
        _assert_pdf_and_caption(out_dir / f"{stem}.pdf")


def test_main_driver_figure_subset(out_dir: Path, synthetic_runs: Path,
                                   tmp_path: Path) -> None:
    """Calling with --figures fig5 fig1 should produce exactly those PDFs
    and skip the others."""
    rc = bpf.main([
        "--runs-dir", str(synthetic_runs),
        "--parquet", str(tmp_path / "no_parquet.parquet"),
        "--layers-csv", str(tmp_path / "no_layers.csv"),
        "--cube-dir", str(tmp_path / "no_cube"),
        "--out-dir", str(out_dir),
        "--figures", "fig5", "fig1",
    ])
    assert rc == 0
    _assert_pdf_and_caption(out_dir / "fig1_concept.pdf")
    _assert_pdf_and_caption(out_dir / "fig5_model_inversion.pdf")
    assert not (out_dir / "fig2_study_area.pdf").exists()
    assert not (out_dir / "fig6_conformal_heatmap.pdf").exists()


# ============================================================
# Fig 7 stripe-fill (snap-assignment NaN-stripe artifact)
# ============================================================


def test_fill_thin_nan_stripes_fills_width1_gaps_keeps_wide_nodata() -> None:
    """The stripe-fill must (i) erase 1-cell interior NaN columns/rows
    bounded on both sides by data, and (ii) leave genuine wide (>=2-cell)
    no-data blocks NaN so the transparency contract holds."""
    lat = np.linspace(34.0, 35.0, 10)
    lon = np.linspace(139.0, 140.0, 10)
    grid = np.full((len(lat), len(lon)), 5.0, dtype="float64")
    # Inject a 1-cell vertical stripe (column 4) and a 1-cell horizontal
    # stripe (row 6) -- exactly the snap artefact.
    grid[:, 4] = np.nan
    grid[6, :] = np.nan
    # Inject a genuine WIDE no-data block (cols 7-9 over rows 0-2): >=2 wide.
    grid[0:3, 7:10] = np.nan

    filled = bpf._fill_thin_nan_stripes(grid.copy(), lat, lon)

    # The 1-cell stripes are gone everywhere EXCEPT where they cross the
    # genuine wide no-data block (those cells are no longer width-1 gaps).
    # Column 4 over rows 3..9 (clear of the wide block) must be filled.
    assert np.all(np.isfinite(filled[3:, 4])), (
        "1-cell vertical stripe not filled")
    # Row 6 over cols 0..6 (clear of the wide block) must be filled.
    assert np.all(np.isfinite(filled[6, 0:7])), (
        "1-cell horizontal stripe not filled")
    # The wide no-data block must remain NaN (transparency preserved).
    assert np.all(np.isnan(filled[0:3, 7:10])), (
        "wide no-data block must stay NaN, not be filled")
    # Filled stripe cells inherit the adjacent data value (no smearing).
    assert np.allclose(filled[3:, 4], 5.0)


def test_fill_thin_nan_stripes_handles_leading_dims() -> None:
    """The fill loops leading (statistic, depth) dims and fills each
    (lat, lon) plane independently."""
    plane = np.full((2, 3, 8, 8), 7.0, dtype="float64")
    plane[..., 3] = np.nan  # 1-cell vertical stripe in every plane
    filled = bpf._fill_thin_nan_stripes(plane.copy(), np.arange(8.0),
                                        np.arange(8.0))
    assert np.all(np.isfinite(filled)), "stripe persisted in some plane"
    assert np.allclose(filled, 7.0)


def test_load_cube_dataarray_has_no_interior_1cell_stripes(
        tmp_path: Path) -> None:
    """End-to-end: the assembled cube from the realistic per-mid-lat tile
    grid must carry NO interior width-1 NaN stripes after the fill."""
    cube_root = tmp_path / "cube_dir"
    _write_unsnapped_tile_grid(cube_root, with_std=True)
    cube = bpf._load_cube_dataarray(cube_root)
    mean = cube.sel(statistic="mean").isel(depth=2).values
    finite = np.isfinite(mean)
    # An interior width-1 vertical stripe = a NaN column whose left+right
    # neighbours are both finite over some row.
    interior = mean[:, 1:-1]
    left = finite[:, :-2]
    right = finite[:, 2:]
    nan_mid = ~np.isfinite(interior)
    bad_v = nan_mid & left & right
    assert not bad_v.any(), (
        f"{int(bad_v.sum())} interior 1-cell vertical NaN stripes remain")
    # And horizontal.
    interior_h = mean[1:-1, :]
    up = finite[:-2, :]
    down = finite[2:, :]
    nan_mid_h = ~np.isfinite(interior_h)
    bad_h = nan_mid_h & up & down
    assert not bad_h.any(), (
        f"{int(bad_h.sum())} interior 1-cell horizontal NaN stripes remain")


# ============================================================
# Block-mean display downsample (kills residual tile-seam stripes)
# ============================================================


def test_block_mean_2d_shape_and_mean() -> None:
    """A 4x4 array downsampled by factor 2 -> 2x2, each cell the mean of its
    2x2 block."""
    arr = np.arange(16, dtype="float64").reshape(4, 4)
    out = bpf._block_mean_2d(arr, 2)
    assert out.shape == (2, 2)
    # Top-left block = mean(0,1,4,5) = 2.5; bottom-right = mean(10,11,14,15).
    assert np.isclose(out[0, 0], (0 + 1 + 4 + 5) / 4)
    assert np.isclose(out[1, 1], (10 + 11 + 14 + 15) / 4)


def test_block_mean_2d_is_nan_aware() -> None:
    """Isolated NaN seam cells inside an otherwise-finite block are ignored
    (np.nanmean over the finite cells), so seams get averaged away."""
    arr = np.ones((4, 4), dtype="float64") * 3.0
    arr[0, 0] = np.nan  # one NaN in the top-left 2x2 block
    out = bpf._block_mean_2d(arr, 2)
    assert out.shape == (2, 2)
    # Block still resolves to 3.0 (mean of the 3 finite cells), not NaN.
    assert np.isclose(out[0, 0], 3.0)
    assert not np.isnan(out).any()


def test_block_mean_2d_all_nan_block_stays_nan() -> None:
    """A block whose every cell is NaN stays NaN (rendered transparent via
    set_bad), while finite blocks resolve normally."""
    arr = np.ones((4, 4), dtype="float64") * 5.0
    arr[0:2, 0:2] = np.nan  # entire top-left 2x2 block is NaN
    out = bpf._block_mean_2d(arr, 2)
    assert out.shape == (2, 2)
    assert np.isnan(out[0, 0]), "all-NaN block must stay NaN"
    assert np.isfinite(out[0, 1]) and np.isclose(out[0, 1], 5.0)


def test_block_mean_2d_pads_non_multiple_extent() -> None:
    """A 5x5 array with factor 2 pads (with NaN) to 6x6 -> 3x3; the padded
    edge cells still resolve from their finite members."""
    arr = np.ones((5, 5), dtype="float64") * 2.0
    out = bpf._block_mean_2d(arr, 2)
    assert out.shape == (3, 3)
    # The padded edge blocks each contain finite cells, so all resolve to 2.0.
    assert np.allclose(out, 2.0)


def test_block_mean_2d_factor_le_1_is_identity() -> None:
    arr = np.arange(9, dtype="float64").reshape(3, 3)
    assert bpf._block_mean_2d(arr, 1) is arr
    assert bpf._block_mean_2d(arr, 0) is arr


def test_downsample_axis_block_centres() -> None:
    """The coordinate axis downsamples to block-centre means so the image
    extent (min/max) tracks the data block-mean grid."""
    axis = np.array([0.0, 1.0, 2.0, 3.0])
    out = bpf._downsample_axis(axis, 2)
    assert out.shape == (2,)
    assert np.allclose(out, [0.5, 2.5])


def test_block_subsample_codes_2d_shape_matches_block_mean() -> None:
    """Categorical down-sample picks block centres and yields the SAME shape
    as the NaN-mean block-mean (so regime grid + std grid align cell-for-cell
    in fig8 panel (b))."""
    codes = np.zeros((8, 8), dtype="int16")
    codes[:, 4:] = 5
    std = np.ones((8, 8), dtype="float64")
    out_codes = bpf._block_subsample_codes_2d(codes, 4)
    out_std = bpf._block_mean_2d(std, 4)
    assert out_codes.shape == out_std.shape
    # Left half stays 0, right half stays 5 (no fractional regimes invented).
    assert set(np.unique(out_codes).tolist()) <= {0, 5}


# ============================================================
# Fig 8 panel (b) -- conformal interval half-width
# ============================================================


def _write_regime_nc(cube_root: Path, lat: np.ndarray,
                     lon: np.ndarray) -> None:
    import xarray as xr

    maps_dir = cube_root / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    # Split the grid into two regime blocks so the multiplier varies.
    codes = np.zeros((len(lat), len(lon)), dtype="int16")
    codes[:, len(lon) // 2:] = 5  # METAMORPHIC half
    da = xr.DataArray(codes, dims=("lat", "lon"),
                      coords={"lat": lat, "lon": lon}, name="regime_code")
    da.to_netcdf(maps_dir / "regime_code.nc")


def test_build_conformal_halfwidth_varies_across_regimes() -> None:
    """The half-width panel = std * q_group[regime] must produce a
    SPATIALLY-VARYING field (>1 distinct value), not the near-flat std."""
    std = np.full((4, 6), 7.58, dtype="float64")
    regime = np.zeros((4, 6), dtype="int16")
    regime[:, 3:] = 5  # second half is METAMORPHIC
    q_table = {0: 2.04, 5: 2.88}  # ALLUVIAL vs METAMORPHIC at alpha=0.95
    hw = bpf._build_conformal_halfwidth(std, regime, q_table, q_marginal=2.26)
    assert hw is not None
    distinct = np.unique(np.round(hw, 4))
    assert distinct.size >= 2, (
        f"half-width is near-flat ({distinct}); expected regime variation")
    assert np.allclose(hw[:, :3], 7.58 * 2.04)
    assert np.allclose(hw[:, 3:], 7.58 * 2.88)


def test_build_conformal_halfwidth_falls_back_without_table() -> None:
    """With no q-table the half-width degrades to a single fallback
    multiplier (marginal q, else Gaussian z) and stays finite."""
    std = np.full((3, 3), 7.5, dtype="float64")
    hw = bpf._build_conformal_halfwidth(std, None, {}, q_marginal=2.0)
    assert hw is not None and np.allclose(hw, 7.5 * 2.0)
    hw2 = bpf._build_conformal_halfwidth(std, None, {}, q_marginal=None)
    assert hw2 is not None and np.allclose(hw2, 7.5 * bpf._GAUSSIAN_Z_95)


def test_build_conformal_halfwidth_propagates_nan() -> None:
    """No-data (NaN) std cells stay NaN in the half-width (transparent)."""
    std = np.full((2, 2), 7.5, dtype="float64")
    std[0, 0] = np.nan
    regime = np.zeros((2, 2), dtype="int16")
    hw = bpf._build_conformal_halfwidth(std, regime, {0: 2.0}, q_marginal=2.0)
    assert hw is not None and np.isnan(hw[0, 0])
    assert np.isfinite(hw[1, 1])


def test_load_conformal_multipliers_reads_persisted_table(
        tmp_path: Path) -> None:
    """When conformal_mondrian.json carries quantiles_per_group, the loader
    reads q_group + q_marginal directly without touching predictions.npz."""
    runs = tmp_path / "runs"
    rd = runs / "dkl_national_full_v2"
    rd.mkdir(parents=True)
    payload = {
        "alphas": [0.5, 0.8, 0.9, 0.95],
        "quantiles_per_group": {
            "0": {"0.9": 1.5, "0.95": 2.04},
            "5": {"0.9": 2.1, "0.95": 2.88},
        },
        "quantiles_marginal": {"0.9": 1.7, "0.95": 2.26},
    }
    (rd / "conformal_mondrian.json").write_text(json.dumps(payload))
    table, q_marg = bpf._load_conformal_multipliers(runs, alpha=0.9)
    assert table == {0: 1.5, 5: 2.1}
    assert q_marg == 1.7


def test_fig8_band_uses_persisted_conformal_table(
        out_dir: Path, tmp_path: Path) -> None:
    """End-to-end: fig8 with a persisted per-regime multiplier table renders
    a populated PDF whose profile band is widened by the loaded q_group
    (the per-regime conformal multiplier resolution is still exercised)."""
    cube_root = tmp_path / "cube_dir"
    _write_tile_zarrs(cube_root, with_std=True, n_tiles=2)
    _write_lpi_nc(cube_root)
    runs = tmp_path / "runs"
    rd = runs / "dkl_national_full_v2"
    rd.mkdir(parents=True)
    (rd / "conformal_mondrian.json").write_text(json.dumps({
        "alphas": [0.9],
        "quantiles_per_group": {"0": {"0.9": 1.5}, "5": {"0.9": 2.1}},
        "quantiles_marginal": {"0.9": 1.7},
    }))
    out = out_dir / "fig8_uncertainty.pdf"
    bpf.fig8_depth_profiles_and_lpi(out, cube_root, runs_dir=runs)
    _assert_pdf_and_caption(out)
    assert out.stat().st_size > 4_000


# ============================================================
# Fig 8 panel (b) -- conformal table resolution (NFS / override)
# ============================================================
#
# On the render pod the Mondrian table is NOT under the figure-build
# --runs-dir; it lives on NFS at a cube-dir-adjacent runs/ tree. Without
# the NFS / --conformal-json resolution, panel (b) fell back to a flat
# Gaussian-z multiplier (the bug these tests guard).


def _write_conformal_table(json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({
        "alphas": [0.9, 0.95],
        "quantiles_per_group": {
            "0": {"0.9": 1.5, "0.95": 2.04},
            "5": {"0.9": 2.1, "0.95": 2.88},
        },
        "quantiles_marginal": {"0.9": 1.7, "0.95": 2.26},
    }))


def test_resolve_conformal_json_prefers_explicit_override(
        tmp_path: Path) -> None:
    """An explicit --conformal-json (file OR directory) wins over the
    runs-dir candidate."""
    runs = tmp_path / "runs"
    _write_conformal_table(runs / "dkl_national_full_v2"
                           / "conformal_mondrian.json")
    override = tmp_path / "elsewhere" / "conformal_mondrian.json"
    _write_conformal_table(override)
    # File form.
    got = bpf._resolve_conformal_json(
        runs, "dkl_national_full_v2", conformal_json=override)
    assert got == override
    # Directory form (resolver appends conformal_mondrian.json).
    got_dir = bpf._resolve_conformal_json(
        runs, "dkl_national_full_v2", conformal_json=override.parent)
    assert got_dir == override


def test_resolve_conformal_json_finds_cube_adjacent_nfs(
        tmp_path: Path) -> None:
    """With no override and an empty runs-dir, the resolver finds the table
    on a cube-dir-adjacent ``runs/<run>/`` tree (the pod NFS layout:
    products/<cube> and runs/<run> share a mount root)."""
    root = tmp_path / "mnt" / "nas" / "geo-estimation"
    cube_dir = root / "products" / "national_cube_japan_1km_v2hero"
    cube_dir.mkdir(parents=True)
    nfs_table = root / "runs" / "dkl_national_full_v2" / "conformal_mondrian.json"
    _write_conformal_table(nfs_table)
    empty_runs = tmp_path / "workspace" / "data" / "runs"
    empty_runs.mkdir(parents=True)
    got = bpf._resolve_conformal_json(
        empty_runs, "dkl_national_full_v2", cube_dir=cube_dir)
    assert got == nfs_table


def test_resolve_conformal_json_returns_none_when_absent(
        tmp_path: Path) -> None:
    empty = tmp_path / "nope"
    empty.mkdir()
    assert bpf._resolve_conformal_json(
        empty, "dkl_national_full_v2", cube_dir=tmp_path / "no_cube") is None


def test_load_conformal_multipliers_resolves_via_cube_dir_nfs(
        tmp_path: Path) -> None:
    """``_load_conformal_multipliers`` reads the per-regime table from the
    cube-dir-adjacent NFS tree even when ``runs_dir`` does not hold it."""
    root = tmp_path / "nas"
    cube_dir = root / "products" / "cube_x"
    cube_dir.mkdir(parents=True)
    _write_conformal_table(
        root / "runs" / "dkl_national_full_v2" / "conformal_mondrian.json")
    empty_runs = tmp_path / "local_runs"
    empty_runs.mkdir()
    table, q_marg = bpf._load_conformal_multipliers(
        empty_runs, alpha=0.95, cube_dir=cube_dir)
    assert table == {0: 2.04, 5: 2.88}
    assert q_marg == 2.26


def test_load_conformal_multipliers_explicit_json_overrides_runs_dir(
        tmp_path: Path) -> None:
    """``conformal_json`` short-circuits the runs-dir lookup."""
    override = tmp_path / "custom" / "conformal_mondrian.json"
    _write_conformal_table(override)
    table, q_marg = bpf._load_conformal_multipliers(
        None, alpha=0.9, conformal_json=override)
    assert table == {0: 1.5, 5: 2.1}
    assert q_marg == 1.7


def test_regime_grid_from_registry_degrades_without_parquet() -> None:
    """The registry regime sampler returns None (not raises) when the
    boring parquet is absent, so panel (b) can fall back to the marginal
    multiplier."""
    lat = np.linspace(34.0, 36.0, 5)
    lon = np.linspace(139.0, 141.0, 5)
    assert bpf._regime_grid_from_registry(lat, lon, None) is None
    assert bpf._regime_grid_from_registry(
        lat, lon, Path("/no/such/borings.parquet")) is None


def _write_boring_parquet(path: Path) -> None:
    """Fabricate a tiny enriched boring parquet with a west/east regime
    split so the registry KNN sampler returns a spatially-varying grid."""
    import pandas as pd

    rows = []
    # West cluster: ALLUVIAL (code 0). East cluster: METAMORPHIC (code 5).
    for lat in np.linspace(34.2, 35.8, 6):
        for lon in np.linspace(139.1, 139.4, 4):  # west
            rows.append((lat, lon, 0))
        for lon in np.linspace(140.6, 140.9, 4):  # east
            rows.append((lat, lon, 5))
    df = pd.DataFrame(rows, columns=["latitude_deg", "longitude_deg",
                                     "regime_code"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def test_regime_grid_from_registry_samples_spatial_variation(
        tmp_path: Path) -> None:
    """With a real boring parquet, the registry sampler (boring:knn) returns
    a populated regime grid that varies west-to-east -- the genuine
    'sample AIST regime per display cell via the covariate registry' path
    that drives panel (b) when no regime_code.nc is shipped."""
    pq = tmp_path / "borings.parquet"
    _write_boring_parquet(pq)
    lat = np.linspace(34.3, 35.7, 8)
    lon = np.linspace(139.2, 140.8, 12)
    grid = bpf._regime_grid_from_registry(lat, lon, pq,
                                          knn_max_distance_km=200.0)
    assert grid is not None, "registry sampler returned None with a real parquet"
    assert grid.shape == (lat.size, lon.size)
    # West columns resolve to ALLUVIAL (0), east columns to METAMORPHIC (5).
    assert grid[:, 0].tolist() == [0] * lat.size
    assert grid[:, -1].tolist() == [5] * lat.size
    assert set(np.unique(grid).tolist()) <= {0, 5}


def test_fig8_band_registry_path_renders_with_parquet(
        out_dir: Path, tmp_path: Path) -> None:
    """End-to-end: a real boring parquet + an explicit conformal table ->
    the per-site band multiplier is looked up via the registry-sampled
    regime code, and the composite renders populated."""
    cube_root = tmp_path / "cube_dir"
    _write_tile_zarrs(cube_root, with_std=True, n_tiles=2)
    _write_lpi_nc(cube_root)
    pq = tmp_path / "borings.parquet"
    _write_boring_parquet(pq)
    override = tmp_path / "conf" / "conformal_mondrian.json"
    _write_conformal_table(override)
    out = out_dir / "fig8_uncertainty.pdf"
    bpf.fig8_depth_profiles_and_lpi(out, cube_root, runs_dir=None,
                                    conformal_json=override, boring_parquet=pq)
    _assert_pdf_and_caption(out)
    assert out.stat().st_size > 4_000


def test_fig8_band_uses_nfs_table_without_parquet(
        out_dir: Path, tmp_path: Path) -> None:
    """End-to-end pod scenario: conformal table on a cube-dir-adjacent NFS
    tree (not under runs_dir), no boring parquet. The composite must still
    render populated -- resolving the multiplier table off NFS and degrading
    the band to the marginal multiplier (no site regime)."""
    root = tmp_path / "nas"
    cube_root = root / "products" / "cube_x"
    _write_tile_zarrs(cube_root, with_std=True, n_tiles=2)
    _write_lpi_nc(cube_root)
    _write_conformal_table(
        root / "runs" / "dkl_national_full_v2" / "conformal_mondrian.json")
    empty_runs = tmp_path / "workspace_runs"
    empty_runs.mkdir()
    out = out_dir / "fig8_uncertainty.pdf"
    bpf.fig8_depth_profiles_and_lpi(out, cube_root, runs_dir=empty_runs)
    _assert_pdf_and_caption(out)
    assert out.stat().st_size > 4_000


def test_fig8_accepts_explicit_conformal_json(
        out_dir: Path, tmp_path: Path) -> None:
    """fig8 honours an explicit ``conformal_json`` path (the --conformal-json
    CLI arg) for the per-site band multiplier."""
    cube_root = tmp_path / "cube_dir"
    _write_tile_zarrs(cube_root, with_std=True, n_tiles=2)
    _write_lpi_nc(cube_root)
    override = tmp_path / "anywhere" / "conformal_mondrian.json"
    _write_conformal_table(override)
    out = out_dir / "fig8_uncertainty.pdf"
    bpf.fig8_depth_profiles_and_lpi(out, cube_root, runs_dir=None,
                                    conformal_json=override)
    _assert_pdf_and_caption(out)
    assert out.stat().st_size > 4_000


def test_display_block_factor_decoupled_from_stripe_removal() -> None:
    """The duplication ripple is now removed at the SOURCE by the
    linear-interp regrid in ``_load_cube_dataarray`` (median-spacing target
    grid), so the display block-mean no longer carries stripe removal. It is
    kept small (>=1, <=4) purely for PDF file-size; relying on it for stripe
    attenuation (the old factor=6) is the regression this guards against."""
    assert 1 <= bpf._DISPLAY_BLOCK_FACTOR <= 4


def test_main_driver_threads_conformal_json_arg(
        out_dir: Path, tmp_path: Path, synthetic_runs: Path) -> None:
    """The --conformal-json CLI arg is parsed and reaches fig8 without
    raising (rendered as a placeholder here since the cube is absent)."""
    override = tmp_path / "conf" / "conformal_mondrian.json"
    _write_conformal_table(override)
    rc = bpf.main([
        "--runs-dir", str(synthetic_runs),
        "--parquet", str(tmp_path / "no_parquet.parquet"),
        "--layers-csv", str(tmp_path / "no_layers.csv"),
        "--cube-dir", str(tmp_path / "no_cube"),
        "--out-dir", str(out_dir),
        "--figures", "fig8",
        "--conformal-json", str(override),
    ])
    assert rc == 0
    _assert_pdf_and_caption(out_dir / "fig8_uncertainty.pdf")


# ============================================================
# Linear-interp regrid kills the nearest-snap duplication ripple
# ============================================================


def _write_two_tiles_differing_lon_spacing(
        cube_root: Path, lat_b: float = 70.0, n_lon: int = 72,
        n_lat: int = 14, base: float = 2.0, slope: float = 80.0) -> None:
    """Two side-by-side lon tiles (same lat band) with DIFFERENT native lon
    spacing -- the precondition that drove the snap-duplication crosshatch.

    Tile A uses the native spacing at lat=34 deg (fine); tile B uses the
    coarser spacing at ``lat_b`` deg. The OLD loader built a single shared
    global axis at ``cos(lat_min)`` (= tile A, the finest), then snapped tile
    B onto it via nearest-neighbour -- because tile B's native cells are far
    coarser than that axis, ~every few columns got DUPLICATED, painting a
    periodic ripple. The underlying field is a smooth pure-lon ramp (constant
    in lat), so the assembled per-lon mean profile is a straight line and ANY
    high-frequency residual is a loader artifact, not real signal.
    """
    import xarray as xr

    cube_dir = cube_root / "cube"
    cube_dir.mkdir(parents=True, exist_ok=True)
    depths = np.array([0.0, 5.0, 20.0], dtype=float)
    statistics = ["mean", "std"]
    res_m, earth = 1000.0, 111320.0
    d_lat = res_m / earth
    lat = 34.0 + np.arange(n_lat) * d_lat
    d_lon_a = res_m / (earth * float(np.cos(np.deg2rad(34.0))))
    d_lon_b = res_m / (earth * float(np.cos(np.deg2rad(lat_b))))

    def field(lo: np.ndarray) -> np.ndarray:
        LO, _ = np.meshgrid(lo, lat)
        return base + slope * (LO - 139.0)  # pure lon ramp, flat in lat

    for ti, (lon0, dlon) in enumerate(
            [(139.0, d_lon_a), (139.0 + n_lon * d_lon_a, d_lon_b)]):
        lon = lon0 + np.arange(n_lon) * dlon
        m = field(lon).astype("float32")
        data = np.empty((2, len(depths), n_lat, n_lon), dtype="float32")
        for di in range(len(depths)):
            data[0, di] = m
            data[1, di] = 7.5
        da = xr.DataArray(
            data, dims=("statistic", "depth", "lat", "lon"),
            coords={"statistic": statistics, "depth": depths,
                    "lat": lat, "lon": lon},
            name="prediction")
        da.to_zarr(cube_dir / f"tile_{ti:04d}.zarr", mode="w")


def _nearest_snap_assemble(cube_root: Path) -> np.ndarray:
    """Emulate the OLD nearest-snap loader: build a single global axis at the
    finest (``cos(lat_min)``) spacing, snap each tile via ``searchsorted``,
    assign into a preallocated array, fill width-1 stripes. Returns the
    mean-statistic, depth=1 plane. This is the baseline the linear-interp
    regrid must beat."""
    import xarray as xr

    bgla, snap = bpf._import_cube_snap_helpers()
    tiles = sorted((cube_root / "cube").glob("tile_*.zarr"))
    cubes = [xr.open_dataarray(t, engine="zarr") for t in tiles]
    all_lat = np.concatenate([c.coords["lat"].values for c in cubes])
    all_lon = np.concatenate([c.coords["lon"].values for c in cubes])
    la, lo = bgla(float(all_lat.min()), float(all_lat.max()),
                  float(all_lon.min()), float(all_lon.max()), 1000.0)
    cubes = [snap(c, la, lo) for c in cubes]
    tmpl = cubes[0]
    ng = tuple(d for d in tmpl.dims if d not in ("lat", "lon"))
    comb = np.full(tuple(tmpl.sizes[d] for d in ng) + (len(la), len(lo)),
                   np.nan, dtype=np.float64)
    for t in cubes:
        tt = t.transpose(*ng, "lat", "lon")
        li = np.searchsorted(la, tt.coords["lat"].values)
        oi = np.searchsorted(lo, tt.coords["lon"].values)
        comb[(slice(None),) * len(ng) + np.ix_(li, oi)] = tt.values
    comb = bpf._fill_thin_nan_stripes(comb, la, lo)
    return comb[0, 1]  # statistic=mean, depth index 1


def _meanprofile_autocorr_ripple(arr: np.ndarray) -> float:
    """Column/row-mean autocorrelation ripple amplitude.

    Collapse each axis to a 1D mean profile, remove the smooth (degree-1)
    trend, and report the relative residual RMS (``std(resid)/mean``) WEIGHTED
    by the strongest autocorrelation sidelobe (lag>=1) so a genuinely periodic
    ripple scores high while incoherent float noise does not. Returns the max
    over the two axes."""
    import warnings as _warnings

    a = np.asarray(arr, dtype=np.float64)
    out = 0.0
    for axis in (0, 1):
        with _warnings.catch_warnings():
            # All-NaN columns/rows -> NaN mean (dropped below); silence the
            # benign "Mean of empty slice" RuntimeWarning.
            _warnings.simplefilter("ignore", category=RuntimeWarning)
            prof = np.nanmean(a, axis=axis)
        prof = prof[np.isfinite(prof)]
        if prof.size < 12:
            continue
        x = np.arange(prof.size)
        resid = prof - np.polyval(np.polyfit(x, prof, 1), x)
        base = float(np.nanmean(np.abs(prof)))
        if base <= 0 or np.allclose(resid, 0.0):
            continue
        r = resid - resid.mean()
        ac = np.correlate(r, r, mode="full")[r.size - 1:]
        ac = ac / ac[0]
        lags = ac[1:max(2, r.size // 2)]
        peak = float(np.max(np.abs(lags))) if lags.size else 0.0
        out = max(out, float(np.std(resid) / base) * np.sqrt(max(peak, 0.0)))
    return out


def test_nearest_snap_baseline_has_strong_duplication_ripple(
        tmp_path: Path) -> None:
    """Guard the fixture itself: the OLD nearest-snap assembly of two tiles
    with differing native lon spacing DOES paint a strong periodic ripple
    (>> 0.02) -- this is the artifact the linear-interp regrid must remove. If
    this baseline ever drops below the threshold the comparison test would be
    vacuous."""
    cube_root = tmp_path / "cube_dir"
    _write_two_tiles_differing_lon_spacing(cube_root)
    baseline = _nearest_snap_assemble(cube_root)
    amp = _meanprofile_autocorr_ripple(baseline)
    assert amp > 0.02, (
        f"nearest-snap baseline ripple {amp:.4f} is not above 0.02; the "
        "fixture no longer reproduces the duplication crosshatch")


def test_load_cube_dataarray_linear_interp_kills_duplication_ripple(
        tmp_path: Path) -> None:
    """Regression for the crosshatch ripple: assembling two tiles with
    DIFFERENT native lon spacing via ``_load_cube_dataarray`` (linear-interp
    onto a median-spacing target grid) must yield a column/row-mean
    autocorrelation ripple amplitude < 0.02 -- far below the nearest-snap
    baseline (~0.1) that periodically duplicated source columns."""
    cube_root = tmp_path / "cube_dir"
    _write_two_tiles_differing_lon_spacing(cube_root)

    cube = bpf._load_cube_dataarray(cube_root)
    mean = cube.sel(statistic="mean").isel(depth=1).values

    interp_amp = _meanprofile_autocorr_ripple(mean)
    baseline_amp = _meanprofile_autocorr_ripple(
        _nearest_snap_assemble(cube_root))

    assert interp_amp < 0.02, (
        f"linear-interp ripple {interp_amp:.4f} still >= 0.02; the "
        "duplication crosshatch was not removed")
    # And it is a real improvement over the nearest-snap baseline.
    assert interp_amp < baseline_amp, (
        f"linear-interp ripple {interp_amp:.4f} is not below the "
        f"nearest-snap baseline {baseline_amp:.4f}")

    # The regridded axes stay regular + monotonic and the data survives.
    lat_ax = cube.coords["lat"].values
    lon_ax = cube.coords["lon"].values
    assert np.all(np.diff(lat_ax) > 0) and np.all(np.diff(lon_ax) > 0)
    assert np.allclose(np.diff(lon_ax), np.diff(lon_ax)[0], rtol=1e-6)
    assert np.isfinite(mean).mean() > 0.8
    # Sigma channel preserved (not collapsed).
    std = cube.sel(statistic="std").isel(depth=1).values
    fin = std[np.isfinite(std)]
    assert fin.size > 0 and np.allclose(fin, 7.5, atol=1e-2)


def test_load_cube_target_grid_not_finer_than_native(tmp_path: Path) -> None:
    """The target lon spacing must be the MEDIAN native tile spacing -- never
    finer than a representative tile (a finer target is exactly what let the
    old nearest-snap loader duplicate source columns)."""
    import xarray as xr

    cube_root = tmp_path / "cube_dir"
    _write_two_tiles_differing_lon_spacing(cube_root)
    tiles = sorted((cube_root / "cube").glob("tile_*.zarr"))
    native_dlon = sorted(
        float(np.median(np.diff(xr.open_dataarray(t, engine="zarr")
                                .coords["lon"].values)))
        for t in tiles)
    cube = bpf._load_cube_dataarray(cube_root)
    target_dlon = float(np.median(np.diff(cube.coords["lon"].values)))
    # Target spacing sits within the native spread (>= the finest tile),
    # i.e. it is NOT finer than the most fine-grained tile.
    assert target_dlon >= native_dlon[0] * (1.0 - 1e-6), (
        f"target d_lon {target_dlon:.6g} is finer than the finest native "
        f"tile {native_dlon[0]:.6g}; would re-introduce duplication")


# ============================================================
# Conformal hardening: predictions.npz recompute via cube-dir NFS
# ============================================================


def _write_predictions_npz(run_dir: Path, n: int = 4000,
                           seed: int = 0) -> None:
    """Fabricate a predictions.npz matching the schema
    ``run_mondrian_recal_national.py`` reads (pred_mean / pred_std / y_true /
    regime) with several well-populated regimes so the Mondrian recompute
    yields a multi-regime multiplier table."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    regime = rng.integers(0, 4, size=n).astype(np.int16)
    y_true = rng.normal(5.0, 2.0, size=n).astype(np.float32)
    pred_mean = (y_true + rng.normal(0.0, 1.0, size=n)).astype(np.float32)
    # Regime-dependent residual spread so the per-group quantiles differ.
    pred_std = (1.0 + 0.5 * regime).astype(np.float32)
    np.savez(run_dir / "predictions.npz",
             pred_mean=pred_mean, pred_std=pred_std, y_true=y_true,
             regime=regime, baseline_pred=np.zeros_like(pred_mean))


def test_resolve_predictions_npz_finds_cube_adjacent_nfs(
        tmp_path: Path) -> None:
    """With no --runs-dir, the resolver finds predictions.npz on the
    cube-dir-adjacent ``runs/<run>/`` tree (the pod NFS layout)."""
    root = tmp_path / "mnt" / "nas" / "geo-estimation"
    cube_dir = root / "products" / "national_cube_japan_1km_v2hero"
    cube_dir.mkdir(parents=True)
    npz = root / "runs" / "dkl_national_full_v2" / "predictions.npz"
    _write_predictions_npz(npz.parent)
    got = bpf._resolve_predictions_npz(
        None, "dkl_national_full_v2", cube_dir=cube_dir)
    assert got == npz


def test_resolve_predictions_npz_returns_none_when_absent(
        tmp_path: Path) -> None:
    assert bpf._resolve_predictions_npz(
        tmp_path / "nope", "dkl_national_full_v2",
        cube_dir=tmp_path / "no_cube") is None


def test_load_conformal_multipliers_recomputes_from_cube_adjacent_npz(
        tmp_path: Path) -> None:
    """The headline conformal-hardening contract: with NO --runs-dir, NO
    conformal_mondrian.json, but a predictions.npz on the cube-dir-adjacent
    NFS runs tree, ``_load_conformal_multipliers`` RECOMPUTES the per-regime
    multipliers (does NOT regress to the flat Gaussian fallback)."""
    root = tmp_path / "nas"
    cube_dir = root / "products" / "cube_x"
    cube_dir.mkdir(parents=True)
    _write_predictions_npz(
        root / "runs" / "dkl_national_full_v2", n=6000, seed=1)
    # runs_dir omitted entirely; resolution must go through cube_dir.
    table, q_marg = bpf._load_conformal_multipliers(
        None, alpha=0.9, cube_dir=cube_dir)
    assert len(table) >= 2, (
        f"expected a multi-regime multiplier table, got {table}")
    # All recomputed multipliers are positive, finite half-width scalers.
    assert all(np.isfinite(v) and v > 0 for v in table.values())


def test_fig8_band_recomputes_from_cube_adjacent_npz_without_runs_dir(
        out_dir: Path, tmp_path: Path) -> None:
    """End-to-end pod scenario for the standard render command (cube-dir
    only): NO --runs-dir, NO conformal JSON, but a predictions.npz on the
    cube-dir-adjacent NFS tree + a boring parquet. The per-site band must
    use the RECOMPUTED per-regime multipliers rather than the flat Gaussian
    fallback, and the composite renders populated."""
    root = tmp_path / "nas"
    cube_root = root / "products" / "cube_x"
    _write_tile_zarrs(cube_root, with_std=True, n_tiles=2)
    _write_lpi_nc(cube_root)
    _write_predictions_npz(
        root / "runs" / "dkl_national_full_v2", n=5000, seed=2)
    pq = tmp_path / "borings.parquet"
    _write_boring_parquet(pq)
    out = out_dir / "fig8_uncertainty.pdf"
    # runs_dir omitted -> resolution must reach predictions.npz via cube_dir.
    bpf.fig8_depth_profiles_and_lpi(out, cube_root, runs_dir=None,
                                    boring_parquet=pq)
    _assert_pdf_and_caption(out)
    assert out.stat().st_size > 4_000


def test_smooth_nan_aware_crushes_ripple_preserves_nan_and_mean():
    """NaN-aware Gaussian low-pass: crushes the baked-in crosshatch ripple
    (the real block-mean + smooth path drives a 9-native-cell ripple to
    <1% residual), preserves large no-data (ocean) regions as NaN, and
    holds the field mean within 1%."""
    import numpy as np
    from scripts.build_paper2_figs import _smooth_nan_aware, _block_mean_2d

    # Real display path: block-mean(2) then Gaussian smooth on a 9-cell ripple.
    x = np.arange(240)
    field = (15.0 + 2.0 * np.sin(2 * np.pi * x / 9.0))[None, :].repeat(240, 0)
    sm = _smooth_nan_aware(_block_mean_2d(field, 2))
    ratio = float(np.std(sm[60, 20:100]) / np.std(field[120, :]))
    assert ratio < 0.12, f"ripple not crushed: ratio={ratio}"

    # Large no-data region (ocean) stays NaN at its interior.
    a = np.full((80, 80), 10.0)
    a[20:60, 20:60] = np.nan
    assert np.isnan(_smooth_nan_aware(a)[40, 40])
    assert np.isfinite(_smooth_nan_aware(a)[0, 0])

    # Mean preserved within 1% on a smooth field.
    b = np.random.RandomState(0).rand(60, 60) * 5 + 10
    assert abs(np.nanmean(_smooth_nan_aware(b)) - np.nanmean(b)) / np.nanmean(b) < 0.01

    # sigma <= 0 is an identity passthrough.
    assert _smooth_nan_aware(a, 0) is a


def test_smooth_along_distance_attenuates_column_spike_keeps_depth_and_nan():
    """Fig 7 transect smoothing: 1-D NaN-aware Gaussian ALONG the distance
    axis (axis=1) only. It must (i) leave the depth structure (axis=0)
    unchanged when the section has no along-distance variation, (ii) strongly
    attenuate a synthetic single-column vertical spike, and (iii) preserve NaN
    no-data columns."""
    import numpy as np
    from scripts.build_paper2_figs import (
        _smooth_along_distance, _TRANSECT_SMOOTH_SIGMA)

    n_depth, n_lon = 40, 100

    # (i) DEPTH structure untouched: a section whose value depends ONLY on
    # depth (constant along distance) must come back essentially unchanged,
    # because smoothing happens along distance and there is nothing to mix.
    depth_profile = np.linspace(2.0, 30.0, n_depth)  # soft surface -> stiff
    section = np.repeat(depth_profile[:, None], n_lon, axis=1)
    sm = _smooth_along_distance(section)
    assert np.allclose(sm, section, atol=1e-9), "depth structure was altered"
    # The per-distance column profile (the depth axis) is preserved exactly.
    assert np.allclose(sm[:, n_lon // 2], depth_profile, atol=1e-9)

    # (ii) A single bright vertical column (single-borehole IDW spike) on an
    # otherwise flat background is strongly attenuated along distance.
    flat = np.full((n_depth, n_lon), 5.0)
    spike = flat.copy()
    col = n_lon // 2
    spike[:, col] = 25.0  # full-depth bright column
    sm_spike = _smooth_along_distance(spike)
    amp_before = spike[n_depth // 2, col] - 5.0
    amp_after = sm_spike[n_depth // 2, col] - 5.0
    assert amp_after < 0.4 * amp_before, (
        f"column spike not attenuated: before={amp_before}, after={amp_after}")
    # Far from the spike the background is essentially untouched.
    assert abs(sm_spike[n_depth // 2, 5] - 5.0) < 0.1

    # (iii) A WIDE no-data band (e.g. an offshore stretch of the transect,
    # much wider than the kernel) keeps NaN at its interior; the smoothing does
    # not bleed finite data across a large gap with no local support. A single
    # isolated missing column, by contrast, is legitimately filled from its
    # neighbours (desirable 1-column speckle fill, not a bleed across a real
    # no-data region).
    wide_gap = np.full((n_depth, n_lon), 8.0)
    wide_gap[:, 30:50] = np.nan  # 20-column band >> kernel FWHM (~7 at sigma 3)
    sm_wide = _smooth_along_distance(wide_gap)
    assert np.isnan(sm_wide[n_depth // 2, 40]), "wide no-data band not preserved"
    assert np.isfinite(sm_wide[n_depth // 2, 0])
    single_gap = np.full((n_depth, n_lon), 8.0)
    single_gap[:, 12] = np.nan
    assert np.isfinite(_smooth_along_distance(single_gap)[n_depth // 2, 12]), \
        "single isolated missing column should be filled, not left NaN"

    # sigma <= 0 is an identity passthrough; a 1-column section is too thin.
    assert _smooth_along_distance(section, 0) is section
    one_col = np.full((n_depth, 1), 3.0)
    assert _smooth_along_distance(one_col) is one_col
    # The constant is the documented modest cross-section sigma.
    assert _TRANSECT_SMOOTH_SIGMA == 3.0


# ============================================================
# Sparse tile loader (_load_tiles_covering) -- ESTALE-resistant
# ============================================================


def _write_tile_field(cube_dir: Path, ti: int,
                      lat0: float, lon0: float,
                      n: int = 6, step: float = 0.1,
                      with_std: bool = True,
                      mean_fill: float | None = None) -> None:
    """Write one ``tile_{ti:04d}.zarr`` covering ``[lat0, lat0+n*step) x
    [lon0, lon0+n*step)`` with a per-tile-distinct mean fill so a downstream
    assertion can verify WHICH tiles were selected by value."""
    import xarray as xr

    depths = np.array([0.0, 3.0, 5.0, 7.0, 13.0, 20.0], dtype=float)
    statistics = ["mean", "std"] if with_std else ["mean"]
    lat = lat0 + np.arange(n) * step
    lon = lon0 + np.arange(n) * step
    fill = float(ti + 1) if mean_fill is None else float(mean_fill)
    data = np.empty((len(statistics), len(depths), n, n), dtype="float32")
    data[0, ...] = fill
    if with_std:
        data[1, ...] = 7.58
    da = xr.DataArray(
        data,
        dims=("statistic", "depth", "lat", "lon"),
        coords={"statistic": statistics, "depth": depths,
                "lat": lat, "lon": lon},
        name="prediction",
    )
    da.to_zarr(cube_dir / f"tile_{ti:04d}.zarr", mode="w")


def _write_spread_tile_grid(cube_root: Path,
                            with_std: bool = True) -> dict[int, tuple]:
    """Fabricate a 3x3 grid of disjoint tiles spread across a wide
    lat/lon footprint (mimicking the national cube's many tiles), each
    with a distinct mean fill. Returns ``{tile_idx: (lat0, lon0)}`` so a
    test can reason about which tiles a given window should select.
    """
    cube_dir = cube_root / "cube"
    cube_dir.mkdir(parents=True, exist_ok=True)
    n, step = 6, 0.1
    span = n * step  # 0.6 deg per tile
    lat0s = [34.0, 35.0, 36.0]   # three lat bands, gap between them
    lon0s = [135.0, 138.0, 139.5]  # three lon columns
    placement: dict[int, tuple] = {}
    ti = 0
    for lat0 in lat0s:
        for lon0 in lon0s:
            _write_tile_field(cube_dir, ti, lat0, lon0, n=n, step=step,
                              with_std=with_std, mean_fill=float(10 + ti))
            placement[ti] = (lat0, lon0, lat0 + span, lon0 + span)
            ti += 1
    return placement


def test_load_tiles_covering_selects_only_intersecting_subset(
        tmp_path: Path) -> None:
    """``_load_tiles_covering`` must combine ONLY the tiles whose footprint
    intersects the requested window -- not the full grid."""
    cube_root = tmp_path / "cube_dir"
    placement = _write_spread_tile_grid(cube_root, with_std=True)
    # A tiny bbox inside the lat=35.0 band / lon=139.5 column tile only.
    # That tile is index 5 (lat0=35.0, lon0=139.5), mean_fill = 15.0.
    target = None
    for ti, (lat0, lon0, lat1, lon1) in placement.items():
        if abs(lat0 - 35.0) < 1e-9 and abs(lon0 - 139.5) < 1e-9:
            target = ti
    assert target is not None
    # Tile 5 spans lat 35.0..35.5, lon 139.5..140.0; pick a bbox well inside.
    cube = bpf._load_tiles_covering(
        cube_root, (35.25, 35.35), (139.75, 139.85))
    mean_da = cube.sel(statistic="mean")
    # The assembled window carries only the target tile's fill (10 + 5 = 15).
    vals = np.asarray(mean_da.values)
    finite = vals[np.isfinite(vals)]
    assert finite.size > 0
    assert np.allclose(finite, float(10 + target)), (
        f"expected only tile {target}'s fill {10 + target}, got "
        f"{np.unique(finite)}")
    # The assembled lat/lon envelope must stay within the selected tile's
    # footprint -- it must NOT span the far-away lat=34 / lon=135 tiles.
    assert float(mean_da.coords["lat"].min()) >= 34.9
    assert float(mean_da.coords["lon"].min()) >= 139.4


def test_load_tiles_covering_band_selects_one_lat_row(
        tmp_path: Path) -> None:
    """A const-lat band over the full lon span selects the 3 tiles in that
    lat row (the fig7 transect access pattern), not all 9."""
    cube_root = tmp_path / "cube_dir"
    placement = _write_spread_tile_grid(cube_root, with_std=True)
    # lat band around 35.3 (inside the 35.0..35.6 row); full lon span.
    cube = bpf._load_tiles_covering(
        cube_root, (35.25, 35.35), (134.0, 140.5))
    mean_da = cube.sel(statistic="mean")
    finite = np.asarray(mean_da.values)
    finite = finite[np.isfinite(finite)]
    # The three lat=35 tiles are indices 3,4,5 -> fills 13,14,15.
    row_fills = {float(10 + ti) for ti, (lat0, *_rest) in placement.items()
                 if abs(lat0 - 35.0) < 1e-9}
    assert set(np.unique(finite)).issubset(row_fills), (
        f"band pulled tiles outside the lat row: {np.unique(finite)} "
        f"vs expected {row_fills}")
    # All three row tiles contributed (full lon span intersects all three).
    assert set(np.unique(finite)) == row_fills


def test_load_tiles_covering_raises_when_no_tiles(tmp_path: Path) -> None:
    """No tile zarrs at all -> FileNotFoundError (caller -> placeholder)."""
    cube_root = tmp_path / "empty_cube"
    (cube_root / "cube").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        bpf._load_tiles_covering(cube_root, (35.0, 36.0), (139.0, 140.0))


def test_load_tiles_covering_raises_when_window_misses_all(
        tmp_path: Path) -> None:
    """Tiles exist but the window intersects none -> ValueError."""
    cube_root = tmp_path / "cube_dir"
    _write_spread_tile_grid(cube_root, with_std=True)
    with pytest.raises(ValueError):
        # Far north of every tile.
        bpf._load_tiles_covering(cube_root, (45.0, 45.1), (139.0, 140.0))


def test_load_tiles_covering_skips_bad_tile_with_warning(
        tmp_path: Path) -> None:
    """A single corrupt tile in the intersecting set is skipped (warning,
    not fatal); the good tiles still assemble."""
    cube_root = tmp_path / "cube_dir"
    cube_dir = cube_root / "cube"
    cube_dir.mkdir(parents=True)
    # Two good tiles in the same lat row + lon span.
    _write_tile_field(cube_dir, 0, 35.0, 139.0, mean_fill=20.0)
    _write_tile_field(cube_dir, 1, 35.0, 139.6, mean_fill=21.0)
    # A corrupt tile that LOOKS like a tile zarr but cannot be opened.
    bad = cube_dir / "tile_0002.zarr"
    bad.mkdir()
    (bad / "zarr.json").write_text("{ this is not valid zarr metadata")
    # Window over the two good tiles' lat row + a lon span hitting both.
    cube = bpf._load_tiles_covering(
        cube_root, (35.0, 35.6), (139.0, 140.2))
    mean_da = cube.sel(statistic="mean")
    finite = np.asarray(mean_da.values)
    finite = finite[np.isfinite(finite)]
    assert finite.size > 0
    # Only the two good fills survive; the corrupt tile contributed nothing.
    assert set(np.unique(finite)).issubset({20.0, 21.0})


def test_fig7_transect_renders_from_sparse_band(
        out_dir: Path, tmp_path: Path) -> None:
    """End-to-end: fig7 over a wide 3x3 spread grid renders a populated
    transect by loading ONLY the lat-band tiles (sparse path)."""
    cube_root = tmp_path / "cube_dir"
    _write_spread_tile_grid(cube_root, with_std=True)
    out = out_dir / "fig7_cube_slices.pdf"
    # Transect along the lat=35 row, across its lon=139.5 tile.
    bpf.fig7_cube_transect(out, cube_root, lat=35.3, lon0=139.55,
                           lon1=140.05, n_samples=12)
    _assert_pdf_and_caption(out)
    assert out.stat().st_size > 4_000, (
        f"{out} looks like a placeholder; expected a real sparse-band render")


def test_fig8_depth_profiles_render_per_site_sparse(
        out_dir: Path, tmp_path: Path) -> None:
    """End-to-end: fig8 depth profiles over a spread grid where each of the
    three default sites lives in a DIFFERENT tile -- each loaded via its own
    tiny per-site bbox (sparse path), not a full-cube read."""
    import xarray as xr

    cube_root = tmp_path / "cube_dir"
    cube_dir = cube_root / "cube"
    cube_dir.mkdir(parents=True)
    # One tile per default site (_PROFILE_SITES: Tokyo Bay 35.6,139.8 /
    # Japan Alps 36.2,138.2 / Osaka 34.7,135.5), each a distinct tile.
    sites = [(35.55, 139.75), (36.15, 138.15), (34.65, 135.45)]
    for ti, (lat0, lon0) in enumerate(sites):
        _write_tile_field(cube_dir, ti, lat0, lon0, n=6, step=0.1,
                          with_std=True, mean_fill=float(8 + ti))
    _write_lpi_nc(cube_root)
    out = out_dir / "fig8_uncertainty.pdf"
    bpf.fig8_depth_profiles_and_lpi(out, cube_root)
    _assert_pdf_and_caption(out)
    assert out.stat().st_size > 4_000


# --------------------------------------------------------------------------
# Japan basemap: coastline asset + geographic aspect
#
# The basemap used to be ``_JAPAN_RINGS`` -- 100 hand-digitised vertices its
# own comment calls "a per-island convex-ish hull" -- drawn with the default
# aspect. At locator-inset scale that does not read as Japan. These tests pin
# the replacement: the MLIT-derived asset is preferred, the hull survives only
# as a fallback, and lon/lat are drawn at a true mid-latitude ratio.
# --------------------------------------------------------------------------


def _reset_coastline_cache() -> None:
    bpf._COASTLINE_CACHE = None


def test_coastline_asset_is_present_and_richer_than_the_fallback_hull() -> None:
    _reset_coastline_cache()
    try:
        rings = bpf._japan_coastline()
        n_pts = sum(len(r) for r in rings)
        hull_pts = sum(len(r) for r in bpf._JAPAN_RINGS)
        assert n_pts > 10 * hull_pts, (
            f"coastline has only {n_pts} vertices vs {hull_pts} in the "
            "fallback hull -- the MLIT asset was probably not loaded. "
            "Rebuild with `python -m scripts.build_japan_coastline`.")
        lons = [p[0] for r in rings for p in r]
        lats = [p[1] for r in rings for p in r]
        # Japan's national coastline extent, generously bounded.
        assert 122.0 < min(lons) and max(lons) < 155.0
        assert 20.0 < min(lats) and max(lats) < 46.5
    finally:
        _reset_coastline_cache()


def test_coastline_falls_back_to_hull_when_asset_missing(
        tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(bpf, "_COASTLINE_ASSET", tmp_path / "absent.json")
    _reset_coastline_cache()
    try:
        rings = bpf._japan_coastline()
        assert len(rings) == len(bpf._JAPAN_RINGS)
        assert sum(len(r) for r in rings) == sum(
            len(r) for r in bpf._JAPAN_RINGS)
    finally:
        _reset_coastline_cache()


def test_geographic_aspect_stretches_latitude_by_one_over_cos() -> None:
    import math

    # At 35 N a degree of longitude is cos(35) ~ 0.819 of a degree of
    # latitude on the ground, so the display aspect must be ~1.22 -- NOT the
    # 1.0 that aspect="equal" would impose.
    assert bpf._geographic_aspect((34.0, 36.0)) == pytest.approx(
        1.0 / math.cos(math.radians(35.0)), rel=1e-6)
    assert bpf._geographic_aspect((24.0, 46.0)) > 1.0
    # Equator degenerates to 1.0.
    assert bpf._geographic_aspect((-1.0, 1.0)) == pytest.approx(1.0, rel=1e-6)


def test_fig7_locator_is_a_sibling_panel_not_an_overlay(
        out_dir: Path, tmp_path: Path, monkeypatch: Any) -> None:
    """The locator must occupy its own axes column, clear of the data.

    Its previous ``ax.inset_axes([0.66, 0.62, 0.32, 0.36])`` placed an opaque
    white box over the shallow 0-7 m band of the Boso half of the section.
    Capture the real figure on its way to disk and assert the locator's axes
    does not intersect the cross-section's.
    """
    import matplotlib.pyplot as plt

    captured: dict[str, Any] = {}
    real_save = bpf._save_pdf

    def _capture(fig, out_path, caption):  # noqa: ANN001
        fig.canvas.draw()
        captured["axes"] = [(ax.get_title(), ax.get_ylabel(),
                             ax.get_position()) for ax in fig.axes]
        return real_save(fig, out_path, caption)

    monkeypatch.setattr(bpf, "_save_pdf", _capture)

    cube_root = tmp_path / "cube_dir"
    _write_tile_zarrs(cube_root, with_std=True, n_tiles=2)
    out = out_dir / "fig7_cube_slices.pdf"
    bpf.fig7_cube_transect(out, cube_root, lat=35.5, lon0=139.05,
                           lon1=139.95, n_samples=8)
    _assert_pdf_and_caption(out)

    axes = captured["axes"]
    locator = [pos for title, _, pos in axes if title == "locator"]
    section = [pos for _, ylabel, pos in axes if ylabel.startswith("Depth")]
    assert len(locator) == 1, f"expected one locator axes, got {len(locator)}"
    assert len(section) == 1, f"expected one section axes, got {len(section)}"
    lb, sb = locator[0], section[0]
    assert lb.x0 >= sb.x1, (
        f"locator (x0={lb.x0:.3f}) starts before the cross-section ends "
        f"(x1={sb.x1:.3f}) -- it overlays the data")
    plt.close("all")
