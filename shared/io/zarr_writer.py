"""Minimal Zarr cube writer for xarray prediction outputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import xarray as xr


def write_zarr_cube(
    da: "xr.DataArray",
    path: Path,
    *,
    chunks: dict[str, int],
    compress: str = "zstd",
    compress_level: int = 5,
) -> Path:
    """Write an ``xarray.DataArray`` as a minimal Zarr v3 cube."""
    import zarr

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data_name = da.name or "values"
    chunk_shape = tuple(_chunk_for_dim(da, dim, chunks) for dim in da.dims)
    compressor = _make_compressor(compress, compress_level)

    group = zarr.open_group(path, mode="w", zarr_format=3)
    values = np.asarray(da.data)
    array = _create_array(
        group,
        data_name,
        values,
        chunk_shape,
        compressor,
        tuple(str(dim) for dim in da.dims),
    )
    array.attrs.update(
        {
            "_ARRAY_DIMENSIONS": list(da.dims),
            **{str(k): _json_attr(v) for k, v in da.attrs.items()},
        }
    )

    for dim in da.dims:
        if dim not in da.coords:
            continue
        coord_values = np.asarray(da.coords[dim].data)
        coord = _create_array(
            group,
            str(dim),
            coord_values,
            coord_values.shape,
            None,
            (str(dim),),
        )
        coord.attrs.update({"_ARRAY_DIMENSIONS": [str(dim)]})

    group.attrs.update(
        {
            "Conventions": "CF-1.8",
            "xarray_dataarray_name": str(data_name),
        }
    )
    return path


def _chunk_for_dim(da: "xr.DataArray", dim: str, chunks: dict[str, int]) -> int:
    size = int(da.sizes[dim])
    chunk = int(chunks.get(dim, size))
    if chunk <= 0:
        raise ValueError(f"Chunk size for {dim!r} must be positive, got {chunk}.")
    return min(chunk, size)


def _make_compressor(compress: str, compress_level: int) -> Any:
    """Return a Zarr-version-aware compressor.

    On Zarr v3 we return a ``zarr.codecs`` codec instance; on v2 we fall
    back to the numcodecs equivalent. ``None`` disables compression.
    """
    if compress_level < 0:
        raise ValueError(f"compress_level must be non-negative, got {compress_level}.")

    name = compress.lower()
    if name in {"none", "uncompressed"}:
        return None

    try:
        import zarr.codecs as zc  # only present in zarr >= 3
    except ImportError:
        zc = None  # type: ignore[assignment]

    if zc is not None:
        if name in {"zstd", "blosc-zstd"}:
            return zc.BloscCodec(cname="zstd", clevel=compress_level, shuffle="bitshuffle")
        if name in {"lz4", "blosc", "blosc-lz4"}:
            return zc.BloscCodec(cname="lz4", clevel=compress_level, shuffle="bitshuffle")
        if name in {"gzip", "gz"}:
            return zc.GzipCodec(level=compress_level)

    from numcodecs import Blosc, GZip

    if name in {"zstd", "blosc-zstd"}:
        return Blosc(cname="zstd", clevel=compress_level, shuffle=Blosc.BITSHUFFLE)
    if name in {"lz4", "blosc", "blosc-lz4"}:
        return Blosc(cname="lz4", clevel=compress_level, shuffle=Blosc.BITSHUFFLE)
    if name in {"gzip", "gz"}:
        return GZip(level=compress_level)
    raise ValueError(f"Unsupported Zarr compressor: {compress!r}.")


def _create_array(
    group: Any,
    name: str,
    data: np.ndarray,
    chunks: tuple[int, ...],
    compressor: Any,
    dimension_names: tuple[str, ...],
) -> Any:
    """Create a Zarr array, handling both the v3 and legacy v2 group APIs.

    Zarr v3 (the project's default since 2025) requires ``create_array``
    to be called *without* ``data`` and the data written separately; older
    versions accept ``data=`` directly. We probe at call time.
    """
    # Preferred Zarr v3 path: create_array(name, shape, dtype, ...) then array[:] = data.
    try:
        kwargs: dict[str, Any] = {
            "shape": tuple(data.shape),
            "chunks": chunks,
            "dtype": data.dtype,
            "overwrite": True,
        }
        if compressor is not None:
            kwargs["compressors"] = [compressor]
        if dimension_names:
            kwargs["dimension_names"] = list(dimension_names)
        array = group.create_array(name, **kwargs)
        array[:] = data
        return array
    except TypeError:
        pass  # fall back to legacy API below

    # Legacy v2 path.
    kwargs = {
        "data": data,
        "chunks": chunks,
        "overwrite": True,
        "shape": tuple(data.shape),
        "dtype": data.dtype,
    }
    if compressor is not None:
        kwargs["compressor"] = compressor
    return group.create_dataset(name, **kwargs)


def _json_attr(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_attr(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_attr(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = ["write_zarr_cube"]
