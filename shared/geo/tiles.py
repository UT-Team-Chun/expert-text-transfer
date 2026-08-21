"""Japanese standard mesh code helpers."""

from __future__ import annotations

import math

import numpy as np

_PRIMARY_LAT_DEG = 2.0 / 3.0
_PRIMARY_LON_DEG = 1.0
_SECONDARY_LAT_DEG = _PRIMARY_LAT_DEG / 8.0
_SECONDARY_LON_DEG = _PRIMARY_LON_DEG / 8.0
_QUARTER_LAT_DEG = _SECONDARY_LAT_DEG / 10.0
_QUARTER_LON_DEG = _SECONDARY_LON_DEG / 10.0


def primary_mesh_code(lat: float, lon: float) -> str:
    """Return the 4-digit primary mesh code containing ``lat, lon``."""
    lat_code = math.floor(float(lat) / _PRIMARY_LAT_DEG)
    lon_code = math.floor(float(lon)) - 100
    _validate_primary_codes(lat_code, lon_code)
    return f"{lat_code:02d}{lon_code:02d}"


def secondary_mesh_code(lat: float, lon: float) -> str:
    """Return the 6-digit secondary mesh code containing ``lat, lon``."""
    primary = primary_mesh_code(lat, lon)
    lat_min, lon_min, _, _ = mesh_bounds(primary)
    row = _bounded_digit((float(lat) - lat_min) / _SECONDARY_LAT_DEG, 8)
    col = _bounded_digit((float(lon) - lon_min) / _SECONDARY_LON_DEG, 8)
    return f"{primary}{row}{col}"


def quarter_mesh_code(lat: float, lon: float) -> str:
    """Return the 8-digit 1/4 mesh code containing ``lat, lon``."""
    secondary = secondary_mesh_code(lat, lon)
    lat_min, lon_min, _, _ = mesh_bounds(secondary)
    row = _bounded_digit((float(lat) - lat_min) / _QUARTER_LAT_DEG, 10)
    col = _bounded_digit((float(lon) - lon_min) / _QUARTER_LON_DEG, 10)
    return f"{secondary}{row}{col}"


def mesh_bounds(mesh_code: str) -> tuple[float, float, float, float]:
    """Return ``(lat_min, lon_min, lat_max, lon_max)`` for a mesh code."""
    code = str(mesh_code)
    if len(code) not in {4, 6, 8} or not code.isdigit():
        raise ValueError("mesh_code must be a 4-, 6-, or 8-digit string.")

    lat_code = int(code[:2])
    lon_code = int(code[2:4])
    _validate_primary_codes(lat_code, lon_code)

    lat_min = lat_code * _PRIMARY_LAT_DEG
    lon_min = lon_code + 100.0
    lat_span = _PRIMARY_LAT_DEG
    lon_span = _PRIMARY_LON_DEG

    if len(code) >= 6:
        row = int(code[4])
        col = int(code[5])
        if row > 7 or col > 7:
            raise ValueError("Secondary mesh digits must be in 0..7.")
        lat_min += row * _SECONDARY_LAT_DEG
        lon_min += col * _SECONDARY_LON_DEG
        lat_span = _SECONDARY_LAT_DEG
        lon_span = _SECONDARY_LON_DEG

    if len(code) == 8:
        row = int(code[6])
        col = int(code[7])
        lat_min += row * _QUARTER_LAT_DEG
        lon_min += col * _QUARTER_LON_DEG
        lat_span = _QUARTER_LAT_DEG
        lon_span = _QUARTER_LON_DEG

    return lat_min, lon_min, lat_min + lat_span, lon_min + lon_span


def _bounded_digit(value: float, subdivisions: int) -> int:
    return min(subdivisions - 1, max(0, math.floor(value)))


def _validate_primary_codes(lat_code: int, lon_code: int) -> None:
    if not 0 <= lat_code <= 99 or not 0 <= lon_code <= 99:
        raise ValueError(
            f"Coordinates are outside 2-digit Japanese mesh code range: "
            f"lat_code={lat_code}, lon_code={lon_code}."
        )


def adjacent_secondary_mesh_codes(code: str, ring: int = 1) -> set[str]:
    """Return the secondary mesh codes within ``ring`` steps of ``code``.

    A 1-mesh ring (the default) returns up to 8 neighbours (the 3x3
    moore neighbourhood minus the centre). Crossings into adjacent
    primary mesh cells are handled by re-encoding the
    (primary_lat, primary_lon, sub_row, sub_col) tuple.

    Args:
        code: a 6-digit secondary mesh code as returned by
            :func:`secondary_mesh_code`.
        ring: neighbourhood radius in mesh cells. ``ring=1`` returns
            the immediate 8 neighbours; ``ring=2`` returns the 24
            neighbours in the 5x5 minus centre, etc.

    Returns:
        ``set[str]`` of mesh codes, never including ``code`` itself.
        Codes that would fall outside the valid Japanese mesh range
        (lat or lon code <0 or >99) are silently dropped.
    """
    if len(str(code)) != 6 or not str(code).isdigit():
        raise ValueError(
            f"Expected 6-digit secondary mesh code, got {code!r}"
        )
    lat_code = int(code[:2])
    lon_code = int(code[2:4])
    row = int(code[4])
    col = int(code[5])

    out: set[str] = set()
    for d_row in range(-ring, ring + 1):
        for d_col in range(-ring, ring + 1):
            if d_row == 0 and d_col == 0:
                continue
            nrow = row + d_row
            ncol = col + d_col
            nlat = lat_code
            nlon = lon_code
            # Handle row underflow / overflow → step into adjacent
            # primary lat cell
            while nrow < 0:
                nrow += 8
                nlat -= 1
            while nrow > 7:
                nrow -= 8
                nlat += 1
            while ncol < 0:
                ncol += 8
                nlon -= 1
            while ncol > 7:
                ncol -= 8
                nlon += 1
            if not (0 <= nlat <= 99 and 0 <= nlon <= 99):
                continue
            out.add(f"{nlat:02d}{nlon:02d}{nrow}{ncol}")
    return out


def secondary_mesh_key_array(lat, lon) -> np.ndarray:
    """Vectorised integer secondary-mesh cell keys for fast grouping.

    Returns one ``int64`` per point such that two points share a key **iff**
    they fall in the same secondary mesh cell as :func:`secondary_mesh_code`
    (the key is a bijection of the ``(lat_code, lon_code, row, col)`` tuple).

    Unlike :func:`secondary_mesh_code` this does not format the 6-digit string
    or validate the Japan bounds, so it is safe and fast on million-row arrays
    (used by the leave-region-out mesh-disjoint calibration split). Points
    outside Japan simply get their own keys rather than raising.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat_code = np.floor(lat / _PRIMARY_LAT_DEG).astype(np.int64)
    lon_code = np.floor(lon).astype(np.int64) - 100
    lat_min = lat_code * _PRIMARY_LAT_DEG
    lon_min = lon_code.astype(np.float64) + 100.0
    row = np.clip(np.floor((lat - lat_min) / _SECONDARY_LAT_DEG), 0, 7).astype(np.int64)
    col = np.clip(np.floor((lon - lon_min) / _SECONDARY_LON_DEG), 0, 7).astype(np.int64)
    # lon_code in 0..99 keeps (lat_code*100 + lon_code) unique per primary cell;
    # row,col in 0..7 keep the final key unique per secondary cell.
    return ((lat_code * 100 + lon_code) * 10 + row) * 10 + col


__all__ = [
    "primary_mesh_code",
    "secondary_mesh_code",
    "quarter_mesh_code",
    "mesh_bounds",
    "adjacent_secondary_mesh_codes",
    "secondary_mesh_key_array",
]
