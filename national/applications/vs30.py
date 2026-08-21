"""V_s30 (time-averaged shear-wave velocity, upper 30 m) from SPT N.

Paper B' Pillar 5 site-amplification map. Converts a depth-indexed SPT
N-value profile (the foundation-model output) to a single per-location
``V_s30`` and the corresponding NEHRP site class. V_s30 is the
international standard input to building-code site amplification
factors (ASCE 7, Japanese AIJ, Eurocode 8).

Two N-to-Vs empirical relations are supported:

* **Imai & Tonouchi 1982** (default): the Japanese standard, used by
  J-SHIS for the national V_s30 model. ``Vs = 97.0 · N^0.314`` m/s,
  applicable to mixed-lithology Japanese soils.
* **Wair, DeJong, Shantz 2012**: a regional aggregate calibrated on
  California data, included for cross-comparison. ``Vs = 30 · N^0.215``
  (Holocene) or ``Vs = 79 · N^0.434`` (Pleistocene).

NEHRP site class thresholds (ASCE 7-22 §11.4.2 / J-SHIS conversion):
    A: V_s30 > 1500 m/s    rock
    B: 760 < V_s30 ≤ 1500  weathered rock
    C: 360 < V_s30 ≤ 760   very dense soil / soft rock
    D: 180 < V_s30 ≤ 360   stiff soil (typical Tokyo basin floor)
    E: V_s30 ≤ 180         soft soil (typical liquefaction-prone area)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# ============================================================
# N -> Vs empirical relations
# ============================================================


def vs_from_n_imai_tonouchi(n_value: np.ndarray) -> np.ndarray:
    """Imai & Tonouchi 1982 ``Vs = 97.0 · N^0.314`` m/s.

    Japanese-soils standard. Used by J-SHIS / NIED for the national
    V_s30 model. Valid for SPT N ∈ [1, 100]; clamped at the low end
    so a zero N does not produce Vs = 0.
    """
    n = np.asarray(n_value, dtype=np.float64)
    n_safe = np.maximum(n, 1.0)
    return 97.0 * np.power(n_safe, 0.314)


def vs_from_n_wair_2012(n_value: np.ndarray, *, age: str = "holocene") -> np.ndarray:
    """Wair, DeJong, Shantz 2012 aggregate California relation.

    Less applicable to Japanese soils but useful for cross-region
    Vs comparison in the Paper B' generalisation audit.
    """
    n = np.asarray(n_value, dtype=np.float64)
    n_safe = np.maximum(n, 1.0)
    if age == "holocene":
        return 30.0 * np.power(n_safe, 0.215)
    if age == "pleistocene":
        return 79.0 * np.power(n_safe, 0.434)
    raise ValueError(f"Unknown age category: {age!r}; use 'holocene' or 'pleistocene'.")


# ============================================================
# V_s30 averaging
# ============================================================


def vs30_from_profile(
    depth_m: np.ndarray,
    n_value: np.ndarray,
    *,
    relation: Literal["imai_tonouchi", "wair_holocene", "wair_pleistocene"] = "imai_tonouchi",
) -> float:
    """Time-averaged shear-wave velocity over the upper 30 m.

    ``V_s30 = 30 / Σ (h_i / V_s,i)`` where the sum is over the layers
    intersecting the 0-30 m window. Layers deeper than 30 m are
    truncated; layers shallower contribute their full thickness.

    If the profile does not reach 30 m, the deepest layer's Vs is
    extrapolated to fill the remainder (standard ASCE 7 / J-SHIS
    convention).

    Args:
        depth_m: 1-D array of bottom-of-layer depths (m), ascending.
        n_value: SPT N at each layer.
        relation: which empirical N->Vs to use.

    Returns:
        V_s30 in m/s. Raises ValueError on misshapen / non-monotone
        inputs or if depth_m has fewer than two samples.
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    n_value = np.asarray(n_value, dtype=np.float64)
    if depth_m.shape != n_value.shape:
        raise ValueError(f"shape mismatch: {depth_m.shape} vs {n_value.shape}")
    if depth_m.ndim != 1:
        raise ValueError(f"depth_m must be 1-D; got {depth_m.shape}")
    if depth_m.size < 2:
        raise ValueError("need at least two depth samples")
    if np.any(np.diff(depth_m) <= 0):
        raise ValueError("depth_m must be strictly ascending")

    if relation == "imai_tonouchi":
        vs = vs_from_n_imai_tonouchi(n_value)
    elif relation == "wair_holocene":
        vs = vs_from_n_wair_2012(n_value, age="holocene")
    elif relation == "wair_pleistocene":
        vs = vs_from_n_wair_2012(n_value, age="pleistocene")
    else:
        raise ValueError(f"Unknown relation: {relation!r}")

    # Walk the layers, accumulating travel-time over the 0..30 m window.
    travel_time = 0.0
    z_prev = 0.0
    for i, z_curr in enumerate(depth_m):
        if z_prev >= 30.0:
            break
        h = min(z_curr, 30.0) - z_prev
        if h <= 0:
            z_prev = z_curr
            continue
        travel_time += h / vs[i]
        z_prev = z_curr
    # Fill the remainder with the deepest sample's Vs (extrapolation).
    if z_prev < 30.0:
        travel_time += (30.0 - z_prev) / vs[-1]
    return 30.0 / travel_time


# ============================================================
# NEHRP site class
# ============================================================


@dataclass(frozen=True)
class NehrpClass:
    """One NEHRP site-class entry."""

    code: str
    label: str
    vs30_lo: float  # exclusive lower bound
    vs30_hi: float  # inclusive upper bound (np.inf for A)


_NEHRP_CLASSES: tuple[NehrpClass, ...] = (
    NehrpClass("A", "Hard rock", 1500.0, float("inf")),
    NehrpClass("B", "Rock", 760.0, 1500.0),
    NehrpClass("C", "Very dense soil / soft rock", 360.0, 760.0),
    NehrpClass("D", "Stiff soil", 180.0, 360.0),
    NehrpClass("E", "Soft clay / loose sand", 0.0, 180.0),
)


def nehrp_class(vs30_m_s: float) -> NehrpClass:
    """Return the NEHRP site class for a V_s30 value (m/s)."""
    if vs30_m_s <= 0:
        raise ValueError(f"vs30_m_s must be positive; got {vs30_m_s}")
    for cls in _NEHRP_CLASSES:
        if cls.vs30_lo < vs30_m_s <= cls.vs30_hi:
            return cls
    # Should be unreachable (Class A covers up to infinity).
    raise RuntimeError(f"NEHRP class lookup failed for V_s30={vs30_m_s}")


__all__ = [
    "NehrpClass",
    "vs_from_n_imai_tonouchi",
    "vs_from_n_wair_2012",
    "vs30_from_profile",
    "nehrp_class",
]
