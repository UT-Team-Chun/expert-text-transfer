"""Foundation bearing-stratum analysis from SPT N profiles.

Paper B' Pillar 5 engineering application focused on foundation
design. Two complementary primitives:

* :func:`depth_to_bearing_stratum`: the depth at which N first reaches
  a design threshold (default ``N_design = 30``, the Japanese AIJ
  practice value for permanent residential / commercial spread
  footings).
* :func:`meyerhof_allowable_bearing_kpa`: Meyerhof 1956 allowable
  bearing pressure for cohesionless soil at the design stratum,
  parameterised by N, footing width B, and embedment depth D. The
  bearing pressure controls the maximum imposed load before
  excessive settlement (>25 mm); the result is the standard input to
  spread-footing design.

Both work on the (depth, N) profile output by the foundation model;
the cube-level driver wraps them into a national map by predicting
the profile at each (lat, lon) and applying the helper per location.

References:
    Meyerhof G.G. (1956)
        "Penetration tests and bearing capacity of cohesionless soils."
        J. Soil Mech. Found. Div. ASCE 82(SM1): 1-19.
    Bowles J.E. (1996)
        *Foundation Analysis and Design*, 5th ed. McGraw-Hill.
    AIJ (2001)
        *Recommendations for Design of Building Foundations*.
"""

from __future__ import annotations

import numpy as np


def depth_to_bearing_stratum(
    depth_m: np.ndarray,
    n_value: np.ndarray,
    *,
    n_design: float = 30.0,
) -> float:
    """First depth at which ``N >= n_design``.

    Implements the standard "depth to bearing stratum" computation used
    in Japanese residential / commercial foundation design. The
    classic AIJ practice value is ``N_design = 30`` for permanent
    spread footings. Returns ``float('inf')`` if no layer reaches the
    threshold within the profile (treated downstream as "deep
    foundation required").

    Args:
        depth_m: 1-D array of layer depths (m), ascending.
        n_value: SPT N at each layer.
        n_design: design threshold. Default 30 (AIJ).

    Returns:
        Depth in metres of the first crossing, or ``inf``.
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    n_value = np.asarray(n_value, dtype=np.float64)
    if depth_m.shape != n_value.shape:
        raise ValueError(f"shape mismatch: {depth_m.shape} vs {n_value.shape}")
    if depth_m.ndim != 1 or depth_m.size == 0:
        raise ValueError(f"depth_m must be 1-D non-empty; got {depth_m.shape}")
    above = n_value >= float(n_design)
    if not above.any():
        return float("inf")
    return float(depth_m[np.argmax(above)])


def meyerhof_allowable_bearing_kpa(
    n_value: float,
    footing_width_m: float,
    *,
    embedment_depth_m: float = 1.0,
    settlement_limit_mm: float = 25.0,
    water_correction: bool = False,
) -> float:
    """Meyerhof 1956 allowable bearing pressure on cohesionless soil.

    Formula (Bowles 1996 Eqn 4-3, SI units, settlement-limited regime):

    .. math::
        q_a = 11.98 \\cdot N \\cdot K_d \\cdot (S / 25)
        \\quad \\text{for } B \\le 1.22\\text{ m}

        q_a = 7.99 \\cdot N \\cdot K_d \\cdot ((B + 0.305) / B)^2 \\cdot (S / 25)
        \\quad \\text{for } B > 1.22\\text{ m}

    where :math:`K_d = 1 + 0.33 D / B \\le 1.33` is the embedment
    factor and :math:`S` is the allowable settlement in mm.

    Water table within ``B`` of the footing base halves the allowable
    bearing per Meyerhof's submerged-soil correction (only when
    ``water_correction=True``; default off because the call site is
    usually expected to handle saturation explicitly via a separate
    correction).

    Args:
        n_value: SPT N at the bearing stratum (post-overburden if the
            caller has already corrected; raw N is also acceptable
            since Meyerhof's regression is empirical).
        footing_width_m: footing breadth ``B``. Typical residential
            spread footing ~1 m, commercial mat foundation ~5-10 m.
        embedment_depth_m: depth to the base of the footing ``D``.
            Standard residential is ~1 m (frost line + finish floor).
        settlement_limit_mm: allowable settlement ``S``. Code default
            25 mm (AIJ); bridge / heavy-load designs use 50 mm.
        water_correction: if True, halve the result to account for
            water-table effects (Bowles 1996 §4-7).

    Returns:
        Allowable bearing pressure ``q_a`` in kPa.
    """
    if n_value <= 0:
        raise ValueError(f"n_value must be positive; got {n_value}")
    if footing_width_m <= 0:
        raise ValueError(f"footing_width_m must be positive; got {footing_width_m}")
    if embedment_depth_m < 0:
        raise ValueError(f"embedment_depth_m must be non-negative; got {embedment_depth_m}")
    if settlement_limit_mm <= 0:
        raise ValueError(
            f"settlement_limit_mm must be positive; got {settlement_limit_mm}"
        )

    kd = min(1.33, 1.0 + 0.33 * embedment_depth_m / footing_width_m)
    settlement_factor = settlement_limit_mm / 25.0
    if footing_width_m <= 1.22:
        qa = 11.98 * n_value * kd * settlement_factor
    else:
        ratio = (footing_width_m + 0.305) / footing_width_m
        qa = 7.99 * n_value * kd * (ratio ** 2) * settlement_factor

    if water_correction:
        qa *= 0.5
    return float(qa)


def bearing_capacity_class(qa_kpa: float) -> str:
    """Qualitative bearing capacity bucket for headline maps.

    Buckets (Japanese / international shallow-foundation practice):
        very_poor:  qa < 100 kPa    -- deep foundation required
        poor:       100 ≤ qa < 200  -- light residential only
        good:       200 ≤ qa < 400  -- typical commercial
        excellent:  qa ≥ 400        -- heavy industrial
    """
    if qa_kpa < 100.0:
        return "very_poor"
    if qa_kpa < 200.0:
        return "poor"
    if qa_kpa < 400.0:
        return "good"
    return "excellent"


__all__ = [
    "depth_to_bearing_stratum",
    "meyerhof_allowable_bearing_kpa",
    "bearing_capacity_class",
]
