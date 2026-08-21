"""Liquefaction Potential Index (LPI) per Iwasaki et al. 1984.

Paper B' Pillar 5 engineering application. Takes a depth-indexed SPT
N-value profile (the foundation-model output) plus groundwater depth
+ peak ground acceleration + fines content, returns a single scalar
LPI value summarising the boring's liquefaction susceptibility under
the assumed earthquake load. Above the water table the soil is treated
as non-liquefiable (water content insufficient); below the water table
each depth contributes ``F · w(z)`` to the LPI integral, where
``F = 1 - FL`` clipped to 0 when the factor-of-safety ``FL ≥ 1``.

References:
    Iwasaki T., Tatsuoka F., Tokida K., Yasuda S. (1984)
        "A practical method for assessing soil liquefaction potential
        based on case studies at various sites in Japan."
        Proc. 2nd Int. Conf. on Microzonation, San Francisco.
    Youd T.L., Idriss I.M. et al. (2001)
        "Liquefaction Resistance of Soils: Summary Report from the 1996
        and 1998 NCEER/NSF Workshops."
        J. Geotech. Geoenviron. Eng. 127(10): 817-833.
    Seed H.B., Idriss I.M. (1971)
        "Simplified Procedure for Evaluating Soil Liquefaction Potential."
        J. Soil Mech. Found. Div. ASCE 97(SM9): 1249-1273.

Convention: depths are positive downwards in metres, stresses in kPa,
PGA in g. Single-boring API is the lowest layer; a separate
``scripts/build_liquefaction_lpi_map.py`` driver vectorises this over
the national 3D N cube.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================
# Constants
# ============================================================


# Atmospheric pressure (kPa) used to non-dimensionalise the
# overburden-stress correction.
_P_ATM_KPA = 101.325

# Standard saturated and dry unit weights (kN/m^3) for Japanese soils
# in the upper 20 m. Iwasaki et al. 1984 use these as default values for
# their regional regression; project-specific γ_sat from CPT or lab
# tests would be plugged through the API.
_GAMMA_SAT_DEFAULT_KN_M3 = 19.0
_GAMMA_DRY_DEFAULT_KN_M3 = 16.0

# LPI integration ceiling. Iwasaki's original method integrates from
# the ground surface to 20 m; soils below 20 m are too deep to cause
# surface manifestation regardless of liquefaction.
_LPI_DEPTH_LIMIT_M = 20.0

# Maximum overburden-stress correction CN (Liao & Whitman 1986 cap
# adopted by Youd et al. 2001 to prevent unrealistically high
# corrections in very shallow rows).
_CN_MAX = 1.7


# ============================================================
# Stresses
# ============================================================


def vertical_stresses(
    depth_m: np.ndarray,
    water_table_m: float,
    gamma_sat_kn_m3: float = _GAMMA_SAT_DEFAULT_KN_M3,
    gamma_dry_kn_m3: float = _GAMMA_DRY_DEFAULT_KN_M3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute total + effective vertical stress at each depth.

    Above the water table the soil is unsaturated (γ_dry); below, the
    soil is saturated (γ_sat) and pore pressure ``u = γ_w · (z -
    water_table)`` (γ_w = 9.81 kN/m^3) is subtracted from total stress
    to get effective stress. Water table at 0 m means saturated from
    the surface; ``water_table_m`` ≥ ``max(depth_m)`` makes the entire
    profile dry-side.

    Args:
        depth_m: 1-D array of depths in metres.
        water_table_m: groundwater depth in metres (positive downward
            from ground surface). NaN treated as "no water encountered"
            -> all soil dry.
        gamma_sat_kn_m3: saturated unit weight (kN/m^3), default 19.
        gamma_dry_kn_m3: dry unit weight (kN/m^3), default 16.

    Returns:
        Tuple ``(sigma_v_total, sigma_v_eff)`` arrays in kPa, both
        shaped like ``depth_m``.
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    if depth_m.ndim != 1:
        raise ValueError(f"depth_m must be 1-D; got {depth_m.shape}")
    if np.any(depth_m < 0):
        raise ValueError("depth_m must be non-negative")

    # NaN water table -> treat as below the deepest sample (entire profile dry).
    if water_table_m != water_table_m:  # NaN check
        water_table_m = float("inf")
    water_table_m = float(water_table_m)

    sigma_total = np.empty_like(depth_m)
    sigma_eff = np.empty_like(depth_m)
    gamma_w = 9.81  # kN/m^3

    for i, z in enumerate(depth_m):
        if z <= water_table_m:
            # Entirely dry above (or at) the water table.
            sigma_total[i] = gamma_dry_kn_m3 * z
            sigma_eff[i] = sigma_total[i]
        else:
            # Mixed: dry layer to water table, then saturated below.
            dry_part = gamma_dry_kn_m3 * water_table_m
            sat_part = gamma_sat_kn_m3 * (z - water_table_m)
            sigma_total[i] = dry_part + sat_part
            u = gamma_w * (z - water_table_m)
            sigma_eff[i] = sigma_total[i] - u
    return sigma_total, sigma_eff


# ============================================================
# Cyclic Stress Ratio (CSR)
# ============================================================


def stress_reduction_factor(depth_m: np.ndarray) -> np.ndarray:
    """Liao & Whitman 1986 / Youd 2001 simplified ``r_d`` curve.

    Two-piece linear approximation valid through 23 m -- the LPI
    integration only goes to 20 m so the deeper branch is rarely
    exercised but kept for API completeness.
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    rd = np.where(
        depth_m < 9.15,
        1.0 - 0.00765 * depth_m,
        1.174 - 0.0267 * depth_m,
    )
    # Clip to a small positive floor so a numerical fluctuation past 23 m
    # cannot drive CSR negative.
    return np.clip(rd, 0.05, 1.0)


def cyclic_stress_ratio(
    depth_m: np.ndarray,
    water_table_m: float,
    pga_g: float,
    *,
    gamma_sat_kn_m3: float = _GAMMA_SAT_DEFAULT_KN_M3,
    gamma_dry_kn_m3: float = _GAMMA_DRY_DEFAULT_KN_M3,
) -> np.ndarray:
    """Seed-Idriss 1971 simplified CSR at each depth.

    ``CSR = 0.65 · (a_max / g) · (σ_v / σ_v') · r_d``

    Returns CSR (dimensionless) shaped like ``depth_m``. Above the
    water table σ_v = σ_v' so the ratio becomes 1; CSR there is
    informational only since :func:`iwasaki_lpi` masks the dry zone
    out of the integral.
    """
    sigma_total, sigma_eff = vertical_stresses(
        depth_m,
        water_table_m,
        gamma_sat_kn_m3=gamma_sat_kn_m3,
        gamma_dry_kn_m3=gamma_dry_kn_m3,
    )
    rd = stress_reduction_factor(depth_m)
    # Floor σ_v' to a small positive value so a numerical 0 at depth=0
    # cannot divide by zero (the dry zone gets masked downstream anyway).
    sigma_eff_safe = np.maximum(sigma_eff, 1e-3)
    return 0.65 * float(pga_g) * (sigma_total / sigma_eff_safe) * rd


# ============================================================
# Corrected blow count N1_60
# ============================================================


def overburden_correction_cn(sigma_eff_kpa: np.ndarray) -> np.ndarray:
    """Liao & Whitman 1986 ``C_N = sqrt(P_atm / σ_v')`` capped at 1.7."""
    sigma_eff = np.asarray(sigma_eff_kpa, dtype=np.float64)
    sigma_eff_safe = np.maximum(sigma_eff, 1.0)  # 1 kPa floor
    cn = np.sqrt(_P_ATM_KPA / sigma_eff_safe)
    return np.minimum(cn, _CN_MAX)


def n1_60(
    n_value: np.ndarray,
    sigma_eff_kpa: np.ndarray,
    *,
    energy_ratio: float = 1.0,
) -> np.ndarray:
    """Overburden + energy-normalised SPT blow count.

    Japanese SPT typically already uses a 60% rod-energy hammer
    (``energy_ratio=1.0`` by Iwasaki et al. convention). US SPT with a
    safety hammer would pass ``energy_ratio = 1.25`` etc.
    """
    n = np.asarray(n_value, dtype=np.float64)
    cn = overburden_correction_cn(sigma_eff_kpa)
    return n * cn * float(energy_ratio)


# ============================================================
# Cyclic Resistance Ratio (CRR)
# ============================================================


def cyclic_resistance_ratio(
    n1_60_value: np.ndarray,
    fines_content_pct: float = 5.0,
    magnitude: float = 7.5,
) -> np.ndarray:
    """Youd et al. 2001 CRR_7.5 + fines-content + magnitude scaling.

    Steps (operating on each N1_60 element):

    1. Apply the Idriss & Boulanger 2008 fines-content correction to
       get the "clean-sand equivalent" N1_60_cs. Two-piece linear
       fit valid for ``Fc ∈ [5%, 35%]``; outside that the curve is
       flat (extrapolation is unreliable in the original literature).
    2. Compute CRR_7.5 via the standard 4-term Youd-Idriss formula
       valid for N1_60_cs < 30 (saturated cohesionless soils). Above
       30 the soil is too dense to liquefy; CRR pinned to 2.0
       (Iwasaki convention: a value high enough that FL = R/L >> 1
       regardless of typical CSR).
    3. Apply the magnitude scaling factor (MSF) to scale CRR from the
       reference M_w=7.5 to the user-specified earthquake magnitude.

    Args:
        n1_60_value: 1-D array of corrected blow counts.
        fines_content_pct: percent passing No. 200 sieve. Default 5
           (clean sand). The function is constant for Fc < 5 and
           saturates for Fc > 35.
        magnitude: moment magnitude ``M_w``. Default 7.5 (the standard
            reference; for Japan's Nankai trough scenario use ~8.5).

    Returns:
        CRR (dimensionless) shaped like ``n1_60_value``.
    """
    n = np.asarray(n1_60_value, dtype=np.float64)

    # ---- Fines-content correction (Idriss & Boulanger 2008) -------
    fc = float(fines_content_pct)
    fc = max(5.0, min(35.0, fc))
    # Delta-N: additive shift on the clean-sand equivalent.
    if fc <= 5.0:
        dn = 0.0
    elif fc < 15.0:
        dn = 1.5 * (fc - 5.0) / 10.0
    else:
        dn = 1.5 + 4.0 * (fc - 15.0) / 20.0
    n_cs = n + dn

    # ---- CRR_7.5 from Youd-Idriss 2001 ---------------------------
    # Mask the elements where N1_60_cs is too dense to liquefy.
    safe = n_cs < 30.0
    safe_n = np.where(safe, n_cs, 29.99)  # avoid /0 in dense rows
    crr_75 = (
        1.0 / (34.0 - safe_n)
        + safe_n / 135.0
        + 50.0 / (10.0 * safe_n + 45.0) ** 2
        - 1.0 / 200.0
    )

    # ---- Magnitude scaling factor (MSF, Youd 2001) ---------------
    # MSF = 10^2.24 / M_w^2.56 for M_w ∈ [5.5, 8.5].
    msf = (10.0 ** 2.24) / (max(5.5, min(8.5, magnitude)) ** 2.56)
    crr = crr_75 * msf
    # Dense soils -> pin to a "non-liquefiable" CRR after MSF so the
    # downstream FL = CRR / CSR is comfortably >> 1 regardless of
    # earthquake magnitude. 2.0 is the Iwasaki convention.
    return np.where(safe, crr, 2.0)


# ============================================================
# Iwasaki LPI integral
# ============================================================


@dataclass(frozen=True)
class LpiResult:
    """LPI plus per-depth diagnostics so a caller can audit the integral."""

    lpi: float
    depths_m: np.ndarray
    fl: np.ndarray
    csr: np.ndarray
    crr: np.ndarray
    n1_60: np.ndarray
    saturated_mask: np.ndarray  # True where the soil is below the water table


def iwasaki_lpi(
    depth_m: np.ndarray,
    n_value: np.ndarray,
    water_table_m: float,
    pga_g: float,
    *,
    fines_content_pct: float = 5.0,
    magnitude: float = 7.5,
    gamma_sat_kn_m3: float = _GAMMA_SAT_DEFAULT_KN_M3,
    gamma_dry_kn_m3: float = _GAMMA_DRY_DEFAULT_KN_M3,
    energy_ratio: float = 1.0,
) -> LpiResult:
    """Compute the Liquefaction Potential Index for one SPT profile.

    Trapezoidal integration of ``F · w(z)`` from ``z = 0`` to
    ``z = min(20, max(depth_m))``:

    * ``F = max(0, 1 - FL)``  (zero where soil is non-liquefiable),
    * ``FL = CRR / CSR``  (factor of safety against liquefaction),
    * ``w(z) = max(0, 10 - 0.5 · z)``  (linear weighting, peaks at z=0,
      reaches 0 at z = 20 m).

    Depths above the water table contribute ``F = 0`` to the integral
    regardless of N, since unsaturated soils don't liquefy.

    Iwasaki's qualitative scale:
        LPI = 0       -- liquefaction very unlikely
        0 < LPI ≤ 5   -- low risk
        5 < LPI ≤ 15  -- high risk
        LPI > 15      -- very high risk; surface manifestation almost certain

    Args:
        depth_m: 1-D array of depths (m), sorted ascending. Anything beyond
            20 m is ignored.
        n_value: SPT N-value at each depth (raw blow count, NOT yet
            overburden / energy corrected; the function does the
            normalisation internally).
        water_table_m: groundwater depth (m). NaN -> profile treated as dry
            (LPI = 0).
        pga_g: peak ground acceleration in g (typical scenario values:
            0.15 g moderate / 0.30 g severe / 0.50 g extreme; Japanese
            building code design level is around 0.20-0.30 g).
        fines_content_pct: percent passing No. 200 sieve. Default 5
            (clean sand worst case). Pillar 3 BERT embeddings could in
            principle predict Fc per-layer but that wiring is future
            work; for now this is a per-profile constant.
        magnitude: scenario M_w. Default 7.5 (Tokimatsu-Yoshimi reference).
        gamma_sat_kn_m3 / gamma_dry_kn_m3: unit weights.
        energy_ratio: SPT rod-energy correction. 1.0 for Japanese hammer.

    Returns:
        :class:`LpiResult` carrying the scalar LPI and per-depth
        diagnostics (FL, CSR, CRR, N1_60, saturated mask).

    Raises:
        ValueError: if input shapes are inconsistent, depths are not
            monotone ascending, or PGA is non-positive.
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    n_value = np.asarray(n_value, dtype=np.float64)
    if depth_m.shape != n_value.shape:
        raise ValueError(
            f"depth_m {depth_m.shape} and n_value {n_value.shape} must align"
        )
    if depth_m.ndim != 1:
        raise ValueError(f"depth_m must be 1-D; got {depth_m.shape}")
    if depth_m.size < 2:
        raise ValueError("Need at least two depth samples for trapezoidal integration")
    if np.any(np.diff(depth_m) <= 0):
        raise ValueError("depth_m must be strictly ascending")
    if pga_g <= 0:
        raise ValueError(f"pga_g must be positive; got {pga_g}")

    sigma_total, sigma_eff = vertical_stresses(
        depth_m,
        water_table_m,
        gamma_sat_kn_m3=gamma_sat_kn_m3,
        gamma_dry_kn_m3=gamma_dry_kn_m3,
    )
    csr = cyclic_stress_ratio(
        depth_m,
        water_table_m,
        pga_g,
        gamma_sat_kn_m3=gamma_sat_kn_m3,
        gamma_dry_kn_m3=gamma_dry_kn_m3,
    )
    n1_60_v = n1_60(n_value, sigma_eff, energy_ratio=energy_ratio)
    crr = cyclic_resistance_ratio(
        n1_60_v, fines_content_pct=fines_content_pct, magnitude=magnitude
    )

    # Factor of safety; safe-clipped CSR floor of 1e-3 avoids /0 in
    # the dry-zone rows (which the saturated mask drops anyway).
    csr_safe = np.maximum(csr, 1e-3)
    fl = crr / csr_safe

    # Saturation mask: rows with z > water_table_m. NaN water table
    # produces an all-False mask -> LPI = 0.
    if water_table_m != water_table_m:  # NaN
        saturated = np.zeros_like(depth_m, dtype=bool)
    else:
        saturated = depth_m > float(water_table_m)

    # F = max(0, 1 - FL) only in the saturated zone, else 0.
    f = np.where(saturated, np.maximum(0.0, 1.0 - fl), 0.0)
    # Weight function w(z) = 10 - 0.5 z, clipped at 0.
    w = np.maximum(0.0, 10.0 - 0.5 * depth_m)

    integrand = f * w

    # Trapezoidal integration, clipped to the 0..20 m window. Manual
    # trapezoidal sum keeps us compatible with NumPy <2.0 (which only
    # has the deprecated `np.trapz`) and NumPy >=2.0 (which renames
    # to `np.trapezoid` and removes `np.trapz`).
    mask_z = depth_m <= _LPI_DEPTH_LIMIT_M
    if mask_z.sum() < 2:
        lpi = 0.0
    else:
        z_int = depth_m[mask_z]
        i_int = integrand[mask_z]
        lpi = float(np.sum(0.5 * (i_int[:-1] + i_int[1:]) * np.diff(z_int)))

    return LpiResult(
        lpi=lpi,
        depths_m=depth_m,
        fl=fl,
        csr=csr,
        crr=crr,
        n1_60=n1_60_v,
        saturated_mask=saturated,
    )


def lpi_risk_class(lpi: float) -> str:
    """Iwasaki's qualitative LPI risk class.

    Returns one of {"very_low", "low", "high", "very_high"}.
    """
    if lpi <= 0.0:
        return "very_low"
    if lpi <= 5.0:
        return "low"
    if lpi <= 15.0:
        return "high"
    return "very_high"


__all__ = [
    "LpiResult",
    "vertical_stresses",
    "stress_reduction_factor",
    "cyclic_stress_ratio",
    "overburden_correction_cn",
    "n1_60",
    "cyclic_resistance_ratio",
    "iwasaki_lpi",
    "lpi_risk_class",
]
