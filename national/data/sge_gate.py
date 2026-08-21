"""Support-Gated Encodings (SGE): fixed geometric gates for the Fourier
coordinate channels — pre-registered in
``docs/research/2026-07-12_sge_preregistration.md``.

The taxonomy result (harm = in-support informativeness x out-of-support
behaviour) says a component whose out-of-support behaviour is CONSTANT is
fail-safe. SGE converts the fail-arbitrary Fourier coordinate block into
that class by wrapping it in a gate that drives it to zero (== the
zero_fourier representation) outside training support:

    phi_SGE(x) = g(x) * phi_Fourier(x_std),
    g(x) = sigmoid((t - DI(x)) / s)

with — per the pre-registration, all constants fixed, nothing trained:

* ``DI(x)`` = L2 distance from x to its nearest neighbour in a FIXED
  10,000-row training-coordinate subsample (numpy seed 4242), divided by
  the mean pairwise training distance (``dbar``, exact on a deterministic
  <=5,000-row subsample of the reference set) — the Meyer & Pebesma (2021)
  dissimilarity index as instrumented by ``scripts.nmi_aoa_audit``;
* ``t`` = Q3 + 1.5*IQR of the TRAINING rows' own cross-region DI (nearest
  neighbour excluding same-LRO-region rows, mirroring the LRO CV
  structure; documented LOO-NN fallback when fewer than
  ``SGE_MIN_CROSS_GROUP_DI`` rows carry a cross-region DI — degenerate
  single-region data, e.g. non-JP archives whose folds are label columns);
* ``s`` = IQR / 2.

Gate SPACE (the round-2 AOA-reversal design principle): the gate must live
in a fail-safe space, so DI is computed in RAW train-standardized lat/lon
for modes ``raw`` / ``test_only``. Mode ``fourier`` (arm D, the mechanism
placebo — predicted to FAIL) computes DI in the 16-band Fourier feature
space of the standardized coords instead, where near-aliasing blinds the
support estimate (the AOA screen went MORE optimistic 8/8 in that space).

Helper provenance: the DI/threshold primitives mirror
``scripts.nmi_aoa_audit`` and are REUSED from it when that module is
importable (local dev / CI). On the frozen 2026-07-07 cluster image the
``scripts.nmi_aoa_audit`` module does not exist (this file is injected via
the nmi_sge preexec), so bit-equal local mirrors (``*_local``) are the
fallback; ``tests/national/test_sge_gate.py`` pins the parity so the two
paths can never diverge silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

LOG = logging.getLogger("national.data.sge_gate")

# Pre-registered constants (docs/research/2026-07-12_sge_preregistration.md).
# "nyquist" / "lowpass" / "nyquist_rev" are the MS-SGE per-band extension
# (docs/research/2026-07-13_msge_preregistration.md): arm G (per-band
# Nyquist gates), arm H (static 100 km-equivalent mask), arm I (reversed
# placebo). "nyquist2" is the round-3 G2 rule (docs/research/
# 2026-07-13_r3_preregistration.md, P-R3a/b): the handicap-free gate
# g_k = exp(-(d_NN/(0.5*lambda_k))^2) — exactly 1 at d = 0, removing the
# sigmoid rule's analytic in-support ceiling sigma(2) ~ 0.88 that sank
# P-MS1; same lambda ladder, same Nyquist scale c (MSGE_C), no new
# constants, per-band dropout unchanged.
SGE_GATE_MODES: tuple[str, ...] = (
    "off", "raw", "fourier", "test_only",
    "nyquist", "lowpass", "nyquist_rev", "nyquist2",
)
MSGE_BAND_MODES: tuple[str, ...] = ("nyquist", "lowpass", "nyquist_rev", "nyquist2")
SGE_REF_SEED = 4242
SGE_REF_SIZE = 10_000
SGE_DBAR_MAX_POINTS = 5_000
# The placebo space mirrors scripts.nmi_arch_sweep's A2 arm exactly
# (16 bands, scale 4.0 — parity-tested in test_sge_gate.py).
SGE_FOURIER_BANDS = 16
SGE_FOURIER_SCALE = 4.0
SGE_MIN_CROSS_GROUP_DI = 20  # mirrors nmi_aoa_audit.MIN_CROSS_GROUP_DI
SGE_NONE_LABEL = "__none__"

# ---- MS-SGE per-band constants (docs/research/2026-07-13_msge_prereg) -------
# g_k(x) = sigmoid((MSGE_C * lambda_k - d_NN(x)) / (MSGE_W * lambda_k)),
# c = 0.5 (local Nyquist, fixed a priori), w = 0.25 (softness, fixed).
MSGE_C = 0.5
MSGE_W = 0.25
# Arm H (lowpass): keep only bands whose wavelength is >= the standardized
# equivalent of this many km (fixed BEFORE runs by the pre-registration).
MSGE_LOWPASS_CUT_KM = 100.0
# km <-> degree conversion at the Earth's surface: 1 degree of latitude
# (and 1 degree of longitude AT THE EQUATOR) spans ~111.32 km; longitude
# degrees shrink by cos(latitude).
KM_PER_DEG_LAT = 111.32

# In-region holdout (new inregion_rmse metric, in EVERY summary): a fixed
# 5% random holdout drawn from the TRAINING-region rows, disjoint from
# training, numpy seed 777.
INREGION_SEED = 777
INREGION_FRACTION = 0.05

# Sigmoid exponent clip: keeps g strictly inside (0, 1) and finite in
# float64 (exp(-30) ~ 9.4e-14 is still resolvable next to 1.0; a larger
# clip would round g to exactly 1.0 and break the fail-safe endpoint
# contract g in (0, 1)).
_SIGMOID_CLIP = 30.0


# ----------------------------------------------------------------------------
# Arm plan resolution (--sge-gate x --gate-dropout -> what the trainer does)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SGEPlan:
    """Resolved behaviour of one (--sge-gate, --gate-dropout) combination.

    ============ =================== =========== ============= ==============
    arm          mode                gate column gate_space    eval-only gate
    ============ =================== =========== ============= ==============
    base         off, p=0            no          —             no
    drop         off, p>0            yes (==1)   None          no
    sge          raw, p              yes (DI)    "raw"         no
    sgef         fourier, p          yes (DI)    "fourier"     no
    testgate     test_only, p==0     no          "raw"         yes
    msge         nyquist, p          yes (bands) None*         no
    lowpass      lowpass, p==0       yes (bands) None*         no
    msgerev      nyquist_rev, p      yes (bands) None*         no
    msge2        nyquist2, p         yes (bands) None*         no
    ============ =================== =========== ============= ==============

    *Band modes carry ``band_mode`` instead of ``gate_space``: the scalar
    ``gate_space`` field keeps its original meaning (which space the SCALAR
    DI gate is estimated in) so the round-1 code paths are untouched; the
    per-band gates always measure d_NN in raw standardized space.
    """

    mode: str
    append_gate_column: bool
    gate_space: str | None
    train_dropout: float
    eval_gate_only: bool
    band_mode: str | None = None

    @property
    def needs_gates(self) -> bool:
        """True when a gate model must be fitted at data load."""
        return self.gate_space is not None or self.band_mode is not None


def resolve_sge_plan(mode: str, gate_dropout: float) -> SGEPlan:
    """Map the two CLI flags onto the executable plan; fail loud on
    combinations the pre-registration does not define."""
    if mode not in SGE_GATE_MODES:
        raise ValueError(f"Unknown --sge-gate {mode!r}; choose from {SGE_GATE_MODES}")
    p = float(gate_dropout)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"--gate-dropout must be in [0, 1]; got {p}")
    if mode == "test_only" and p > 0.0:
        raise ValueError(
            "--gate-dropout is a TRAINING-time mechanism (modes raw/fourier, "
            "or mode off for the pure-dropout 'drop' arm); test_only trains "
            "exactly like base, so dropout would silently change the train "
            "path — refuse instead."
        )
    if mode == "lowpass" and p > 0.0:
        raise ValueError(
            "--sge-gate lowpass is the STATIC arm H of the MS-SGE "
            "pre-registration (docs/research/2026-07-13_msge_preregistration"
            ".md): a fixed 100 km-equivalent band mask with NO adaptivity "
            "and NO dropout. Refusing --gate-dropout > 0 instead of "
            "silently training a different arm."
        )
    if mode in MSGE_BAND_MODES:
        # Per-band gates (arms G / H / I). The gate columns are a fixed
        # geometric function of the training coordinates; dropout (G / I
        # only) is per band per row, applied by the trainer.
        return SGEPlan(mode=mode, append_gate_column=True, gate_space=None,
                       train_dropout=p, eval_gate_only=False, band_mode=mode)
    if mode == "off":
        if p > 0.0:
            # 'drop' arm: no gate ever (constant 1 at test), but the Fourier
            # block is zeroed per-row with prob p during training.
            return SGEPlan(mode=mode, append_gate_column=True, gate_space=None,
                           train_dropout=p, eval_gate_only=False)
        return SGEPlan(mode=mode, append_gate_column=False, gate_space=None,
                       train_dropout=0.0, eval_gate_only=False)
    if mode == "test_only":
        return SGEPlan(mode=mode, append_gate_column=False, gate_space="raw",
                       train_dropout=0.0, eval_gate_only=True)
    # raw / fourier
    return SGEPlan(mode=mode, append_gate_column=True, gate_space=mode,
                   train_dropout=p, eval_gate_only=False)


# ----------------------------------------------------------------------------
# Gate math
# ----------------------------------------------------------------------------


def gate_threshold_softness(di_train: np.ndarray) -> tuple[float, float]:
    """(t, s) from the training rows' own DI distribution:
    t = Q3 + 1.5*IQR (Meyer & Pebesma's boxplot-whisker AOA threshold),
    s = IQR/2 (pre-registered softness), floored to stay strictly positive
    on degenerate (constant-DI) inputs."""
    di_train = np.asarray(di_train, dtype=np.float64)
    q1, q3 = np.percentile(di_train, [25.0, 75.0])
    iqr = q3 - q1
    threshold = float(q3 + 1.5 * iqr)
    softness = float(max(iqr / 2.0, 1e-9))
    return threshold, softness


def sge_gate_values(di: np.ndarray, threshold: float, softness: float) -> np.ndarray:
    """g = sigmoid((t - DI)/s), exponent clipped so g in (0, 1) strictly."""
    z = (float(threshold) - np.asarray(di, dtype=np.float64)) / float(softness)
    z = np.clip(z, -_SIGMOID_CLIP, _SIGMOID_CLIP)
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------
# Local mirrors of the nmi_aoa_audit DI primitives.
#
# Kept bit-equal to scripts.nmi_aoa_audit (parity-tested); present so the
# frozen cluster image (which pre-dates nmi_aoa_audit) can run this module
# stand-alone.
# ----------------------------------------------------------------------------


def _scale_columns_train_only_local(
    train_x: np.ndarray, test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column z-score with TRAINING statistics only (M&P step 1)."""
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    mean = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (train_x - mean) / sd, (test_x - mean) / sd


def _mean_pairwise_distance_local(x: np.ndarray, *, max_points: int, seed: int) -> float:
    """``dbar``: mean pairwise L2 distance (DI denominator); exact on a
    deterministic <=max_points subsample."""
    from sklearn.metrics import pairwise_distances

    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        raise ValueError("Need at least 2 training points for a pairwise mean.")
    if max_points > 1 and len(x) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=max_points, replace=False)
        idx.sort()
        x = x[idx]
    d = pairwise_distances(x, metric="euclidean")
    n = len(x)
    return float(d.sum() / (n * (n - 1)))


def _nn_distance_to_train_local(train_x: np.ndarray, query_x: np.ndarray) -> np.ndarray:
    """L2 distance from each query point to its nearest TRAINING point."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="euclidean")
    nn.fit(np.asarray(train_x, dtype=np.float64))
    dist, _ = nn.kneighbors(np.asarray(query_x, dtype=np.float64))
    return dist[:, 0].astype(np.float64)


def _loo_nn_distance_local(x: np.ndarray, query_idx: np.ndarray | None = None) -> np.ndarray:
    """Leave-one-out NN distance within ``x``."""
    from sklearn.neighbors import NearestNeighbors

    x = np.asarray(x, dtype=np.float64)
    q = x if query_idx is None else x[query_idx]
    nn = NearestNeighbors(n_neighbors=2, algorithm="brute", metric="euclidean")
    nn.fit(x)
    dist, idx = nn.kneighbors(q)
    own = np.arange(len(x)) if query_idx is None else np.asarray(query_idx)
    take_second = idx[:, 0] == own
    return np.where(take_second, dist[:, 1], dist[:, 0]).astype(np.float64)


def _cross_group_nn_distance_local(
    x: np.ndarray,
    groups: np.ndarray,
    *,
    none_label: str = SGE_NONE_LABEL,
    query_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Per-row L2 distance to the nearest row of a DIFFERENT group (the
    LRO-CV-mirroring NN of Meyer & Pebesma); NaN for ``none_label`` holders
    and degenerate single-group inputs."""
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors

    x = np.asarray(x, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    if query_mask is None:
        query_mask = np.ones(len(x), dtype=bool)
    out = np.full(len(x), np.nan, dtype=np.float64)
    for g in pd.unique(groups):
        if g == none_label:
            continue
        holders = (groups == g) & query_mask
        if not holders.any():
            continue
        candidates = groups != g
        if not candidates.any():
            continue
        nn = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="euclidean")
        nn.fit(x[candidates])
        dist, _ = nn.kneighbors(x[holders])
        out[holders] = dist[:, 0]
    return out


def _region_labels_local(
    lat: np.ndarray,
    lon: np.ndarray,
    regions: dict[str, tuple[float, float, float, float]] | None = None,
    *,
    none_label: str = SGE_NONE_LABEL,
) -> np.ndarray:
    """LRO region label per row from the JP bounding boxes (first match in
    insertion order), ``none_label`` outside every box."""
    if regions is None:
        from national.evaluation.leave_region_out import DEFAULT_REGIONS

        regions = DEFAULT_REGIONS
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    labels = np.full(len(lat), none_label, dtype=object)
    unassigned = np.ones(len(lat), dtype=bool)
    for name, (lat_min, lat_max, lon_min, lon_max) in regions.items():
        inside = (
            (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
        )
        take = inside & unassigned
        labels[take] = name
        unassigned &= ~take
    return labels


def _fourier_map_local(
    x_std: np.ndarray,
    *,
    n_bands: int = SGE_FOURIER_BANDS,
    scale: float = SGE_FOURIER_SCALE,
) -> np.ndarray:
    """The placebo space's coordinate map — bit-equal mirror of
    ``scripts.nmi_arch_sweep.fourier_features`` (which itself wraps the
    paper's ``_FourierFeatures``). Returns ``(N, 2*n_bands)`` float64."""
    import torch

    from national.models.foundation import _FourierFeatures

    ff = _FourierFeatures(n_bands, scale)
    with torch.no_grad():
        z = ff(torch.as_tensor(np.asarray(x_std, dtype=np.float32)))
    return z.numpy().astype(np.float64, copy=False)


# REUSE the nmi_aoa_audit originals where importable (local dev / CI); the
# frozen cluster image lacks scripts.nmi_aoa_audit, so fall back to the
# bit-equal mirrors above. Parity is pinned by test_sge_gate.py either way.
try:  # pragma: no cover - exercised implicitly by both environments
    from scripts.nmi_aoa_audit import (  # type: ignore
        cross_group_nn_distance as _cross_group_nn_distance,
        loo_nn_distance as _loo_nn_distance,
        mean_pairwise_distance as _mean_pairwise_distance,
        nn_distance_to_train as _nn_distance_to_train,
        region_labels as _region_labels,
        scale_columns_train_only as _scale_columns_train_only,
    )

    _AOA_HELPER_SOURCE = "scripts.nmi_aoa_audit"
except Exception:  # ImportError on the frozen image; anything else -> mirror too
    _cross_group_nn_distance = _cross_group_nn_distance_local
    _loo_nn_distance = _loo_nn_distance_local
    _mean_pairwise_distance = _mean_pairwise_distance_local
    _nn_distance_to_train = _nn_distance_to_train_local
    _region_labels = _region_labels_local
    _scale_columns_train_only = _scale_columns_train_only_local
    _AOA_HELPER_SOURCE = "local_mirror"


# ----------------------------------------------------------------------------
# The gate model
# ----------------------------------------------------------------------------


@dataclass
class SGEGateModel:
    """Fitted (i.e. computed — nothing is trained) support gate.

    Frozen geometric function of the training coordinates: featurize into
    the gate space, DI = NN distance to the fixed reference subsample over
    ``dbar``, g = sigmoid((threshold - DI)/softness).
    """

    space: str
    lat_mean: float
    lat_std: float
    lon_mean: float
    lon_std: float
    ref_feats: np.ndarray
    dbar: float
    threshold: float
    softness: float
    threshold_method: str
    di_train_median: float
    di_train_q1: float
    di_train_q3: float
    n_ref: int
    ref_seed: int
    ref_size: int
    n_dims: int

    def _featurize(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat_std = (np.asarray(lat, dtype=np.float64) - self.lat_mean) / self.lat_std
        lon_std = (np.asarray(lon, dtype=np.float64) - self.lon_mean) / self.lon_std
        if self.space == "raw":
            return np.column_stack([lat_std, lon_std])
        if self.space == "fourier":
            return np.concatenate(
                [_fourier_map_local(lat_std), _fourier_map_local(lon_std)], axis=1
            )
        raise ValueError(f"Unknown gate space {self.space!r}")

    def di(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Dissimilarity index of arbitrary query coordinates."""
        feats = self._featurize(lat, lon)
        return _nn_distance_to_train(self.ref_feats, feats) / self.dbar

    def gate(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """g(x) in (0, 1) for arbitrary query coordinates."""
        return sge_gate_values(self.di(lat, lon), self.threshold, self.softness)

    def stats(self) -> dict[str, Any]:
        """Provenance payload persisted into summary.json."""
        return {
            "space": self.space,
            "threshold": self.threshold,
            "softness": self.softness,
            "di_train_median": self.di_train_median,
            "di_train_q1": self.di_train_q1,
            "di_train_q3": self.di_train_q3,
            "dbar": self.dbar,
            "threshold_method": self.threshold_method,
            "n_ref": self.n_ref,
            "n_dims": self.n_dims,
            "ref_seed": self.ref_seed,
            "ref_size": self.ref_size,
            "helper_source": _AOA_HELPER_SOURCE,
        }


def fit_sge_gate(
    train_lat: np.ndarray,
    train_lon: np.ndarray,
    *,
    space: str,
    ref_size: int = SGE_REF_SIZE,
    ref_seed: int = SGE_REF_SEED,
    dbar_max_points: int = SGE_DBAR_MAX_POINTS,
    regions: dict[str, tuple[float, float, float, float]] | None = None,
) -> SGEGateModel:
    """Compute the frozen gate from the TRAINING coordinates.

    Steps (all deterministic, all constants pre-registered):

    1. standardize lat/lon by the full training mean/sd;
    2. draw the fixed reference subsample (<= ``ref_size`` rows, numpy
       ``default_rng(ref_seed)``) and featurize it into the gate space;
    3. ``dbar`` = exact mean pairwise distance on a deterministic
       <= ``dbar_max_points`` subsample of the reference features;
    4. threshold DI distribution = the reference rows' own cross-region NN
       distance (LRO region boxes; LOO-NN fallback recorded when fewer than
       ``SGE_MIN_CROSS_GROUP_DI`` rows carry one) over ``dbar``;
    5. t = Q3 + 1.5*IQR, s = IQR/2.
    """
    if space not in ("raw", "fourier"):
        raise ValueError(f"Unknown gate space {space!r}; choose 'raw' or 'fourier'")
    train_lat = np.asarray(train_lat, dtype=np.float64)
    train_lon = np.asarray(train_lon, dtype=np.float64)
    if train_lat.shape != train_lon.shape or train_lat.ndim != 1:
        raise ValueError("train_lat / train_lon must be equal-length 1-D arrays")
    n_train = len(train_lat)
    if n_train < 2:
        raise ValueError("Need at least 2 training rows to fit a gate.")

    lat_mean = float(train_lat.mean())
    lat_std = float(train_lat.std()) or 1.0
    lat_std = lat_std if lat_std >= 1e-12 else 1.0
    lon_mean = float(train_lon.mean())
    lon_std = float(train_lon.std())
    lon_std = lon_std if lon_std >= 1e-12 else 1.0

    # 2. Fixed reference subsample (seed 4242).
    rng = np.random.default_rng(ref_seed)
    if n_train > ref_size:
        ref_idx = rng.choice(n_train, size=ref_size, replace=False)
        ref_idx.sort()
    else:
        ref_idx = np.arange(n_train)
    ref_lat = train_lat[ref_idx]
    ref_lon = train_lon[ref_idx]

    model = SGEGateModel(
        space=space,
        lat_mean=lat_mean, lat_std=lat_std, lon_mean=lon_mean, lon_std=lon_std,
        ref_feats=np.empty((0, 0)), dbar=1.0, threshold=0.0, softness=1.0,
        threshold_method="", di_train_median=0.0, di_train_q1=0.0,
        di_train_q3=0.0, n_ref=len(ref_idx), ref_seed=int(ref_seed),
        ref_size=int(ref_size), n_dims=0,
    )
    ref_feats = model._featurize(ref_lat, ref_lon)
    model.ref_feats = ref_feats
    model.n_dims = int(ref_feats.shape[1])

    # 3. dbar (exact on <= dbar_max_points deterministic subsample).
    model.dbar = _mean_pairwise_distance(
        ref_feats, max_points=dbar_max_points, seed=ref_seed
    )

    # 4. Training rows' own cross-region DI (mirrors nmi_aoa_audit: holders
    # capped at the reference set, all reference rows stay candidates).
    groups = _region_labels(ref_lat, ref_lon, regions)
    d_cross = _cross_group_nn_distance(ref_feats, groups)
    valid = d_cross[np.isfinite(d_cross)]
    if len(valid) >= SGE_MIN_CROSS_GROUP_DI or (
        len(valid) == len(ref_feats) and len(valid) > 0
    ):
        di_train = valid / model.dbar
        model.threshold_method = "cross_region"
    else:
        di_train = _loo_nn_distance(ref_feats) / model.dbar
        model.threshold_method = "loo_nn"
        LOG.info(
            "SGE gate: cross-region DI degenerate (%d valid rows < %d) -- "
            "documented LOO-NN threshold fallback.",
            len(valid), SGE_MIN_CROSS_GROUP_DI,
        )

    # 5. Threshold + softness from the quantiles.
    model.threshold, model.softness = gate_threshold_softness(di_train)
    q1, q3 = np.percentile(np.asarray(di_train, dtype=np.float64), [25.0, 75.0])
    model.di_train_median = float(np.median(di_train))
    model.di_train_q1 = float(q1)
    model.di_train_q3 = float(q3)
    return model


# ----------------------------------------------------------------------------
# MS-SGE: per-band Nyquist gates (docs/research/2026-07-13_msge_prereg)
# ----------------------------------------------------------------------------


def msge_band_wavelengths_deg(n_bands: int, fourier_scale: float) -> np.ndarray:
    """Band wavelengths in INPUT DEGREES, read off the ACTUAL
    ``_FourierFeatures`` parametrization (prereg risk (iii): derive from the
    implementation, never assume).

    The encoder (``ResMLPEncoder._featurize``) feeds ``x_rad = x_deg *
    pi/180`` into ``_FourierFeatures``, whose band-k frequency is
    ``freqs[k] = 2^k * 2^fourier_scale`` cycles per radian. Band k's period
    is therefore ``2*pi / freqs[k]`` radians of input =
    ``(2*pi / freqs[k]) * (180/pi) = 360 / freqs[k]`` degrees. We read the
    ``freqs`` buffer from the module itself so this can never drift from
    the model code."""
    from national.models.foundation import _FourierFeatures

    ff = _FourierFeatures(int(n_bands), float(fourier_scale))
    freqs = ff.freqs.detach().cpu().numpy().astype(np.float64)
    return 360.0 / freqs


def msge_gate_values(
    dnn: np.ndarray, lambda_std: np.ndarray, *, reverse: bool = False
) -> np.ndarray:
    """Per-band gate matrix ``(N, n_bands)``.

    Forward (arm G, ``nyquist``):
        g_k(x) = sigmoid((0.5 * lambda_k - d_NN(x)) / (0.25 * lambda_k))
    Reversed (arm I, ``nyquist_rev``) is the SIGN-FLIPPED threshold rule:
        g'_k(x) = sigmoid((d_NN(x) - 0.5 * lambda_k) / (0.25 * lambda_k))
    which equals ``1 - g_k(x)`` exactly (sigmoid antisymmetry; the exponent
    clip is symmetric), i.e. it closes long-wavelength bands and keeps the
    short ones — the pre-registered placebo semantics.

    ``dnn`` and ``lambda_std`` must share units (both in the raw
    train-standardized coordinate space here). The exponent clip keeps every
    gate strictly inside (0, 1) — the fail-safe endpoint contract."""
    d = np.asarray(dnn, dtype=np.float64).reshape(-1, 1)
    lam = np.asarray(lambda_std, dtype=np.float64).reshape(1, -1)
    if np.any(lam <= 0.0):
        raise ValueError("lambda_std must be strictly positive")
    z = (MSGE_C * lam - d) / (MSGE_W * lam)
    if reverse:
        z = -z
    z = np.clip(z, -_SIGMOID_CLIP, _SIGMOID_CLIP)
    return 1.0 / (1.0 + np.exp(-z))


def msge_g2_gate_values(dnn: np.ndarray, lambda_std: np.ndarray) -> np.ndarray:
    """Round-3 G2 rule (arm ``msge2``, mode ``nyquist2`` — P-R3a/b of
    docs/research/2026-07-13_r3_preregistration.md):

        g_k(x) = exp(-(d_NN(x) / (0.5 * lambda_k))^2)

    The handicap-free replacement for the sigmoid Nyquist rule: EXACTLY 1
    at d_NN = 0 (the sigmoid rule caps in-support gates at sigma(c/w) =
    sigma(2) ~ 0.88 — the analytic ceiling that sank P-MS1), Gaussian
    falloff with the SAME pre-registered Nyquist scale ``MSGE_C`` = 0.5
    and no new constants (the softness ``MSGE_W`` has no analogue here —
    the falloff shape IS the rule). The exponent is floored at
    ``-_SIGMOID_CLIP`` so the far field stays strictly positive and
    finite in float64 (the same tail magnitude, exp(-30) ~ 9.4e-14, as
    the sigmoid rule's clipped tail); the near field genuinely reaches
    1.0 — that is the point of G2, so the (0, 1)-strict endpoint contract
    of the sigmoid rules is deliberately relaxed to (0, 1] here.

    ``dnn`` and ``lambda_std`` must share units (both in the raw
    train-standardized coordinate space)."""
    d = np.asarray(dnn, dtype=np.float64).reshape(-1, 1)
    lam = np.asarray(lambda_std, dtype=np.float64).reshape(1, -1)
    if np.any(lam <= 0.0):
        raise ValueError("lambda_std must be strictly positive")
    z = -np.square(d / (MSGE_C * lam))
    z = np.maximum(z, -_SIGMOID_CLIP)
    return np.exp(z)


@dataclass
class MSGEGateModel:
    """Fitted (computed — nothing is trained) multi-scale per-band gate.

    Wraps the round-1 raw-space :class:`SGEGateModel` for its FIXED
    reference geometry — the SAME 10,000-row training-coordinate subsample
    (numpy seed 4242) and the same raw train-standardized (lat, lon)
    featurization — so ``d_NN`` is one distance shared by all bands; only
    the thresholds differ (0.5 * lambda_k per band).

    Unit conventions (recorded in ``stats()`` for the summary):

    * ``lambda_deg[k]`` — band period in input degrees (see
      :func:`msge_band_wavelengths_deg`).
    * ``deg_to_std`` — mean of the two per-axis degree->standardized
      conversions, ``0.5 * (1/sd_lat + 1/sd_lon)`` with the training
      standard deviations (the same stats that define the d_NN space; the
      mean is the documented convention for collapsing the lat/lon
      anisotropy into the single isotropic L2 gate distance).
    * ``km_to_std`` — mean of the two per-axis km->standardized
      conversions, ``0.5 * (1/(111.32 * sd_lat) + 1/(111.32 *
      cos(lat_mean) * sd_lon))`` (1 km along latitude spans 1/111.32 deg;
      along longitude 1/(111.32 * cos(lat_mean)) deg).
    * ``lambda_std = lambda_deg * deg_to_std`` — the gate thresholds.
    * ``lambda_km = lambda_std / km_to_std`` — km equivalents (used by the
      arm-H lowpass cut, which is *defined* in standardized units:
      keep band k iff ``lambda_std[k] >= lowpass_cut_km * km_to_std``).
    """

    mode: str
    base: SGEGateModel
    n_bands: int
    fourier_scale: float
    lambda_deg: np.ndarray
    lambda_std: np.ndarray
    lambda_km: np.ndarray
    deg_to_std: float
    km_to_std: float
    lowpass_cut_km: float
    lowpass_keep: np.ndarray

    def dnn(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Raw-space NN distance to the fixed reference subsample, in
        train-standardized coordinate units (the pre-dbar distance of
        ``SGEGateModel.di`` — dbar normalization is not needed here because
        the lambda thresholds already live in the same standardized
        space)."""
        feats = self.base._featurize(lat, lon)
        return _nn_distance_to_train(self.base.ref_feats, feats)

    def gates(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Per-band gate matrix ``(N, n_bands)`` for arbitrary queries."""
        n = len(np.asarray(lat))
        if self.mode == "lowpass":
            # STATIC mask — no adaptivity, identical for every row.
            return np.tile(
                self.lowpass_keep.astype(np.float64), (n, 1)
            )
        d = self.dnn(lat, lon)
        if self.mode == "nyquist2":
            # Round-3 G2 rule: same d_NN geometry and lambda ladder, the
            # Gaussian handicap-free falloff instead of the sigmoid.
            return msge_g2_gate_values(d, self.lambda_std)
        return msge_gate_values(
            d, self.lambda_std, reverse=(self.mode == "nyquist_rev")
        )

    def stats(self) -> dict[str, Any]:
        """Provenance payload persisted into summary.json."""
        return {
            "band_mode": self.mode,
            # Which analytic rule produced the gates: the round-2 sigmoid
            # family (nyquist / nyquist_rev / lowpass mask) or the round-3
            # Gaussian G2 rule. Harvest code keys the ceiling analysis on
            # this.
            "gate_rule": "g2_gauss" if self.mode == "nyquist2" else "sigmoid",
            "space": "raw",
            "n_bands": int(self.n_bands),
            "fourier_scale": float(self.fourier_scale),
            "nyquist_c": MSGE_C,
            "softness_w": MSGE_W,
            "lambda_deg": [float(v) for v in self.lambda_deg],
            "lambda_std": [float(v) for v in self.lambda_std],
            "lambda_km": [float(v) for v in self.lambda_km],
            "deg_to_std": float(self.deg_to_std),
            "km_to_std": float(self.km_to_std),
            "lowpass_cut_km": float(self.lowpass_cut_km),
            "lowpass_keep": [bool(v) for v in self.lowpass_keep],
            "lat_mean": self.base.lat_mean,
            "lat_std": self.base.lat_std,
            "lon_mean": self.base.lon_mean,
            "lon_std": self.base.lon_std,
            "n_ref": self.base.n_ref,
            "ref_seed": self.base.ref_seed,
            "ref_size": self.base.ref_size,
            "helper_source": _AOA_HELPER_SOURCE,
        }


def fit_msge_gate(
    train_lat: np.ndarray,
    train_lon: np.ndarray,
    *,
    mode: str,
    n_bands: int,
    fourier_scale: float = 4.0,
    lowpass_cut_km: float = MSGE_LOWPASS_CUT_KM,
    ref_size: int = SGE_REF_SIZE,
    ref_seed: int = SGE_REF_SEED,
    dbar_max_points: int = SGE_DBAR_MAX_POINTS,
    regions: dict[str, tuple[float, float, float, float]] | None = None,
) -> MSGEGateModel:
    """Compute the frozen per-band gate from the TRAINING coordinates.

    The reference geometry is delegated to :func:`fit_sge_gate` with
    ``space="raw"`` — bit-identical standardization, reference subsample
    (seed 4242) and NN machinery to the round-1 scalar gate, so "the SAME
    raw-space NN distance" is guaranteed by construction rather than by
    parallel implementation. (The scalar model's DI threshold/softness are
    computed alongside but unused by the per-band rule.)
    """
    if mode not in MSGE_BAND_MODES:
        raise ValueError(
            f"Unknown MS-SGE band mode {mode!r}; choose from {MSGE_BAND_MODES}"
        )
    if int(n_bands) < 1:
        raise ValueError(f"n_bands must be >= 1; got {n_bands}")
    base = fit_sge_gate(
        train_lat, train_lon, space="raw",
        ref_size=ref_size, ref_seed=ref_seed,
        dbar_max_points=dbar_max_points, regions=regions,
    )
    lambda_deg = msge_band_wavelengths_deg(n_bands, fourier_scale)
    deg_to_std = 0.5 * (1.0 / base.lat_std + 1.0 / base.lon_std)
    cos_lat = float(np.cos(np.radians(base.lat_mean)))
    if cos_lat <= 0.0:
        raise ValueError(
            f"Degenerate mean latitude {base.lat_mean:.2f} deg (cos <= 0); "
            "the km->standardized conversion is undefined."
        )
    km_to_std = 0.5 * (
        1.0 / (KM_PER_DEG_LAT * base.lat_std)
        + 1.0 / (KM_PER_DEG_LAT * cos_lat * base.lon_std)
    )
    lambda_std = lambda_deg * deg_to_std
    lambda_km = lambda_std / km_to_std
    lowpass_keep = lambda_std >= float(lowpass_cut_km) * km_to_std
    model = MSGEGateModel(
        mode=mode,
        base=base,
        n_bands=int(n_bands),
        fourier_scale=float(fourier_scale),
        lambda_deg=lambda_deg,
        lambda_std=lambda_std,
        lambda_km=lambda_km,
        deg_to_std=float(deg_to_std),
        km_to_std=float(km_to_std),
        lowpass_cut_km=float(lowpass_cut_km),
        lowpass_keep=lowpass_keep,
    )
    if mode == "lowpass" and not (0 < int(lowpass_keep.sum()) < int(n_bands)):
        LOG.warning(
            "MS-SGE lowpass mask is degenerate (%d/%d bands kept) — the "
            "km->standardized conversion (%.3g) or the data extent looks "
            "unusual; proceeding (mask is pre-registered as-is).",
            int(lowpass_keep.sum()), int(n_bands), km_to_std,
        )
    return model


# ----------------------------------------------------------------------------
# In-region holdout (inregion_rmse metric)
# ----------------------------------------------------------------------------


def draw_inregion_holdout(
    positions: np.ndarray,
    *,
    seed: int = INREGION_SEED,
    fraction: float = INREGION_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Split training positions into (train, in-region holdout).

    A fixed ``fraction`` (default 5%) random draw with ``default_rng(seed)``
    (default 777) — an isolated generator, so the draw never perturbs the
    global numpy RNG stream and the training trajectory of arms that differ
    only in eval-time behaviour stays bit-identical. Returns sorted,
    disjoint position arrays whose union is ``positions``.
    """
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1:
        raise ValueError("positions must be 1-D")
    n = len(positions)
    if n < 2:
        return positions, np.empty(0, dtype=np.int64)
    n_hold = max(1, int(round(float(fraction) * n)))
    if n_hold >= n:
        raise ValueError(f"in-region holdout ({n_hold}) would consume all {n} rows")
    rng = np.random.default_rng(seed)
    hold_mask = np.zeros(n, dtype=bool)
    hold_mask[rng.choice(n, size=n_hold, replace=False)] = True
    holdout = np.sort(positions[hold_mask])
    train = np.sort(positions[~hold_mask])
    return train, holdout


__all__ = [
    "SGE_GATE_MODES",
    "MSGE_BAND_MODES",
    "SGE_REF_SEED",
    "SGE_REF_SIZE",
    "SGE_DBAR_MAX_POINTS",
    "SGE_FOURIER_BANDS",
    "SGE_FOURIER_SCALE",
    "SGE_MIN_CROSS_GROUP_DI",
    "MSGE_C",
    "MSGE_W",
    "MSGE_LOWPASS_CUT_KM",
    "KM_PER_DEG_LAT",
    "INREGION_SEED",
    "INREGION_FRACTION",
    "SGEPlan",
    "resolve_sge_plan",
    "gate_threshold_softness",
    "sge_gate_values",
    "msge_band_wavelengths_deg",
    "msge_gate_values",
    "msge_g2_gate_values",
    "SGEGateModel",
    "MSGEGateModel",
    "fit_sge_gate",
    "fit_msge_gate",
    "draw_inregion_holdout",
]
