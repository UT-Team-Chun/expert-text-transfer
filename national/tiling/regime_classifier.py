"""Map a surface-geology code to a coarse lithology regime.

The DKL+SVGP head is modulated per regime (FiLM-style). Regimes are a coarser
partition than the ~150-class surface geology so that each regime has enough
data for stable per-regime hyperparameter learning.

Two regime-assignment paths exist in the project, by design:

- The *legend* path (:func:`national.data.derived.lithology.regime_from_legend`)
  resolves a regime from the AIST human-readable legend fields (age / group /
  lithology strings). The enrichment pipeline uses this.
- The *integer-code* path here maps a raw AIST ``geology_code`` to a regime via
  a precomputed lookup table. ``ingest/geology.py`` uses this when rasterising
  the seamless map to a GeoPackage ``lithology_group`` column.

No lookup is bundled in-tree because the AIST code vocabulary is dataset-version
specific; the table is produced as a sidecar JSON by ``ingest/geology.py`` and
loaded with :func:`load_lookup`.
"""

from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path


class Regime(IntEnum):
    """Lithology regime labels used to modulate the SVGP head."""

    ALLUVIAL = 0
    DILUVIAL = 1
    VOLCANIC_ASH = 2
    SEDIMENTARY = 3
    IGNEOUS = 4
    METAMORPHIC = 5
    LIMESTONE = 6  # 主に沖縄
    UNKNOWN = 7


def regime_from_geology_code(code: int, lookup: dict[int, int]) -> Regime:
    """Map a surface-geology integer code to a :class:`Regime`.

    Args:
        code: int code from the surface-geology ingest (see ``codes.json``).
        lookup: ``geology_code -> regime int`` table from :func:`load_lookup`.
            An empty dict is valid and maps every code to ``UNKNOWN``.

    Returns:
        Regime label. Falls back to ``Regime.UNKNOWN`` for codes absent from
        the lookup.

    Raises:
        ValueError: if ``lookup`` is ``None`` (no table bundled in-tree; load
            one with :func:`load_lookup`).
    """
    if lookup is None:
        raise ValueError(
            "A geology_code -> regime lookup is required. Load one with "
            "load_lookup(path); ingest/geology.py writes the table as a sidecar "
            "JSON. No table is bundled because the AIST code vocabulary is "
            "dataset-version specific."
        )
    return Regime(lookup.get(int(code), int(Regime.UNKNOWN)))


def load_lookup(path: Path) -> dict[int, int]:
    """Load and validate a ``geology_code -> regime int`` lookup JSON.

    Args:
        path: JSON file mapping (stringified) integer geology codes to integer
            :class:`Regime` values.

    Returns:
        ``dict[int, int]`` with integer keys and validated regime values.

    Raises:
        ValueError: if any value is not a valid :class:`Regime` integer.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    valid = {int(member) for member in Regime}
    out: dict[int, int] = {}
    for key, value in raw.items():
        regime_int = int(value)
        if regime_int not in valid:
            raise ValueError(
                f"Lookup entry {key!r} -> {value!r} is not a valid Regime "
                f"(expected 0..{int(Regime.UNKNOWN)})."
            )
        out[int(key)] = regime_int
    return out


__all__ = ["Regime", "regime_from_geology_code", "load_lookup"]
