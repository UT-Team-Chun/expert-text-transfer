"""KuniJiban soil-description text extraction (Paper B' Pillar 3 starter).

Pulls the per-layer geologist's narrative (``<観察記事_記事>``) from raw
KuniJiban XML files into a per-boring text record. The narrative is
genuinely descriptive Japanese-language geological observation:
particle size, weathering state, gravel angularity, water content
assessment, color, and so on. None of this information is captured by
the structured AIST regime / age / lithology fields, which means a
Japanese-language sentence embedding (multilingual BERT, RoBERTa) over
the concatenated narrative is expected to carry encoder-relevant signal
the foundation model currently ignores.

KuniJiban XML structure (representative)::

  <観察記事>
    <観察記事_上端深度>2.50</観察記事_上端深度>
    <観察記事_下端深度>3.45</観察記事_下端深度>
    <観察記事_記事>礫混じり砂。中砂〜粗砂主体。φ0.5〜4cmの亜円礫〜亜角礫を20%程度含む。
                  0.6〜1.0m,2.45〜3.10m間は細粒分をやや多く含む。</観察記事_記事>
  </観察記事>

The same encoding ladder + DOCTYPE stripping as
:mod:`groundwater_xml` is reused (shift_jis dominates v1.10-v3.00).
A 200-file random-sample audit on the live corpus found 122 / 200 =
61% of borings carry at least one non-empty narrative; the typical
file has 3-8 layer descriptions.

This module owns ONLY extraction. The BERT-embedding pipeline lives in
``scripts/embed_soil_text.py`` (Paper B' Pillar 3 BERT layer); the
foundation-encoder wiring lives in ``national/data/boring_dataset.py``
via ``feature_columns``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

LOG = logging.getLogger("national.data.derived.soil_text_xml")

# Shared with :mod:`groundwater_xml` -- kept duplicated locally rather than
# importing because the two extractors are intentionally independent (one
# can change without breaking the other). Same encoding ladder.
_ENCODING_CANDIDATES = ("utf-8", "shift_jis", "cp932")
_DOCTYPE_RE = re.compile(r"<!DOCTYPE[^>]*>\s*", re.MULTILINE)


def _read_xml_text(path: Path) -> str:
    last_err: Exception | None = None
    for encoding in _ENCODING_CANDIDATES:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
    if last_err is not None:
        raise last_err
    return path.read_text()  # pragma: no cover


def _parse_xml_root(path: Path) -> ET.Element:
    text = _read_xml_text(path)
    cleaned = _DOCTYPE_RE.sub("", text, count=1)
    parser = ET.XMLParser(target=ET.TreeBuilder())
    parser.feed(cleaned)
    return parser.close()


# ============================================================
# Coordinate parsing (DMS -> decimal degrees)
# ============================================================


def _coerce_float(text: str | None) -> float | None:
    if text is None:
        return None
    cleaned = text.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _find_text(parent: ET.Element, tag: str) -> str | None:
    el = parent.find(tag)
    if el is None or el.text is None:
        return None
    val = el.text.strip()
    return val or None


def _parse_lat_lon(root: ET.Element) -> tuple[float | None, float | None]:
    block = root.find(".//経度緯度情報")
    if block is None:
        return None, None
    lat_d = _coerce_float(_find_text(block, "緯度_度"))
    lat_m = _coerce_float(_find_text(block, "緯度_分")) or 0.0
    lat_s = _coerce_float(_find_text(block, "緯度_秒")) or 0.0
    lon_d = _coerce_float(_find_text(block, "経度_度"))
    lon_m = _coerce_float(_find_text(block, "経度_分")) or 0.0
    lon_s = _coerce_float(_find_text(block, "経度_秒")) or 0.0
    if lat_d is None or lon_d is None:
        return None, None
    return (
        lat_d + lat_m / 60.0 + lat_s / 3600.0,
        lon_d + lon_m / 60.0 + lon_s / 3600.0,
    )


# ============================================================
# Soil-text extraction
# ============================================================


@dataclass(frozen=True)
class SoilTextRecord:
    """One per-boring soil-description narrative record."""

    file_path: str
    latitude_deg: float
    longitude_deg: float
    # Concatenated narrative across all layers in observation order,
    # joined with " || " so a downstream BERT can attend across layers.
    # Empty string if the boring carries no usable text.
    observation_text: str
    n_layers: int
    char_length: int


@dataclass(frozen=True)
class SoilTextLayerRecord:
    """One per-LAYER soil-description narrative record.

    Each ``<観察記事>`` block in the XML emits one record. Depth bounds
    let the downstream pipeline join the per-depth boring data
    (the v4 parquet) to the matching layer's embedding via the depth
    interval ``[depth_top_m, depth_bottom_m)``.

    Compared to the per-boring concatenated representation, this is
    the engineering-relevant granularity: a 2.5 m sample sees the
    "fine sand" description while a 12 m sample sees "weathered
    granite", instead of both averaging to the boring-level cohort.
    """

    file_path: str
    latitude_deg: float
    longitude_deg: float
    layer_idx: int  # 0-based index within the file, document order
    depth_top_m: float
    depth_bottom_m: float
    observation_text: str
    char_length: int


# Cleanup regex: collapse CR/LF + leading/trailing whitespace inside one
# narrative chunk. We deliberately KEEP all unicode characters (Japanese
# punctuation, fullwidth digits, the 〜 range marker) -- the BERT
# tokenizer will handle those. Stripping them here would lose signal.
_WS_RE = re.compile(r"\s+", re.UNICODE)
_LAYER_SEP = " || "


def _layer_texts(root: ET.Element) -> list[str]:
    """Pull every non-empty ``<観察記事_記事>`` value in document order."""
    out: list[str] = []
    for el in root.findall(".//観察記事_記事"):
        if el.text is None:
            continue
        cleaned = _WS_RE.sub(" ", el.text).strip()
        if cleaned:
            out.append(cleaned)
    return out


def extract_soil_text(path: Path) -> SoilTextRecord:
    """Extract one :class:`SoilTextRecord` from a KuniJiban XML.

    Returns a record with ``observation_text=""`` and
    ``n_layers=0`` if the file has no usable narrative. Errors are NOT
    raised; one bad XML in a 191 k-file corpus walk produces a
    NaN-coords + empty-text record rather than killing the iteration.
    """
    try:
        root = _parse_xml_root(path)
    except (ET.ParseError, UnicodeDecodeError, OSError) as exc:
        LOG.warning("Failed to parse %s: %s", path, exc)
        return SoilTextRecord(
            file_path=str(path),
            latitude_deg=float("nan"),
            longitude_deg=float("nan"),
            observation_text="",
            n_layers=0,
            char_length=0,
        )
    lat, lon = _parse_lat_lon(root)
    if lat is None or lon is None:
        return SoilTextRecord(
            file_path=str(path),
            latitude_deg=float("nan"),
            longitude_deg=float("nan"),
            observation_text="",
            n_layers=0,
            char_length=0,
        )
    layers = _layer_texts(root)
    combined = _LAYER_SEP.join(layers)
    return SoilTextRecord(
        file_path=str(path),
        latitude_deg=lat,
        longitude_deg=lon,
        observation_text=combined,
        n_layers=len(layers),
        char_length=len(combined),
    )


def _layer_blocks(root: ET.Element) -> list[ET.Element]:
    """Return every ``<観察記事>`` element in document order."""
    return list(root.findall(".//観察記事"))


def extract_soil_text_layers(path: Path) -> list[SoilTextLayerRecord]:
    """Extract per-layer soil-text records from one KuniJiban XML.

    Returns an empty list if the file is unparseable, has invalid
    coords, or has no usable ``<観察記事_記事>`` element with both a
    non-empty text and parseable depth bounds. Errors are NOT raised
    so a 191 k-file corpus walk doesn't die on bad XML.

    Layers with an empty / unparseable ``観察記事_記事`` are skipped.
    Layers without parseable depth bounds are skipped (they cannot
    be matched to a per-depth row downstream).
    """
    try:
        root = _parse_xml_root(path)
    except (ET.ParseError, UnicodeDecodeError, OSError) as exc:
        LOG.warning("Failed to parse %s: %s", path, exc)
        return []
    lat, lon = _parse_lat_lon(root)
    if lat is None or lon is None:
        return []

    out: list[SoilTextLayerRecord] = []
    for idx, block in enumerate(_layer_blocks(root)):
        text_el = block.find("観察記事_記事")
        if text_el is None or text_el.text is None:
            continue
        cleaned = _WS_RE.sub(" ", text_el.text).strip()
        if not cleaned:
            continue
        # Explicit None checks instead of `or` -- xml.etree.Element evaluates
        # falsy when it has no children even if its .text is non-empty
        # (DeprecationWarning in Python 3.12; behaviour-changing in future).
        top_el = block.find("観察記事_上端深度")
        bot_el = block.find("観察記事_下端深度")
        top = _coerce_float(top_el.text) if top_el is not None else None
        bot = _coerce_float(bot_el.text) if bot_el is not None else None
        if top is None or bot is None:
            continue
        if bot <= top:
            # Degenerate / mis-ordered bounds: skip rather than emit
            # a zero-thickness layer that confuses the downstream
            # interval join.
            continue
        out.append(
            SoilTextLayerRecord(
                file_path=str(path),
                latitude_deg=lat,
                longitude_deg=lon,
                layer_idx=idx,
                depth_top_m=top,
                depth_bottom_m=bot,
                observation_text=cleaned,
                char_length=len(cleaned),
            )
        )
    return out


__all__ = [
    "SoilTextRecord",
    "SoilTextLayerRecord",
    "extract_soil_text",
    "extract_soil_text_layers",
]
