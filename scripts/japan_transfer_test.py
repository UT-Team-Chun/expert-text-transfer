#!/usr/bin/env python
"""Bring the JAPAN domain to the same leak-proof bar as the UK/storm replications.

The original Japanese content effect (-7.3%) used the cluster pipeline with a Sarashina
embedding and a single GLOBAL PCA basis (fit on all layers, incl. held-out regions) and
a single shuffle. To match the UK and storm replications --- and to answer the capstone
review's request for per-fold PCA + multi-seed + a significance test on Japan --- this
driver rebuilds the (text, SPT-N, region) table locally and runs the identical
``_evaluate_lro`` harness (per-fold PCA fit on training regions only, multi-seed,
shuffled-embedding null) with the SAME multilingual embedder used for UK/storm.

It reconstructs the text<->SPT join from primary sources (no cluster artefacts needed):
  * per-layer narratives + coordinates + depth intervals come from
    ``data/features/derived/soil_text_layers.csv`` (file_path, lat, lon, depth_top/bottom);
  * SPT N-values come from re-parsing each borehole's KuniJiban XML
    (``verification.r_okauchi.v2_extract_location_with_river.extract_spt_measurements``),
    joined to the layer that contains each SPT depth, by ``file_path`` + depth-interval.
Regions are the eight standard Japanese blocks (leave_region_out.DEFAULT_REGIONS).
Text-bearing rows only, so the comparison is uniform with the ~100%-text UK/storm corpora
(the has_text-provenance confound is reported separately in the original decomposition).

Usage:
  python -m scripts.japan_transfer_test --layers ../data/features/derived/soil_text_layers.csv \
      --out ../docs/research/2026-06-21_japan_transfer_leakproof.json --cache-dir <cache> \
      --per-region-files 700
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from national.evaluation.leave_region_out import DEFAULT_REGIONS
from scripts.uk_transfer_test import _evaluate_lro, embed_texts

LOG = logging.getLogger("japan_transfer_test")
REPO = Path(__file__).resolve().parents[2]


def _region_of(lat: float, lon: float) -> str | None:
    for name, (la0, la1, lo0, lo1) in DEFAULT_REGIONS.items():
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return name
    return None


def _resolve_xml(file_path: str) -> Path:
    """soil_text_layers file_path is e.g. '../data/kunijiban/xml/107283_xml.html'
    (relative to backend/). Resolve against the repo root robustly."""
    p = file_path.lstrip("./")
    if p.startswith("data/"):
        return REPO / p
    return (REPO / "backend" / file_path).resolve()


# Increment tags by DTD family (v2 extractor): the <=3.00 DTDs use 10-20/20-30
# split, the 4.00 DTD uses 100-200/200-300. We try both per measurement so no
# DTD-version detection (whose import chain pulls in the full poc package) is needed.
_INCREMENT_TAGSETS = (
    ("標準貫入試験_10_20打撃回数", "標準貫入試験_20_30打撃回数"),
    ("標準貫入試験_100_200打撃回数", "標準貫入試験_200_300打撃回数"),
)


def _spt_rows_for_file(xml_path: Path, layers: pd.DataFrame) -> list[dict]:
    """Parse SPT (depth, N) from one XML and attach the containing layer's text.

    Reuses the v2 extractor's pure helpers (load_xml_root handles the encoding
    ladder + DOCTYPE strip); SPT N is the sum of blow-count increments, trying
    both DTD tag families and taking the larger non-zero total.
    """
    from verification.r_okauchi.v2_extract_location_with_river import (
        find_text,
        load_xml_root,
        parse_float,
        parse_int,
    )
    try:
        root = load_xml_root(xml_path)
    except Exception:  # noqa: BLE001 (a single malformed borehole must not abort the sweep)
        return []
    out = []
    lay = layers.sort_values("depth_top_m")
    for element in root.findall(".//標準貫入試験"):
        d = parse_float(find_text(element, ("標準貫入試験_開始深度",)))
        if d is None:
            continue
        n = None
        for tags in _INCREMENT_TAGSETS:
            incs = [parse_int(find_text(element, (t,))) or 0 for t in tags]
            if any(incs):
                n = float(sum(incs))
                break
        if n is None or not (0 < n <= 100):
            continue
        hit = lay[(lay.depth_top_m <= d) & (d < lay.depth_bottom_m)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        text = str(row.observation_text or "").strip()
        if len(text) < 1:
            continue
        out.append({"latitude_deg": float(row.latitude_deg),
                    "longitude_deg": float(row.longitude_deg),
                    "depth_from_surface": float(d), "n_value": float(n), "text": text,
                    # Borehole identity (XML basename): the unit for grouped
                    # permutation nulls and block bootstraps (prereg P-T1/T3).
                    "boring_file": xml_path.name})
    return out


def build_dataset(layers_csv: Path, per_region_files: int, seed: int = 42,
                  cache: Path | None = None) -> pd.DataFrame:
    """(text, SPT-N, region, boring_file) table from primary sources.

    ``cache``: parquet path. When it exists the frame is loaded from it and
    NO XML is parsed -- this is what lets the cluster run the analysis
    without the 10 GB KuniJiban corpus (only ~123k files' worth of parsed
    rows travel, as a single parquet). When it does not exist the frame is
    built from the XMLs and written there. The cache key is the caller's
    responsibility: bake (per_region_files, seed) into the filename.

    ``per_region_files``: stratified per-region file budget; pass ``0`` (or a
    negative value) to take EVERY text-bearing file -- the full-population
    path the pre-registered primary analysis uses. ``seed`` controls the
    subsample draw so multiple independent subsamples can be requested
    (the historical fixed draw is seed=42, 500 files/region).
    """
    if cache is not None and cache.exists():
        df = pd.read_parquet(cache)
        LOG.info("build_dataset: loaded %d rows from cache %s", len(df), cache)
        return df
    # A subsample can be derived from the full-population cache without
    # re-parsing any XML: the sampling unit is the file, and the full cache
    # already holds every text-bearing file's rows.
    if cache is not None and per_region_files > 0:
        full = cache.parent / f"japan_dataset_full_seed{seed}.parquet"
        if not full.exists():  # the full build is seed-independent
            cands = sorted(cache.parent.glob("japan_dataset_full_seed*.parquet"))
            full = cands[0] if cands else full
        if full.exists():
            src = pd.read_parquet(full)
            file_region = src.groupby("boring_file").region.first()
            rng = np.random.default_rng(seed)
            chosen: list[str] = []
            for r in sorted(set(file_region.values)):
                files = file_region[file_region == r].index.to_numpy()
                take = (files if len(files) <= per_region_files
                        else rng.choice(files, per_region_files, replace=False))
                chosen.extend(take.tolist())
            df = src[src.boring_file.isin(set(chosen))].reset_index(drop=True)
            LOG.info("build_dataset: subsampled %d rows / %d files from the "
                     "full cache %s (seed %d)", len(df), len(chosen), full, seed)
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache, index=False)
            return df
    lay = pd.read_csv(layers_csv)
    lay = lay[lay.observation_text.fillna("").str.len() >= 1].copy()
    lay["region"] = [(_region_of(la, lo) or "other")
                     for la, lo in zip(lay.latitude_deg, lay.longitude_deg)]
    lay = lay[lay.region != "other"]
    # one region per file (by its first layer); stratified file sample per region
    file_region = lay.groupby("file_path").region.first()
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for r in sorted(set(file_region.values)):
        files = file_region[file_region == r].index.to_numpy()
        if per_region_files <= 0 or len(files) <= per_region_files:
            take = files
        else:
            take = rng.choice(files, per_region_files, replace=False)
        chosen.extend(take.tolist())
    LOG.info("sampled %d files across %d regions", len(chosen), file_region.nunique())
    by_file = {fp: g for fp, g in lay[lay.file_path.isin(set(chosen))].groupby("file_path")}
    rows: list[dict] = []
    for i, fp in enumerate(chosen):
        g = by_file.get(fp)
        if g is None:
            continue
        rows.extend(_spt_rows_for_file(_resolve_xml(fp), g))
        if (i + 1) % 500 == 0:
            LOG.info("  parsed %d/%d files, %d SPT rows", i + 1, len(chosen), len(rows))
    df = pd.DataFrame(rows)
    df["region"] = [(_region_of(la, lo) or "other")
                    for la, lo in zip(df.latitude_deg, df.longitude_deg)]
    df = df[df.region != "other"].reset_index(drop=True)
    counts = df.region.value_counts()
    keep = counts[counts >= 200].index.tolist()
    df = df[df.region.isin(keep)].reset_index(drop=True)
    LOG.info("Japan dataset: %d SPT rows, %d regions %s", len(df), df.region.nunique(),
             df.region.value_counts().to_dict())
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        LOG.info("build_dataset: wrote cache %s", cache)
    return df


def run(layers_csv: Path, out: Path, cache_dir: Path, per_region_files: int = 700,
        seeds: list[int] | None = None) -> dict:
    seeds = seeds or [42, 43, 44, 45, 46]
    df = build_dataset(layers_csv, per_region_files)
    emb = embed_texts(df["text"].tolist(), cache_dir / "japan_e5_emb.npy")
    base = ["depth_from_surface", "latitude_deg", "longitude_deg"]
    per = _evaluate_lro(df, base, emb, seeds)

    regions = sorted(per.keys())
    nt = {r: per[r]["no_text"][0] for r in regions}
    tx = {r: per[r]["text"][0] for r in regions}
    sh = {r: per[r]["shuffled"][0] for r in regions}
    content_r = {r: 100 * (tx[r] - sh[r]) / sh[r] for r in regions}

    def agg(d):
        v = list(d.values())
        return round(float(np.mean(v)), 3), round(float(np.std(v)), 3)

    diffs = [tx[r] - sh[r] for r in regions]
    n_neg = sum(d < 0 for d in diffs)
    from math import comb
    sign_p = sum(comb(len(diffs), k) for k in range(n_neg, len(diffs) + 1)) / 2 ** len(diffs)
    wilcox_p = None
    try:
        from scipy.stats import wilcoxon
        wilcox_p = float(wilcoxon(diffs, alternative="less").pvalue)
    except Exception:  # noqa: BLE001
        pass

    results = {
        "config": {"domain": "Japan KuniJiban boreholes (local reconstruction)",
                   "target": "SPT N-value", "split": "leave-region-out (8 Japanese blocks)",
                   "embedder": "intfloat/multilingual-e5-base (uniform with UK/storm)",
                   "leak_proof_per_fold_pca": True, "text_bearing_only": True,
                   "seeds": seeds, "baseline": base, "n_regions": len(regions),
                   "n_rows": len(df), "per_region_files": per_region_files},
        "no_text": {"mean_rmse": agg(nt)[0], "std_across_regions": agg(nt)[1]},
        "text": {"mean_rmse": agg(tx)[0], "std_across_regions": agg(tx)[1],
                 "per_region": {r: round(tx[r], 3) for r in regions}},
        "shuffled": {"mean_rmse": agg(sh)[0], "std_across_regions": agg(sh)[1],
                     "per_region": {r: round(sh[r], 3) for r in regions}},
        "per_region_content_pct": {r: round(content_r[r], 1) for r in regions},
        "per_region_n": {r: int((df.region == r).sum()) for r in regions},
        "deltas": {
            "text_vs_notext_pct": round(100 * (agg(tx)[0] - agg(nt)[0]) / agg(nt)[0], 1),
            "shuffled_vs_notext_pct": round(100 * (agg(sh)[0] - agg(nt)[0]) / agg(nt)[0], 1),
            "content_text_vs_shuffled_pct": round(100 * (agg(tx)[0] - agg(sh)[0]) / agg(sh)[0], 1),
        },
        "content_significance": {
            "n_regions_negative": f"{n_neg}/{len(diffs)}",
            "sign_test_p_one_sided": round(sign_p, 5),
            "wilcoxon_p_one_sided": (round(wilcox_p, 5) if wilcox_p is not None else None),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    LOG.info("JAPAN leak-proof content %.1f%% | %s neg | sign-p %s | %s",
             results["deltas"]["content_text_vs_shuffled_pct"],
             results["content_significance"]["n_regions_negative"], sign_p,
             json.dumps(results["deltas"]))
    print(json.dumps(results, indent=2))
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=Path,
                   default=REPO / "data/features/derived/soil_text_layers.csv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--per-region-files", type=int, default=700)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run(a.layers, a.out, a.cache_dir, a.per_region_files, a.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
