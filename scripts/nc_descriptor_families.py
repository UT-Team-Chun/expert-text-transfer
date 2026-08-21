#!/usr/bin/env python
"""Descriptor-family mechanism analysis (P-T10): which observations transfer?

Moves the paper from "text helps" to "THESE geological observations carry
the transferable signal". Uses the structured parser as the transfer channel
(language-model-free, so no embedding confound), in its TEXT-DERIVED-ONLY
configuration (``include_archive_codes=False`` -- the published parser rung
mixed AIST archive codes into the Japan block; both variants are reported
here so the two are never conflated again):

- ``parser_text_only``  : all text-derived families
- ``parser_with_codes`` : legacy configuration (archive codes included)
- ``minus_<family>``    : leave-one-family-out over grain_size / sorting /
                          lith_class / weathering / water_state|secondary /
                          angularity / colour / composition_pct

Each arm runs the leak-proof LRO harness (per-fold PCA, multi-seed,
row-shuffled null -- the family deltas are point-estimate mechanism
localisation, prereg marks P-T10 exploratory). The mechanism figure (new
Fig. 4) plots the per-family attenuation.

CLI::

    cd backend
    .venv/bin/python -m scripts.nc_descriptor_families --domain japan \
        --out ../docs/research/2026-08-12_descriptor_families_japan.json
"""
from __future__ import annotations

# OpenMP cap BEFORE any sklearn import: HGB's parallel splitter spin-waits at
# barriers on sub-100k-row fits (observed 6.6 effective threads at ~0 speedup).
# threadpoolctl cannot fix this reliably -- libomp is loaded lazily after the
# limiter enumerates libraries -- so pin it at process start. Override with
# OMP_NUM_THREADS in the environment for large-data runs.
import os
# Measured on the fit this protocol actually runs (M-series, 16 cores):
# 1 thread 5.35 s, 2 threads 9.59 s, 4 threads 9.96 s, 8 threads 15.80 s.
# `sample` shows the multi-thread time goes into __kmp_barrier ->
# cthread_yield: at this data size the histogram/splitting kernels hit the
# OpenMP barrier far more often than they do useful work. One thread per
# process is faster, and leaves the cores free to run shards in parallel.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "0")


import argparse
import json
import logging
from pathlib import Path

import numpy as np

from scripts.text_leakage_controls import load_domain, structured_families
from scripts.uk_transfer_test import _evaluate_lro

LOG = logging.getLogger("nc_descriptor_families")
REPO = Path(__file__).resolve().parents[2]


def _content(per: dict) -> dict:
    regions = sorted(per.keys())
    tx = np.array([per[r]["text"][0] for r in regions])
    sh = np.array([per[r]["shuffled"][0] for r in regions])
    eff = 100.0 * (tx.mean() - sh.mean()) / sh.mean()
    per_region = {r: round(100.0 * (per[r]["text"][0] - per[r]["shuffled"][0])
                           / per[r]["shuffled"][0], 3) for r in regions}
    return {"content_pct": round(float(eff), 3),
            "regions_negative": f"{int((tx < sh).sum())}/{len(regions)}",
            "per_region": per_region}


def run(domain: str, out: Path, cache_dir: Path, *, seeds: list[int],
        per_region_files: int, sample_seed: int) -> dict:
    df, base, _ = load_domain(domain, cache_dir,
                              per_region_files=per_region_files,
                              sample_seed=sample_seed)
    raw = df["text"].tolist()

    # AIST codes for the legacy variant (Japan only)
    if domain == "japan":
        from scripts.nc_grouped_null import attach_v4id_covariates
        dfc = attach_v4id_covariates(df)
        hit = dfc["regime_code"].notna()
        dfc = dfc[hit].reset_index(drop=True)
        raw = [t for t, h in zip(raw, hit) if h]
        df = dfc
    fams_text = structured_families(raw, df, domain, include_archive_codes=False)
    fams_code = structured_families(raw, df, domain, include_archive_codes=True)

    def _concat(fams: dict, drop: str | None = None) -> np.ndarray:
        return np.concatenate(
            [m for k, m in fams.items() if k != drop], axis=1)

    res: dict = {"arms": {}}
    if True:
        for name, feats in (
            ("parser_text_only", _concat(fams_text)),
            ("parser_with_codes", _concat(fams_code)),
        ):
            per = _evaluate_lro(df, base, feats, seeds)
            res["arms"][name] = {"n_features": int(feats.shape[1]),
                                 **_content(per)}
            LOG.info("%-22s (%3dD): %+0.2f%% (%s)", name, feats.shape[1],
                     res["arms"][name]["content_pct"],
                     res["arms"][name]["regions_negative"])
        for fam in fams_text:
            feats = _concat(fams_text, drop=fam)
            per = _evaluate_lro(df, base, feats, seeds)
            res["arms"][f"minus_{fam}"] = {
                "n_features": int(feats.shape[1]), **_content(per)}
            full = res["arms"]["parser_text_only"]["content_pct"]
            att = res["arms"][f"minus_{fam}"]["content_pct"] - full
            res["arms"][f"minus_{fam}"]["attenuation_pp"] = round(float(att), 3)
            LOG.info("minus_%-16s: %+0.2f%% (attenuation %+0.2f pp)", fam,
                     res["arms"][f"minus_{fam}"]["content_pct"], att)

    res["config"] = {
        "domain": domain, "seeds": seeds, "n_rows": int(len(df)),
        "per_region_files": per_region_files, "sample_seed": sample_seed,
        "prereg": "docs/research/2026-08-11_nc_text_preregistration.md "
                  "P-T10 (exploratory)",
    }
    res["_provenance"] = {
        "purpose": "P-T10 descriptor-family mechanism localisation; also "
                   "separates the parser rung into text-only vs +archive-code "
                   "variants (the published -16.2%/-4.4% used the mixed one)",
        "script": "backend/scripts/nc_descriptor_families.py",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    LOG.info("wrote %s", out)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", choices=("japan", "uk"), default="japan")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path,
                    default=REPO / "data/features/derived/nc_cache")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--per-region-files", type=int, default=500)
    ap.add_argument("--sample-seed", type=int, default=42)
    a = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    run(a.domain, a.out, a.cache_dir, seeds=a.seeds,
        per_region_files=a.per_region_files, sample_seed=a.sample_seed)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
