#!/usr/bin/env python
"""NC revision [MAJOR, geoscience] — overburden/depth confound on the text content effect.

Depth is in every baseline, and SPT-N rises with effective overburden stress; a reviewer worries
the lithology text may be re-encoding depth-correlated compactness rather than independent
geological content. We test this directly: re-run the leak-proof content effect (text vs global
shuffled-embedding null, per-fold PCA, multi-seed) WITHIN narrow depth bands. Within a band depth
is ~constant, so the text cannot be proxying depth-stress. If the content effect survives in each
band, it is not a re-encoding of overburden already available through the depth covariate.

Reuses the PROVEN harness scripts.uk_transfer_test._evaluate_lro (unchanged) on depth-band subsets;
embedding is the headline lithology_only representation (cache shared with nc_null_controls).

Run (CPU):
  cd backend
  uv run python -m scripts.nc_depth_band --domain japan \
      --out ../docs/research/2026-06-24_depth_band_japan.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from scripts.text_leakage_controls import load_domain, strip_text
from scripts.uk_transfer_test import _evaluate_lro, embed_texts

LOG = logging.getLogger("nc_depth_band")
REPO = Path(__file__).resolve().parents[2]
_BANDS = [(0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 1e9)]


def _content(per: dict) -> dict:
    regions = sorted(per.keys())
    tx = {r: per[r]["text"][0] for r in regions}
    sh = {r: per[r]["shuffled"][0] for r in regions}
    nt = {r: per[r]["no_text"][0] for r in regions}
    diffs = [tx[r] - sh[r] for r in regions]
    n_neg = sum(d < 0 for d in diffs)
    mean = lambda d: float(np.mean(list(d.values())))
    return {"n_regions": len(regions), "no_text": round(mean(nt), 3), "text": round(mean(tx), 3),
            "shuffled": round(mean(sh), 3),
            "content_pct": round(100 * (mean(tx) - mean(sh)) / mean(sh), 2),
            "regions_negative": f"{n_neg}/{len(regions)}",
            "per_region_content": {r: round(100 * (tx[r] - sh[r]) / sh[r], 1) for r in regions}}


def run(domain: str, out: Path, cache_dir: Path, seeds: list[int]) -> dict:
    df, base, _ = load_domain(domain, cache_dir)
    texts = [strip_text(t, domain, "lithology_only") for t in df["text"].tolist()]
    from scripts.text_leakage_controls import STRIP_VOCAB_VERSION
    emb = embed_texts(texts, cache_dir / f"ncnull_{domain}_lithology_only_{STRIP_VOCAB_VERSION}_e5.npy")
    depth = df["depth_from_surface"].to_numpy()
    res = {"config": {"domain": domain, "n_rows": len(df), "seeds": seeds,
                      "representation": "lithology_only", "bands_m": _BANDS},
           "all_depths": _content(_evaluate_lro(df.reset_index(drop=True), base, emb, seeds)),
           "by_band": {}}
    for lo, hi in _BANDS:
        m = (depth >= lo) & (depth < hi)
        sub = df[m].reset_index(drop=True)
        # need >=2 regions with enough rows for an LRO content estimate
        vc = sub["region"].value_counts()
        if (vc >= 60).sum() < 2:
            res["by_band"][f"{lo:g}-{hi:g}m"] = {"skipped": "too few rows/regions", "n": int(m.sum())}
            continue
        per = _evaluate_lro(sub, base, emb[m.to_numpy() if hasattr(m, "to_numpy") else m], seeds)
        c = _content(per); c["n"] = int(m.sum())
        res["by_band"][f"{lo:g}-{hi:g}m"] = c
        LOG.info("band %g-%g m: n=%d content %.1f%% (%s)", lo, hi, m.sum(),
                 c["content_pct"], c["regions_negative"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({"all_depths": res["all_depths"]["content_pct"],
                      "by_band": {k: v.get("content_pct") for k, v in res["by_band"].items()}}, indent=2))
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["japan", "uk"], default="japan")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=REPO / "data/features/derived")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run(a.domain, a.out, a.cache_dir, a.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
