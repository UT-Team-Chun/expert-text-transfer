#!/usr/bin/env python
"""Third-domain breadth test: does the text-content transfer principle hold in a
*non-geotechnical* domain --- NOAA Storm Events (atmospheric hazards)?

The "words generalize, coordinates memorize" principle was shown for Japanese and UK
boreholes (geologist lithology text -> SPT-N). An NCE&E editor's breadth bar: two
borehole archives are one earth-science task. This script tests the SAME principle in
a maximally distant domain: NWS forecasters' free-text storm reports
(``EVENT_NARRATIVE``) predicting a co-located measured magnitude (hail diameter,
inches), under leave-STATE-out, with the identical leak-proof protocol used for the UK
(per-fold PCA fit on train states only, multi-seed, shuffled-embedding null,
significance test).

Coordinates (lat/lon) of a hail event carry little transferable information about its
size; the forecaster's prose ("quarter size", "golf ball size") does, and that content
transfers across the state boundary where coordinates cannot. If the content effect
replicates here, the principle spans subsurface geology AND atmospheric hazards.

Usage:
  python -m scripts.storm_transfer_test --storm-dir <dir of StormEvents_details*.csv.gz> \
      --out <result.json> --cache-dir <cache> --event-type Hail
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.uk_transfer_test import _evaluate_lro, embed_texts

LOG = logging.getLogger("storm_transfer_test")

# Colloquial + numeric hail-size cues that *directly* name the diameter. The
# --strip-size robustness variant removes these so the residual content effect
# reflects what the LLM extracts BEYOND the literal size word (a leakage control
# against the "the narrative just states the answer" critique).
_SIZE_WORDS = (r"pea|marble|mothball|penny|dime|nickel|quarter|half[\s-]?dollar|"
               r"walnut|ping[\s-]?pong|golf[\s-]?ball|golfball|hen[\s-]?egg|egg|"
               r"lime|tennis[\s-]?ball|baseball|apple|softball|grapefruit|"
               r"teacup|tea[\s-]?cup|cd|dvd|"
               r"quarter[\s-]?size|nickel[\s-]?size|dollar[\s-]?size")
_SIZE_RE = re.compile(
    r"(\b(?:%s)\b|\d+(?:\.\d+)?\s*(?:in(?:ch(?:es)?)?|\"|cm|mm)\b|"
    r"\b(?:quarter|half|three[\s-]?quarter)s?\b)" % _SIZE_WORDS, re.IGNORECASE)


def _strip_size(text: str) -> str:
    return _SIZE_RE.sub(" ", text)


def ingest(storm_dir: Path, event_type: str = "Hail", min_chars: int = 20,
           min_state: int = 200, strip_size: bool = False) -> pd.DataFrame:
    """Build (latitude, longitude, n_value=target, region=state, text) from the NOAA
    StormEvents detail CSVs. Target is MAGNITUDE (hail diameter, inches) for the chosen
    event type; region is the US state for leave-region-out."""
    cols = ["STATE", "EVENT_TYPE", "MAGNITUDE", "BEGIN_LAT", "BEGIN_LON", "EVENT_NARRATIVE"]
    files = sorted(glob.glob(str(storm_dir / "StormEvents_details*.csv.gz")))
    if not files:
        raise FileNotFoundError(f"no StormEvents_details*.csv.gz under {storm_dir}")
    df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False)
                    for f in files], ignore_index=True)
    df = df[df["EVENT_TYPE"] == event_type].copy()
    df["text"] = df["EVENT_NARRATIVE"].fillna("").astype(str)
    if strip_size:
        df["text"] = df["text"].map(_strip_size)
        LOG.info("strip_size ON: removed explicit hail-size descriptors from narratives")
    df = df[(df["MAGNITUDE"].notna()) & (df["MAGNITUDE"] > 0)]
    df = df[df["BEGIN_LAT"].notna() & df["BEGIN_LON"].notna()]
    df = df[df["text"].str.len() >= min_chars]
    out = pd.DataFrame({
        "latitude_deg": df["BEGIN_LAT"].to_numpy(np.float32),
        "longitude_deg": df["BEGIN_LON"].to_numpy(np.float32),
        "n_value": df["MAGNITUDE"].to_numpy(np.float32),  # hail diameter (in)
        "region": df["STATE"].astype(str).to_numpy(),
        "text": df["text"].to_numpy(),
    }).reset_index(drop=True)
    counts = out["region"].value_counts()
    keep = counts[counts >= min_state].index.tolist()
    out = out[out["region"].isin(keep)].reset_index(drop=True)
    LOG.info("storm %s: %d events, %d states (>=%d)", event_type, len(out),
             out["region"].nunique(), min_state)
    return out


def run(storm_dir: Path, out: Path, cache_dir: Path, event_type: str = "Hail",
        seeds: list[int] | None = None, strip_size: bool = False) -> dict:
    seeds = seeds or [42, 43, 44, 45, 46]
    df = ingest(storm_dir, event_type=event_type, strip_size=strip_size)
    suffix = "_nosize" if strip_size else ""
    emb = embed_texts(df["text"].tolist(),
                      cache_dir / f"storm_{event_type.lower()}{suffix}_e5_emb.npy")
    base = ["latitude_deg", "longitude_deg"]  # coordinates = the memorize-able baseline
    per = _evaluate_lro(df, base, emb, seeds)

    regions = sorted(per.keys())
    nt = {r: per[r]["no_text"][0] for r in regions}
    tx = {r: per[r]["text"][0] for r in regions}
    sh = {r: per[r]["shuffled"][0] for r in regions}
    content_r = {r: 100 * (tx[r] - sh[r]) / sh[r] for r in regions}
    n_ev = {r: int((df["region"] == r).sum()) for r in regions}

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
        "config": {"domain": "NOAA Storm Events (atmospheric hazards)",
                   "event_type": event_type, "target": "MAGNITUDE (hail diameter, in)",
                   "split": "leave-state-out", "leak_proof_per_fold_pca": True,
                   "strip_size_keywords": strip_size,
                   "seeds": seeds, "baseline": base, "n_regions": len(regions),
                   "n_events": len(df)},
        "no_text": {"mean_rmse": agg(nt)[0], "std_across_regions": agg(nt)[1]},
        "text": {"mean_rmse": agg(tx)[0], "std_across_regions": agg(tx)[1],
                 "per_region": {r: round(tx[r], 3) for r in regions}},
        "shuffled": {"mean_rmse": agg(sh)[0], "std_across_regions": agg(sh)[1],
                     "per_region": {r: round(sh[r], 3) for r in regions}},
        "per_region_content_pct": {r: round(content_r[r], 1) for r in regions},
        "per_region_n": n_ev,
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
    LOG.info("STORM content effect %.1f%% | %s neg | sign-p %s | %s",
             results["deltas"]["content_text_vs_shuffled_pct"],
             results["content_significance"]["n_regions_negative"], sign_p,
             json.dumps(results["deltas"]))
    print(json.dumps(results, indent=2))
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--storm-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--event-type", default="Hail")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--strip-size", action="store_true",
                   help="Robustness control: remove explicit hail-size descriptors "
                        "from narratives before embedding (tests whether content "
                        "survives beyond the literally-stated size).")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run(a.storm_dir, a.out, a.cache_dir, a.event_type, a.seeds, a.strip_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
