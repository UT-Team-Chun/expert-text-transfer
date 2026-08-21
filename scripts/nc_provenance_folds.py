#!/usr/bin/env python
"""Provenance-transfer folds: leave-{contractor,client,year,DTD,project}-out (P-T4).

The strongest archival answer to the logging-provenance objection: if the
text effect were carried by contractor templates, client conventions, survey
eras or DTD-version formatting, it would collapse when entire provenance
groups are held out. The metadata comes from the national header pass
(``scripts/extract_kunijiban_metadata.py``: project 99.9%, contractor 99.9%,
client 99.9%, year 99.8% coverage), joined by exact borehole identity.

Fold families (labels held out one at a time; train = everything else):

- ``contractor``: the top-N contractors by borehole count
- ``client``:     the top-N ordering agencies
- ``year``:       decade bins (<=1979, 1980s, 1990s, 2000s, 2010s, >=2020)
- ``dtd``:        the six DTD versions
- ``project``:    the top-N largest projects

Arms and inference reuse the pre-registered grouped-null engine
(``nc_grouped_null.evaluate_grouped``): rich non-text baseline + KNN prior,
strength-stripped v2 text, borehole-block null with GEOGRAPHIC-region strata
(the fold family changes; the null's strata stay geographic), row-null
alongside, (1+r)/(1+n) per fold. NOTE: this runner reports a Stouffer
combination across the units of a family (``stouffer_p_block``); the paper's
primary combination for the leave-one-region-out estimand is the
Bonferroni-corrected minimum with Cauchy alongside (see
``nc_grouped_null``), because those folds are dependent. Family units here
are far less overlapping, but the Stouffer figure is still a secondary
descriptor and the paper does not quote it.

Bar (prereg P-T4): per family, mean content effect negative AND >=70% of
held-out units negative.

CLI::

    cd backend
    .venv/bin/python -m scripts.nc_provenance_folds \
        --per-region-files 500 --n-perm-block 200 --n-perm-row 50 \
        --out ../docs/research/2026-08-12_provenance_folds_japan.json
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
import pandas as pd

from scripts.nc_grouped_null import (
    REPO,
    _hash_texts,
    _stouffer,
    build_rich_features,
    evaluate_grouped,
)
from scripts.text_leakage_controls import (
    STRIP_VOCAB_VERSION,
    load_domain,
    strip_text,
)
from scripts.uk_transfer_test import embed_texts

LOG = logging.getLogger("nc_provenance_folds")

MIN_FOLD_ROWS = 300  # a held-out unit smaller than this is skipped


def _year_bin(y: float | None) -> str | None:
    if y is None or (isinstance(y, float) and np.isnan(y)):
        return None
    y = int(y)
    if y <= 1979:
        return "<=1979"
    if y >= 2020:
        return ">=2020"
    return f"{(y // 10) * 10}s"


def attach_provenance(df: pd.DataFrame) -> pd.DataFrame:
    meta = pd.read_parquet(
        REPO / "data/features/derived/kunijiban_metadata.parquet",
        columns=["boring_file", "project_key", "orderer_name",
                 "surveyor_name", "survey_year", "dtd_version"])
    meta["boring_file"] = meta["boring_file"].astype(str)
    out = df.copy()
    out["boring_file"] = out["boring_file"].astype(str)
    out = out.merge(meta, on="boring_file", how="left", validate="m:1")
    out["year_bin"] = [_year_bin(v) for v in out["survey_year"]]
    cov = {c: round(100.0 * out[c].notna().mean(), 1)
           for c in ("project_key", "orderer_name", "surveyor_name",
                     "year_bin", "dtd_version")}
    LOG.info("provenance coverage %%: %s", cov)
    return out


def _top_labels(df: pd.DataFrame, col: str, top_n: int) -> list[str]:
    by_boreholes = (
        df.drop_duplicates("boring_file").groupby(col, observed=True)
        .size().sort_values(ascending=False)
    )
    labels = []
    for lab in by_boreholes.index[:top_n]:
        if (df[col] == lab).sum() >= MIN_FOLD_ROWS:
            labels.append(str(lab))
    return labels


def run(out: Path, cache_dir: Path, *, per_region_files: int,
        sample_seed: int, seeds: list[int], n_perm_block: int,
        n_perm_row: int, top_n: int,
        families_filter: list[str] | None = None,
        fold_labels_filter: list[str] | None = None,
        shard: bool = False, list_folds: bool = False) -> dict:
    """Run the provenance-fold families.

    ``families_filter`` and ``fold_labels_filter`` shard the run. The full job
    is 5 families x ~8 held-out units x ~700 model fits, so it is worth
    fanning out one process per (family, unit); as in ``nc_grouped_null``,
    sharding is statistically inert because each permutation seeds its own
    generator from ``100_000 * seed + p``, independent of loop order. The fold
    label set a family offers is computed from the WHOLE frame before
    filtering, so a shard sees the same folds -- and therefore the same
    training complement -- as it would in a full run.
    """
    df, _thin, _ = load_domain("japan", cache_dir,
                               per_region_files=per_region_files,
                               sample_seed=sample_seed)
    df, base = build_rich_features(df, "japan")
    df = attach_provenance(df)
    df = df.rename(columns={"region": "geo_region"})

    texts = [strip_text(t, "japan", "lithology_only") for t in df["text"].tolist()]
    cache = cache_dir / (
        f"prov_japan_lithonly_{STRIP_VOCAB_VERSION}_{_hash_texts(texts)}_e5.npy")
    emb = embed_texts(texts, cache)

    families = {
        "contractor": ("surveyor_name", _top_labels(df, "surveyor_name", top_n)),
        "client": ("orderer_name", _top_labels(df, "orderer_name", top_n)),
        "year": ("year_bin", sorted(x for x in df["year_bin"].dropna().unique())),
        "dtd": ("dtd_version", sorted(x for x in df["dtd_version"].dropna().unique()
                                      if (df["dtd_version"] == x).sum() >= MIN_FOLD_ROWS)),
        "project": ("project_key", _top_labels(df, "project_key", top_n)),
    }

    if list_folds:
        # Emitted so a sharding driver can enumerate (family, fold) pairs
        # without hard-coding them; the label sets depend on the subsample.
        print(json.dumps({k: {"fold_col": v[0], "labels": list(v[1])}
                          for k, v in families.items()},
                         ensure_ascii=False, indent=2))
        return {"families": {k: list(v[1]) for k, v in families.items()}}

    if families_filter is not None:
        unknown = set(families_filter) - set(families)
        if unknown:
            raise SystemExit(f"unknown famil(y/ies) {sorted(unknown)}; "
                             f"available: {sorted(families)}")
        families = {k: v for k, v in families.items() if k in families_filter}

    res: dict = {"families": {}}
    for fam, (col, labels) in families.items():
        if not labels:
            LOG.warning("family %s: no viable fold labels; skipped", fam)
            continue
        LOG.info("family %-10s: %d folds %s", fam, len(labels),
                 labels[:4] + (["..."] if len(labels) > 4 else []))
        sub = df.dropna(subset=[col]).reset_index(drop=True)
        sub_emb = emb[df[col].notna().to_numpy()]
        r = evaluate_grouped(
            sub, base, sub_emb, seeds=seeds,
            n_perm_block=n_perm_block, n_perm_row=n_perm_row,
            use_knn=True, fold_col=col, strata_col="geo_region",
            fold_labels=labels, regions_filter=fold_labels_filter,
            return_parts=shard,
        )
        if shard:
            res["families"][fam] = {
                "fold_col": col,
                "per_fold": r["per_region"],
                "fold_p_block": r["fold_p_block"],
                "fold_p_row": r["fold_p_row"],
            }
            LOG.info("family %-10s: shard wrote %d fold(s) %s", fam,
                     len(r["per_region"]), sorted(r["per_region"]))
            continue
        c = [d["content_pct_block"] for d in r["per_region"].values()]
        res["families"][fam] = {
            "fold_col": col,
            "n_folds": len(r["per_region"]),
            "mean_content_pct_block": round(float(np.mean(c)), 3) if c else None,
            "units_negative": f"{int(np.sum(np.array(c) < 0))}/{len(c)}",
            "stouffer_p_block": r["summary"]["stouffer_p_block"] if c else None,
            "per_fold": r["per_region"],
        }
        LOG.info("family %-10s: mean %+0.2f%%, %s negative", fam,
                 res["families"][fam]["mean_content_pct_block"] or 0.0,
                 res["families"][fam]["units_negative"])

    res["config"] = {
        "per_region_files": per_region_files, "sample_seed": sample_seed,
        "seeds": seeds, "n_perm_block": n_perm_block, "n_perm_row": n_perm_row,
        "top_n": top_n, "min_fold_rows": MIN_FOLD_ROWS,
        "families_filter": families_filter,
        "fold_labels_filter": fold_labels_filter,
        "strip": f"lithology_only {STRIP_VOCAB_VERSION}",
        "prereg": "docs/research/2026-08-11_nc_text_preregistration.md P-T4",
    }
    res["_provenance"] = {
        "purpose": "P-T4 provenance-transfer folds (leave-contractor/client/"
                   "year/DTD/project-out)",
        "metadata": "data/features/derived/kunijiban_metadata.parquet "
                    "(national header pass)",
        "script": "backend/scripts/nc_provenance_folds.py",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    LOG.info("wrote %s", out)
    return res


def combine(shard_dir: Path, out: Path) -> dict:
    """Combine per-(family, fold) shards into the final P-T4 result.

    Per-fold numbers are bit-identical to an unsharded run, so this reproduces
    the sequential output exactly; only the family-level aggregation (mean
    effect, units-negative count, Stouffer p) happens here.
    """
    fams: dict = {}
    config: dict | None = None
    out_r = out.resolve()
    shards = sorted(q for q in shard_dir.glob("*.json")
                    if q.resolve() != out_r)
    if not shards:
        raise SystemExit(f"no shards under {shard_dir}")
    used: list[Path] = []
    skipped: list[str] = []
    for q in shards:
        d = json.loads(q.read_text())
        fams_in = d.get("families")
        # A shard dir also collects incidental JSON (a --list-folds fold map,
        # a previous combined output). Recognise a real shard by its shape
        # rather than by filename, and skip the rest loudly instead of
        # crashing on the first key that is missing.
        if "combined_from_shards" in (d.get("config") or {}):
            # A previous combined output left in the shard directory. Its
            # per_fold blocks are the same folds, so ingesting it would trip
            # the double-count guard on a re-run.
            LOG.warning("skipping %s: already-combined output", q.name)
            continue
        if not isinstance(fams_in, dict) or not all(
                isinstance(v, dict) and "per_fold" in v
                for v in fams_in.values()):
            LOG.warning("skipping %s: not a --shard output", q.name)
            continue
        used.append(q)
        for fam, blk in fams_in.items():
            e = fams.setdefault(fam, {"fold_col": blk["fold_col"],
                                      "per_fold": {}, "fold_p_block": [],
                                      "fold_p_row": []})
            dup = set(blk["per_fold"]) & set(e["per_fold"])
            if dup:
                raise SystemExit(f"family {fam}: fold(s) {sorted(dup)} appear "
                                 f"in more than one shard; refusing to "
                                 f"double-count")
            e["per_fold"].update(blk["per_fold"])
            e["fold_p_block"] += blk.get("fold_p_block", [])
            e["fold_p_row"] += blk.get("fold_p_row", [])
        config = config or d.get("config")

    res: dict = {"families": {}}
    for fam, e in sorted(fams.items()):
        c = [d["content_pct_block"] for d in e["per_fold"].values()]
        res["families"][fam] = {
            "fold_col": e["fold_col"],
            "n_folds": len(e["per_fold"]),
            "mean_content_pct_block": round(float(np.mean(c)), 3) if c else None,
            "units_negative": f"{int(np.sum(np.array(c) < 0))}/{len(c)}",
            "stouffer_p_block": _stouffer(e["fold_p_block"]) if c else None,
            "per_fold": e["per_fold"],
        }
        LOG.info("family %-10s: mean %+0.2f%%, %s negative, Stouffer p=%.4g",
                 fam, res["families"][fam]["mean_content_pct_block"] or 0.0,
                 res["families"][fam]["units_negative"],
                 res["families"][fam]["stouffer_p_block"] or float("nan"))
    res["config"] = config or {}
    if not used:
        raise SystemExit(f"no --shard outputs under {shard_dir} (looked at "
                         f"{len(shards)} JSON file(s))")
    res["config"]["combined_from_shards"] = [q.name for q in used]
    res["_provenance"] = {
        "purpose": "P-T4 provenance-transfer folds (leave-contractor/client/"
                   "year/DTD/project-out), combined from per-fold shards",
        "metadata": "data/features/derived/kunijiban_metadata.parquet "
                    "(national header pass)",
        "sharding": "statistically inert: each permutation seeds its own "
                    "generator from 100_000 * seed + permutation index",
        "script": "backend/scripts/nc_provenance_folds.py",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    LOG.info("wrote %s from %d shards", out, len(used))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path,
                    default=REPO / "data/features/derived/nc_cache")
    ap.add_argument("--per-region-files", type=int, default=500)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-perm-block", type=int, default=200)
    ap.add_argument("--n-perm-row", type=int, default=50)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--families", nargs="+", default=None,
                    help="Restrict to these fold families (sharding).")
    ap.add_argument("--fold-labels", nargs="+", default=None,
                    help="Restrict to these held-out units within the "
                         "selected family (sharding).")
    ap.add_argument("--list-folds", action="store_true",
                    help="Print the (family -> fold labels) map as JSON and "
                         "exit; used to enumerate shards.")
    ap.add_argument("--shard", action="store_true",
                    help="Write raw per-fold parts for later --combine.")
    ap.add_argument("--combine", type=Path, default=None,
                    help="Directory of shard JSONs to combine into --out.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.combine is not None:
        combine(args.combine, args.out)
        return 0
    run(args.out, args.cache_dir, per_region_files=args.per_region_files,
        sample_seed=args.sample_seed, seeds=args.seeds,
        n_perm_block=args.n_perm_block, n_perm_row=args.n_perm_row,
        top_n=args.top_n, families_filter=args.families,
        fold_labels_filter=args.fold_labels, shard=args.shard,
        list_folds=args.list_folds)
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(main())
