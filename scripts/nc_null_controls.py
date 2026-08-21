#!/usr/bin/env python
"""NC revision — the [BLOCKER] control: stratified shuffled-embedding nulls.

The published content effect uses a GLOBAL row-permutation null: (text - global_shuffle).
A reviewer (NC, ML lens) objects that the global shuffle destroys ALL embedding<->row
structure, so it credits as "content" both (a) genuine fine-grained lithology->strength
information AND (b) the embedding merely acting as a soft geology/region CLUSTER INDEX whose
out-of-region usefulness is distributional cluster-matching -- the same mechanism the paper
indicts as "coordinate memorization", re-expressed in lithology space.

Fix: STRATIFIED nulls that permute embeddings only WITHIN a stratum, preserving the
stratum<->embedding-distribution mapping (the cluster-ID information) while destroying
fine within-stratum row detail:
  - global        : permute across all rows           -> (text-global)        = total embedding signal (the published headline)
  - within_region : permute within training-region    -> (text-within_region) = signal beyond region-cluster identity
  - within_class  : permute within coarse litho class  -> (text-within_class)  = signal beyond a coarse lithology LABEL

If the content effect SURVIVES under within_class (and within_region), the transfer is
genuinely fine-grained cross-distributional content, not cluster-matching -> the causal claim
is airtight. If it COLLAPSES toward the dictionary-one-hot level, much of the headline "content"
was a lithology cluster index, and the headline must come down.

Also addresses the [MAJOR] "null as one draw per seed" point: each null is estimated as a
distribution over ``--n-perm`` permutations (permutation p-value), retiring the degenerate
combinatorial sign test.

Run (CPU, ~1-1.5 h japan, ~20 min uk):
  cd backend
  uv run python -m scripts.nc_null_controls --domain japan --representation lithology_only \
      --out ../docs/research/2026-06-24_within_region_null_japan.json --n-perm 100
  uv run python -m scripts.nc_null_controls --domain uk --representation lithology_only \
      --out ../docs/research/2026-06-24_within_region_null_uk.json --n-perm 100
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.text_leakage_controls import load_domain, strip_text
from scripts.uk_transfer_test import _fit_rmse, embed_texts

LOG = logging.getLogger("nc_null_controls")
REPO = Path(__file__).resolve().parents[2]

# --- coarse lithology class derived from the description text (the within_class stratum).
# This is deliberately the COARSE label the embedding might merely be re-encoding; the
# within_class null asks whether the embedding carries information beyond it.
_JP_CLASS = [  # (label, regex) — first match wins, coarse-to-fine priority
    ("rock", re.compile(r"岩|礫岩|砂岩|泥岩|凝灰|花崗|玄武|安山")),
    ("organic", re.compile(r"腐植|有機|泥炭|ピート")),
    ("gravel", re.compile(r"礫|れき|玉石|砂礫")),
    ("sand", re.compile(r"砂|細砂|中砂|粗砂")),
    ("silt", re.compile(r"シルト")),
    ("clay", re.compile(r"粘土|粘性|シルト質粘土|ねんど")),
    ("loam", re.compile(r"ローム|火山灰|関東ローム")),
    ("mud", re.compile(r"泥|どろ")),
]
_UK_CLASSES = ["CLAY", "SILT", "SAND", "GRAVEL", "PEAT", "COBBLES", "BOULDERS", "CHALK",
               "MUDSTONE", "SANDSTONE", "SILTSTONE", "LIMESTONE", "TILL"]
_uk_cls_re = re.compile(r"\b(" + "|".join(_UK_CLASSES) + r")\b")


def _coarse_class(texts: list[str], domain: str) -> np.ndarray:
    """Coarse primary-lithology label per row (the within_class stratum)."""
    out = []
    if domain == "uk":
        for t in texts:
            # BS5930 capitalises ONLY the principal soil/rock; match the ORIGINAL text so
            # lowercase descriptive mentions (e.g. "of sandstone") do not masquerade as the
            # principal noun. The principal is conventionally the last CAPS lithology word.
            ms = _uk_cls_re.findall(t or "")
            out.append(ms[-1] if ms else "other")
    else:
        for t in texts:
            lab = "other"
            for name, rx in _JP_CLASS:
                if rx.search(t or ""):
                    lab = name
                    break
            out.append(lab)
    return np.asarray(out)


def _strat_perm(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permutation index that shuffles ONLY within each stratum label."""
    out = np.arange(len(strata))
    for lab in np.unique(strata):
        m = np.where(strata == lab)[0]
        if m.size > 1:
            out[m] = rng.permutation(m)
    return out


def _evaluate_stratified_nulls(df: pd.DataFrame, base: list[str], emb: np.ndarray,
                               cls: np.ndarray, n_perm: int, hgb_seed: int,
                               text_seeds: list[int], pca_dim: int = 64) -> dict:
    from sklearn.decomposition import PCA
    regions = sorted(df["region"].unique())
    region_arr = df["region"].to_numpy()
    Xb = df[base].to_numpy(np.float64)
    y = df["n_value"].to_numpy(np.float64)
    per: dict[str, dict] = {}
    for r in regions:
        te = (region_arr == r)
        tr = ~te
        if te.sum() < 30 or tr.sum() < 100:
            continue
        k = min(pca_dim, emb.shape[1], int(tr.sum()))
        pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(emb[tr])
        red_tr, red_te = pca.transform(emb[tr]), pca.transform(emb[te])
        reg_tr = region_arr[tr]
        cls_tr, cls_te = cls[tr], cls[te]
        # anchors (a few HGB seeds)
        no_text = [_fit_rmse(Xb[tr], y[tr], Xb[te], y[te], s) for s in text_seeds]
        text = [_fit_rmse(np.hstack([Xb[tr], red_tr]), y[tr],
                          np.hstack([Xb[te], red_te]), y[te], s) for s in text_seeds]
        text_mean = float(np.mean(text))
        nulls = {"global": [], "within_region": [], "within_class": []}
        for p in range(n_perm):
            rng = np.random.default_rng(1000 * hgb_seed + p)
            # train-side permutations
            ptr_g = rng.permutation(len(red_tr))
            ptr_r = _strat_perm(reg_tr, rng)
            ptr_c = _strat_perm(cls_tr, rng)
            # test-side: test is one region, so within_region == global; within_class strat.
            pte_g = rng.permutation(len(red_te))
            pte_c = _strat_perm(cls_te, rng)
            for name, ptr, pte in (("global", ptr_g, pte_g),
                                   ("within_region", ptr_r, pte_g),
                                   ("within_class", ptr_c, pte_c)):
                nulls[name].append(_fit_rmse(np.hstack([Xb[tr], red_tr[ptr]]), y[tr],
                                             np.hstack([Xb[te], red_te[pte]]), y[te], hgb_seed))
        per[r] = {
            "n_test": int(te.sum()), "n_train": int(tr.sum()),
            "no_text": float(np.mean(no_text)), "text": text_mean, "text_sd": float(np.std(text)),
            "nulls": {name: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                             "content_pct": round(100 * (text_mean - float(np.mean(v))) / float(np.mean(v)), 2),
                             "perm_p": round(float(np.mean(np.asarray(v) <= text_mean)), 4)}
                      for name, v in nulls.items()},
        }
    return per


def _aggregate(per: dict) -> dict:
    regions = sorted(per.keys())
    out = {"n_regions": len(regions), "per_region": per, "summary": {}}
    for name in ("global", "within_region", "within_class"):
        cps = [per[r]["nulls"][name]["content_pct"] for r in regions]
        n_neg = sum(c < 0 for c in cps)
        # pooled permutation p: mean over regions of per-region perm_p
        pps = [per[r]["nulls"][name]["perm_p"] for r in regions]
        out["summary"][name] = {
            "mean_content_pct": round(float(np.mean(cps)), 2),
            "regions_negative": f"{n_neg}/{len(regions)}",
            "per_region_content_pct": {r: per[r]["nulls"][name]["content_pct"] for r in regions},
            "mean_perm_p": round(float(np.mean(pps)), 4),
        }
    return out


def run(domain: str, representation: str, out: Path, cache_dir: Path,
        n_perm: int, text_seeds: list[int]) -> dict:
    # canonical loader (handles uk + japan; sets df["text"], the 500/region JP subsample)
    df, base, _ = load_domain(domain, cache_dir)
    raw = df["text"].tolist()
    # representation: strip strength/N for lithology_only (the headline rep)
    if representation == "lithology_only":
        texts = [strip_text(t, domain, "lithology_only") for t in raw]
    else:
        texts = raw
    cls = _coarse_class(raw, domain)
    LOG.info("%s: %d rows, %d regions, base=%s, classes=%s", domain, len(df),
             df["region"].nunique(), base, dict(zip(*np.unique(cls, return_counts=True))))
    from scripts.text_leakage_controls import STRIP_VOCAB_VERSION
    ver = "" if representation == "full" else f"_{STRIP_VOCAB_VERSION}"
    emb = embed_texts(texts, cache_dir / f"ncnull_{domain}_{representation}{ver}_e5.npy")
    per = _evaluate_stratified_nulls(df, base, emb, cls, n_perm, hgb_seed=42, text_seeds=text_seeds)
    res = _aggregate(per)
    res["config"] = {"domain": domain, "representation": representation, "n_rows": len(df),
                     "n_perm": n_perm, "text_seeds": text_seeds, "base": base,
                     "class_strata": sorted(set(cls.tolist()))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    s = res["summary"]
    LOG.info("%s content%%: global %.1f (%s) | within_region %.1f (%s) | within_class %.1f (%s)",
             domain, s["global"]["mean_content_pct"], s["global"]["regions_negative"],
             s["within_region"]["mean_content_pct"], s["within_region"]["regions_negative"],
             s["within_class"]["mean_content_pct"], s["within_class"]["regions_negative"])
    print(json.dumps(res["summary"], indent=2))
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["japan", "uk"], required=True)
    p.add_argument("--representation", choices=["lithology_only", "full"], default="lithology_only")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=REPO / "data/features/derived")
    p.add_argument("--n-perm", type=int, default=100)
    p.add_argument("--text-seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run(a.domain, a.representation, a.out, a.cache_dir, a.n_perm, a.text_seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
