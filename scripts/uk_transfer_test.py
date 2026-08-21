#!/usr/bin/env python
"""Cross-national make-or-break: does the geologist-text content effect transfer to the UK?

Leave-one-region-out over UK macro-regions, HGB (model-agnostic), three feature sets:
  no_text  : [depth, lat, lon]
  text     : + 64-D PCA of multilingual-e5 embeddings of the BS5930 lithology description
  shuffled : + 64-D PCA of the SAME embeddings, row-permuted (content destroyed)

The content effect = text vs shuffled (capacity + any has_text signal cancel, since UK is
~100% text-bearing -> has_text is ~constant, so this is a CLEANER content isolation than Japan).
If text < shuffled under UK leave-region-out, the geological-narrative content transfers across
the national boundary -> "words generalize, coordinates memorize" stands as a principle.

Usage: python -m scripts.uk_transfer_test --parquet <uk.parquet> --out <result.json>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("uk_transfer_test")
_MODEL = "intfloat/multilingual-e5-base"


def embed_texts(texts: list[str], cache: Path | None) -> np.ndarray:
    if cache and cache.exists():
        LOG.info("embedding cache hit %s", cache)
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    LOG.info("loading %s (CPU)", _MODEL)
    m = SentenceTransformer(_MODEL, device="cpu")
    # e5 models expect a prefix; "passage:" is the document/feature side.
    inp = [f"passage: {t}" for t in texts]
    emb = m.encode(inp, batch_size=64, show_progress_bar=True,
                   normalize_embeddings=False).astype(np.float32)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
    return emb


def pca64(emb: np.ndarray, dim: int = 64, seed: int = 42) -> np.ndarray:
    from sklearn.decomposition import PCA
    k = min(dim, emb.shape[1], emb.shape[0])
    p = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    red = p.fit_transform(emb).astype(np.float32)
    LOG.info("PCA -> %dD, variance kept %.3f", k, float(p.explained_variance_ratio_.sum()))
    return red


def _fit_rmse(Xtr, ytr, Xte, yte, seed: int) -> float:
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                      max_depth=None, random_state=seed)
    m.fit(Xtr.astype(np.float64), ytr)
    pred = m.predict(Xte.astype(np.float64))
    return float(np.sqrt(np.mean((pred - yte) ** 2)))


def _evaluate_lro(df: pd.DataFrame, base: list[str], emb: np.ndarray,
                  seeds: list[int], pca_dim: int = 64,
                  arms: tuple[str, ...] = ("no_text", "text", "shuffled")) -> dict:
    """Leave-region-out RMSE with LEAK-PROOF per-fold PCA and a multi-seed null.

    For each held-out region the 64-D PCA basis is fit on the TRAIN regions'
    embeddings ONLY and applied to the held-out region (no held-out information
    enters the basis). The shuffled-embedding null permutes the per-fold
    embeddings independently within train and within test, so content is
    destroyed while feature count / dimensionality / per-fold PCA structure are
    preserved. Each (region, mode) is evaluated over ``seeds`` (HGB random_state
    + shuffle permutation), giving a seed-mean and -std per region.

    ``arms`` restricts which of ``{"no_text", "text", "shuffled"}`` are
    computed (default: all three, matching every existing caller's implicit
    contract). When neither ``"text"`` nor ``"shuffled"`` is requested (e.g.
    an ablation that only needs the ``no_text`` arm, such as the coordinate
    ablation in :func:`run`), the per-fold PCA fit on ``emb`` is skipped
    entirely -- ``emb`` may then be a placeholder with 0 columns.
    """
    from sklearn.decomposition import PCA
    regions = sorted(df["region"].unique())
    Xb = df[base].to_numpy(np.float64)
    y = df["n_value"].to_numpy(np.float64)
    needs_emb = ("text" in arms) or ("shuffled" in arms)
    per = {}  # region -> mode -> list over seeds
    for r in regions:
        te = (df["region"] == r).to_numpy()
        tr = ~te
        if te.sum() < 30 or tr.sum() < 100:
            continue
        if needs_emb:
            k = min(pca_dim, emb.shape[1], int(tr.sum()))
            pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(emb[tr])
            red_tr, red_te = pca.transform(emb[tr]), pca.transform(emb[te])
        modes: dict[str, list[float]] = {m: [] for m in arms}
        for s in seeds:
            if "no_text" in arms:
                modes["no_text"].append(_fit_rmse(Xb[tr], y[tr], Xb[te], y[te], s))
            if "text" in arms:
                modes["text"].append(_fit_rmse(np.hstack([Xb[tr], red_tr]), y[tr],
                                               np.hstack([Xb[te], red_te]), y[te], s))
            if "shuffled" in arms:
                g = np.random.default_rng(s)
                ptr, pte = g.permutation(len(red_tr)), g.permutation(len(red_te))
                modes["shuffled"].append(_fit_rmse(np.hstack([Xb[tr], red_tr[ptr]]), y[tr],
                                                   np.hstack([Xb[te], red_te[pte]]), y[te], s))
        per[r] = {m: (float(np.mean(v)), float(np.std(v))) for m, v in modes.items()}
    return per


def run(parquet: Path, out: Path, cache_dir: Path, seeds: list[int] | None = None,
        no_coords: bool = False, arms: tuple[str, ...] | None = None) -> dict:
    """Run the leave-region-out content-effect harness.

    ``no_coords`` drops ``latitude_deg``/``longitude_deg`` from the base
    feature list (the W2 ±coords ablation:
    ``docs/research/2026-07-09_nmi_universality_preregistration.md`` P-W2).
    ``arms`` restricts which of ``{"no_text", "text", "shuffled"}`` are
    computed; when neither ``"text"`` nor ``"shuffled"`` is requested the
    (expensive) sentence-embedding step is skipped entirely -- this is the
    mode the ±coords ablation uses, since it only needs ``no_text``.
    """
    seeds = seeds or [42, 43, 44, 45, 46]
    arms = tuple(arms) if arms else ("no_text", "text", "shuffled")
    df = pd.read_parquet(parquet)
    counts = df["region"].value_counts()
    keep = counts[counts >= 200].index.tolist()
    df = df[df["region"].isin(keep)].reset_index(drop=True)
    df = df[(df["n_value"] > 0) & (df["n_value"] <= 100)].reset_index(drop=True)
    df = df[df["lith_desc"].str.len() > 0].reset_index(drop=True)
    LOG.info("UK rows %d, regions %s (seeds=%s, arms=%s, no_coords=%s, per-fold PCA)", len(df),
             df["region"].value_counts().to_dict(), seeds, arms, no_coords)

    needs_emb = ("text" in arms) or ("shuffled" in arms)
    emb = (embed_texts(df["lith_desc"].tolist(), cache_dir / "uk_e5_emb.npy") if needs_emb
           else np.zeros((len(df), 0), dtype=np.float32))
    base_candidates = ["depth_from_surface", "ground_level", "latitude_deg", "longitude_deg"]
    if no_coords:
        base_candidates = [c for c in base_candidates if c not in ("latitude_deg", "longitude_deg")]
    base = [c for c in base_candidates if c in df.columns]
    per = _evaluate_lro(df, base, emb, seeds, arms=arms)

    regions = sorted(per.keys())
    # per-region seed-means per mode
    nt = {r: per[r]["no_text"][0] for r in regions} if "no_text" in arms else {}
    tx = {r: per[r]["text"][0] for r in regions} if "text" in arms else {}
    sh = {r: per[r]["shuffled"][0] for r in regions} if "shuffled" in arms else {}
    n_uk = {r: int((df["region"] == r).sum()) for r in regions}

    def agg(d):
        vals = list(d.values())
        return round(float(np.mean(vals)), 3), round(float(np.std(vals)), 3)

    results: dict = {
        "config": {"leak_proof_per_fold_pca": needs_emb, "seeds": seeds, "pca_dim": 64,
                   "baseline": base, "n_regions": len(regions), "no_coords": no_coords,
                   "arms": list(arms)},
        "per_region_n_spt": n_uk,
        "per_region_seed_std": {r: {m: round(per[r][m][1], 3) for m in arms} for r in regions},
    }
    if "no_text" in arms:
        results["no_text"] = {
            "mean_rmse": agg(nt)[0], "std_across_regions": agg(nt)[1],
            "per_region": {r: round(nt[r], 3) for r in regions},
        }
    if "text" in arms:
        results["text"] = {"mean_rmse": agg(tx)[0], "std_across_regions": agg(tx)[1],
                            "per_region": {r: round(tx[r], 3) for r in regions}}
    if "shuffled" in arms:
        results["shuffled"] = {"mean_rmse": agg(sh)[0], "std_across_regions": agg(sh)[1],
                                "per_region": {r: round(sh[r], 3) for r in regions}}

    # content-effect deltas / significance only make sense when both the text
    # and shuffled arms were actually computed (the default, full-content run).
    if "text" in arms and "shuffled" in arms:
        content_r = {r: 100 * (tx[r] - sh[r]) / sh[r] for r in regions}  # per-region content %
        diffs = [tx[r] - sh[r] for r in regions]
        n_neg = sum(d < 0 for d in diffs)
        from math import comb
        sign_p = sum(comb(len(diffs), k) for k in range(n_neg, len(diffs) + 1)) / 2 ** len(diffs)
        wilcox_p = None
        try:
            from scipy.stats import wilcoxon
            wilcox_p = float(wilcoxon(diffs, alternative="less").pvalue)
        except Exception:  # noqa: BLE001 (scipy optional)
            pass
        results["per_region_content_pct"] = {r: round(content_r[r], 1) for r in regions}
        deltas = {"content_text_vs_shuffled_pct": round(100 * (agg(tx)[0] - agg(sh)[0]) / agg(sh)[0], 1)}
        if "no_text" in arms:
            deltas["text_vs_notext_pct"] = round(100 * (agg(tx)[0] - agg(nt)[0]) / agg(nt)[0], 1)
            deltas["shuffled_vs_notext_pct"] = round(100 * (agg(sh)[0] - agg(nt)[0]) / agg(nt)[0], 1)
        results["deltas"] = deltas
        results["content_significance"] = {
            "n_regions_negative": f"{n_neg}/{len(diffs)}",
            "sign_test_p_one_sided": round(sign_p, 4),
            "wilcoxon_p_one_sided": (round(wilcox_p, 4) if wilcox_p is not None else None),
        }
        LOG.info("LEAK-PROOF content effect %.1f%% | %s neg | sign-p %.4f | %s",
                 deltas["content_text_vs_shuffled_pct"],
                 results["content_significance"]["n_regions_negative"], sign_p,
                 json.dumps(deltas))
    else:
        LOG.info("no_coords=%s arms=%s -> no_text mean RMSE %s", no_coords, arms,
                  results.get("no_text", {}).get("mean_rmse"))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument(
        "--no-coords", action="store_true",
        help="Drop latitude_deg/longitude_deg from the base feature list "
        "(W2 +-coords ablation, docs/research/2026-07-09_nmi_universality_preregistration.md P-W2).",
    )
    p.add_argument(
        "--arms", default="no_text,text,shuffled",
        help="Comma-separated subset of {no_text,text,shuffled} to evaluate. "
        "Restricting to 'no_text' skips the sentence-embedding step entirely.",
    )
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s")
    arms = tuple(s.strip() for s in a.arms.split(",") if s.strip())
    unknown_arms = set(arms) - {"no_text", "text", "shuffled"}
    if unknown_arms:
        raise SystemExit(f"Unknown --arms {sorted(unknown_arms)}; choose from no_text,text,shuffled")
    run(a.parquet, a.out, a.cache_dir, a.seeds, no_coords=a.no_coords, arms=arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
