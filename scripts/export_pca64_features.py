#!/usr/bin/env python
"""Export the per-fold PCA-64 text features that the Zenodo bundle ships.

Why this script exists
----------------------
``data_availability.tex`` rests the legal case for redistributing a
derivative of the KuniJiban corpus -- which MLIT's terms forbid us to
redistribute raw -- on one specific property: a frozen language model's output
*projected to 64 principal components* is a non-invertible derived
representation. That argument covers PCA-64 features. It does **not** cover the
full 768-dimensional sentence embeddings, which are what the evaluation
pipeline actually caches on disk, and from which sentence-embedding inversion
can recover a meaningful fraction of the source text.

The pipeline never persisted the PCA-64 features: ``uk_transfer_test``
``_evaluate_lro`` fits ``PCA(n_components=64, random_state=0)`` on each fold's
*training* rows and throws the basis away when the fold ends. So the artefact
the manuscript promises did not exist. This script produces it.

What it guarantees
------------------
The embedding cache name in every released grouped-null artefact is
content-addressed -- ``grouped_<domain>_<tag>_<sha256(texts)[:16]>_e5.npy`` --
so the hash pins the exact text set that produced the reported numbers. This
script recomputes that hash and, when ``--expect-text-hash`` is given, refuses
to write anything unless it matches. That turns "these features correspond to
the published result" from a claim into a check.

The per-fold PCA is reproduced exactly as the evaluator does it: for held-out
region ``r``, ``PCA(n_components=64, svd_solver="randomized", random_state=0)``
fit on the rows of every *other* region, then applied to all rows. Fitting on
train rows only is what makes the released features leak-proof; shipping the
transform of all rows is what lets a referee refit the downstream model.

Usage
-----
    cd backend
    .venv/bin/python -m scripts.export_pca64_features \
        --domain japan --per-region-files 0 \
        --expect-text-hash 22c29123b9f2a14e \
        --device cuda \
        --out ../data/release/pca64/japan_fullpop
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger("scripts.export_pca64_features")

REPO = Path(__file__).resolve().parents[2]
PCA_DIM = 64
#: Must match ``uk_transfer_test._evaluate_lro``; changing it silently would
#: make the released features stop corresponding to the published pipeline.
PCA_RANDOM_STATE = 0
PCA_SVD_SOLVER = "randomized"
_MODEL = "intfloat/multilingual-e5-base"

#: Filled in by :func:`embed` so the manifest can state whether the
#: embedding was recomputed here or reused from a cached array.
_PROVENANCE: dict[str, str] = {}

#: Columns written to ``keys.parquet``, in feature-row order.
#:
#: Without these the released features are anonymous 64-vectors: a referee
#: could not tell which rows form a held-out fold, had no target to regress,
#: and no borehole identifier to block the permutation null or the BCa
#: bootstrap on. Every column here is already public in
#: ``borings_japan_v4id.parquet`` under CC-BY-4.0 and none of them is narrative
#: text, so shipping them leaves the PCA-64 non-invertibility argument -- the
#: basis on which a derivative of the restricted corpus is redistributable --
#: exactly as it was.
#:
#: Together with the PCA-64 features this is the complete input to the
#: evaluation: the one-hot blocks the model consumes are derived from the three
#: code columns, and the per-fold KNN prior is derived from the coordinates,
#: ``n_value`` and ``boring_file``.
KEY_COLUMNS = [
    "boring_file",            # borehole identity: the block unit for the null
    "region",                 # the leave-region-out fold label
    "n_value",                # the regression target
    "latitude_deg", "longitude_deg",   # inputs to the per-fold KNN prior
    "depth_from_surface", "absolute_elevation",
    "river_distance_km", "coast_distance_km",
    "regime_code", "aist_litho_macro_code", "aist_era_code",
]


def _sha256_array(a: np.ndarray) -> str:
    """Hash the embedding matrix itself, not just the features derived from it.

    The manifest already hashes each PCA output, which detects a corrupted
    feature file. It does not detect a feature file that is intact but was
    projected from a differently-derived input.
    """
    h = hashlib.sha256()
    step = 100_000
    for i in range(0, a.shape[0], step):
        h.update(np.ascontiguousarray(a[i:i + step]).tobytes())
    return h.hexdigest()


def hash_texts(texts: list[str]) -> str:
    """Byte-identical to ``nc_grouped_null._hash_texts``."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def embed(texts: list[str], cache: Path | None, device: str,
          batch_size: int = 64, chunk: int = 20_000) -> np.ndarray:
    """Embed with multilingual-e5.

    ``uk_transfer_test.embed_texts`` pins ``device="cpu"``, which is fine for
    the 18k-53k row corpora it was written for and slow for 1.3M. A GPU/MPS
    encode is NOT bit-identical to the CPU one -- measured max absolute
    difference 7.9e-7 over 3,000 real texts, with no row identical -- so the
    device is recorded in the manifest and ``--device cpu`` is what reproduces
    the published arrays. The
    ``passage:`` prefix and the un-normalised output are kept exactly as the
    original, because the downstream PCA is not scale-invariant.
    """
    if cache and cache.exists():
        LOG.info("embedding cache hit: %s", cache)
        _PROVENANCE["embedding_origin"] = f"cache hit: {cache.name}"
        return np.load(cache, mmap_mode="r")
    from sentence_transformers import SentenceTransformer

    LOG.info("loading %s on %s", _MODEL, device)
    m = SentenceTransformer(_MODEL, device=device)
    inp = [f"passage: {t}" for t in texts]

    # Encode in chunks straight into a disk-backed array rather than letting
    # sentence-transformers accumulate every batch in RAM. On the full
    # Japanese population that accumulation is 1,298,728 x 768 x 4 B = 4.0 GB
    # held live for the whole run, which on a memory-pressured machine turns a
    # GPU-bound job into a swap-bound one. Chunking keeps peak resident memory
    # at one chunk and makes the job restartable-in-principle. The encoder is
    # deterministic per input, so chunking cannot change a single value.
    _PROVENANCE["embedding_origin"] = (
        f"computed here on device={device}, batch_size={batch_size}")
    if cache is None:
        return m.encode(inp, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=False).astype(np.float32)

    cache.parent.mkdir(parents=True, exist_ok=True)
    dim = int(m.get_sentence_embedding_dimension())
    tmp = cache.with_suffix(".partial.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                    shape=(len(inp), dim))
    done = 0
    for start in range(0, len(inp), chunk):
        piece = inp[start:start + chunk]
        out[start:start + len(piece)] = m.encode(
            piece, batch_size=batch_size, show_progress_bar=False,
            normalize_embeddings=False).astype(np.float32)
        done += len(piece)
        if (start // chunk) % 5 == 0 or done == len(inp):
            LOG.info("embedded %d/%d (%.1f%%)", done, len(inp),
                     100.0 * done / len(inp))
    out.flush()
    del out
    tmp.replace(cache)
    LOG.info("wrote embedding cache %s", cache)
    return np.load(cache, mmap_mode="r")


def write_keys(df, out: Path) -> dict:
    """Write the row key table that makes the features usable.

    Row order is the feature row order, and that is asserted rather than
    assumed: a silent misalignment here would not crash anything, it would
    just make every released number wrong.
    """
    import pandas as pd

    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"key columns absent from the frame: {missing}")
    keys = df.loc[:, KEY_COLUMNS].reset_index(drop=True)
    path = out / "keys.parquet"
    out.mkdir(parents=True, exist_ok=True)
    keys.to_parquet(path, index=False)
    rec = {
        "file": path.name,
        "columns": list(keys.columns),
        "n_rows": int(len(keys)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "note": ("Row i of every pca64_heldout_*.npy is row i of this table. "
                 "The one-hot regime / lithology / era blocks the model "
                 "consumes are derived from the three code columns; the "
                 "per-fold KNN prior is derived from the coordinates, "
                 "n_value and boring_file."),
    }
    LOG.info("keys.parquet: %d rows x %d cols", len(keys), len(keys.columns))
    return rec


def per_fold_pca64(emb: np.ndarray, regions: np.ndarray,
                   out: Path) -> list[dict]:
    """One PCA-64 basis per held-out region, fit on the other regions.

    Returns one manifest record per fold. The basis itself is deliberately
    NOT written: it would let a holder of the features move back toward the
    768-dimensional space the redistribution argument excludes.
    """
    from sklearn.decomposition import PCA

    out.mkdir(parents=True, exist_ok=True)
    # A surviving manifest from an earlier, differently-shaped run would keep
    # its stale n_rows and sha256 and make a half-written export look intact.
    stale = out / "manifest.json"
    if stale.exists():
        stale.unlink()
    records = []
    for r in sorted(set(regions.tolist())):
        te = regions == r
        tr = ~te
        k = min(PCA_DIM, emb.shape[1], int(tr.sum()))
        if k < PCA_DIM:
            raise ValueError(
                f"fold {r}: only {tr.sum()} training rows, cannot fit "
                f"{PCA_DIM} components")
        # ``emb[tr]`` materialises the training rows; that copy is what the
        # evaluator fits on too, so it is required for an identical basis.
        # The transform, however, is chunked and written straight to disk --
        # holding a second full-size float array would double peak memory for
        # no benefit.
        pca = PCA(n_components=k, svd_solver=PCA_SVD_SOLVER,
                  random_state=PCA_RANDOM_STATE).fit(np.asarray(emb[tr]))
        path = out / f"pca64_heldout_{r}.npy"
        # Write under a temporary name and rename only once the array is
        # complete. open_memmap creates a full-size file with a VALID header
        # before any value is computed, so a killed export used to leave a
        # final-named file that loads cleanly with a zero tail -- which the
        # bundle would then hash and certify as OK.
        tmp = path.with_suffix(".partial.npy")
        feats = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                          shape=(emb.shape[0], k))
        step = 100_000
        for s0 in range(0, emb.shape[0], step):
            feats[s0:s0 + step] = pca.transform(
                np.asarray(emb[s0:s0 + step])).astype(np.float32)
        feats.flush()
        shape = list(feats.shape)
        var_kept = float(pca.explained_variance_ratio_.sum())
        del feats, pca
        tmp.replace(path)
        rec = {
            "held_out_region": str(r),
            "file": path.name,
            "shape": shape,
            "n_train_rows_pca_was_fit_on": int(tr.sum()),
            "n_heldout_rows": int(te.sum()),
            "explained_variance_ratio_sum": var_kept,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        records.append(rec)
        LOG.info("fold %-16s train=%d test=%d var=%.4f -> %s",
                 r, tr.sum(), te.sum(), rec["explained_variance_ratio_sum"],
                 path.name)
    return records


def run(domain: str, out: Path, *, per_region_files: int, sample_seed: int,
        strip_mode: str, device: str, cache_dir: Path,
        expect_text_hash: str | None, batch_size: int,
        keys_only: bool = False) -> dict:
    from scripts.nc_grouped_null import build_rich_features
    from scripts.text_leakage_controls import apply_strip_mode, cache_tag
    from scripts.text_leakage_controls import load_domain

    # This sequence mirrors scripts.nc_grouped_null.run() exactly, and the
    # order is load-bearing. ``build_rich_features`` DROPS rows (55 of
    # 1,298,783 on the Japanese full population, concentrated in three
    # regions), so hashing the texts straight out of ``load_domain`` yields a
    # different text set from the one behind every published number. The
    # content-addressed cache name is what makes that mistake detectable
    # rather than silent -- see ``--expect-text-hash``.
    _PROVENANCE.clear()   # never inherit an earlier call's provenance
    df, _thin_base, _ = load_domain(domain, cache_dir,
                                    per_region_files=per_region_files,
                                    sample_seed=sample_seed)
    n_loaded = len(df)
    df, _base = build_rich_features(df, domain)
    LOG.info("build_rich_features: %d -> %d rows (%d dropped)",
             n_loaded, len(df), n_loaded - len(df))
    texts, strip_stats = apply_strip_mode(
        df["text"].tolist(), domain, strip_mode,
        boring_files=df["boring_file"].astype(str).tolist())

    th = hash_texts(texts)
    LOG.info("text set: %d rows, sha256[:16]=%s", len(texts), th)
    if expect_text_hash and th != expect_text_hash:
        raise SystemExit(
            f"text hash {th} != expected {expect_text_hash}: these texts are "
            "NOT the ones behind the published result, so the exported "
            "features would not correspond to it. Refusing to write.")

    if keys_only:
        # Attach the key table to an export whose features already exist,
        # without paying for the embedding again. The hash gate above still
        # ran, so the keys provably describe the same corpus as the features.
        mpath = out / "manifest.json"
        if not mpath.exists():
            raise SystemExit(f"--keys-only needs an existing export at {out}")
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if manifest.get("text_sha256_16") != th:
            raise SystemExit(
                f"existing export was built from text set "
                f"{manifest.get('text_sha256_16')}, not {th}; refusing to "
                "attach a key table that does not describe its rows")
        if manifest.get("n_rows") != len(df):
            raise SystemExit(
                f"existing export has {manifest.get('n_rows')} rows, frame has "
                f"{len(df)}")
        manifest["keys"] = write_keys(df, out)
        mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                         encoding="utf-8")
        LOG.info("attached keys.parquet to the existing export at %s", out)
        return manifest

    cache = cache_dir / (f"grouped_{domain}_{cache_tag(strip_mode)}_"
                         f"{th}_e5.npy")
    emb = embed(texts, cache, device, batch_size=batch_size)
    if emb.shape[0] != len(texts):
        raise SystemExit(f"embedding rows {emb.shape[0]} != texts {len(texts)}")

    if len(df) != emb.shape[0]:
        raise SystemExit(f"frame rows {len(df)} != embedding rows {emb.shape[0]}")
    keys_rec = write_keys(df, out)
    records = per_fold_pca64(emb, df["region"].to_numpy(), out)

    manifest = {
        "purpose": ("Per-fold PCA-64 text features -- the redistributable "
                    "derived representation named in the Data availability "
                    "statement. The 768-dimensional embeddings they are "
                    "projected from are NOT released for this domain."),
        "domain": domain,
        "n_rows": int(len(texts)),
        "n_boreholes": int(df["boring_file"].nunique()),
        "population": ("full text-bearing population" if per_region_files <= 0
                       else f"{per_region_files} boreholes/region subsample"),
        "sample_seed": sample_seed,
        "strip_mode": strip_mode,
        "strip_stats": strip_stats,
        "text_sha256_16": th,
        "embedding_model": _MODEL,
        "embedding_cache_name": cache.name,
        # The device matters: every published number went through
        # uk_transfer_test.embed_texts, which pins device="cpu". A GPU/MPS
        # re-embed agrees only to fp32 rounding (~8e-7 absolute, measured),
        # and the downstream gradient-boosted splits are discrete, so a
        # re-projection from a different device can move a per-region RMSE in
        # its third significant figure. Recording it makes that auditable
        # instead of invisible.
        # What ACTUALLY produced the array, not the flag that was passed. On a
        # cache hit nothing is embedded here, so recording the requested device
        # would assert a provenance this run did not establish. The published
        # arrays were computed on an amd64 Linux node under
        # uk_transfer_test.embed_texts (CPU-pinned); re-embedding elsewhere
        # agrees only to fp32 rounding, so the sha256 is the thing to check.
        "embedding_origin": _PROVENANCE.get("embedding_origin", "unknown"),
        "embedding_device_requested": device,
        "embedding_sha256": _sha256_array(emb),
        "pca": {"n_components": PCA_DIM, "svd_solver": PCA_SVD_SOLVER,
                "random_state": PCA_RANDOM_STATE,
                "fit_on": "training regions only, per held-out fold",
                "basis_released": False},
        "keys": keys_rec,
        "folds": records,
        "_provenance": {
            "script": "backend/scripts/export_pca64_features.py",
            "matches_evaluator": ("scripts/uk_transfer_test._evaluate_lro "
                                  "fits the identical PCA per fold"),
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("wrote %d folds + manifest to %s", len(records), out)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", default="japan", choices=["japan", "uk"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-region-files", type=int, default=0,
                    help="0 = full text-bearing population")
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--strip-mode", default="lithology_only")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cache-dir", type=Path,
                    default=REPO / "data/features/derived/nc_cache")
    ap.add_argument("--keys-only", action="store_true",
                    help="attach keys.parquet to an existing export without "
                         "re-embedding; the text-hash gate still applies")
    ap.add_argument("--expect-text-hash", default=None,
                    help="refuse to write unless the text set hashes to this")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    run(a.domain, a.out, per_region_files=a.per_region_files,
        sample_seed=a.sample_seed, strip_mode=a.strip_mode, device=a.device,
        cache_dir=a.cache_dir, expect_text_hash=a.expect_text_hash,
        batch_size=a.batch_size, keys_only=a.keys_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
