#!/usr/bin/env python
"""Embed per-boring KuniJiban soil-text narrative into a fixed-D vector.

Paper B' Pillar 3 embedding layer: reads the CSV produced by
``scripts/extract_soil_text_from_xml.py`` and writes a Parquet keyed by
(lat, lon, file_path) carrying a 768-D (or PCA-reduced) embedding per
boring. Downstream the BoringDataset can join the embedding by (lat,
lon) and the foundation encoder receives 64-128 extra continuous
features carrying the expert geological knowledge.

**Default model: ``cl-nagoya/ruri-v3-310m``** (Tsukagoshi & Sasano 2024,
arXiv:2409.07737). Ruri v3 is the 2024 Japanese sentence-embedding SOTA:

* JMTEB benchmark average **77.24** (Retrieval 81.89, STS 81.22).
  Beats Sarashina-1B (75.50) and OpenAI text-embedding-3-large (73.97)
  while being significantly smaller.
* ModernBERT-Ja backbone with FlashAttention 2 and a 100k SentencePiece
  tokenizer -- no external word segmentation required (good for
  mixed-script geological text with φ / 〜 / half/full-width digits).
* **8192-token context** (vs. 512 for the older xlm-roberta-base).
  Our observation narrative has p99 = 2904 chars (~1500 tokens) so
  no truncation needed for any KuniJiban record.
* Apache 2.0 license, safe for the Paper B' public-artefact release.
* 310M params, ~1.2 GB GPU memory at batch=32 -- comfortable on a
  single H100, ~15-20 minutes on 191k records.

Why Ruri over the alternatives:
  - xlm-roberta-base (278M, 2019): old, 512-token limit, multilingual
    overhead -- replaced.
  - intfloat/multilingual-e5-large (560M): multilingual but worse on
    JMTEB than Ruri v3, larger context but slower.
  - cl-nagoya/sup-simcse-ja-base (110M): predecessor, JMTEB ~72.

Ruri v3's `1+3` prefix scheme:
  - ``""`` (empty)        -> general semantic representation
  - ``"トピック: "``       -> classification / clustering / topic
  - ``"検索クエリ: "``     -> retrieval queries
  - ``"検索文書: "``       -> documents to retrieve

Our soil-text use case is "encoder-feature augmentation" -- not
classification, retrieval, or clustering -- so the empty prefix
(general semantic) is the right choice.

The pipeline:

1. Read CSV (190 k rows, ~166 MB).
2. Drop empty-text rows (~35% of corpus is empty after the extractor).
3. Encode via ``sentence-transformers`` (handles tokenisation +
   pooling internally; the Ruri model card recommends this API over
   raw transformers).
4. Optional PCA reduction to ``--pca-dim`` (default 64) so the on-disk
   parquet stays small and the BoringDataset input-column count stays
   tractable. Variance retained is logged.
5. Write Parquet with columns ``file_path, latitude_deg, longitude_deg,
   embed_0, ..., embed_{D-1}, raw_embed_norm``.

Example::

    cd backend
    uv run python -m scripts.embed_soil_text \\
        --soil-text-csv ../data/features/derived/soil_text.csv \\
        --output ../data/features/derived/soil_text_embed_ruri_pca64.parquet \\
        --model cl-nagoya/ruri-v3-310m --batch-size 32 --pca-dim 64
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("scripts.embed_soil_text")


# Ruri v3 prefix tokens. Empty -> general semantic representation, which is
# what we want for encoder-feature augmentation (NOT retrieval).
_RURI_PREFIX_GENERAL = ""
_RURI_PREFIX_TOPIC = "トピック: "
_RURI_PREFIX_QUERY = "検索クエリ: "
_RURI_PREFIX_DOCUMENT = "検索文書: "


def _resolve_prefix(prefix_mode: str) -> str:
    """Pick a Ruri v3 prefix string. ``general`` returns ``""``."""
    return {
        "general": _RURI_PREFIX_GENERAL,
        "topic": _RURI_PREFIX_TOPIC,
        "query": _RURI_PREFIX_QUERY,
        "document": _RURI_PREFIX_DOCUMENT,
    }[prefix_mode]


_PRECISION_TO_DTYPE = {
    # Resolved lazily inside run() so importing this module never imports torch.
    "fp32": None,
    "fp16": None,
    "bf16": None,
}


def _resolve_dtype(precision: str):
    """Map a precision string to a torch dtype (None for fp32, no cast)."""
    import torch

    return {
        "fp32": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]


def run(
    soil_text_csv: Path,
    output_parquet: Path,
    *,
    model_name: str = "cl-nagoya/ruri-v3-310m",
    batch_size: int = 256,
    pca_dim: int | None = 64,
    device: str = "auto",
    limit: int | None = None,
    prefix_mode: str = "general",
    precision: str = "auto",
    pca_solver: str = "randomized",
    wandb_enabled: bool = False,
    wandb_project: str = "geo-paperB-national",
    wandb_run_name: str | None = None,
) -> int:
    """Embed every non-empty boring / per-layer narrative and write the Parquet.

    Returns the number of embeddings written.

    Performance knobs (H100-tuned defaults):
      * ``batch_size=256``: per-layer median input is ~30 tokens, so a 310M
        encoder fits ~256 sequences comfortably in 80 GB. Sarashina-1B at
        ~3 GB / sample headroom uses ``batch_size=128`` (the cell config sets
        that explicitly).
      * ``precision='auto'`` -> ``bf16`` on CUDA (numerically safer than fp16
        for FlashAttention2 ModernBERT-Ja), ``fp32`` on MPS/CPU. ``bf16`` is
        roughly 2x faster than fp32 on H100 and uses half the memory.
      * ``pca_solver='randomized'``: full SVD on (1.15M, 768) is several
        minutes; randomized SVD targeting just the top-64 components is
        seconds with essentially identical variance retained.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if precision == "auto":
        precision = "bf16" if device == "cuda" else "fp32"
    LOG.info(
        "Device=%s, precision=%s, model=%s, batch_size=%d",
        device, precision, model_name, batch_size,
    )

    wandb_run = None
    if wandb_enabled:
        import wandb

        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name or output_parquet.stem,
            config={
                "stage": "pillar3_text_embedding",
                "model_name": model_name,
                "batch_size": batch_size,
                "pca_dim": pca_dim,
                "device": device,
                "precision": precision,
                "pca_solver": pca_solver,
                "prefix_mode": prefix_mode,
                "soil_text_csv": str(soil_text_csv),
                "output_parquet": str(output_parquet),
                "limit": limit,
            },
        )
        LOG.info("W&B run: %s", wandb_run.url)

    LOG.info("Reading soil-text CSV %s", soil_text_csv)
    df = pd.read_csv(soil_text_csv)
    LOG.info("Loaded %d records", len(df))
    # Two CSV schemas supported:
    #  per_boring: columns include `n_layers`; one row per file; the text
    #              is the per-file concatenated narrative.
    #  per_layer : columns include `layer_idx`, `depth_top_m`,
    #              `depth_bottom_m`; one row per <観察記事> block; the
    #              text is a single layer's narrative.
    # Drop rows whose observation_text is empty / NaN either way.
    df = df[df["observation_text"].notna()].reset_index(drop=True)
    df = df[df["observation_text"].str.len() > 0].reset_index(drop=True)
    if "n_layers" in df.columns:
        # per_boring: also drop rows where n_layers == 0 (text would be "").
        df = df[df["n_layers"] > 0].reset_index(drop=True)
    LOG.info("After empty-text drop: %d records", len(df))
    is_per_layer = "layer_idx" in df.columns and "depth_top_m" in df.columns
    LOG.info("Schema: %s", "per_layer" if is_per_layer else "per_boring")
    if limit is not None:
        df = df.head(limit).reset_index(drop=True)
        LOG.info("Limited to %d records via --limit", len(df))

    prefix = _resolve_prefix(prefix_mode)
    if prefix:
        LOG.info("Ruri prefix mode: %r (text prepended)", prefix_mode)

    LOG.info("Loading sentence-transformer %s", model_name)
    model_kwargs: dict = {}
    dtype = _resolve_dtype(precision)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    # sentence-transformers >=3.0 forwards model_kwargs to the underlying HF
    # AutoModel.from_pretrained, so bf16 weights land on the GPU without a
    # separate .half() / .to(dtype) pass that would have to cast attention
    # masks and rotary buffers too.
    model = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs)
    if device == "cuda":
        # Log starting GPU memory so we can confirm batch_size is well-tuned
        # for the chosen precision (if mem usage stays <50% we left perf
        # on the table).
        torch.cuda.empty_cache()
        LOG.info(
            "CUDA mem after model load: alloc=%.2f GB / reserved=%.2f GB / total=%.2f GB",
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved() / 1e9,
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    texts = (prefix + s for s in df["observation_text"].tolist())
    texts_list = list(texts)  # collect once for sentence-transformers API
    n = len(texts_list)

    t0 = time.time()
    LOG.info("Encoding %d texts in batches of %d", n, batch_size)
    raw_embeds = model.encode(
        texts_list,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # keep raw for PCA; cosine-norm doesn't help PCA
    ).astype(np.float32, copy=False)
    elapsed = time.time() - t0
    rec_per_s = n / max(elapsed, 1.0)
    LOG.info(
        "Encoding done in %.1f s (%.1f rec/s, %d records, embed_dim=%d)",
        elapsed, rec_per_s, n, raw_embeds.shape[1],
    )
    if device == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        LOG.info(
            "Peak CUDA mem during encoding: %.2f / %.2f GB (%.0f%% utilised)",
            peak_alloc, total_mem, 100.0 * peak_alloc / total_mem,
        )
        if wandb_run is not None:
            wandb_run.summary["peak_gpu_mem_gb"] = peak_alloc
            wandb_run.summary["peak_gpu_mem_pct"] = 100.0 * peak_alloc / total_mem

    embed_dim = raw_embeds.shape[1]
    raw_norms = np.linalg.norm(raw_embeds, axis=-1)

    if wandb_run is not None:
        wandb_run.summary["encode_wall_clock_s"] = elapsed
        wandb_run.summary["encode_records_per_s"] = n / max(elapsed, 1.0)
        wandb_run.summary["n_records"] = n
        wandb_run.summary["embed_dim_raw"] = embed_dim
        wandb_run.summary["raw_norm_mean"] = float(raw_norms.mean())
        wandb_run.summary["raw_norm_std"] = float(raw_norms.std())

    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    def _write_parquet(emb: np.ndarray, out_path: Path, d_out: int) -> None:
        cols: dict[str, np.ndarray] = {
            "file_path": df["file_path"].values,
            "latitude_deg": df["latitude_deg"].astype(np.float64).values,
            "longitude_deg": df["longitude_deg"].astype(np.float64).values,
            "raw_embed_norm": raw_norms,
        }
        # Carry per-layer columns through when the input CSV was layered,
        # so the downstream depth-interval join can match each layer's
        # embedding to per-depth boring rows without a second CSV read.
        if is_per_layer:
            cols["layer_idx"] = df["layer_idx"].astype(np.int32).values
            cols["depth_top_m"] = df["depth_top_m"].astype(np.float32).values
            cols["depth_bottom_m"] = df["depth_bottom_m"].astype(np.float32).values
        df_out = pd.DataFrame(cols)
        for k in range(d_out):
            df_out[f"embed_{k}"] = emb[:, k]
        df_out.to_parquet(out_path, index=False)

    # --- Always write the raw full-D embedding -----------------------
    # This lets downstream consumers (encoder ablation, alternative dim
    # reduction, sentence-similarity sanity checks) reuse the same
    # inference output without paying the BERT forward cost again.
    # File stem convention: <output>_full.parquet next to <output>.
    full_path = output_parquet.with_name(output_parquet.stem + "_full.parquet")
    _write_parquet(raw_embeds, full_path, embed_dim)
    LOG.info(
        "Wrote %d rows × %d raw embed cols to %s",
        n, embed_dim, full_path,
    )

    # --- Write the PCA-reduced parquet if requested -----------------
    var_kept = 1.0
    d_out = embed_dim
    if pca_dim is not None and pca_dim < embed_dim:
        from sklearn.decomposition import PCA

        LOG.info(
            "Fitting PCA to %d -> %d dims on %d rows (solver=%s)",
            embed_dim, pca_dim, n, pca_solver,
        )
        # Randomized SVD targets just the top-k singular components and
        # is O(n * d * k) instead of O(n * d^2). On (1.15M, 768) -> 64
        # it's seconds instead of minutes, with variance retained that
        # matches the full SVD to ~4 decimal places.
        t_pca = time.time()
        pca = PCA(
            n_components=int(pca_dim),
            svd_solver=pca_solver,
            random_state=42,
        )
        reduced = pca.fit_transform(raw_embeds).astype(np.float32)
        var_kept = float(pca.explained_variance_ratio_.sum())
        LOG.info("PCA fit done in %.1f s", time.time() - t_pca)
        LOG.info("PCA variance retained: %.3f", var_kept)
        _write_parquet(reduced, output_parquet, int(pca_dim))
        d_out = int(pca_dim)
        LOG.info(
            "Wrote %d rows × %d PCA-reduced embed cols (variance %.3f) to %s",
            n, d_out, var_kept, output_parquet,
        )
    else:
        # No PCA requested: the canonical output path holds the raw
        # full-D parquet (same as <stem>_full.parquet, hardlinked /
        # copied for symmetry of the downstream API).
        _write_parquet(raw_embeds, output_parquet, embed_dim)

    if wandb_run is not None:
        wandb_run.summary["embed_dim_out"] = d_out
        wandb_run.summary["pca_variance_retained"] = var_kept
        wandb_run.summary["output_parquet"] = str(output_parquet)
        wandb_run.summary["output_parquet_full"] = str(full_path)
        wandb_run.finish()
    return n


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--soil-text-csv",
        type=Path,
        default=repo / "data" / "features" / "derived" / "soil_text.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repo
            / "data"
            / "features"
            / "derived"
            / "soil_text_embed_ruri_pca64.parquet"
        ),
    )
    parser.add_argument(
        "--model",
        default="cl-nagoya/ruri-v3-310m",
        help="HuggingFace model id. Default ruri-v3-310m (2024 Japanese SOTA, "
             "JMTEB avg 77.24, Apache 2.0).",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=64,
        help="If > 0, PCA-reduce the raw 768-D embedding to this many dims. "
             "Set to 0 to disable PCA and keep the full 768-D output.",
    )
    parser.add_argument(
        "--pca-solver",
        choices=["randomized", "full", "arpack", "auto"],
        default="randomized",
        help="sklearn PCA svd_solver. `randomized` is ~10-100x faster than "
             "`full` on (1M+, 768) inputs and gives indistinguishable variance.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
        help="Inference precision. `auto` -> bf16 on CUDA, fp32 elsewhere. "
             "bf16 is ~2x faster than fp32 on H100 with FlashAttention2 "
             "(ModernBERT-Ja's preferred path) and uses half the memory.",
    )
    parser.add_argument(
        "--prefix-mode",
        choices=["general", "topic", "query", "document"],
        default="general",
        help="Ruri v3 1+3 prefix scheme. `general` (empty prefix) is the "
             "default for encoder-feature augmentation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap (smoke testing).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases run logging (requires WANDB_API_KEY).",
    )
    parser.add_argument(
        "--wandb-project",
        default="geo-paperB-national",
        help="W&B project. Used only when --wandb is set.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="W&B run name. Defaults to output parquet stem.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s"
    )
    pca_dim = int(args.pca_dim) if args.pca_dim and args.pca_dim > 0 else None
    run(
        args.soil_text_csv,
        args.output,
        model_name=args.model,
        batch_size=int(args.batch_size),
        pca_dim=pca_dim,
        device=args.device,
        limit=args.limit,
        prefix_mode=args.prefix_mode,
        precision=args.precision,
        pca_solver=args.pca_solver,
        wandb_enabled=bool(args.wandb),
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
