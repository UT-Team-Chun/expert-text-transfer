#!/usr/bin/env python
"""Gate-2 leakage control: shuffled-embedding null.

A referee asked whether the per-layer text gain reflects genuine geological
information in the embeddings or merely the presence of 64 extra
correctly-scaled features. This script writes a null-control parquet in which
the ``embed_*`` block is permuted ACROSS rows (each row receives a randomly
chosen other row's embedding) while every other column -- including the
``has_text`` source (``embed_0`` NaN-ness) and the (lat, lon, depth, N,
groundwater) values -- is preserved. Training on the shuffled parquet and
comparing held-out RMSE to the unshuffled v5 run isolates the *content* of the
embeddings from their dimensionality:

* shuffled RMSE ~ no-text RMSE  -> the gain is real semantic signal (good).
* shuffled RMSE ~ v5 RMSE       -> the "gain" was just extra capacity / a
  structured-missingness proxy (the claim collapses).

The permutation is applied as a single row reindex of the whole ``embed_*``
block (not per-column), so each shuffled row carries a *real, internally
consistent* embedding vector from some other borehole -- the strongest form of
the null (it preserves the joint embedding distribution, only breaking the
embedding<->location link). NaN rows (``has_text=0``) are permuted among
themselves and real rows among themselves, so the has_text rate and its
row-alignment are unchanged; otherwise a NaN could land on a real-text row and
silently change which rows count as text-bearing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("shuffle_embeddings")


def _embed_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        (c for c in df.columns if c.startswith("embed_")),
        key=lambda c: int(c.split("_", 1)[1]),
    )


def shuffle_embeddings(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Return a copy of ``df`` with the ``embed_*`` block row-permuted.

    Real-text rows (``embed_0`` finite) are permuted among themselves and
    NaN rows among themselves, so the ``has_text`` indicator stays aligned.
    """
    embed_cols = _embed_cols(df)
    if not embed_cols:
        raise ValueError("no embed_* columns found; is this a v5 parquet?")

    rng = np.random.default_rng(seed)
    out = df.copy()
    emb = df[embed_cols].to_numpy()

    has_text = np.isfinite(df[embed_cols[0]].to_numpy())
    real_idx = np.flatnonzero(has_text)
    nan_idx = np.flatnonzero(~has_text)

    perm = np.arange(len(df))
    if real_idx.size > 1:
        perm[real_idx] = real_idx[rng.permutation(real_idx.size)]
    if nan_idx.size > 1:
        perm[nan_idx] = nan_idx[rng.permutation(nan_idx.size)]

    out[embed_cols] = emb[perm]
    return out


def run(input_parquet: Path, output_parquet: Path, seed: int = 42) -> int:
    df = pd.read_parquet(input_parquet)
    n_real = int(np.isfinite(df[_embed_cols(df)[0]].to_numpy()).sum())
    shuffled = shuffle_embeddings(df, seed=seed)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    shuffled.to_parquet(output_parquet, index=False)
    LOG.info(
        "shuffled %d embed cols over %d rows (%d real-text, seed=%d) -> %s",
        len(_embed_cols(df)), len(df), n_real, seed, output_parquet,
    )
    return len(df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run(args.input_parquet, args.output_parquet, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
