"""Tests for the Gate-2 shuffled-embedding null (scripts.shuffle_embeddings).

The shuffled-embedding control is only a valid null if it (a) actually breaks
the embedding<->row link, (b) preserves every non-embedding column verbatim,
and (c) keeps the has_text indicator (embed_0 NaN-ness) aligned to its
original row -- otherwise a NaN could land on a real-text row and silently
change which rows count as text-bearing, contaminating the matched/unmatched
split. These tests pin all three.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.shuffle_embeddings import shuffle_embeddings


def _synth_v5(n: int = 400, dim: int = 8, real_frac: float = 0.6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    has_text = rng.random(n) < real_frac
    emb[~has_text] = np.nan  # zero-text rows are NaN in embed_*
    data = {
        "latitude_deg": rng.uniform(35, 40, n).astype(np.float32),
        "longitude_deg": rng.uniform(135, 141, n).astype(np.float32),
        "depth_from_surface": rng.uniform(0.5, 25, n).astype(np.float32),
        "n_value": rng.uniform(0, 50, n).astype(np.float32),
    }
    for k in range(dim):
        data[f"embed_{k}"] = emb[:, k]
    return pd.DataFrame(data)


def test_shuffle_preserves_non_embed_columns() -> None:
    df = _synth_v5()
    out = shuffle_embeddings(df, seed=1)
    for col in ("latitude_deg", "longitude_deg", "depth_from_surface", "n_value"):
        np.testing.assert_array_equal(df[col].to_numpy(), out[col].to_numpy())


def test_shuffle_preserves_has_text_alignment() -> None:
    """A row that had text (embed_0 finite) must still have text after the
    shuffle, and a NaN row must stay NaN -- the indicator cannot move."""
    df = _synth_v5()
    before = np.isfinite(df["embed_0"].to_numpy())
    after = np.isfinite(shuffle_embeddings(df, seed=2)["embed_0"].to_numpy())
    np.testing.assert_array_equal(before, after)


def test_shuffle_actually_permutes_real_rows() -> None:
    """The embedding block must change for a non-trivial fraction of real-text
    rows (otherwise it is not a null)."""
    df = _synth_v5()
    out = shuffle_embeddings(df, seed=3)
    real = np.isfinite(df["embed_0"].to_numpy())
    emb_cols = [c for c in df.columns if c.startswith("embed_")]
    before = df.loc[real, emb_cols].to_numpy()
    after = out.loc[real, emb_cols].to_numpy()
    changed = ~np.all(before == after, axis=1)
    assert changed.mean() > 0.5, "shuffle barely moved any real embeddings"


def test_shuffled_rows_carry_intact_real_vectors() -> None:
    """Each shuffled real row's embedding must be a real (non-NaN) vector
    drawn from some original real row -- no NaN bleed into text rows, and the
    multiset of real embedding rows is preserved."""
    df = _synth_v5()
    out = shuffle_embeddings(df, seed=4)
    emb_cols = [c for c in df.columns if c.startswith("embed_")]
    real = np.isfinite(out["embed_0"].to_numpy())
    after_real = out.loc[real, emb_cols].to_numpy()
    assert np.isfinite(after_real).all(), "NaN bled into a text row"
    # The set of real embedding vectors is a permutation of the original set.
    before_real = df.loc[np.isfinite(df["embed_0"].to_numpy()), emb_cols].to_numpy()
    np.testing.assert_array_equal(
        np.sort(before_real, axis=0), np.sort(after_real, axis=0)
    )


def test_shuffle_is_deterministic_under_seed() -> None:
    df = _synth_v5()
    a = shuffle_embeddings(df, seed=7)
    b = shuffle_embeddings(df, seed=7)
    pd.testing.assert_frame_equal(a, b)
