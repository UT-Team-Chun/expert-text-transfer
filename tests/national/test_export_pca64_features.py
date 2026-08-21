"""Tests for ``scripts/export_pca64_features``.

This exporter produces the artefact ``data_availability.tex`` promises and the
pipeline never persisted: the per-fold PCA-64 text features. Two properties
carry the whole claim, and both are tested here.

1. **The features are what the evaluator uses.** If this script's PCA differed
   in any way from ``uk_transfer_test._evaluate_lro`` -- a different component
   count, solver, seed, or a basis fit on all rows instead of the training
   fold -- the released features would not correspond to the published
   numbers, and a referee refitting from them would get something else.
2. **It cannot silently export the wrong corpus.** The embedding cache name in
   every released artefact is content-addressed on the texts, so the hash pins
   the exact text set behind a published number. The exporter refuses to write
   when the hash does not match the one asserted on the command line.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import export_pca64_features as X


def _fake(n=400, d=96, regions=("a", "b", "c"), seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, d)).astype(np.float32)
    reg = np.array([regions[i % len(regions)] for i in range(n)])
    return emb, reg


def test_hash_matches_the_grouped_null_implementation():
    """Byte-for-byte, or the exporter's gate would test the wrong thing."""
    from scripts.nc_grouped_null import _hash_texts

    for texts in ([], ["a"], ["粘土質シルト", "fine SAND", ""],
                  ["x" * 300, "　", "emoji \U0001F600"]):
        assert X.hash_texts(texts) == _hash_texts(texts)


def test_pca_constants_match_the_evaluator():
    """These are pinned so a later edit to one side breaks the build."""
    import inspect

    from scripts import uk_transfer_test

    src = inspect.getsource(uk_transfer_test._evaluate_lro)
    assert 'svd_solver="randomized"' in src
    assert "random_state=0" in src
    assert X.PCA_DIM == 64
    assert X.PCA_SVD_SOLVER == "randomized"
    assert X.PCA_RANDOM_STATE == 0
    sig = inspect.signature(uk_transfer_test._evaluate_lro)
    assert sig.parameters["pca_dim"].default == X.PCA_DIM


def test_exported_features_equal_the_evaluator_computation(tmp_path):
    """The decisive equivalence: same basis, same numbers, per fold."""
    from sklearn.decomposition import PCA

    emb, reg = _fake()
    records = X.per_fold_pca64(emb, reg, tmp_path)
    assert len(records) == len(set(reg.tolist()))

    for rec in records:
        r = rec["held_out_region"]
        te = reg == r
        tr = ~te
        # Exactly what uk_transfer_test._evaluate_lro does for this fold.
        k = min(64, emb.shape[1], int(tr.sum()))
        pca = PCA(n_components=k, svd_solver="randomized",
                  random_state=0).fit(emb[tr])
        red_tr, red_te = pca.transform(emb[tr]), pca.transform(emb[te])

        got = np.load(tmp_path / rec["file"])
        assert got.shape == (len(emb), k)
        np.testing.assert_array_equal(got[tr], red_tr.astype(np.float32))
        np.testing.assert_array_equal(got[te], red_te.astype(np.float32))
        assert rec["n_train_rows_pca_was_fit_on"] == int(tr.sum())
        assert rec["n_heldout_rows"] == int(te.sum())


def test_pca_is_fit_on_training_rows_only(tmp_path):
    """A basis fit on all rows would leak the held-out region into the features."""
    from sklearn.decomposition import PCA

    emb, reg = _fake(seed=3)
    records = X.per_fold_pca64(emb, reg, tmp_path)
    rec = records[0]
    r = rec["held_out_region"]
    te = reg == r
    all_rows = PCA(n_components=64, svd_solver="randomized",
                   random_state=0).fit(emb).transform(emb).astype(np.float32)
    got = np.load(tmp_path / rec["file"])
    assert not np.allclose(got[te], all_rows[te]), (
        "features match a basis fit on ALL rows -- the fold is not leak-proof")


def test_the_pca_basis_is_not_shipped(tmp_path):
    """Shipping the basis would walk the features back toward 768-d."""
    emb, reg = _fake()
    X.per_fold_pca64(emb, reg, tmp_path)
    written = {p.name for p in tmp_path.iterdir()}
    assert all(n.startswith("pca64_heldout_") and n.endswith(".npy")
               for n in written), written
    assert not any("component" in n or "basis" in n or n.endswith(".pkl")
                   for n in written)


def test_too_few_training_rows_is_an_error(tmp_path):
    emb, reg = _fake(n=60, d=96, regions=("a", "b"))
    with pytest.raises(ValueError, match="cannot fit"):
        X.per_fold_pca64(emb[:60], np.array(["a"] * 30 + ["b"] * 30), tmp_path)


def test_hash_gate_refuses_to_write_a_mismatched_corpus(tmp_path, monkeypatch):
    """The gate must fire BEFORE embedding, and must write nothing."""
    import pandas as pd

    df = pd.DataFrame({"text": ["a", "b", "c"], "region": ["x", "x", "y"],
                       "boring_file": ["f1", "f1", "f2"]})
    monkeypatch.setattr(X, "__doc__", X.__doc__)  # no-op, keeps linters quiet

    import scripts.nc_grouped_null as NGN
    import scripts.text_leakage_controls as TLC

    monkeypatch.setattr(TLC, "load_domain", lambda *a, **k: (df, [], []))
    # run() imports these at call time, so patching the modules is enough.
    monkeypatch.setattr(NGN, "build_rich_features", lambda d, dom: (d, []))
    monkeypatch.setattr(TLC, "apply_strip_mode",
                        lambda texts, *a, **k: (list(texts), {"mode": "t"}))

    def _boom(*a, **k):  # pragma: no cover -- must never be reached
        raise AssertionError("embedding ran despite a hash mismatch")

    monkeypatch.setattr(X, "embed", _boom)

    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="NOT the ones behind the published"):
        X.run("japan", out, per_region_files=0, sample_seed=42,
              strip_mode="lithology_only", device="cpu", cache_dir=tmp_path,
              expect_text_hash="deadbeefdeadbeef", batch_size=8)
    assert not out.exists() or not list(out.iterdir()), \
        "the exporter wrote output despite refusing"


def test_manifest_records_the_policy_and_the_provenance(tmp_path):
    emb, reg = _fake()
    records = X.per_fold_pca64(emb, reg, tmp_path)
    # per_fold_pca64 writes the arrays; run() writes the manifest. Rebuild the
    # manifest shape here so the contract is pinned without a full run.
    for rec in records:
        assert set(rec) >= {"held_out_region", "file", "shape", "sha256",
                            "explained_variance_ratio_sum",
                            "n_train_rows_pca_was_fit_on", "n_heldout_rows"}
        blob = (tmp_path / rec["file"]).read_bytes()
        import hashlib
        assert rec["sha256"] == hashlib.sha256(blob).hexdigest()


def test_run_hashes_the_post_enrichment_frame(tmp_path, monkeypatch):
    """Behavioural, not textual: the dropped rows must leave the hash.

    ``build_rich_features`` drops rows (55 of 1,298,783 on the real Japanese
    population, because they carry no joined geological covariate). An earlier
    version of this test only checked that the call appeared in the source, so
    it stayed green when the call was made and its result thrown away -- the
    exporter then hashed 1,298,783 rows and produced 56215b220d666c31 instead
    of the published 22c29123b9f2a14e.
    """
    import pandas as pd

    import scripts.nc_grouped_null as NGN
    import scripts.text_leakage_controls as TLC

    df = pd.DataFrame({"text": ["a", "b", "c"], "region": ["x", "x", "y"],
                       "boring_file": ["f1", "f1", "f2"]})
    monkeypatch.setattr(TLC, "load_domain", lambda *a, **k: (df, [], []))
    # drops the last row, exactly as the real enrichment drops uncovered rows
    monkeypatch.setattr(NGN, "build_rich_features",
                        lambda d, dom: (d.iloc[:2].reset_index(drop=True), []))
    monkeypatch.setattr(TLC, "apply_strip_mode",
                        lambda texts, *a, **k: (list(texts), {"mode": "t"}))
    monkeypatch.setattr(X, "embed", lambda *a, **k: pytest.fail(
        "embedding ran despite a hash mismatch"))

    # The post-enrichment hash is over ["a", "b"], not ["a", "b", "c"].
    with pytest.raises(SystemExit) as ei:
        X.run("japan", tmp_path / "o", per_region_files=0, sample_seed=42,
              strip_mode="lithology_only", device="cpu", cache_dir=tmp_path,
              expect_text_hash=X.hash_texts(["a", "b", "c"]), batch_size=8)
    assert X.hash_texts(["a", "b"]) in str(ei.value), (
        "the exporter hashed the pre-enrichment frame")


def test_cache_name_matches_the_grouped_null_resolver(tmp_path):
    """The exporter must look where the published cache actually lives.

    The cache path format is duplicated between this exporter and
    ``nc_grouped_null.embedding_cache_path``; if they drift, the exporter
    silently re-embeds instead of reusing the published array.
    """
    from scripts.nc_grouped_null import embedding_cache_path

    texts = ["粘土質シルト", "fine SAND", ""]
    for mode in ("lithology_only", "full"):
        expect = embedding_cache_path(tmp_path, "japan", texts, mode)
        from scripts.text_leakage_controls import cache_tag
        got = tmp_path / (f"grouped_japan_{cache_tag(mode)}_"
                          f"{X.hash_texts(texts)}_e5.npy")
        assert got == expect, mode


def test_manifest_is_written_and_records_the_policy(tmp_path, monkeypatch):
    """Drive run() to completion and assert on the FILE, not a return value."""
    import json as _json

    import pandas as pd

    import scripts.nc_grouped_null as NGN
    import scripts.text_leakage_controls as TLC

    n, dim = 300, 96   # >64 training rows per fold, or PCA-64 cannot fit
    df = pd.DataFrame({
        "text": [f"t{i}" for i in range(n)],
        "region": [("a", "b", "c")[i % 3] for i in range(n)],
        "boring_file": [f"f{i // 3}" for i in range(n)],
        "n_value": np.arange(n, dtype=float),
        "latitude_deg": np.linspace(35, 36, n),
        "longitude_deg": np.linspace(139, 140, n),
        "depth_from_surface": np.linspace(1, 20, n),
        "absolute_elevation": np.linspace(0, 50, n),
        "river_distance_km": np.linspace(0, 5, n),
        "coast_distance_km": np.linspace(0, 9, n),
        "regime_code": np.zeros(n, dtype=int),
        "aist_litho_macro_code": np.ones(n, dtype=int),
        "aist_era_code": np.full(n, 2, dtype=int),
    })
    monkeypatch.setattr(TLC, "load_domain", lambda *a, **k: (df, [], []))
    monkeypatch.setattr(NGN, "build_rich_features", lambda d, dom: (d, []))
    monkeypatch.setattr(TLC, "apply_strip_mode",
                        lambda texts, *a, **k: (list(texts), {"mode": "t"}))
    rng = np.random.default_rng(0)
    monkeypatch.setattr(X, "embed",
                        lambda *a, **k: rng.normal(size=(n, dim)).astype(np.float32))

    out = tmp_path / "export"
    X.run("japan", out, per_region_files=0, sample_seed=42,
          strip_mode="lithology_only", device="cpu", cache_dir=tmp_path,
          expect_text_hash=None, batch_size=8)

    m = _json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["pca"]["basis_released"] is False, (
        "shipping the basis would walk the features back toward 768-d")
    assert m["pca"]["n_components"] == 64
    assert m["text_sha256_16"] == X.hash_texts([f"t{i}" for i in range(n)])
    assert m["embedding_device_requested"] == "cpu"
    # embed() is stubbed here, so nothing recorded an origin -- and the
    # manifest says exactly that rather than inheriting the requested device.
    assert m["embedding_origin"] == "unknown"
    assert len(m["embedding_sha256"]) == 64
    assert m["n_rows"] == n
    assert m["keys"]["n_rows"] == n
    assert set(m["keys"]["columns"]) == set(X.KEY_COLUMNS)
    assert len(m["folds"]) == 3
    assert (out / "keys.parquet").exists()
    assert not list(out.glob("*.partial.npy")), "a partial file survived"


def test_keys_align_with_the_feature_rows(tmp_path):
    """Row i of every fold file must be row i of keys.parquet."""
    import pandas as pd

    n = 30
    df = pd.DataFrame({c: np.arange(n) for c in X.KEY_COLUMNS})
    df["region"] = [("a", "b")[i % 2] for i in range(n)]
    df["boring_file"] = [f"f{i}" for i in range(n)]
    rec = X.write_keys(df, tmp_path)
    got = pd.read_parquet(tmp_path / "keys.parquet")
    assert rec["n_rows"] == n == len(got)
    assert list(got.columns) == X.KEY_COLUMNS
    pd.testing.assert_series_equal(got["region"], df["region"],
                                   check_names=False)


def test_write_keys_rejects_a_frame_missing_a_column(tmp_path):
    import pandas as pd

    df = pd.DataFrame({c: [1] for c in X.KEY_COLUMNS if c != "n_value"})
    with pytest.raises(SystemExit, match="key columns absent"):
        X.write_keys(df, tmp_path)


def test_embed_records_whether_it_computed_or_reused(tmp_path, monkeypatch):
    """The manifest must distinguish a fresh encode from a cache hit.

    Recording the ``--device`` flag as if it were provenance was misleading:
    on a cache hit nothing is embedded at all, and the cached array may have
    been produced on entirely different hardware. The published Japanese
    array, for instance, was computed on an amd64 Linux node, not here.
    """
    class FakeST:
        def __init__(self, *a, **k):
            pass

        def get_sentence_embedding_dimension(self):
            return 4

        def encode(self, batch, **k):
            return np.zeros((len(batch), 4), dtype=np.float32)

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)

    cache = tmp_path / "c.npy"
    X._PROVENANCE.clear()
    X.embed(["a", "b"], cache, "cpu", batch_size=2, chunk=2)
    assert X._PROVENANCE["embedding_origin"].startswith(
        "computed here on device=cpu")

    X._PROVENANCE.clear()
    X.embed(["a", "b"], cache, "mps", batch_size=2, chunk=2)
    assert X._PROVENANCE["embedding_origin"] == f"cache hit: {cache.name}", (
        "a cache hit must not be reported as an encode on the requested device")


def test_run_clears_stale_provenance():
    """A module-global must not leak an earlier call's origin into a manifest."""
    import inspect

    src = inspect.getsource(X.run)
    i_clear = src.index("_PROVENANCE.clear()")
    i_embed = src.index("emb = embed(")
    assert i_clear < i_embed
