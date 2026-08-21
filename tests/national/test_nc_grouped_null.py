"""Mechanics tests for the pre-registered grouped-null runner (P-T1/T3).

Synthetic data only -- no sentence embedding, no corpus files. The toy
construction plants the target signal INSIDE the embedding so the text arm
must beat the nulls, and gives every borehole several layers so the block
null actually has blocks to move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.nc_grouped_null import evaluate_grouped


def _light_model(seed: int):
    """Mechanics tests inject a Ridge instead of the protocol HGB: the
    400-tree HGB thrashes OpenMP barriers on toy data (~6 s/fit, 18 min per
    test run) while contributing nothing to what these tests verify."""
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0, random_state=seed)


def _toy_frame(n_regions: int = 3, boreholes_per: int = 12,
               layers_per: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    emb_rows = []
    for ri in range(n_regions):
        for b in range(boreholes_per):
            lat = 34.0 + ri * 2.0 + rng.normal(0, 0.05)
            lon = 135.0 + ri * 2.0 + rng.normal(0, 0.05)
            b_effect = rng.normal(0, 3.0)
            for k in range(layers_per):
                depth = 1.0 + k
                n_val = np.clip(5 + 2.0 * depth + b_effect + rng.normal(0, 1.0),
                                1, 100)
                rows.append({
                    "region": f"r{ri}",
                    "boring_file": f"r{ri}_b{b}.html",
                    "latitude_deg": lat, "longitude_deg": lon,
                    "depth_from_surface": depth, "n_value": float(n_val),
                })
                # embedding = noisy copy of the signal + noise dims
                e = rng.normal(0, 1.0, size=8)
                e[0] = n_val + rng.normal(0, 0.5)
                emb_rows.append(e)
    return pd.DataFrame(rows), np.asarray(emb_rows, dtype=np.float32)


def test_grouped_runner_structure_and_direction() -> None:
    df, emb = _toy_frame()
    res = evaluate_grouped(
        df, base=["depth_from_surface"], emb=emb,
        seeds=[0], n_perm_block=19, n_perm_row=19,
        use_knn=False, pca_dim=8, model_factory=_light_model,
    )
    s = res["summary"]
    # every fold present with both nulls
    assert set(res["per_region"]) == {"r0", "r1", "r2"}
    for r, d in res["per_region"].items():
        assert d["perm_p_block"] <= 1.0 and d["perm_p_block"] >= 1 / 20
        assert d["n_te"] == 72
    # the planted signal must produce a negative content effect under BOTH
    # nulls, and p floored at 1/(n+1), never 0
    assert s["content_pct_block_mean"] < 0
    assert s["content_pct_row_mean"] < 0
    assert 0 < s["stouffer_p_block"] < 1
    # Dependence-robust combinations must be present and must be WEAKER than
    # Stouffer: LORO folds share most of their training rows, so Stouffer is
    # anti-conservative and is reported only as a secondary figure.
    for k in ("bonferroni_min_p_block", "cauchy_p_block"):
        assert 0 < s[k] <= 1
    assert s["bonferroni_min_p_block"] >= s["stouffer_p_block"]
    assert len(s["fold_p_block"]) == len(res["per_region"])
    # the 8-region resample is reported as a descriptive spread, never as a CI
    spread = s["region_spread_block"]
    assert spread["per_region_min"] <= spread["per_region_max"]
    assert "NOT a 95% CI" in spread["note"]
    bb = s["borehole_block_bootstrap"]
    assert bb is not None and bb["n_boreholes"] == 36
    assert bb["bca_95"][0] <= bb["theta_hat_pct"] <= bb["bca_95"][1]


def test_perm_p_has_additive_correction() -> None:
    """With signal-free embeddings the p-value must be large but the floor
    (1+r)/(1+n) guarantees it can never be reported as 0."""
    df, emb = _toy_frame()
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, size=emb.shape).astype(np.float32)
    res = evaluate_grouped(
        df, base=["depth_from_surface"], emb=noise,
        seeds=[0], n_perm_block=9, n_perm_row=9,
        use_knn=False, pca_dim=8, model_factory=_light_model,
    )
    for d in res["per_region"].values():
        assert d["perm_p_block"] >= 1 / 10
        assert d["perm_p_row"] >= 1 / 10


def test_knn_prior_excludes_own_borehole() -> None:
    from scripts.nc_grouped_null import knn_prior

    df, _ = _toy_frame(n_regions=1, boreholes_per=6, layers_per=3)
    tr = np.ones(len(df), dtype=bool)
    feats = knn_prior(df, tr)
    assert feats.shape == (len(df), 2)
    # own borehole excluded -> min distance strictly positive
    assert np.nanmin(feats[:, 1]) > 0


# --------------------------------------------------------------- sharding


def test_region_shards_reproduce_the_sequential_run(tmp_path) -> None:
    """Sharding by held-out region must be statistically inert.

    Every permutation seeds its own generator from ``100_000 * seed + p``,
    which depends on neither the region nor the loop order, so a region
    computed alone must be bit-identical to the same region computed inside a
    full sequential run -- and the combined summary must match too.
    """
    from scripts.nc_grouped_null import combine, combine_parts

    df, emb = _toy_frame()
    kw = dict(base=["depth_from_surface"], emb=emb, seeds=[0],
              n_perm_block=19, n_perm_row=19, use_knn=False, pca_dim=8,
              model_factory=_light_model)

    whole = evaluate_grouped(df, **kw)

    per, pb, pr, losses = {}, [], [], []
    for r in ("r0", "r1", "r2"):
        parts = evaluate_grouped(df, **kw, regions_filter=[r],
                                 return_parts=True)
        assert set(parts["per_region"]) == {r}
        assert parts["per_region"][r] == whole["per_region"][r], (
            f"shard {r} differs from the sequential run")
        per.update(parts["per_region"])
        pb += parts["fold_p_block"]
        pr += parts["fold_p_row"]
        losses += parts["borehole_losses"]

    assert combine_parts(per, pb, pr, losses)["summary"] == whole["summary"]


def test_combine_refuses_to_double_count_a_region(tmp_path) -> None:
    from scripts.nc_grouped_null import combine

    df, emb = _toy_frame()
    kw = dict(base=["depth_from_surface"], emb=emb, seeds=[0],
              n_perm_block=19, n_perm_row=19, use_knn=False, pca_dim=8,
              model_factory=_light_model)
    parts = evaluate_grouped(df, **kw, regions_filter=["r0"],
                             return_parts=True)
    parts.pop("borehole_losses")
    import json
    for name in ("a.json", "b.json"):  # same region twice
        (tmp_path / name).write_text(json.dumps(parts))
    with pytest.raises(SystemExit, match="more than one shard"):
        combine(tmp_path, tmp_path / "out.json")


def test_unknown_region_filter_is_an_error() -> None:
    df, emb = _toy_frame()
    with pytest.raises(SystemExit, match="unknown region"):
        evaluate_grouped(df, base=["depth_from_surface"], emb=emb, seeds=[0],
                         n_perm_block=5, n_perm_row=5, use_knn=False,
                         pca_dim=8, model_factory=_light_model,
                         regions_filter=["r9"])


def test_artefact_carries_proof_that_the_null_was_a_permutation() -> None:
    """The defect that survived three weeks was invisible in the artefacts.

    Every shard must now state, per region, that its null draws were bijective
    and which strata columns were used, so the property is checkable from the
    released JSON without rerunning anything.
    """
    df, emb = _toy_frame()
    res = evaluate_grouped(
        df, base=["depth_from_surface"], emb=emb, seeds=[0],
        n_perm_block=7, n_perm_row=3, use_knn=False, pca_dim=8,
        model_factory=_light_model,
    )
    for r, d in res["per_region"].items():
        np_ = d["null_permutation"]
        assert np_["all_draws_bijective"] is True, f"{r}: null was not a permutation"
        assert np_["n_draws_checked"] == 7
        assert np_["strata_columns"] == ["region"]
        total = (np_["frac_block_matched_mean"] + np_["frac_rank_fallback_mean"]
                 + np_["frac_self_retained_mean"])
        assert abs(total - 1.0) < 1e-4


def test_provenance_describes_the_null_that_actually_ran() -> None:
    """The provenance blob must be derived, never asserted.

    It previously hardcoded "strata=region ... Stouffer across folds" and went
    on saying so after both had changed, so the summary a reviewer reads first
    contradicted the per-region record underneath it.
    """
    from scripts.nc_grouped_null import _null_provenance

    per = {
        "r0": {"null_permutation": {"strata_columns": ["region", "litho"],
                                    "all_draws_bijective": True,
                                    "n_draws_checked": 1000}},
        "r1": {"null_permutation": {"strata_columns": ["region", "litho"],
                                    "all_draws_bijective": True,
                                    "n_draws_checked": 1000}},
    }
    prov = _null_provenance(per)
    assert prov["strata_columns_used"] == ["litho", "region"]
    assert prov["every_draw_was_a_permutation"] is True
    assert prov["draws_checked_per_region"] == [1000]
    assert "Bonferroni" in prov["fold_combination"]
    assert "secondary" in prov["fold_combination"]

    # one bad region must poison the whole claim, not be averaged away
    per["r1"]["null_permutation"]["all_draws_bijective"] = False
    assert _null_provenance(per)["every_draw_was_a_permutation"] is False
