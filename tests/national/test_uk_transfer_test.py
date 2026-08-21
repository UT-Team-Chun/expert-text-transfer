"""Tests for scripts.uk_transfer_test -- specifically the W2 +-coords ablation
support added on top of the existing leave-region-out content-effect harness
(docs/research/2026-07-09_nmi_universality_preregistration.md, P-W2).

No network / sentence-transformers access: every test restricts ``arms`` to
``("no_text",)`` so ``embed_texts`` (which downloads
``intfloat/multilingual-e5-base``) is never invoked -- exercising exactly the
code path the W2 ablation driver uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.uk_transfer_test import _evaluate_lro, main, run

# Region counts must clear the >=200 keep-threshold in `run`; region+train
# sizes must also clear `_evaluate_lro`'s te>=30 / tr>=100 per-fold minimums.
_N_PER_REGION = 220
_REGIONS = ("midlands", "north_england", "scotland")


def _synthetic_uk_dataframe(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = _N_PER_REGION * len(_REGIONS)
    region = np.repeat(_REGIONS, _N_PER_REGION)
    # Distinct lat/lon bands per region so the columns carry real signal.
    lat_offset = {"midlands": 52.5, "north_england": 54.0, "scotland": 57.0}
    lat = np.array([lat_offset[r] for r in region]) + rng.normal(0, 0.2, size=n)
    lon = rng.uniform(-3.0, -1.0, size=n)
    depth = rng.uniform(0.0, 20.0, size=n)
    ground_level = rng.normal(50.0, 10.0, size=n)
    n_value = np.clip(5.0 + 0.7 * depth + rng.normal(0.0, 2.0, size=n), 1, 99)
    lith_desc = np.array(["Stiff brown CLAY with occasional gravel"] * n)
    return pd.DataFrame(
        {
            "region": region,
            "latitude_deg": lat,
            "longitude_deg": lon,
            "depth_from_surface": depth,
            "ground_level": ground_level,
            "n_value": n_value,
            "lith_desc": lith_desc,
        }
    )


@pytest.fixture
def synthetic_parquet(tmp_path: Path) -> Path:
    df = _synthetic_uk_dataframe()
    path = tmp_path / "uk_synthetic.parquet"
    df.to_parquet(path)
    return path


# ============================================================================
# Feature-list assembly (the flag's literal contract)
# ============================================================================


def test_no_coords_flag_drops_lat_lon_from_base(synthetic_parquet: Path, tmp_path: Path) -> None:
    results = run(
        synthetic_parquet, tmp_path / "with_coords.json", tmp_path / "cache",
        seeds=[42], no_coords=False, arms=("no_text",),
    )
    assert results["config"]["baseline"] == ["depth_from_surface", "ground_level",
                                              "latitude_deg", "longitude_deg"]

    results_no_coords = run(
        synthetic_parquet, tmp_path / "without_coords.json", tmp_path / "cache",
        seeds=[42], no_coords=True, arms=("no_text",),
    )
    assert results_no_coords["config"]["baseline"] == ["depth_from_surface", "ground_level"]
    assert "latitude_deg" not in results_no_coords["config"]["baseline"]
    assert "longitude_deg" not in results_no_coords["config"]["baseline"]


def test_no_coords_flag_is_a_pure_subtraction(synthetic_parquet: Path, tmp_path: Path) -> None:
    """The only difference between the two base lists is the two coordinate
    columns -- no_coords must not perturb any other feature."""
    with_coords = run(
        synthetic_parquet, tmp_path / "a.json", tmp_path / "cache",
        seeds=[42], no_coords=False, arms=("no_text",),
    )["config"]["baseline"]
    without_coords = run(
        synthetic_parquet, tmp_path / "b.json", tmp_path / "cache",
        seeds=[42], no_coords=True, arms=("no_text",),
    )["config"]["baseline"]
    assert set(with_coords) - set(without_coords) == {"latitude_deg", "longitude_deg"}


# ============================================================================
# arms restriction (skips the embedding step; only no_text is computed)
# ============================================================================


def test_no_text_only_arm_skips_text_and_shuffled_keys(synthetic_parquet: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    results = run(synthetic_parquet, out, tmp_path / "cache", seeds=[42, 43, 44],
                  no_coords=False, arms=("no_text",))
    assert "no_text" in results
    assert "text" not in results
    assert "shuffled" not in results
    assert "deltas" not in results
    assert "content_significance" not in results
    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk == results


def test_no_text_only_arm_reports_all_regions_and_seed_stats(
    synthetic_parquet: Path, tmp_path: Path,
) -> None:
    results = run(synthetic_parquet, tmp_path / "out.json", tmp_path / "cache",
                  seeds=[42, 43, 44], no_coords=True, arms=("no_text",))
    per_region = results["no_text"]["per_region"]
    assert set(per_region.keys()) == set(_REGIONS)
    for rmse in per_region.values():
        assert np.isfinite(rmse) and rmse > 0
    assert set(results["per_region_n_spt"].keys()) == set(_REGIONS)
    for n in results["per_region_n_spt"].values():
        assert n == _N_PER_REGION
    for region_std in results["per_region_seed_std"].values():
        assert set(region_std.keys()) == {"no_text"}


def test_full_arms_default_unchanged_shape(synthetic_parquet: Path, tmp_path: Path, monkeypatch) -> None:
    """Default arms=(no_text,text,shuffled) must still request the embedding
    step (backward compatibility for every other caller of `run`)."""
    calls: list[bool] = []

    def _fake_embed(texts, cache):
        calls.append(True)
        rng = np.random.default_rng(0)
        return rng.normal(size=(len(texts), 8)).astype(np.float32)

    monkeypatch.setattr("scripts.uk_transfer_test.embed_texts", _fake_embed)
    results = run(synthetic_parquet, tmp_path / "out.json", tmp_path / "cache", seeds=[42])
    assert calls, "default arms must still trigger embed_texts"
    assert "text" in results and "shuffled" in results and "deltas" in results


# ============================================================================
# _evaluate_lro: arms parameter does not require a real embedding matrix
# ============================================================================


def test_evaluate_lro_no_text_arm_accepts_zero_width_embedding() -> None:
    df = _synthetic_uk_dataframe()
    base = ["depth_from_surface", "ground_level"]
    placeholder_emb = np.zeros((len(df), 0), dtype=np.float32)
    per = _evaluate_lro(df, base, placeholder_emb, seeds=[42], arms=("no_text",))
    assert set(per.keys()) == set(_REGIONS)
    for region_result in per.values():
        assert set(region_result.keys()) == {"no_text"}
        mean, std = region_result["no_text"]
        assert np.isfinite(mean) and mean > 0
        assert std >= 0.0


def test_evaluate_lro_default_arms_matches_prior_behavior() -> None:
    """Calling _evaluate_lro positionally (as every existing caller does)
    must still compute all three modes."""
    df = _synthetic_uk_dataframe()
    base = ["depth_from_surface", "ground_level", "latitude_deg", "longitude_deg"]
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(len(df), 8)).astype(np.float32)
    per = _evaluate_lro(df, base, emb, [42])
    for region_result in per.values():
        assert set(region_result.keys()) == {"no_text", "text", "shuffled"}


# ============================================================================
# CLI wiring
# ============================================================================


def test_cli_no_coords_and_arms_flags(synthetic_parquet: Path, tmp_path: Path) -> None:
    out = tmp_path / "cli_out.json"
    rc = main([
        "--parquet", str(synthetic_parquet),
        "--out", str(out),
        "--cache-dir", str(tmp_path / "cache"),
        "--seeds", "42",
        "--no-coords",
        "--arms", "no_text",
        "--log-level", "WARNING",
    ])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["config"]["no_coords"] is True
    assert report["config"]["arms"] == ["no_text"]
    assert "latitude_deg" not in report["config"]["baseline"]


def test_cli_unknown_arm_raises_system_exit(synthetic_parquet: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([
            "--parquet", str(synthetic_parquet),
            "--out", str(tmp_path / "out.json"),
            "--cache-dir", str(tmp_path / "cache"),
            "--arms", "not_a_real_arm",
        ])
