"""Tests for ``scripts/build_paper2_data_bundle``.

The bundle is what ``data_availability.tex`` promises referees. The failure
mode that matters is not a crash -- it is a bundle that builds happily while
quietly missing something the manuscript says is in it. These tests pin the
properties that prevent that:

- the 35-run manifest of ZENODO.md resolves, including the ``region_hgb`` ->
  ``region_hgb_lro`` rename the appendix cites;
- a declared layer that is absent aborts the build rather than being skipped;
- the only way to omit something is to register it in ``KNOWN_ABSENT``, and
  registered absences reach both ``manifest.json`` and the README;
- the manifest describes the tree exactly, in both directions;
- Japan's embedding policy stays PCA-64-only, because relaxing it would put
  full-dimensional embeddings of an MLIT-restricted corpus into a public
  deposit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_paper2_data_bundle as B

# This builder assembles a deposit out of run artefacts (`data/runs/`, which is
# gitignored), the manuscript directory and the bundled inference demo. In the
# companion CODE repository none of those are present -- correctly -- and that
# repo's build runs this suite as its gate, so skip rather than fail there.
_HAVE_INPUTS = (
    (B.REPO / "data/runs").is_dir()
    and (B.PAPER / "ZENODO.md").exists()
    and (B.BACKEND / "notebooks/zenodo_inference_demo.py").exists()
)
pytestmark = pytest.mark.skipif(
    not _HAVE_INPUTS,
    reason="run artefacts / manuscript / demo not present (expected in the "
           "code-only companion repository)")


# ---------------------------------------------------------------- run set --

def test_run_manifest_resolves_to_the_thirty_five_of_zenodo_md():
    runs = B.collect_runs(B.REPO / "data/runs")
    assert len(runs) == 35
    names = [n for _, n in runs]
    assert len(set(names)) == 35, "duplicate run names would collide in runs/"
    # (B) the rename the manuscript's appendix depends on
    assert B.HGB_BUNDLE_NAME in names
    assert not any(n == "region_hgb" for n in names)
    hgb_src = dict((n, s) for s, n in runs)[B.HGB_BUNDLE_NAME]
    assert hgb_src == B.REPO / B.HGB_SRC_REL
    # (C) and (D)
    assert all(n in names for n in B.PT7_RUNS)
    assert all(n in names for n in B.PT9_RUNS)
    assert len(B.PT9_RUNS) == 8


def test_calibrated_cell_count_is_enforced(tmp_path):
    """23 is a contract with Appendix C and Fig S1, not an incidental count."""
    (tmp_path / "dkl_national_a").mkdir()
    (tmp_path / "dkl_national_a" / "conformal_mondrian.json").write_text("{}")
    with pytest.raises(SystemExit, match="expected 23 calibrated cells"):
        B.collect_runs(tmp_path)


# ------------------------------------------------------------- strictness --

def test_declared_but_absent_layer_aborts(tmp_path):
    b = B.Bundle(tmp_path)
    with pytest.raises(SystemExit, match="declared bundle layer is missing"):
        b.add(tmp_path / "nope.json", "data/nope.json", "hint")


def test_duplicate_bundle_path_aborts(tmp_path):
    src = tmp_path / "a.json"
    src.write_text("{}")
    b = B.Bundle(tmp_path / "out")
    b.add(src, "data/a.json", "hint")
    with pytest.raises(SystemExit, match="duplicate bundle path"):
        b.add(src, "data/a.json", "hint")


def test_validate_rejects_a_file_the_manifest_does_not_list(tmp_path):
    src = tmp_path / "a.json"
    src.write_text("{}")
    out = tmp_path / "out"
    b = B.Bundle(out)
    b.add(src, "data/a.json", "hint")
    (out / "data" / "stowaway.csv").write_text("x\n")
    with pytest.raises(SystemExit, match="manifest does not describe the tree"):
        b.validate()


def test_validate_rejects_a_corrupted_copy(tmp_path):
    src = tmp_path / "a.json"
    src.write_text("{}")
    out = tmp_path / "out"
    b = B.Bundle(out)
    b.add(src, "data/a.json", "hint")
    (out / "data" / "a.json").write_text("tampered")
    with pytest.raises(SystemExit, match="sha256 mismatch after copy"):
        b.validate()


def test_build_refuses_to_overwrite(tmp_path):
    (tmp_path / "existing").mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        B.build(tmp_path / "existing", pca64_japan=None,
                runs_dir=B.REPO / "data/runs")


# ------------------------------------------------------------- disclosure --

def test_known_absences_name_the_storm_table_and_the_japan_embeddings():
    keys = set(B.KNOWN_ABSENT)
    assert any("storm" in k for k in keys)
    assert "embeddings/japan/full_768d" in keys
    for k, why in B.KNOWN_ABSENT.items():
        assert len(why) > 80, f"{k}: an absence needs a real reason, not a stub"


def test_readme_and_manifest_carry_every_declared_absence():
    manifest = {"release": "r", "license": "CC-BY-4.0", "n_files": 1,
                "total_bytes": 10, "embedding_policy": B.EMBEDDING_POLICY,
                "declared_absent": B.KNOWN_ABSENT,
                "layers": [{"path": "data/x.csv", "sha256": "0", "size_bytes": 10,
                            "content_type": "csv", "citation_hint": "h"}]}
    readme = B._readme(manifest)
    for path in B.KNOWN_ABSENT:
        assert path in readme, f"{path} absent from the generated README"
    assert "Declared absences" in readme


def test_japan_embedding_policy_is_pca64_only():
    """A referee-facing legal property, not a preference.

    ``data_availability.tex`` justifies redistributing a derivative of the
    MLIT-restricted corpus on PCA-64 non-invertibility. Shipping 768-d
    embeddings would leave that justification not covering what ships.
    """
    assert B.EMBEDDING_POLICY["japan"] == "pca64_only"
    assert B.EMBEDDING_POLICY["uk"] == "full_768_and_source"
    assert not any("japan" in Path(p).parts[-1] and p.endswith("_e5.npy")
                   for p in B.UK_EMBEDDINGS), \
        "a Japan 768-d cache leaked into the UK embedding list"


# ----------------------------------------------------------------- misc ----

def test_content_types_cover_what_zenodo_md_declares():
    for suffix, expect in [(".pt", "torch_pickle"), (".npz", "numpy_npz"),
                           (".json", "json"), (".csv", "csv"),
                           (".parquet", "parquet"), (".md", "markdown"),
                           (".png", "png"), (".txt", "text"),
                           (".npy", "binary"), (".zzz", "binary")]:
        assert B._content_type(suffix) == expect


def test_provenance_and_prereg_are_the_companion_repo_sets():
    """ZENODO.md says these are the same sets; imports make that structural."""
    from scripts.build_paper2_companion_repo import PREREG, RESULTS
    src = Path(B.__file__).read_text(encoding="utf-8")
    assert "from scripts.build_paper2_companion_repo import PREREG, RESULTS" in src
    assert len(PREREG) == 3
    for p in PREREG + RESULTS:
        assert (B.REPO / p).exists(), f"companion set names a missing file: {p}"


def test_add_tree_refuses_a_missing_or_partial_directory(tmp_path):
    """rglob over a nonexistent path yields nothing and raises nothing.

    Without these guards the builder produced a deposit whose manifest,
    README and bundled ZENODO.md all advertised a layer the tree lacked.
    """
    b = B.Bundle(tmp_path / "out")
    with pytest.raises(SystemExit, match="is not a directory"):
        b.add_tree(tmp_path / "nope", "embeddings/japan", "hint")

    src = tmp_path / "half"
    src.mkdir()
    (src / "pca64_heldout_a.npy").write_bytes(b"x")
    # manifest.json is written last by the exporter, so its absence is exactly
    # the signal that an export is still running or died partway.
    with pytest.raises(SystemExit, match="missing manifest.json"):
        b.add_tree(src, "embeddings/japan", "hint",
                   require=("manifest.json", "keys.parquet"))

    (src / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="missing keys.parquet"):
        b.add_tree(src, "embeddings/japan", "hint",
                   require=("manifest.json", "keys.parquet"))

    (src / "keys.parquet").write_bytes(b"x")
    with pytest.raises(SystemExit, match="expected at least"):
        b.add_tree(src, "embeddings/japan", "hint", min_files=10,
                   require=("manifest.json", "keys.parquet"))


def test_pca64_japan_is_required():
    import inspect

    sig = inspect.signature(B.build)
    assert sig.parameters["pca64_japan"].default is inspect.Parameter.empty, (
        "an optional Japan layer let the builder ship a deposit that "
        "advertised features it did not contain")
    src = Path(B.__file__).read_text(encoding="utf-8")
    assert '"--pca64-japan", type=Path, required=True' in src


def test_restricted_layer_table_is_declared_absent():
    """source_data rows name it as their source, so its absence must be stated."""
    assert "data/soil_text_layers.csv" in B.KNOWN_ABSENT
    why = B.KNOWN_ABSENT["data/soil_text_layers.csv"]
    assert "narrative" in why and "MLIT" in why
    # and it must never be added to the shipped data layer
    assert not any("soil_text_layers" in rel for rel, _n, _h in B.DATA_LAYER)


@pytest.mark.slow
def test_end_to_end_build_validates(tmp_path):
    out = tmp_path / "bundle"
    manifest = B.build(out, pca64_japan=B.REPO / "data/release/pca64/japan_fullpop",
                       runs_dir=B.REPO / "data/runs", skip_source_data=True)
    listed = {l["path"] for l in manifest["layers"]}
    # positive assertions: emptying MODEL_FILES / DATA_LAYER / UK_EMBEDDINGS
    # and dropping predictions.npz from RUN_FILES all passed the old
    # self-referential check, producing a 4 MB "valid" bundle.
    for required in ("model/foundation_model.pt", "model/foundation_model.pt.meta.json",
                     "data/borings_japan_v4id.parquet",
                     "data/kunijiban_metadata.parquet",
                     "transfer/uk_bgs_spt_full.parquet",
                     "embeddings/japan/keys.parquet",
                     "ZENODO.md", "zenodo_inference_demo.py"):
        assert required in listed, f"{required} absent from the deposit"
    assert any(p.startswith("embeddings/uk/") and p.endswith(".npy")
               for p in listed), "no UK 768-d embedding shipped"
    assert sum(1 for p in listed if p.endswith("/predictions.npz")) >= 30, (
        "predictions.npz missing from most evaluation cells")
    assert len([p for p in listed if p.startswith("runs/")]) >= 90
    assert len([p for p in listed if p.startswith("prereg/")]) == 3
    assert manifest["n_files"] == len(manifest["layers"])
    assert (out / "manifest.json").exists()
    assert (out / "release_validated.txt").read_text().startswith("OK\n")
    on_disk = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    listed = {l["path"] for l in manifest["layers"]}
    assert on_disk - listed == {"manifest.json", "README.md",
                                "release_validated.txt"}
    loaded = json.loads((out / "manifest.json").read_text())
    assert loaded["declared_absent"] == B.KNOWN_ABSENT


def test_nested_manifests_stay_under_the_completeness_check(tmp_path):
    """source_data/ and embeddings/japan/ ship their own manifest.json.

    Excluding the builder's own root files by *name* also excluded those, so
    they were listed in the manifest but treated as absent from the tree --
    and, worse, a stowaway named ``manifest.json`` anywhere in the tree would
    have been invisible to the check.
    """
    src = tmp_path / "m.json"
    src.write_text("{}")
    out = tmp_path / "out"
    b = B.Bundle(out)
    b.add(src, "source_data/manifest.json", "nested payload")
    b.add(src, "embeddings/japan/manifest.json", "nested payload")
    b.validate()  # must not raise

    # and a genuine stowaway with that name is still caught
    (out / "source_data" / "extra").mkdir(parents=True, exist_ok=True)
    (out / "source_data" / "extra" / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="manifest does not describe the tree"):
        b.validate()


# --------------------------------------------------------- inference demo --
#
# ZENODO.md tells a referee to run zenodo_inference_demo.py as their first
# action against the bundle. It was broken in four ways at once -- two wrong
# paths, a wrong keyword signature and a method that does not exist -- and a
# blanket ``except Exception`` turned the last two into a silent "raw std"
# result, so the demo exited 0 while reporting uncalibrated numbers in a column
# labelled calibrated. These pin each one.

_DEMO = B.BACKEND / "notebooks/zenodo_inference_demo.py"


def test_demo_reads_paths_the_builder_actually_writes():
    src = _DEMO.read_text(encoding="utf-8")
    # the parquet the builder ships, not its pre-identity-spine predecessor
    shipped = {name for _rel, name, _hint in B.DATA_LAYER}
    assert "borings_japan_v4id.parquet" in shipped
    assert 'release / "data" / "borings_japan_v4id.parquet"' in src
    assert '"borings_japan_v4.parquet"' not in src
    # the abolished layout must not come back
    assert 'release / "quantiles"' not in src, (
        "quantiles/ was abolished; artefacts live in runs/<cell>/")
    assert '"runs" / "dkl_national_full"' in src
    assert B.MODEL_RUN_REL.endswith("dkl_national_full")


def test_demo_calls_the_calibrator_that_exists():
    """Both the method name and its keywords, checked against the real class."""
    import inspect

    from national.evaluation.calibration import ConformalCalibrator

    src = _DEMO.read_text(encoding="utf-8")
    assert "cal.radii_for(" not in src, "radii_for() does not exist"
    assert "cal.interval_mondrian(" in src
    assert hasattr(ConformalCalibrator, "interval_mondrian")

    fit = set(inspect.signature(ConformalCalibrator.fit_mondrian).parameters)
    for kw in ("y_true=", "y_pred=", "y_std=", "groups=", "alphas="):
        assert f"cal.fit_mondrian(\n" in src or kw in src
        assert kw.rstrip("=") in fit, f"{kw} is not a fit_mondrian parameter"
    # the names that were wrong before
    assert "mean=mean, std=std" not in src

    iv = set(inspect.signature(ConformalCalibrator.interval_mondrian).parameters)
    for kw in ("y_pred", "y_std", "groups", "alpha"):
        assert kw in iv


def test_demo_does_not_swallow_its_own_bugs():
    """A wrong signature must surface, not degrade to an uncalibrated column."""
    src = _DEMO.read_text(encoding="utf-8")
    i_reraise = src.index("except (TypeError, AttributeError):")
    i_blanket = src.index("except Exception as exc:")
    assert i_reraise < i_blanket, (
        "the blanket handler must come after the re-raise, or it catches first")
    assert "raise" in src[i_reraise:i_blanket]
