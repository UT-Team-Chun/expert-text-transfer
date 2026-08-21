#!/usr/bin/env python
"""Assemble the Paper-2 (Nature Communications) companion repository.

Mirrors the Paper-1 precedent (``UT-Team-Chun/kanto-calibrated-spt-prior``):
a curated, standalone reproduction package containing ONLY the code, tests and
result-provenance artefacts needed to reproduce the paper's numbers --- not the
research monorepo. Layout matches the monorepo's ``backend/`` package structure
(top-level ``national/`` + ``scripts/`` + ``tests/``) so every ``python -m
scripts.<entry>`` invocation works unchanged from the repo root.

The code allowlist is computed, not hand-maintained: the AST import closure of
the entry-point scripts over the internal ``national.*`` / ``scripts.*`` /
``shared.*`` namespaces, plus package ``__init__.py`` files, plus the curated
test set, the result-provenance artefacts and the pre-registration record. A
``manifest.json`` with per-file sha256 is emitted and re-validated after
copying (same FAIR discipline as ``build_zenodo_release.py``).

Tree layout::

    national/ scripts/ shared/ tests/   code (import closure)
    results/                            result-provenance artefacts
    prereg/                             pre-registration, amendment, verdicts
    README.md REPRODUCIBILITY.md ...    from docs/paper/paper_2_national/companion/
    manifest.json                       sha256 + size of every file above

CLI::

    cd backend
    .venv/bin/python -m scripts.build_paper2_companion_repo \
        --output ../../expert-text-transfer

Four checks then run IN the assembled tree, and the build fails loudly if any
of them does:

1. ``compileall`` -- everything parses;
2. an import probe over every shipped module -- everything *imports*, which
   compileall cannot see (this is what catches a module whose dependency the
   allowlist never copied);
3. the included pytest subset;
4. manifest completeness in BOTH directions -- no shipped file is unlisted and
   no listed file is absent -- re-asserted after the build's own
   ``__pycache__``/``.pytest_cache`` by-products are removed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger("build_companion")
BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent

# ---------------------------------------------------------------- allowlists
ENTRY_SCRIPTS = [
    # ---- PRIMARY analysis (round 4, pre-registered): borehole-block
    # permutation null, provenance folds, borehole-budgeted few-shot curve,
    # descriptor-family mechanism, grouped conformal. These are the entries
    # named first in code_availability.tex.
    "nc_grouped_null", "nc_provenance_folds", "nc_fewshot_curve",
    "nc_descriptor_families", "run_mondrian_recal_grouped",
    # borehole identity spine + archive-header pass + join audit: the data
    # substrate the primary analysis joins on (code_availability item 1)
    "attach_identity_to_parquet", "extract_kunijiban_metadata",
    "audit_text_join",
    # transfer + leakage + row-level-null harness. nc_null_controls is the
    # EARLIER row-level shuffled-embedding null that the borehole-block null
    # of nc_grouped_null supersedes; it ships because the paper reports the
    # row-vs-block sensitivity contrast (P-T3).
    "uk_transfer_test", "japan_transfer_test", "storm_transfer_test",
    "text_leakage_controls", "nc_null_controls", "nc_geo_ablation",
    "nc_depth_band", "shuffle_embeddings",
    # round-3 additions: cross-archive transfer, rich-baseline ladder + KNN rung,
    # missingness/IPW
    "nc_cross_archive", "nc_rich_baseline", "nc_knn_prior", "nc_missingness",
    # text pipeline
    "extract_soil_text_from_xml", "embed_soil_text", "join_soil_text_to_parquet",
    # training + coordinate ablation + conformal
    "train_kanto_smoke", "train_lmc_national", "run_leave_region_out",
    "run_mondrian_recal_lmc", "visualize_results",
    # data enrichment (named in the manuscript's corpus-filter footnote)
    "enrich_borings",
    # ---- how the released deposit itself is produced. ZENODO.md hands the
    # referee the export_pca64_features command verbatim, and
    # data_availability.tex promises per-figure source data; without these
    # three the code record documents an analysis whose artefacts a reader
    # cannot regenerate.
    "export_pca64_features",      # the per-fold PCA-64 text features
    "export_figure_source_data",  # per-figure source data CSVs
    "build_paper2_data_bundle",   # assembles the data record described by ZENODO.md
    # cube products (named in SI S7 pipeline)
    "predict_national_cube",
    # figures + SI tables: build_paper2_figs renders everything except the
    # forest plot (build_forest_plot); the per-regime coverage tables of the
    # SI come from build_mondrian_recal_appendix_table; build_japan_coastline
    # is the provenance of the committed coastline asset in DATA below.
    "build_paper2_figs", "build_forest_plot",
    "build_mondrian_recal_appendix_table", "build_japan_coastline",
    # this builder ships itself: code_availability.tex points referees at
    # build_paper2_companion_repo.py as the "committed, auditable builder"
    # of the repository they are reading, so it has to be in there.
    "build_paper2_companion_repo",
]

#: Non-Python runtime files the AST closure cannot discover. Path is relative
#: to backend/ and preserved in the companion tree.
DATA = [
    # Simplified MLIT C23 national coastline drawn by the fig7/fig8 basemaps;
    # without it the figure renderer silently falls back to a 100-vertex hull.
    "national/data/assets/japan_coastline.json",
]

TESTS = [
    # --- primary analysis (round 4). test_grouped_null is the load-bearing
    # one: it asserts block_permutation_indices returns a PERMUTATION on
    # ragged frames and that the defective predecessor kept for bias
    # measurement (legacy_clipped_block_permutation) does not.
    "tests/national/test_grouped_null.py",
    "tests/national/test_nc_grouped_null.py",
    "tests/national/test_nc_provenance_folds.py",
    "tests/national/test_nc_fewshot_curve.py",
    "tests/national/test_run_mondrian_recal_grouped.py",
    "tests/national/test_identity_spine.py",
    "tests/national/test_structured_families.py",
    # --- earlier rounds (row-level nulls, transfer harnesses, conformal)
    "tests/national/test_nc_null_controls.py",
    "tests/national/test_shuffle_embeddings.py",
    "tests/national/test_calibration.py",
    "tests/national/test_japan_transfer.py",
    "tests/national/test_uk_transfer_test.py",
    "tests/national/test_leave_region_out_runner.py",
    "tests/national/test_storm_transfer.py",
    "tests/national/test_text_leakage_controls.py",
    "tests/national/test_run_mondrian_recal_lmc.py",
    "tests/national/test_build_paper2_figs.py",
    # --- the deposit machinery. These ship with their scripts because the
    # in-tree test run is this build's own gate: without them the gate does
    # not cover the code that produces the released artefacts. The two
    # artefact-reproduction suites skip cleanly when data/runs/ is absent,
    # which it is for anyone who has only the code record.
    "tests/national/test_export_pca64_features.py",
    "tests/national/test_export_figure_source_data.py",
    "tests/national/test_build_paper2_data_bundle.py",
    "tests/national/test_pt7_artefacts_reproduce.py",
    "tests/national/test_pt9_artefacts_reproduce.py",
]
# NOTE: tests/national/test_extract_kunijiban_metadata.py is deliberately NOT
# shipped. It parametrises over the six representative KuniJiban XMLs in
# data/sample_xml/, which are raw archive records we may not redistribute
# (see Data availability); in the assembled tree it would only ever report
# "skipped", which is worse than being honestly absent.

# static repo docs (README, LICENSE, CITATION.cff, ...) maintained in the monorepo
TEMPLATE_DIR = REPO / "docs/paper/paper_2_national/companion"

#: Pre-registration record, shipped VERBATIM under ``prereg/``. The
#: pre-registration was committed before any of the confirmatory computations
#: were run; the amendment records the deviations (including the block-null
#: defect and its repair); the verdicts record CONFIRMED/REFUTED against the
#: unaltered bars. code_availability.tex promises exactly these three.
PREREG = [
    "docs/research/2026-08-11_nc_text_preregistration.md",
    "docs/research/2026-08-14_nc_text_prereg_amendment_1.md",
    "docs/research/2026-08-18_nc_text_prereg_verdicts.md",
]

# result-provenance JSONs: every headline number in the manuscript traces here
RESULTS = [
    # ================= round 4: the pre-registered primary analysis ========
    # Borehole-block permutation null, CORRECTED (bijective) machinery,
    # 2026-08-18. The 2026-08-12/2026-08-13 grouped-null artefacts are the
    # output of the defective clipped-block predecessor and are deliberately
    # NOT shipped: they would put superseded numbers in a referee's hands.
    # P-T1: Japan permutation stage, three independent subsample draws
    "docs/research/2026-08-18_grouped_null_japan_s42.json",
    "docs/research/2026-08-18_grouped_null_japan_s43.json",
    "docs/research/2026-08-18_grouped_null_japan_s44.json",
    # P-T1: Japan full text-bearing population -- point estimate + BCa interval
    "docs/research/2026-08-18_grouped_null_japan_fullpop.json",
    # P-T2: UK replication (full UK text-bearing population, region strata)
    "docs/research/2026-08-18_grouped_null_uk.json",
    # P-T5: depersonalisation control -- proper nouns, header substrings and
    # per-project template boilerplate stripped before embedding, then the
    # P-T1 protocol rerun unchanged
    "docs/research/2026-08-18_grouped_null_japan_pt5_deperson.json",
    # P-T4: provenance folds (client / year / contractor / DTD / project)
    "docs/research/2026-08-18_provenance_folds_japan.json",
    # P-T9: the genuinely coordinate-free arm (--zero-fourier --no-residual-geo)
    "docs/research/2026-08-18_pt9_coordfree.json",
    # P-T7: in-distribution effect on the exact borehole-identity join
    "docs/research/2026-08-14_pt7_identity_join_3fold.json",
    # P-T6: borehole-budgeted cross-archive few-shot curve (+ its note)
    "docs/research/2026-08-12_fewshot_borehole_curve.json",
    "docs/research/2026-08-12_pt6_fewshot_borehole_curve.md",
    # P-T8: conformal coverage under row / borehole / site calibration splits
    "docs/research/2026-08-12_conformal_grouped_split.json",
    "docs/research/2026-08-12_conformal_grouped_split_lro.json",
    "docs/research/2026-08-12_pt8_conformal_grouped_split.md",
    # P-T10: descriptor-family mechanism (exploratory) (+ its note)
    "docs/research/2026-08-12_descriptor_families_japan.json",
    "docs/research/2026-08-12_descriptor_families_uk.json",
    "docs/research/2026-08-12_pt10_descriptor_families.md",
    # Borehole-identity join audit (SI "Join audit")
    "docs/research/2026-08-11_join_audit.json",
    "docs/research/2026-08-11_join_audit.md",
    # ================= rounds 1-3: superseded-but-still-reported ===========
    "docs/research/2026-06-21_japan_transfer_leakproof.json",
    "docs/research/2026-06-21_uk_transfer_leakproof.json",
    "docs/research/2026-06-21_storm_transfer_3rd_domain.json",
    "docs/research/2026-06-21_storm_transfer_nosize.json",
    "docs/research/2026-06-23_factorial_table.md",
    "docs/research/2026-06-21_text_leakage_japan.json",
    "docs/research/2026-06-21_text_leakage_uk.json",
    "docs/research/2026-06-21_region_bootstrap_ci.json",
    "docs/research/2026-06-23_structured_litho_baseline.json",
    "docs/research/2026-06-23_coord_ablation_hgb.json",
    "docs/research/2026-06-23_geological_province_split.json",
    "docs/research/2026-06-23_geological_split_subset_control.json",
    "docs/research/2026-06-23_target_harmonization.json",
    "docs/research/2026-06-24_within_region_null_japan.json",
    "docs/research/2026-06-24_within_region_null_uk.json",
    "docs/research/2026-06-24_geo_fold_lm_and_regime_ablation.json",
    "docs/research/2026-06-24_depth_band_japan.json",
    "docs/research/2026-06-29_national_dkl_coord_ablation.json",
    # round-3 (strip-vocabulary v2 + new analyses; canonical from 2026-07-04)
    "docs/research/2026-07-04_leakage_audit.json",
    "docs/research/2026-07-04_leakage_v2_japan.json",
    "docs/research/2026-07-04_leakage_v2_uk.json",
    "docs/research/2026-07-04_within_null_v2_japan.json",
    "docs/research/2026-07-04_within_null_v2_uk.json",
    "docs/research/2026-07-04_rich_baseline_ladder.json",
    "docs/research/2026-07-04_knn_prior_rung.json",
    "docs/research/2026-07-04_cross_archive_transfer.json",
    "docs/research/2026-07-04_geo_fold_v2.json",
    "docs/research/2026-07-04_depth_band_v2_japan.json",
    "docs/research/2026-07-04_missingness_ipw.json",
    # ---- artefacts added 2026-08-18 after a number-by-number audit of the
    # manuscript against the shipped set. Each closes a specific traceability
    # gap; the section they back is named.
    # Kanto from-scratch coordinate 2x2 (Results, "coordinates memorize" table)
    "docs/research/2026-06-21_coord_ablation_trainfromscratch.json",
    # conformal calibration budget: mean/worst coverage gap vs n_cal
    # (Results "calibrated uncertainty"; Conclusion)
    "docs/research/2026-06-21_conformal_budget.json",
    # P-T4 fifth family: leave-project-out. Measurable only on the FULL
    # text-bearing population (exactly one project clears the 300-row fold
    # minimum on the balanced subsample), so it runs sharded, one artefact
    # per held-out project. All eight folds of Table "Leave-project-out"
    # ship; shipping fewer would put a table in a referee's hands whose rows
    # they cannot check one by one.
    "docs/research/project__p00-1cd5faba.json",
    "docs/research/project__p01-e3827f1b.json",
    "docs/research/project__p02-482f081d.json",
    "docs/research/project__p03-41f8797b.json",
    "docs/research/project__p04-553e5e57.json",
    "docs/research/project__p05-f4e89d17.json",
    "docs/research/project__p06-4df448a6.json",
    "docs/research/project__p07-cfa26457.json",
    # MINE mutual-information audit, random-init null (Limitations)
    "docs/research/2026-06-20_mine_random_init_null.md",
    # UK corpus coverage: ground-level and lithology-text fractions (Data)
    "docs/research/2026-06-20_uk_cross_national_transfer.md",
    # ---- research notes the MANUSCRIPT CITES BY NAME, or that are the sole
    # provenance of a printed table. Shipped as markdown because that is the
    # form the numbers exist in; none was ever a JSON.
    # cited in Data: the AIST national geology cache pass
    "docs/research/2026-05-28_phase_c_national_aist_cache.md",
    # cited in Appendix B: the three-model national leave-region-out table
    "docs/research/2026-05-27_phase_c_national_lro.md",
    # cited in Appendix D: the null-result ablations
    "docs/research/2026-05-28_phase_c_dkl_v2_and_ablations.md",
    # per-region leave-region-out detail (SI transfer table, Appendix B)
    "docs/research/2026-05-29_phase_c_dkl_lro_8way.md",
    # per-regime Mondrian recalibration cells (Results calibration, Appendix C)
    "docs/research/2026-05-29_phase_c_mondrian_recal.md",
    # test-time-adaptation 24-cell sweep (SI transfer detail)
    "docs/research/2026-06-01_phase_4_tta_24cell_sweep_results.md",
    # content-vs-capacity decomposition of the LRO text effect (Results)
    "docs/research/2026-06-19_lro_text_transfer_b1_decomposition.md",
    # per-layer text embedding pass: corpus counts, JMTEB scores (Data)
    "docs/research/2026-05-29_phase_d_per_layer_text_embedding.md",
    # data-pipeline landing counts (Data)
    "docs/research/2026-05-29_phase_d_data_pipelines_landed.md",
    # spatially blocked 3-fold text-vs-no-text table (Data, Limitations)
    "docs/research/2026-06-02_phase_3_ruri_l2_3fold.md",
]
# NOT SHIPPED, deliberately: docs/research/results_table.md. Appendix A cites
# it by name as the source of the 27-cell sweep, but it is the monorepo's
# cross-project canonical table and carries 18 rows of unpublished results
# belonging to the sister (NMI) manuscript. Releasing it with this submission
# would disclose that paper before its own submission. The Paper-2 rows it
# holds are covered by the phase notes above; the citation in Appendix A is
# reported to the authors as needing repointing.


def _imports_of(path: Path) -> set[str]:
    """Modules imported by ``path``, excluding OPTIONAL imports.

    An import inside a ``try:`` block is a guarded, best-effort dependency by
    construction -- e.g. ``national/data/sge_gate.py`` prefers
    ``scripts.nmi_aoa_audit`` when importable but carries bit-equal local
    mirrors for environments without it. Following such imports would drag
    the NMI companion paper's analysis scripts into this paper's repository,
    violating the "only what reproduces THIS paper" contract.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:  # pragma: no cover
        return set()
    optional: set[ast.AST] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Try):
            for child in n.body:
                for sub in ast.walk(child):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        optional.add(sub)
    out: set[str] = set()
    for n in ast.walk(tree):
        if n in optional:
            continue
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
    return out


def _resolve(mod: str) -> Path | None:
    p = BACKEND / (mod.replace(".", "/") + ".py")
    if p.exists():
        return p
    p = BACKEND / mod.replace(".", "/") / "__init__.py"
    if p.exists():
        return p
    return None


def compute_closure() -> set[Path]:
    queue = [BACKEND / f"scripts/{e}.py" for e in ENTRY_SCRIPTS]
    missing = [p for p in queue if not p.exists()]
    if missing:
        raise SystemExit(f"missing entry scripts: {missing}")
    queue += [BACKEND / t for t in TESTS]
    closure: set[Path] = set(queue)
    seen: set[Path] = set()
    while queue:
        f = queue.pop()
        if f in seen:
            continue
        seen.add(f)
        for m in _imports_of(f):
            # ``shared`` is the monorepo's cross-package utility namespace
            # (shared.geo.tiles supplies the JIS mesh codes that the spatial
            # k-fold and the tile manager import at MODULE level). Omitting it
            # shipped national/evaluation/spatial_kfold.py and
            # national/tiling/tile_manager.py in a state where importing them
            # raised ModuleNotFoundError.
            if m.startswith(("national", "scripts", "shared")):
                r = _resolve(m)
                if r and r not in closure:
                    closure.add(r)
                    queue.append(r)
    # package __init__.py for every package directory touched
    for f in list(closure):
        d = f.parent
        while d != BACKEND and BACKEND in d.parents or d == BACKEND / "scripts":
            init = d / "__init__.py"
            if init.exists():
                closure.add(init)
            if d == BACKEND:
                break
            d = d.parent
    return closure


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: Build by-products the checks create inside the assembled tree.
_TRANSIENT_DIRS = {"__pycache__", ".pytest_cache", ".benchmarks"}


#: The template directory is copied verbatim into the package root, so its
#: contents are published without further review. Rather than trying to
#: enumerate what must not ship, this names the complete set that may: any
#: other file appearing there fails the build. That keeps stray working notes,
#: editor state and local configuration out of a public release by default.
_TEMPLATE_ALLOWED = frozenset({
    ".gitignore", "AUTHORS.md", "CITATION.cff", "LICENSE", "README.md",
    "REPRODUCIBILITY.md", "pyproject.toml",
})


def _clean_build_artifacts(output: Path) -> None:
    for d in sorted(output.rglob("*"), key=lambda p: -len(p.parts)):
        if d.is_dir() and d.name in _TRANSIENT_DIRS:
            shutil.rmtree(d)


_IMPORT_PROBE = r'''
import importlib, pathlib, sys, traceback
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root))
bad = []
for p in sorted(root.rglob("*.py")):
    rel = p.relative_to(root)
    if rel.parts[0] not in ("national", "scripts", "shared"):
        continue
    mod = ".".join(rel.with_suffix("").parts)
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    try:
        importlib.import_module(mod)
    except ModuleNotFoundError as e:
        bad.append((mod, f"ModuleNotFoundError: {e.name}"))
    except Exception as e:                      # noqa: BLE001
        bad.append((mod, f"{type(e).__name__}: {e}"))
for mod, err in bad:
    print(f"{mod}\t{err}")
sys.exit(1 if bad else 0)
'''


def _assert_every_module_imports(output: Path) -> None:
    """Every shipped module must import in the assembled tree.

    ``compileall`` only proves the files parse. It cannot see that a module
    imports a package the allowlist never copied -- which is exactly how
    ``national/evaluation/spatial_kfold.py`` and ``national/tiling/
    tile_manager.py`` once shipped importing ``shared.geo.tiles`` from a
    package that was not in the tree. A referee would have found that; this
    check finds it first.
    """
    LOG.info("import probe (every shipped module) ...")
    r = subprocess.run([sys.executable, "-c", _IMPORT_PROBE, str(output)],
                       cwd=output, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise SystemExit("modules that do not import in the assembled tree:\n"
                         + (r.stdout or r.stderr))
    LOG.info("import probe: all shipped modules import cleanly")


def _assert_manifest_is_complete(output: Path, manifest: dict[str, dict]) -> None:
    """Every file in the tree must be in the manifest, and vice versa.

    The manifest is the repository's integrity claim, so a file that is
    shipped but unlisted is exactly as bad as a listed file that is absent.
    The original builder only checked one direction.
    """
    on_disk = {
        str(p.relative_to(output))
        for p in output.rglob("*")
        if p.is_file() and not any(part in _TRANSIENT_DIRS for part in p.parts)
    }
    on_disk.discard("manifest.json")  # the manifest cannot hash itself
    listed = set(manifest)
    unlisted = sorted(on_disk - listed)
    missing = sorted(listed - on_disk)
    if unlisted or missing:
        raise SystemExit(
            f"manifest does not describe the tree: "
            f"{len(unlisted)} shipped-but-unlisted {unlisted[:10]}, "
            f"{len(missing)} listed-but-absent {missing[:10]}"
        )
    LOG.info("manifest completeness: %d files on disk, all listed", len(on_disk))


def build(output: Path, run_checks: bool = True) -> None:
    closure = compute_closure()
    LOG.info("code closure: %d files", len(closure))
    if output.exists():
        raise SystemExit(f"output {output} already exists; remove it first")
    (output / "results").mkdir(parents=True)

    manifest: dict[str, dict] = {}
    for src in sorted(closure):
        rel = src.relative_to(BACKEND)
        dst = output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest[str(rel)] = {"sha256": _sha256(src), "bytes": src.stat().st_size}
    for rel_str in DATA:
        src = BACKEND / rel_str
        if not src.exists():
            raise SystemExit(f"missing data asset: {rel_str}")
        dst = output / rel_str
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest[rel_str] = {"sha256": _sha256(src), "bytes": src.stat().st_size}
    for r in RESULTS:
        src = REPO / r
        if not src.exists():
            raise SystemExit(f"missing result artefact: {r}")
        dst = output / "results" / src.name
        shutil.copy2(src, dst)
        manifest[f"results/{src.name}"] = {"sha256": _sha256(src), "bytes": src.stat().st_size}

    # pre-registration record, verbatim
    (output / "prereg").mkdir(parents=True, exist_ok=True)
    for p in PREREG:
        src = REPO / p
        if not src.exists():
            raise SystemExit(f"missing pre-registration document: {p}")
        dst = output / "prereg" / src.name
        shutil.copy2(src, dst)
        manifest[f"prereg/{src.name}"] = {"sha256": _sha256(src), "bytes": src.stat().st_size}

    # static repo docs from the template dir (README.md, LICENSE, CITATION.cff, ...)
    if TEMPLATE_DIR.exists():
        for src in sorted(TEMPLATE_DIR.iterdir()):
            if not src.is_file():
                continue
            if src.name not in _TEMPLATE_ALLOWED:
                raise SystemExit(
                    f"{TEMPLATE_DIR / src.name} is not in _TEMPLATE_ALLOWED and "
                    "the template directory is published verbatim. Add it "
                    "deliberately, or move it out of the template directory.")
            dst = output / src.name
            shutil.copy2(src, dst)
            manifest[src.name] = {"sha256": _sha256(src), "bytes": src.stat().st_size}

    # re-validate copies
    for rel, meta in manifest.items():
        if _sha256(output / rel) != meta["sha256"]:  # pragma: no cover
            raise SystemExit(f"sha256 mismatch after copy: {rel}")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    LOG.info("manifest: %d files, all sha256 round-trip OK", len(manifest))
    _assert_manifest_is_complete(output, manifest)

    if run_checks:
        LOG.info("compileall ...")
        subprocess.run([sys.executable, "-m", "compileall", "-q", str(output)], check=True)
        _assert_every_module_imports(output)
        LOG.info("pytest (included subset, in assembled tree) ...")
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
                       cwd=output, check=True)
        # The checks litter the tree with bytecode and pytest state. Those are
        # build by-products, not shipped files; leaving them behind would make
        # the manifest an incomplete description of the tree (and put ~1 MB of
        # .pyc in a referee's clone). Remove them, then re-assert completeness.
        _clean_build_artifacts(output)
        _assert_manifest_is_complete(output, manifest)
    LOG.info("companion repo assembled at %s", output)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--skip-checks", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")
    build(a.output.resolve(), run_checks=not a.skip_checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
