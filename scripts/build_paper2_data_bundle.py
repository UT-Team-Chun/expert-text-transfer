#!/usr/bin/env python
"""Assemble the Zenodo **data** bundle for Paper 2 (national text transfer).

Why this exists
---------------
``docs/paper/paper_2_national/ZENODO.md`` specifies a bundle laid out as
``model/ data/ embeddings/ transfer/ runs/ provenance/ prereg/ source_data/``,
and ``data_availability.tex`` promises its contents to referees. Nothing built
it. The pre-existing ``scripts/build_zenodo_release.py`` is a *different*,
superseded package (GeoFM-JP v1.0): it **requires** ``--cube-dir`` and
``--maps-dir`` for the 3-D probabilistic cube and the engineering maps, which
the 2026-08-12 scope note removed from this paper, and it emits the
``predictions/`` + ``quantiles/`` split that ZENODO.md records as abolished.
That script is left alone -- it still describes its own release -- and this one
implements the bundle this paper actually promises.

Design rules
------------
1. **Declared but absent is a build failure.** Every layer this script declares
   must exist, or the build aborts. A bundle that silently omits something
   ``data_availability.tex`` promises is worse than no bundle, because the
   omission is discovered by a referee rather than by us.
2. **Deliberate absences are declared, not silent.** The one thing the
   manuscript mentions that this repo cannot ship (the NOAA storm parsed table)
   is registered in :data:`KNOWN_ABSENT` with a reason, written into
   ``manifest.json`` under ``declared_absent`` and into the generated README.
   The bundle therefore states what it does not contain.
3. **Cross-deposit contracts hold by construction.** ZENODO.md says
   ``provenance/`` and ``prereg/`` are the same sets as the companion code
   repository's ``results/`` and ``prereg/``. Rather than restate them, this
   module imports ``RESULTS`` and ``PREREG`` from
   ``scripts.build_paper2_companion_repo``, so the two deposits cannot drift.
4. **Japan ships PCA-64 only.** See :data:`EMBEDDING_POLICY`.

CLI::

    cd backend
    .venv/bin/python -m scripts.build_paper2_data_bundle \
        --pca64-japan ../data/release/pca64/japan_fullpop \
        --output ../data/releases/paper2_data_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

LOG = logging.getLogger("scripts.build_paper2_data_bundle")

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
PAPER = REPO / "docs/paper/paper_2_national"

RELEASE_NAME = "GeoFM-JP v1.0 -- Paper 2 data bundle"
RELEASE_LICENSE = "CC-BY-4.0"

# --------------------------------------------------------------------------
# Embedding policy.
#
# ``data_availability.tex`` justifies redistributing a derivative of the
# KuniJiban corpus -- which MLIT forbids us to redistribute raw -- on the
# non-invertibility of a *64-principal-component projection*. That argument
# covers PCA-64 features. It does NOT cover the 768-dimensional sentence
# embeddings the evaluation pipeline caches, from which embedding-inversion
# attacks can recover a meaningful fraction of the source text. So:
#
#   Japan (KuniJiban, MLIT non-redistribution)  -> per-fold PCA-64 ONLY.
#   UK (BGS AGS, Open Government Licence)       -> full 768-d, because the
#       source text is itself openly redistributable, which lets a referee
#       reproduce the UK replication end to end INCLUDING its permutation null.
#
# The asymmetry is deliberate and is stated in the generated README.
# --------------------------------------------------------------------------
EMBEDDING_POLICY = {
    "japan": "pca64_only",
    "uk": "full_768_and_source",
}

#: Files a run directory contributes, per ZENODO.md. ``checkpoints/`` and
#: ``wandb/`` are training internals: not needed to reproduce, and they
#: dominate the byte count.
RUN_FILES = ("summary.json", "conformal_mondrian.json", "predictions.npz",
             "diagnostics.png", "results.json")

#: (D) the coordinate-free arm, retrieved from the utens NAS on 2026-08-19.
PT9_RUNS = [f"dkl_natlro_pt9_{r}_{a}"
            for r in ("chubu", "kansai", "kyushu_okinawa", "tohoku")
            for a in ("coordfree", "zfonly")]

#: (C) the identity-join 3-fold cells behind P-T7.
PT7_RUNS = [f"dkl_national_lmc_v5id_sarashina_l2_kfold{i}" for i in (0, 1, 2)]

#: (B) the HGB baseline. ZENODO.md: appendix_b cites ``runs/region_hgb_lro/``
#: but the artefact lives at ``leave_region_out/region_hgb/``. The bundle is
#: renamed to match the manuscript, because the manuscript is what a referee
#: reads first.
HGB_SRC_REL = "data/runs/leave_region_out/region_hgb"
HGB_BUNDLE_NAME = "region_hgb_lro"

#: The model layer. ZENODO.md (M): the only national run holding all four of
#: weights + meta + summary + diagnostics.
MODEL_RUN_REL = "data/runs/dkl_national_full"
#: data_availability.tex says "foundation_model.pt plus .meta.json", so the
#: in-run name is changed on the way in to match the sentence a referee reads.
MODEL_FILES = {
    "foundation_model.pt": "foundation_model.pt",
    "foundation_model.meta.json": "foundation_model.pt.meta.json",
    "summary.json": "summary.json",
    "diagnostics.png": "diagnostics.png",
}

#: ``data/`` layer: (repo-relative source, bundle name, citation hint).
DATA_LAYER = [
    ("data/features/borings_japan_v4id.parquet", "borings_japan_v4id.parquet",
     "Methods, National data stack -- enriched national feature parquet with "
     "the borehole-identity spine (boring_file is the borehole key)"),
    ("data/features/derived/kunijiban_metadata.parquet",
     "kunijiban_metadata.parquet",
     "Methods, Provenance folds -- archive header table (project / client / "
     "contractor / survey year / DTD version)"),
    ("data/features/derived/groundwater_depth.csv", "groundwater_depth.csv",
     "Methods, National data stack -- structured groundwater-depth table"),
]

#: ``transfer/`` layer.
TRANSFER_LAYER = [
    ("data/features/uk_bgs_spt_full.parquet", "uk_bgs_spt_full.parquet",
     "Cross-archive transfer -- UK BGS AGS parsed table (Open Government "
     "Licence), the source of the replication arm"),
]
#: Transfer result tables + their driver outputs, from docs/research.
TRANSFER_RESULTS = [
    "docs/research/2026-06-20_uk_transfer_full.json",
    "docs/research/2026-06-20_uk_transfer_fair.json",
    "docs/research/2026-06-20_uk_transfer_result.json",
    "docs/research/2026-06-21_uk_transfer_leakproof.json",
    "docs/research/2026-06-21_japan_transfer_leakproof.json",
    "docs/research/2026-06-21_storm_transfer_3rd_domain.json",
    "docs/research/2026-06-21_storm_transfer_nosize.json",
]

#: UK 768-d embedding caches (shippable: OGL source).
UK_EMBEDDINGS = [
    "data/features/derived/nc_cache/grouped_uk_lithonly_v2_d1c3ec425705441c_e5.npy",
]

#: Things the manuscript mentions that this repo cannot ship, with the reason
#: that goes into the manifest and the README. Adding an entry here is a
#: deliberate, reviewable act; it is the ONLY way to omit something.
KNOWN_ABSENT: dict[str, str] = {
    "transfer/storm_events_parsed.csv": (
        "The NOAA Storm Events parsed table is not checked into this project: "
        "scripts/storm_transfer_test.py rebuilds it on each run from "
        "--storm-dir, and no copy of the 2026-06-21 build survives. The storm "
        "analysis is a supplementary positive control, and every number it "
        "contributes is reproduced per-state in "
        "provenance/2026-06-21_storm_transfer_3rd_domain.json and "
        "provenance/2026-06-21_storm_transfer_nosize.json, whose _provenance "
        "block names the exact public-domain source (NCEI detail CSVs "
        "2015-2019, EVENT_TYPE=Hail) and filter. The corpus is US public "
        "domain and can be rebuilt with the released script."),
    "embeddings/storm/": (
        "Same cause as transfer/storm_events_parsed.csv: the storm embeddings "
        "were never persisted."),
    "data/soil_text_layers.csv": (
        "Withheld deliberately. Rows of source_data/fig7/ and source_data/fig8/ "
        "name this table as their source, so its absence is stated here rather "
        "than left for a reader to discover: it carries the verbatim KuniJiban "
        "layer narrative, which MLIT's terms forbid us to redistribute and "
        "which the Data availability statement's 'does not contain the raw "
        "narrative text' claim depends on us NOT shipping. What ships is what "
        "is derived from it -- the histogram bin heights in source_data/, and "
        "the non-invertible per-fold PCA-64 features in embeddings/japan/."),
    "embeddings/japan/full_768d": (
        "Deliberately withheld. The KuniJiban corpus is under MLIT "
        "non-redistribution terms, and the Data availability statement's legal "
        "basis for releasing a derivative is the non-invertibility of the "
        "64-component projection. Full 768-dimensional sentence embeddings are "
        "not covered by that argument, so only the per-fold PCA-64 features "
        "ship for Japan. The UK arm, whose source is Open Government Licence, "
        "ships its full-dimensional embeddings instead."),
}

_CONTENT_TYPES = {
    ".pt": "torch_pickle", ".pth": "torch_pickle", ".npz": "numpy_npz",
    ".npy": "binary", ".json": "json", ".csv": "csv", ".parquet": "parquet",
    ".md": "markdown", ".txt": "text", ".png": "png", ".py": "text",
}


def _content_type(suffix: str) -> str:
    return _CONTENT_TYPES.get(suffix.lower(), "binary")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Bundle:
    """Accumulates the tree and its manifest, then validates the copies."""

    output: Path
    layers: dict[str, dict] = field(default_factory=dict)

    def add(self, src: Path, rel: str, hint: str) -> None:
        if not src.exists():
            raise SystemExit(
                f"declared bundle layer is missing on disk: {rel} <- {src}\n"
                "Every layer this builder declares is promised by "
                "data_availability.tex. Fix the input or register the absence "
                "in KNOWN_ABSENT with a reason.")
        if rel in self.layers:
            raise SystemExit(f"duplicate bundle path: {rel}")
        dst = self.output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self.layers[rel] = {
            "path": rel,
            "sha256": _sha256(src),
            "size_bytes": src.stat().st_size,
            "content_type": _content_type(src.suffix),
            "citation_hint": hint,
        }

    def add_tree(self, src_dir: Path, rel_prefix: str, hint: str,
                 min_files: int = 1, require: tuple[str, ...] = ()) -> int:
        """Copy a directory, refusing a missing or half-written one.

        ``rglob`` over a nonexistent path yields nothing and raises nothing, so
        without these guards the builder happily produced a deposit whose
        manifest and README advertised a layer the tree did not contain.
        """
        if not src_dir.is_dir():
            raise SystemExit(
                f"declared bundle layer is not a directory: {rel_prefix} "
                f"<- {src_dir}")
        for name in require:
            if not (src_dir / name).exists():
                raise SystemExit(
                    f"{src_dir} is missing {name}; refusing to ship a partial "
                    f"{rel_prefix} layer")
        n = 0
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                self.add(p, f"{rel_prefix}/{p.relative_to(src_dir)}", hint)
                n += 1
        if n < min_files:
            raise SystemExit(
                f"{rel_prefix}: expected at least {min_files} files, found {n}")
        return n

    # -- validation -------------------------------------------------------
    def validate(self) -> list[str]:
        """Re-hash every copy, and prove the manifest describes the tree."""
        trace = []
        for rel, meta in sorted(self.layers.items()):
            got = _sha256(self.output / rel)
            if got != meta["sha256"]:
                raise SystemExit(f"sha256 mismatch after copy: {rel}")
            trace.append(f"{got}  {rel}")
        # Exclude the three files the builder itself writes at the ROOT, by
        # path and not by name: source_data/ and embeddings/japan/ each carry
        # their own manifest.json and README.md, which are real payload and
        # must stay under the completeness check.
        self_written = {"manifest.json", "README.md", "release_validated.txt"}
        on_disk = {
            str(p.relative_to(self.output))
            for p in self.output.rglob("*")
            if p.is_file()
            and str(p.relative_to(self.output)) not in self_written
        }
        listed = set(self.layers)
        unlisted, missing = sorted(on_disk - listed), sorted(listed - on_disk)
        if unlisted or missing:
            raise SystemExit(
                f"manifest does not describe the tree: "
                f"{len(unlisted)} unlisted {unlisted[:10]}, "
                f"{len(missing)} absent {missing[:10]}")
        LOG.info("validated %d files; manifest describes the tree exactly",
                 len(self.layers))
        return trace


def _calibrated_cells(runs_dir: Path) -> list[Path]:
    """(A) the 23 calibrated DKL cells -- ZENODO.md's defining glob."""
    return sorted(p.parent for p in runs_dir.glob(
        "dkl_national_*/conformal_mondrian.json"))


def collect_runs(runs_dir: Path) -> list[tuple[Path, str]]:
    """Resolve ZENODO.md's 35-run manifest to (source dir, bundle name)."""
    out: list[tuple[Path, str]] = []
    cal = _calibrated_cells(runs_dir)
    if len(cal) != 23:
        raise SystemExit(
            f"expected 23 calibrated cells (ZENODO.md section A), found "
            f"{len(cal)}: {[p.name for p in cal]}")
    out += [(p, p.name) for p in cal]
    out.append((REPO / HGB_SRC_REL, HGB_BUNDLE_NAME))
    out += [(runs_dir / n, n) for n in PT7_RUNS]
    out += [(runs_dir / n, n) for n in PT9_RUNS]
    names = [n for _, n in out]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate run names: {names}")
    if len(out) != 35:
        raise SystemExit(f"expected 35 runs, resolved {len(out)}")
    return out


def build(output: Path, *, pca64_japan: Path,
          runs_dir: Path, skip_source_data: bool = False) -> dict:
    from scripts.build_paper2_companion_repo import PREREG, RESULTS

    if output.exists():
        raise SystemExit(f"output {output} already exists; remove it first")
    output.mkdir(parents=True)
    b = Bundle(output)

    # ---- model ---------------------------------------------------------
    model_src = REPO / MODEL_RUN_REL
    for name, bundle_name in MODEL_FILES.items():
        b.add(model_src / name, f"model/{bundle_name}",
              "Data availability -- trained DKL+SVGP probabilistic regressor")
    # summary.json is the trainer's own file, shipped byte-for-byte so it stays
    # identical to the run artefact; its run_name is the trainer default and it
    # names no cell. The note about WHICH cell these weights are therefore goes
    # in its own file rather than by editing a run artefact.
    mr = output / "model" / "README.md"
    mr.write_text(MODEL_README, encoding="utf-8")
    b.layers["model/README.md"] = {
        "path": "model/README.md", "sha256": _sha256(mr),
        "size_bytes": mr.stat().st_size, "content_type": "markdown",
        "citation_hint": "Which evaluation cell the released weights come from",
    }

    # ---- data ----------------------------------------------------------
    for rel, name, hint in DATA_LAYER:
        b.add(REPO / rel, f"data/{name}", hint)

    # ---- embeddings ----------------------------------------------------
    # Not optional: without it the deposit's own manifest, README and bundled
    # ZENODO.md all advertise a Japan feature layer that is not there.
    # ``manifest.json`` is written last by the exporter, so requiring it is
    # exactly a completion signal; ``keys.parquet`` is what makes the features
    # joinable to a fold, a target and a borehole at all.
    n = b.add_tree(pca64_japan, "embeddings/japan",
                   "Data availability -- per-fold PCA-64 text features "
                   "(the redistributable derived representation; the "
                   "768-d embeddings they project from are NOT released "
                   "for Japan)",
                   min_files=10,
                   require=("manifest.json", "keys.parquet"))
    LOG.info("embeddings/japan: %d files", n)
    for rel in UK_EMBEDDINGS:
        src = REPO / rel
        b.add(src, f"embeddings/uk/{src.name}",
              "Data availability -- UK per-layer embeddings. Full 768-d is "
              "shippable here because the BGS source text is Open Government "
              "Licence, so the UK replication reproduces end to end including "
              "its permutation null")

    # ---- transfer ------------------------------------------------------
    for rel, name, hint in TRANSFER_LAYER:
        b.add(REPO / rel, f"transfer/{name}", hint)
    for rel in TRANSFER_RESULTS:
        src = REPO / rel
        b.add(src, f"transfer/{src.name}",
              "Cross-archive transfer -- result table and driver output")

    # ---- runs ----------------------------------------------------------
    n_run_files = 0
    for src_dir, bundle_name in collect_runs(runs_dir):
        if not src_dir.is_dir():
            raise SystemExit(f"run directory missing: {src_dir}")
        present = [f for f in RUN_FILES if (src_dir / f).exists()]
        if not present:
            raise SystemExit(
                f"run {bundle_name} has none of {RUN_FILES} in {src_dir}")
        for f in present:
            b.add(src_dir / f, f"runs/{bundle_name}/{f}",
                  "Evaluation cell artefact (leave-region-out / spatial K-fold)")
            n_run_files += 1
    LOG.info("runs: 35 cells, %d files", n_run_files)

    # ---- provenance + prereg (same sets as the companion code repo) -----
    for rel in RESULTS:
        src = REPO / rel
        b.add(src, f"provenance/{src.name}",
              "Provenance record behind a reported number")
    for rel in PREREG:
        src = REPO / rel
        b.add(src, f"prereg/{src.name}",
              "Pre-registration, amendment and verdicts, verbatim")

    # ---- source data ---------------------------------------------------
    if not skip_source_data:
        LOG.info("generating source_data/ ...")
        r = subprocess.run(
            [sys.executable, "-m", "scripts.export_figure_source_data",
             "--out", str(output / "_sd_tmp")],
            cwd=BACKEND, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(
                f"source-data export failed:\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}")
        tmp = output / "_sd_tmp"
        n = b.add_tree(tmp, "source_data",
                       "Source data underlying the figure of this number",
                       min_files=20)
        shutil.rmtree(tmp)
        LOG.info("source_data: %d files", n)

    # ---- root docs -----------------------------------------------------
    b.add(PAPER / "ZENODO.md", "ZENODO.md",
          "Data availability -- manifest schema and figure-to-data crosswalk")
    b.add(BACKEND / "notebooks/zenodo_inference_demo.py",
          "zenodo_inference_demo.py", "Minimal CPU-only inference demo")
    req = output / "requirements.txt"
    req.write_text(REQUIREMENTS, encoding="utf-8")
    b.layers["requirements.txt"] = {
        "path": "requirements.txt", "sha256": _sha256(req),
        "size_bytes": req.stat().st_size, "content_type": "text",
        "citation_hint": "Minimal CPU inference dependency list",
    }

    # ---- manifest + README + validation --------------------------------
    manifest = {
        "release": RELEASE_NAME,
        "license": RELEASE_LICENSE,
        "n_files": len(b.layers),
        "total_bytes": sum(m["size_bytes"] for m in b.layers.values()),
        "embedding_policy": EMBEDDING_POLICY,
        "declared_absent": KNOWN_ABSENT,
        "layers": [b.layers[k] for k in sorted(b.layers)],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "README.md").write_text(_readme(manifest), encoding="utf-8")
    trace = b.validate()
    (output / "release_validated.txt").write_text(
        "OK\n" + "\n".join(trace) + "\n", encoding="utf-8")
    LOG.info("bundle ready: %s (%d files, %.2f GB)", output, len(b.layers),
             manifest["total_bytes"] / 1e9)
    return manifest


MODEL_README = """\
# Released weights

`foundation_model.pt` is the **v1** national cell (`dkl_national_full`): RBF
kernel with a linear mean, random inducing-point initialisation.

It is released rather than the v2 cell because the two are indistinguishable
from fold noise on the national evaluation --- RMSE / MAE 7.544 / 4.657 (v1)
against 7.546 / 4.655 (v2) --- and v1 is the only national run in the project
that carries weights, metadata, training summary and diagnostics together. The
v2 cell's evaluation artefacts ship too, at `runs/dkl_national_full_v2/`.

`summary.json` is the trainer's own output, copied byte-for-byte so that it
remains identical to the run artefact it came from. Read it accordingly: its
`run_name` is the trainer's default (`kanto_smoke`) and names no cell, and its
metrics are the in-region holdout, not the national figures quoted above. The
national numbers come from the evaluation cells under `runs/`.

`foundation_model.pt.meta.json` is the run's `foundation_model.meta.json`,
renamed to match the wording of the paper's Data availability statement.
"""


REQUIREMENTS = """\
# Minimal CPU-only inference dependencies for the Paper 2 data bundle.
# Lower bounds only; see the companion code repository's pyproject.toml for the
# full runtime manifest.
numpy>=1.26
pandas>=2.2
pyarrow>=14.0
scikit-learn>=1.5
torch>=2.5
gpytorch>=1.13
"""


def _readme(manifest: dict) -> str:
    by_top: dict[str, list[dict]] = {}
    for layer in manifest["layers"]:
        by_top.setdefault(layer["path"].split("/")[0] if "/" in layer["path"]
                          else "(root)", []).append(layer)
    lines = [
        f"# {manifest['release']}", "",
        f"Licence: {manifest['license']}. "
        f"{manifest['n_files']} files, "
        f"{manifest['total_bytes'] / 1e9:.2f} GB.", "",
        "`manifest.json` is the source of truth: every file below carries a",
        "sha256, a size and a content type there, and `release_validated.txt`",
        "records the hash of every file as it was written.", "",
        "## Contents", "",
        "| Layer | Files | Bytes |", "|---|---:|---:|",
    ]
    for top in sorted(by_top):
        ls = by_top[top]
        lines.append(f"| `{top}` | {len(ls)} | "
                     f"{sum(x['size_bytes'] for x in ls):,} |")
    lines += [
        "", "## Embeddings: why the two archives are treated differently", "",
        "Japan ships **per-fold PCA-64 features only**. The KuniJiban corpus is",
        "under MLIT non-redistribution terms, and the Data availability",
        "statement's basis for releasing a derivative of it is that a",
        "64-component projection is non-invertible. Full-dimensional sentence",
        "embeddings are not covered by that argument, so they are withheld.",
        "",
        "The UK ships its **full 768-dimensional** embeddings and its parsed",
        "source table, because BGS AGS data is Open Government Licence. The UK",
        "replication can therefore be reproduced end to end, including its",
        "permutation null, from this bundle alone.", "",
        "## Declared absences", "",
        "This bundle states what it does not contain:", "",
    ]
    for path, why in sorted(manifest["declared_absent"].items()):
        lines += [f"- **`{path}`** -- {why}", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pca64-japan", type=Path, required=True,
                    help="directory of per-fold PCA-64 .npy + keys.parquet + "
                         "manifest.json from scripts.export_pca64_features")
    ap.add_argument("--runs-dir", type=Path, default=REPO / "data/runs")
    ap.add_argument("--skip-source-data", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    build(a.output, pca64_japan=a.pca64_japan, runs_dir=a.runs_dir,
          skip_source_data=a.skip_source_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
