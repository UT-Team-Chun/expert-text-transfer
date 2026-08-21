#!/usr/bin/env python
"""Export machine-readable source data for every figure of Paper B'
(national text-transfer manuscript, target venue *Nature Communications*).

``data_availability.tex`` promises that "source data underlying each figure
are provided as machine-readable arrays inside the Zenodo bundle,
cross-referenced by figure number". This module is what makes that promise
true: it writes one directory per figure --

.. code-block:: text

    <out>/manifest.json          -- figure -> files index (row counts, provenance)
    <out>/README.md              -- same index, human-readable
    <out>/fig1/...csv
    <out>/fig2/...csv
    ...
    <out>/figS4/...csv

-- following the ``source_data/fig<N>/`` layout declared by
``docs/paper/paper_2_national/ZENODO.md``.

Design contract
---------------

**The figure list is discovered, never hard-coded.** ``discover_manuscript_figures``
scans every ``\\includegraphics`` inside a ``figure`` environment of
``sections/*.tex`` and resolves the printed figure number from ``main.aux``
(the last successful LaTeX build). A figure added to the manuscript therefore
shows up here automatically, and a figure with no registered exporter is a
hard error rather than a silent omission.

**What is exported is what is plotted.** Each file holds the marks of one
figure/panel with the column names its axes use -- bar heights, curve points,
heatmap cells, histogram bins. Not the upstream population the marks were
reduced from (that ships as its own bundle layer), and not per-fold detail the
figure never draws.

**Provenance survives into the release.** Every row carries two extra
columns:

``source_kind``
    one of ``run_artefact`` (a number computed by a training/evaluation run
    and read back out of that run's artefact), ``research_json`` (a committed
    result JSON under ``docs/research/``), ``derived_dataset`` (a feature
    table shipped in the bundle), or ``collated_constant`` (a value that
    exists only as a Python literal in the plotting script, transcribed there
    by hand from a table or an off-repo run).
``source``
    the artefact path, or ``build_paper2_figs.py::<CONSTANT>`` for a collated
    constant.

The collated-constant rows are the honest part: Figs 3, 8 and S3 rest on
numbers that were transcribed into ``build_paper2_figs.py`` rather than read
back from a run directory, and a reader of the release can see exactly which
rows those are. Where a figure reads a JSON under ``docs/research/``, this
exporter reads the same JSON, so the source data and the figure cannot
diverge; where a figure reads a Python literal, this exporter imports that
literal from ``build_paper2_figs`` rather than keeping a second copy.

**CSV throughout.** Every figure in this manuscript plots a small table --
the largest export is the 30,000-point study-area scatter (~1 MB) and the
next largest is 43 rows. CSV is the format a referee can open in one click
without a toolchain, it diffs, and it carries the per-row provenance columns
that a bare ``.npy`` could not. Histogram panels export binned counts (the
bars that are drawn), not the millions of underlying values.

CLI
---

.. code-block:: bash

    python -m scripts.export_figure_source_data --out ../data/release/source_data
    python -m scripts.export_figure_source_data --figures fig3 figS3 --out /tmp/sd
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

LOG = logging.getLogger("scripts.export_figure_source_data")

REPO = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
PAPER_DIR_DEFAULT = REPO / "docs/paper/paper_2_national"

# --- provenance vocabulary -------------------------------------------------

KIND_RUN = "run_artefact"
KIND_RESEARCH = "research_json"
KIND_DATASET = "derived_dataset"
KIND_CONST = "collated_constant"

PROVENANCE_KINDS: tuple[str, ...] = (
    KIND_RUN, KIND_RESEARCH, KIND_DATASET, KIND_CONST,
)

PROVENANCE_KIND_DOC: dict[str, str] = {
    KIND_RUN: ("read back out of a training/evaluation run artefact "
               "(summary.json / conformal_mondrian.json)"),
    KIND_RESEARCH: "read out of a committed result JSON under docs/research/",
    KIND_DATASET: "computed from a feature table shipped in the bundle",
    KIND_CONST: ("a Python literal in build_paper2_figs.py, transcribed by "
                 "hand from a manuscript table or an off-repo run -- no "
                 "run artefact stands behind it in this repository"),
}

PROVENANCE_COLUMNS: tuple[str, ...] = ("source_kind", "source")


class MissingInput(RuntimeError):
    """An input a figure's source data is derived from is not available."""


class UnregisteredFigure(RuntimeError):
    """The manuscript includes a figure this exporter does not know about."""


# ---------------------------------------------------------------------------
# Manuscript discovery
# ---------------------------------------------------------------------------

_FIGURE_ENV = re.compile(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.S)
_INCLUDEGRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")
_LABEL = re.compile(r"\\label\{([^}]*)\}")


@dataclass(frozen=True)
class ManuscriptFigure:
    """One ``\\includegraphics`` that the manuscript actually typesets."""

    number: str          # printed figure number, e.g. "3" or "S1"
    label: str           # LaTeX label, e.g. "fig:model_inversion"
    graphic: str         # graphic file name, e.g. "fig5_model_inversion.pdf"
    section: str         # sections/*.tex it is included from

    @property
    def slug(self) -> str:
        """Directory name inside the source-data tree (``fig3``, ``figS1``)."""
        return f"fig{self.number}"

    @property
    def sort_key(self) -> tuple[int, float, str]:
        supp = self.number.upper().startswith("S")
        digits = self.number[1:] if supp else self.number
        try:
            n = float(digits)
        except ValueError:
            n = float("inf")
        return (1 if supp else 0, n, self.number)


def _aux_figure_numbers(aux_path: Path) -> dict[str, str]:
    """Map ``\\label`` -> printed figure number from a LaTeX ``.aux``."""
    if not aux_path.exists():
        raise MissingInput(
            f"{aux_path} not found -- run `make` in the paper directory once "
            f"so figure numbers can be resolved")
    out: dict[str, str] = {}
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^{}]*)\}", aux_path.read_text(
            encoding="utf-8", errors="replace")):
        out.setdefault(m.group(1), m.group(2))
    return out


def discover_manuscript_figures(
    paper_dir: Path = PAPER_DIR_DEFAULT,
) -> list[ManuscriptFigure]:
    """Every figure the manuscript typesets, with its printed number.

    Derived from the ``\\includegraphics`` calls inside ``figure``
    environments of ``<paper_dir>/sections/*.tex``, numbered from
    ``<paper_dir>/main.aux``. Nothing here is hard-coded, so a figure added
    to the manuscript is picked up on the next run.
    """
    sections = sorted((paper_dir / "sections").glob("*.tex"))
    if not sections:
        raise MissingInput(f"no sections/*.tex under {paper_dir}")
    numbers = _aux_figure_numbers(paper_dir / "main.aux")

    figures: list[ManuscriptFigure] = []
    for tex in sections:
        body_all = tex.read_text(encoding="utf-8", errors="replace")
        seen_in_env = 0
        for env in _FIGURE_ENV.finditer(body_all):
            body = env.group(1)
            graphics = _INCLUDEGRAPHICS.findall(body)
            seen_in_env += len(graphics)
            if not graphics:
                continue
            labels = _LABEL.findall(body)
            if not labels:
                raise MissingInput(
                    f"{tex.name}: figure including {graphics} carries no "
                    f"\\label, so it cannot be keyed by figure number")
            label = labels[0]
            if label not in numbers:
                raise MissingInput(
                    f"{tex.name}: \\label{{{label}}} has no entry in main.aux "
                    f"-- rebuild the manuscript so figure numbers are current")
            for g in graphics:
                figures.append(ManuscriptFigure(
                    number=numbers[label], label=label, graphic=g,
                    section=tex.name))
        n_total = len(_INCLUDEGRAPHICS.findall(body_all))
        if n_total != seen_in_env:
            raise MissingInput(
                f"{tex.name}: {n_total - seen_in_env} \\includegraphics call(s) "
                f"outside a figure environment -- cannot key them by number")
    figures.sort(key=lambda f: f.sort_key)
    return figures


# ---------------------------------------------------------------------------
# Sibling-module loading
# ---------------------------------------------------------------------------

_SIBLING_CACHE: dict[str, ModuleType] = {}


def _load_sibling(name: str) -> ModuleType:
    """Import a sibling script by path (works in monorepo and companion repo).

    Mirrors ``tests/national/test_build_paper2_figs.py``: loading by file
    location avoids picking up a stale ``scripts.<name>`` from a different
    worktree on ``sys.path``.
    """
    if name in _SIBLING_CACHE:
        return _SIBLING_CACHE[name]
    try:  # never pop a window open; these modules import pyplot at import time
        import matplotlib
        matplotlib.use("Agg", force=False)
    except Exception:  # pragma: no cover - matplotlib always present here
        pass
    src = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_sd_{name}", src)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise MissingInput(f"cannot load {src}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_sd_{name}"] = mod
    spec.loader.exec_module(mod)
    _SIBLING_CACHE[name] = mod
    return mod


# ---------------------------------------------------------------------------
# Export plumbing
# ---------------------------------------------------------------------------

@dataclass
class ExportedFile:
    name: str
    rows: int
    columns: list[str]
    description: str
    source_kinds: list[str]
    sources: list[str]
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "rows": self.rows,
            "columns": self.columns,
            "description": self.description,
            "source_kinds": self.source_kinds,
            "sources": self.sources,
        }
        if self.annotations:
            d["annotations"] = self.annotations
        return d


@dataclass
class ExportContext:
    """Where the exporters read from."""

    repo: Path = REPO
    paper_dir: Path = PAPER_DIR_DEFAULT
    runs_dir: Path = REPO / "data/runs"
    research_dir: Path = REPO / "docs/research"
    parquet: Path = REPO / "data/features/borings_japan_v4id.parquet"
    layers_csv: Path = REPO / "data/features/derived/soil_text_layers.csv"
    max_scatter_points: int = 30_000

    @property
    def bpf(self) -> ModuleType:
        return _load_sibling("build_paper2_figs")

    @property
    def bfp(self) -> ModuleType:
        return _load_sibling("build_forest_plot")

    def research(self, name: str) -> Path:
        p = self.research_dir / name
        if not p.exists():
            raise MissingInput(f"result JSON not found: {p}")
        return p

    def research_json(self, name: str) -> dict[str, Any]:
        return json.loads(self.research(name).read_text())

    def rel(self, path: Path) -> str:
        """Bundle-relative POSIX path, for the ``source`` column.

        The ``source`` column exists so a reader of the Zenodo deposit can find
        the file a row came from, so it is written in the deposit's own
        coordinates, exactly as ZENODO.md's provenance table declares:
        ``runs/<cell>/<file>`` and ``provenance/<name>.json``. Emitting the
        repo path instead produced strings like ``data/runs/...``, and the
        deposit's ``data/`` holds three files -- so the prefix resolved to
        nothing for the reader it was written for.
        """
        try:
            r = path.resolve().relative_to(self.repo.resolve()).as_posix()
        except ValueError:
            return path.as_posix()
        if r.startswith("data/runs/"):
            return r[len("data/"):]
        if r.startswith("docs/research/"):
            return "provenance/" + r[len("docs/research/"):]
        return r


def _write_csv(out_dir: Path, name: str, columns: list[str],
               rows: list[dict[str, Any]], description: str,
               annotations: dict[str, Any] | None = None) -> ExportedFile:
    """Write one CSV and describe it for the manifest."""
    for col in PROVENANCE_COLUMNS:
        if col not in columns:
            raise AssertionError(
                f"{name}: exported files must carry a '{col}' column")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    kinds = sorted({str(r["source_kind"]) for r in rows})
    bad = [k for k in kinds if k not in PROVENANCE_KINDS]
    if bad:
        raise AssertionError(f"{name}: unknown source_kind(s) {bad}")
    LOG.info("wrote %s (%d rows)", path, len(rows))
    return ExportedFile(
        name=name, rows=len(rows), columns=list(columns),
        description=description, source_kinds=kinds,
        sources=sorted({str(r["source"]) for r in rows}),
        annotations=annotations or {},
    )


def _const_source(constant: str) -> str:
    return f"backend/scripts/build_paper2_figs.py::{constant}"


# ---------------------------------------------------------------------------
# Per-figure exporters
#
# Keyed by graphic file name, so the manuscript's ``\includegraphics`` is the
# thing that selects an exporter. Each returns the files it wrote.
# ---------------------------------------------------------------------------

def _export_fig1_deployment(ctx: ExportContext, out: Path) -> list[ExportedFile]:
    """Fig 1 is a schematic. Its only numeric marks are the two subtitle
    annotations, which the figure now prints from module-level constants that
    are themselves read off the join audit; both are exported with the audit
    value alongside so the rounding is visible."""
    bpf = ctx.bpf
    audit_name = bpf.FIG1_ANNOTATION_SOURCE_JSON
    audit = ctx.research_json(audit_name)
    src = ctx.rel(ctx.research(audit_name))
    jd = audit["join_delta"]
    note = ("Fig. 1 is a schematic: every other mark is a labelled box or "
            "arrow encoding the order of the data-generating process, not a "
            "measured quantity.")
    cols = ["quantity", "value", "unit", "value_as_drawn", "artefact_value",
            "drawn_as", "note", "source_kind", "source"]
    rows = [
        {"quantity": "corpus_spt_records",
         "value": bpf.FIG1_CORPUS_RECORDS, "unit": "SPT records",
         "value_as_drawn": f"{bpf.FIG1_CORPUS_RECORDS / 1e6:.2f}M",
         "artefact_value": jd["n_rows"],
         "drawn_as": "subtitle annotation", "note": note,
         "source_kind": KIND_RESEARCH,
         "source": f"{src}#join_delta.n_rows"},
        {"quantity": "text_bearing_share",
         "value": bpf.FIG1_TEXT_BEARING_PCT, "unit": "% of SPT records",
         "value_as_drawn": f"{bpf.FIG1_TEXT_BEARING_PCT}%",
         "artefact_value": jd["file_join_match_rate_pct"],
         "drawn_as": "subtitle annotation", "note": note,
         "source_kind": KIND_RESEARCH,
         "source": f"{src}#join_delta.file_join_match_rate_pct"},
    ]
    return [_write_csv(
        out, "fig1_annotated_quantities.csv", cols, rows,
        "The two numeric quantities annotated on the Fig. 1 schematic, with "
        "the join-audit values they are rounded from.")]


def _export_fig4_lro_gap(ctx: ExportContext, out: Path) -> list[ExportedFile]:
    """Leave-region-out bar chart + its two reference lines."""
    bpf = ctx.bpf
    lro = bpf._load_lro_rmse(ctx.runs_dir)
    missing = [r for r in bpf.LRO_REGIONS if r not in lro]
    if missing:
        raise MissingInput(
            f"missing LRO run summaries under {ctx.runs_dir} for: "
            f"{', '.join(missing)}")
    order = sorted(lro, key=lambda r: lro[r]["rmse"])  # plotted top-to-bottom
    bar_cols = ["region", "region_label", "heldout_rmse_spt_n",
                "heldout_mae_spt_n", "bar_color", "source_kind", "source"]
    bars = [{
        "region": r,
        "region_label": bpf.LRO_REGION_LABELS.get(r, r.replace("_", " / ")),
        "heldout_rmse_spt_n": lro[r]["rmse"],
        "heldout_mae_spt_n": lro[r]["mae"],
        "bar_color": "#d62728" if r == "kyushu_okinawa" else "#1f77b4",
        "source_kind": KIND_RUN,
        "source": (f"runs/dkl_national_lro_{r}/summary.json"
                   f"#spatial_kfold[0]"),
    } for r in order]

    import inspect
    kanto_ref = float(inspect.signature(bpf.fig4_lro_gap)
                      .parameters["kanto_ref_rmse"].default)
    fv2_summary = ctx.runs_dir / "dkl_national_full_v2" / "summary.json"
    national_ref = float(bpf.national_kfold_rmse(ctx.runs_dir))
    ref_cols = ["reference", "reference_label", "heldout_rmse_spt_n",
                "line_style", "source_kind", "source"]
    refs = [
        {"reference": "kanto_in_region", "reference_label": "Kanto in-region",
         "heldout_rmse_spt_n": kanto_ref, "line_style": "dashed",
         "source_kind": KIND_CONST,
         "source": _const_source("fig4_lro_gap(kanto_ref_rmse=...)")
                   + " -- companion Kanto study best cell"},
        {"reference": "national_random_kfold",
         "reference_label": "National random K-fold",
         "heldout_rmse_spt_n": national_ref, "line_style": "dotted",
         "source_kind": KIND_RUN if fv2_summary.exists() else KIND_CONST,
         "source": ("runs/dkl_national_full_v2/summary.json"
                    "#mean(spatial_kfold[].rmse)" if fv2_summary.exists()
                    else _const_source("national_kfold_rmse(fallback=...)"))},
    ]
    return [
        _write_csv(out, "fig2_lro_heldout_rmse.csv", bar_cols, bars,
                   "One bar per leave-region-out cell, in plotted order "
                   "(ascending RMSE, drawn top to bottom)."),
        _write_csv(out, "fig2_reference_lines.csv", ref_cols, refs,
                   "The two vertical reference lines drawn over the bars."),
    ]


def _export_fig5_model_inversion(ctx: ExportContext, out: Path) -> list[ExportedFile]:
    """The ranking-inversion bars. Both panels are collated constants: the
    numbers were transcribed into the plotting script from the manuscript's
    own cross-region section and the companion study's results table, and no
    run directory in this repository stands behind them."""
    bpf = ctx.bpf
    src_a = (_const_source("KANTO_INREGION_BARS")
             + " -- transcribed from sections/07_cross_region_transfer.tex "
               "and the companion Kanto study's results table")
    src_b = (_const_source("LRO_AVERAGE_BARS")
             + " -- transcribed from sections/07_cross_region_transfer.tex "
               "(8-region leave-region-out means)")
    a_cols = ["model", "heldout_rmse_spt_n", "bar_color",
              "source_kind", "source"]
    a_rows = [{"model": r["model"], "heldout_rmse_spt_n": r["rmse"],
               "bar_color": r["color"], "source_kind": KIND_CONST,
               "source": src_a} for r in bpf.KANTO_INREGION_BARS]
    b_cols = ["model", "mean_rmse_spt_n", "std_rmse_spt_n", "bar_color",
              "source_kind", "source"]
    b_rows = [{"model": r["model"], "mean_rmse_spt_n": r["rmse"],
               "std_rmse_spt_n": r["std"], "bar_color": r["color"],
               "source_kind": KIND_CONST, "source": src_b}
              for r in bpf.LRO_AVERAGE_BARS]
    return [
        _write_csv(out, "fig3_panel_a_kanto_in_region.csv", a_cols, a_rows,
                   "Panel (a): in-region Kanto held-out RMSE per model, in "
                   "plotted left-to-right order."),
        _write_csv(out, "fig3_panel_b_lro_average.csv", b_cols, b_rows,
                   "Panel (b): leave-region-out 8-region mean RMSE per model "
                   "with the plotted error-bar half-width (std)."),
    ]


def _export_fig11_leakage_controls(ctx: ExportContext, out: Path) -> list[ExportedFile]:
    """Leakage-control bars, straight out of the two result JSONs the figure
    reads."""
    bpf = ctx.bpf
    files = {"Japan": "2026-06-21_text_leakage_japan.json",
             "UK": "2026-06-21_text_leakage_uk.json"}
    cols = ["domain", "variant", "variant_label", "content_pct", "n_neg",
            "sign_p", "tag_drawn", "source_kind", "source"]
    rows: list[dict[str, Any]] = []
    for domain, fname in files.items():
        data = ctx.research_json(fname)["variants"]
        src = ctx.rel(ctx.research(fname))
        for key, label in bpf._LEAK_ORDER:
            v = data[key]
            sig_ns = (v.get("sign_p") or 1.0) > 0.05
            rows.append({
                "domain": domain, "variant": key,
                "variant_label": label.replace("\n", " "),
                "content_pct": v["content_pct"],
                "n_neg": v.get("n_neg", ""),
                "sign_p": v.get("sign_p", ""),
                "tag_drawn": "n.s." if sig_ns else str(v.get("n_neg", "")),
                "source_kind": KIND_RESEARCH,
                "source": f"{src}#variants.{key}",
            })
    return [_write_csv(
        out, "fig4_leakage_controls.csv", cols, rows,
        "One row per (domain, text representation) bar, in plotted "
        "left-to-right order; `tag_drawn` is the label printed under the bar.")]


def _export_fig4_descriptor_families(ctx: ExportContext, out: Path) -> list[ExportedFile]:
    """Descriptor-mechanism panels, from the P-T10 ablation JSON."""
    bpf = ctx.bpf
    fname = "2026-08-12_descriptor_families_japan.json"
    data = ctx.research_json(fname)
    src = ctx.rel(ctx.research(fname))
    arms = data["arms"]

    a_cols = ["arm", "arm_label", "n_features", "content_pct",
              "regions_negative", "bar_color", "source_kind", "source"]
    a_rows = [
        {"arm": "parser_text_only", "arm_label": "text-derived features only",
         "n_features": arms["parser_text_only"]["n_features"],
         "content_pct": arms["parser_text_only"]["content_pct"],
         "regions_negative": arms["parser_text_only"]["regions_negative"],
         "bar_color": "#2a7ab0", "source_kind": KIND_RESEARCH,
         "source": f"{src}#arms.parser_text_only"},
        {"arm": "parser_with_codes", "arm_label": "+ AIST archive codes",
         "n_features": arms["parser_with_codes"]["n_features"],
         "content_pct": arms["parser_with_codes"]["content_pct"],
         "regions_negative": arms["parser_with_codes"]["regions_negative"],
         "bar_color": "#f0a35e", "source_kind": KIND_RESEARCH,
         "source": f"{src}#arms.parser_with_codes"},
    ]

    full_effect = abs(float(arms["parser_text_only"]["content_pct"]))
    fams = [(key, label, arms[key]) for key, label in bpf._DESCRIPTOR_FAMILIES
            if key in arms]
    fams.sort(key=lambda kv: float(kv[2]["attenuation_pp"]), reverse=True)
    b_cols = ["family", "family_label", "attenuation_pp",
              "effect_surviving_pp", "full_text_only_effect_pp",
              "regions_negative", "source_kind", "source"]
    b_rows = [{
        "family": key, "family_label": label,
        "attenuation_pp": float(arm["attenuation_pp"]),
        "effect_surviving_pp": full_effect - float(arm["attenuation_pp"]),
        "full_text_only_effect_pp": full_effect,
        "regions_negative": arm.get("regions_negative", ""),
        "source_kind": KIND_RESEARCH, "source": f"{src}#arms.{key}",
    } for key, label, arm in fams]
    return [
        _write_csv(out, "fig5_panel_a_parser_rungs.csv", a_cols, a_rows,
                   "Panel (a): the two parser rungs, with the feature count "
                   "printed inside each bar."),
        _write_csv(out, "fig5_panel_b_leave_one_family_out.csv", b_cols,
                   b_rows,
                   "Panel (b): stacked bars, in plotted order (descending "
                   "attenuation, top to bottom); `effect_surviving_pp` is the "
                   "grey segment and `attenuation_pp` the coloured tip.",
                   annotations={"full_text_only_effect_pp": full_effect}),
    ]


def _export_fig5_fewshot_curve(ctx: ExportContext, out: Path) -> list[ExportedFile]:
    """Few-shot curves and the two horizontal references per panel."""
    bpf = ctx.bpf
    fname = "2026-08-12_fewshot_borehole_curve.json"
    data = ctx.research_json(fname)
    src = ctx.rel(ctx.research(fname))
    curve_cols = ["direction", "panel", "n_target_boreholes", "arm",
                  "spearman_rho_mean", "spearman_rho_std",
                  "source_kind", "source"]
    ref_cols = ["direction", "panel", "reference", "spearman_rho",
                "line_style", "source_kind", "source"]
    curve_rows: list[dict[str, Any]] = []
    ref_rows: list[dict[str, Any]] = []
    for key, title in bpf.FEWSHOT_DIRECTIONS:
        blk = data[key]
        panel = title.split(")")[0].strip("( ")
        for budget in bpf.FEWSHOT_BUDGETS:
            cell = blk["fewshot_curve"][f"boreholes={budget}"]
            for arm in ("depth_only", "depth_text"):
                curve_rows.append({
                    "direction": key, "panel": panel,
                    "n_target_boreholes": budget, "arm": arm,
                    "spearman_rho_mean": cell[arm]["spearman_rho_mean"],
                    "spearman_rho_std": cell[arm].get("spearman_rho_std", 0.0),
                    "source_kind": KIND_RESEARCH,
                    "source": (f"{src}#{key}.fewshot_curve."
                               f"boreholes={budget}.{arm}"),
                })
        ref_rows.append({
            "direction": key, "panel": panel,
            "reference": "zero_shot_shuffled_embedding_null",
            "spearman_rho": blk["zero_shot_holdout"]["depth_shuffled"]["spearman_rho"],
            "line_style": "dotted", "source_kind": KIND_RESEARCH,
            "source": f"{src}#{key}.zero_shot_holdout.depth_shuffled"})
        ref_rows.append({
            "direction": key, "panel": panel,
            "reference": "target_trained_depth_only",
            "spearman_rho": blk["reference_target_trained_depth_only"]["spearman_rho"],
            "line_style": "dashdot", "source_kind": KIND_RESEARCH,
            "source": f"{src}#{key}.reference_target_trained_depth_only"})
    return [
        _write_csv(out, "fig6_fewshot_curve.csv", curve_cols, curve_rows,
                   "Curve points; budgets are drawn at evenly spaced "
                   "categorical x positions in the listed order, bands are "
                   "mean +/- 1 s.d. over seeds."),
        _write_csv(out, "fig6_reference_lines.csv", ref_cols, ref_rows,
                   "The two horizontal reference lines drawn in both panels."),
    ]


def _export_fig2_study_area(ctx: ExportContext, out: Path) -> list[ExportedFile]:
    """The plotted scatter subsample and the inset histogram's bars."""
    import numpy as np
    import pandas as pd

    bpf = ctx.bpf
    if not ctx.parquet.exists():
        raise MissingInput(f"feature parquet not found: {ctx.parquet}")
    df = pd.read_parquet(ctx.parquet, columns=["latitude_deg", "longitude_deg",
                                               "regime_code"])
    if len(df) > ctx.max_scatter_points:  # same subsample the figure draws
        df = df.sample(n=ctx.max_scatter_points, random_state=42)
    codes = df["regime_code"].astype("Int64")
    src_pq = ctx.rel(ctx.parquet)
    scatter_cols = ["longitude_deg", "latitude_deg", "regime_code",
                    "regime_name", "source_kind", "source"]
    scatter_rows: list[dict[str, Any]] = []
    per_regime: dict[str, int] = {}
    for code in range(len(bpf.REGIME_NAMES)):
        sub = df.loc[codes == code]
        if len(sub) == 0:
            continue
        per_regime[bpf.REGIME_NAMES[code]] = int(len(sub))
        for lon, lat in zip(sub["longitude_deg"], sub["latitude_deg"]):
            scatter_rows.append({
                "longitude_deg": float(lon), "latitude_deg": float(lat),
                "regime_code": code, "regime_name": bpf.REGIME_NAMES[code],
                "source_kind": KIND_DATASET, "source": src_pq})
    files = [_write_csv(
        out, "fig7_borehole_scatter.csv", scatter_cols, scatter_rows,
        "Every point drawn in the main panel: the random "
        f"{ctx.max_scatter_points:,}-point subsample (seed 42) of the "
        "national feature table, grouped by regime as the figure draws it.",
        annotations={"n_per_regime": per_regime,
                     "subsample_seed": 42,
                     "max_points": ctx.max_scatter_points})]

    if ctx.layers_csv.exists():
        counts = pd.read_csv(ctx.layers_csv, usecols=["file_path"]
                             ).groupby("file_path").size()
        corpus_mean = float(counts.mean())
        clipped = counts[counts < 60].values
        hist, edges = np.histogram(clipped, bins=30)
        hist_cols = ["bin_left", "bin_right", "n_borings",
                     "source_kind", "source"]
        hist_rows = [{"bin_left": float(edges[i]),
                      "bin_right": float(edges[i + 1]),
                      "n_borings": int(hist[i]),
                      "source_kind": KIND_DATASET,
                      "source": ctx.rel(ctx.layers_csv)}
                     for i in range(len(hist))]
        files.append(_write_csv(
            out, "fig7_inset_layers_per_boring_hist.csv", hist_cols,
            hist_rows,
            "Inset histogram bars: layers per boring, 30 bins over borings "
            "with fewer than 60 layers (the tail is clipped for legibility "
            "in the bins only). The annotated mean is over the full corpus.",
            annotations={"corpus_mean_layers_per_boring": corpus_mean,
                         "n_borings_total": int(len(counts)),
                         "histogram_clip_upper": 60, "bins": 30}))
    else:
        raise MissingInput(f"layer text CSV not found: {ctx.layers_csv}")
    return files


def _export_fig3_llm_text_gain(ctx: ExportContext, out: Path) -> list[ExportedFile]:
    """Panels (a)/(b) are a collated constant; panel (c) is a histogram of the
    layer-narrative corpus."""
    import numpy as np
    import pandas as pd

    bpf = ctx.bpf
    src = (_const_source("HELDOUT_3FOLD")
           + " -- 3-fold means collated from the 15 K-fold runs listed in "
             "the constant's comment (/mnt/nas/runs/dkl_national_lmc_v*), "
             "not re-read from a run directory in this repository")
    ab_cols = ["level", "variant", "ell", "m_inducing", "heldout_rmse_n",
               "heldout_mae_n", "heldout_rmse_gw", "bar_color",
               "source_kind", "source"]
    colors = {"v4 (baseline)": "#7f7f7f", "v5 Sarashina": "#1f77b4",
              "v5 Ruri": "#ff7f0e"}
    ab_rows = [{
        "level": f"l={r['ell']}, m={r['m'] // 1000}k",
        "variant": r["variant"], "ell": r["ell"], "m_inducing": r["m"],
        "heldout_rmse_n": r["rmse_n"], "heldout_mae_n": r["mae_n"],
        "heldout_rmse_gw": r["rmse_gw"],
        "bar_color": colors.get(r["variant"], ""),
        "source_kind": KIND_CONST, "source": src,
    } for r in bpf.HELDOUT_3FOLD]
    files = [_write_csv(
        out, "fig8_panel_ab_heldout_3fold.csv", ab_cols, ab_rows,
        "Panels (a) and (b): one row per (inducing-point level, embedding "
        "variant) bar. `heldout_rmse_n` is panel (a), `heldout_rmse_gw` is "
        "panel (b), and `heldout_mae_n` backs panel (a)'s percentage "
        "annotation.")]

    if not ctx.layers_csv.exists():
        raise MissingInput(f"layer text CSV not found: {ctx.layers_csv}")
    cl = pd.read_csv(ctx.layers_csv, usecols=["char_length"])["char_length"]
    cl = cl.clip(upper=200)
    hist, edges = np.histogram(cl.values, bins=40)
    c_cols = ["bin_left", "bin_right", "n_layers", "source_kind", "source"]
    c_rows = [{"bin_left": float(edges[i]), "bin_right": float(edges[i + 1]),
               "n_layers": int(hist[i]), "source_kind": KIND_DATASET,
               "source": ctx.rel(ctx.layers_csv)} for i in range(len(hist))]
    files.append(_write_csv(
        out, "fig8_panel_c_char_length_hist.csv", c_cols, c_rows,
        "Panel (c): the 40 histogram bars of observation_text character "
        "length, clipped at 200 characters exactly as plotted.",
        annotations={"median": float(cl.median()), "mean": float(cl.mean()),
                     "n_layers": int(cl.shape[0]), "clip_upper": 200,
                     "bins": 40}))
    return files


def _export_fig6_conformal_heatmap(ctx: ExportContext, out: Path,
                                   primary_run: str = "dkl_national_full_v2"
                                   ) -> list[ExportedFile]:
    """Heatmap cells for the primary run + the marginal-gap strip."""
    bpf = ctx.bpf
    primary = ctx.runs_dir / primary_run / "conformal_mondrian.json"
    if not primary.exists():
        raise MissingInput(f"conformal table not found: {primary}")
    c = json.loads(primary.read_text())
    n_cal = c.get("n_cal_per_regime", {})
    a_cols = ["alpha", "regime_code", "regime_name", "coverage",
              "coverage_gap", "n_cal", "n_eval", "source_kind", "source"]
    a_rows: list[dict[str, Any]] = []
    for alpha in c["alphas"]:
        per_reg = c["per_regime"].get(str(alpha), {})
        for code, name in enumerate(bpf.REGIME_NAMES):
            entry = per_reg.get(str(code))
            if entry is None:
                continue  # blank heatmap cell -- nothing is drawn
            a_rows.append({
                "alpha": alpha, "regime_code": code, "regime_name": name,
                "coverage": float(entry["coverage"]),
                "coverage_gap": float(entry["coverage"]) - float(alpha),
                "n_cal": n_cal.get(str(code), ""),
                "n_eval": entry.get("n_eval", ""),
                "source_kind": KIND_RUN,
                "source": (f"runs/{primary_run}/conformal_mondrian.json"
                           f"#per_regime.{alpha}.{code}")})

    b_cols = ["run", "abs_gap_mondrian", "alpha", "source_kind", "source"]
    b_rows: list[dict[str, Any]] = []
    for cj in sorted(ctx.runs_dir.glob("dkl_national_*/conformal_mondrian.json")):
        cc = json.loads(cj.read_text())
        mg = cc.get("marginal", {}).get("0.95", {}).get("gap_mondrian")
        if mg is None:
            continue
        b_rows.append({
            "run": cj.parent.name.replace("dkl_national_", ""),
            "abs_gap_mondrian": abs(float(mg)), "alpha": 0.95,
            "source_kind": KIND_RUN,
            "source": (f"runs/{cj.parent.name}/conformal_mondrian.json"
                       f"#marginal.0.95.gap_mondrian")})
    if not b_rows:
        raise MissingInput(
            f"no dkl_national_*/conformal_mondrian.json under {ctx.runs_dir}")
    return [
        _write_csv(out, "figS1_panel_a_per_regime_gap.csv", a_cols, a_rows,
                   f"Panel (a): one row per drawn heatmap cell of the primary "
                   f"run ({primary_run}); `coverage_gap` is the plotted "
                   f"empirical - nominal value.",
                   annotations={"primary_run": primary_run}),
        _write_csv(out, "figS1_panel_b_marginal_gap.csv", b_cols, b_rows,
                   "Panel (b): one point per calibrated cell in the marginal "
                   "coverage-gap strip."),
    ]


def _export_fig_forest_content(ctx: ExportContext, out: Path) -> list[ExportedFile]:
    """Per-held-out-unit points and the family mean/CI diamonds, taken from
    ``build_forest_plot.collect`` so the figure and its source data cannot
    disagree (the bootstrap CI is recomputed by that same function)."""
    fam = ctx.bfp.collect(ctx.research_dir)
    fam_files = {
        "Japan — leave-region-out (8 regions)": "2026-08-18_grouped_null_japan_s42.json",
        "UK — leave-region-out (5 regions)": "2026-08-18_grouped_null_uk.json",
        "Japan — leave-lithology-class-out (LM)": "2026-07-04_geo_fold_v2.json",
        "Japan — leave-geological-era-out (LM)": "2026-07-04_geo_fold_v2.json",
        "US storms — leave-state-out (30 states)": "2026-06-21_storm_transfer_nosize.json",
    }
    u_cols = ["family", "held_out_unit", "content_pct", "point_color",
              "source_kind", "source"]
    m_cols = ["family", "mean_content_pct", "ci_lo", "ci_hi", "n_units",
              "source_kind", "source"]
    unknown = [f["label"] for f in fam if f["label"] not in fam_files]
    if unknown:
        raise MissingInput(
            f"build_forest_plot.collect returned family labels this exporter "
            f"cannot attribute to a result JSON: {unknown}")
    u_rows: list[dict[str, Any]] = []
    m_rows: list[dict[str, Any]] = []
    for f in fam:
        src = f"docs/research/{fam_files[f['label']]}"
        for name, v in sorted(f["units"].items(), key=lambda kv: kv[1]):
            u_rows.append({
                "family": f["label"], "held_out_unit": name,
                "content_pct": float(v),
                "point_color": "#1f6fb2" if v < 0 else "#c0392b",
                "source_kind": KIND_RESEARCH, "source": src})
        m_rows.append({
            "family": f["label"], "mean_content_pct": f["mean"],
            "ci_lo": f["ci"][0], "ci_hi": f["ci"][1],
            "n_units": len(f["units"]),
            "source_kind": KIND_RESEARCH,
            "source": (f"{src} (unit bootstrap, 10^4 resamples, seed 42, "
                       f"recomputed by build_forest_plot._ci)")})
    return [
        _write_csv(out, "figS2_per_unit_content.csv", u_cols, u_rows,
                   "Every plotted point: the genuine-content effect for one "
                   "held-out unit, in plotted order within each family."),
        _write_csv(out, "figS2_family_means.csv", m_cols, m_rows,
                   "The diamond-and-bar rows: family mean with its unit "
                   "bootstrap interval."),
    ]


def _export_fig9_tta_delta_rmse(ctx: ExportContext, out: Path) -> list[ExportedFile]:
    """TTA delta-RMSE bars and the sigma-collapse inset. A collated constant:
    the per-(region, strategy) records live only as a Python literal."""
    bpf = ctx.bpf
    src = (_const_source("TTA_RECORDS")
           + " -- per-(region, strategy) TTA records collated into the "
             "plotting script; no run directory in this repository holds them")
    by_key = {(r["region"], r["strategy"]): r for r in bpf.TTA_RECORDS}
    cols = ["region", "strategy", "strategy_label", "rmse_before",
            "rmse_after", "delta_rmse", "mae_before", "mae_after",
            "mean_std_before", "mean_std_after", "n_adapted_rows",
            "wall_clock_s", "bar_color", "source_kind", "source"]
    rows: list[dict[str, Any]] = []
    for region in bpf.FIG9_REGION_ORDER:
        for strat in bpf.FIG9_STRATEGIES:
            r = by_key.get((region, strat))
            if r is None:
                continue  # no bar is drawn for a missing cell
            rows.append({
                "region": region, "strategy": strat,
                "strategy_label": bpf.FIG9_STRATEGY_LABELS[strat],
                "rmse_before": r["rmse_before"], "rmse_after": r["rmse_after"],
                "delta_rmse": r["rmse_after"] - r["rmse_before"],
                "mae_before": r["mae_before"], "mae_after": r["mae_after"],
                "mean_std_before": r["mean_std_before"],
                "mean_std_after": r["mean_std_after"],
                "n_adapted_rows": r["n_adapted_rows"],
                "wall_clock_s": r["wall_clock_s"],
                "bar_color": bpf.FIG9_STRATEGY_COLORS[strat],
                "source_kind": KIND_CONST, "source": src})
    return [_write_csv(
        out, "figS3_tta_delta_rmse.csv", cols, rows,
        "One row per plotted bar, in x-axis order. `delta_rmse` is the bar "
        "height of the main panel; `mean_std_after` is the point plotted in "
        "the predictive-sigma inset.")]


def _export_fig10_cross_national(ctx: ExportContext, out: Path) -> list[ExportedFile]:
    """Three-domain decomposition (panel a) + per-region points (panel b)."""
    bpf = ctx.bpf
    names = {
        "Japanese boreholes": "2026-06-21_japan_transfer_leakproof.json",
        "UK boreholes": "2026-06-21_uk_transfer_leakproof.json",
        "US storm reports": "2026-06-21_storm_transfer_3rd_domain.json",
    }
    a_cols = ["domain", "no_text_rmse", "shuffled_rmse", "text_rmse",
              "capacity_pct", "content_pct", "drawn_as",
              "source_kind", "source"]
    b_cols = ["domain", "region", "content_pct", "source_kind", "source"]
    a_rows: list[dict[str, Any]] = []
    b_rows: list[dict[str, Any]] = []
    for domain, fname in names.items():
        d = ctx.research_json(fname)
        src = ctx.rel(ctx.research(fname))
        a_rows.append({
            "domain": domain,
            "no_text_rmse": d["no_text"]["mean_rmse"],
            "shuffled_rmse": d["shuffled"]["mean_rmse"],
            "text_rmse": d["text"]["mean_rmse"],
            "capacity_pct": bpf._capacity_pct(d),
            "content_pct": bpf._content_pct(d),
            "drawn_as": "paired bars", "source_kind": KIND_RESEARCH,
            "source": src})
        tpr, spr = d["text"]["per_region"], d["shuffled"]["per_region"]
        for region in tpr:
            if region not in spr:
                continue
            b_rows.append({
                "domain": domain, "region": region,
                "content_pct": 100 * (tpr[region] - spr[region]) / spr[region],
                "source_kind": KIND_RESEARCH,
                "source": f"{src}#text.per_region/shuffled.per_region"})
    ns_name = "2026-06-21_storm_transfer_nosize.json"
    ns = ctx.research_json(ns_name)
    a_rows.append({
        "domain": "US storm reports (hail-size words stripped)",
        "no_text_rmse": ns["no_text"]["mean_rmse"],
        "shuffled_rmse": ns["shuffled"]["mean_rmse"],
        "text_rmse": ns["text"]["mean_rmse"],
        "capacity_pct": bpf._capacity_pct(ns),
        "content_pct": bpf._content_pct(ns),
        "drawn_as": "horizontal marker over the storm content bar",
        "source_kind": KIND_RESEARCH, "source": ctx.rel(ctx.research(ns_name))})
    return [
        _write_csv(out, "figS4_panel_a_decomposition.csv", a_cols, a_rows,
                   "Panel (a): the capacity/provenance-null and "
                   "genuine-content bar heights per domain, with the RMSE "
                   "arms they are computed from, plus the size-stripped "
                   "storm marker."),
        _write_csv(out, "figS4_panel_b_per_region_content.csv", b_cols, b_rows,
                   "Panel (b): every plotted point -- the genuine-content "
                   "effect for one held-out region/state (x jitter is "
                   "cosmetic and seeded, so it is not part of the data)."),
    ]


EXPORTERS: dict[str, Callable[[ExportContext, Path], list[ExportedFile]]] = {
    "fig1_deployment.pdf": _export_fig1_deployment,
    "fig4_lro_gap.pdf": _export_fig4_lro_gap,
    "fig5_model_inversion.pdf": _export_fig5_model_inversion,
    "fig11_leakage_controls.pdf": _export_fig11_leakage_controls,
    "fig4_descriptor_families.pdf": _export_fig4_descriptor_families,
    "fig5_fewshot_curve.pdf": _export_fig5_fewshot_curve,
    "fig2_study_area.pdf": _export_fig2_study_area,
    "fig3_llm_text_gain.pdf": _export_fig3_llm_text_gain,
    "fig6_conformal_heatmap.pdf": _export_fig6_conformal_heatmap,
    "fig_forest_content.pdf": _export_fig_forest_content,
    "fig9_tta_delta_rmse.pdf": _export_fig9_tta_delta_rmse,
    "fig10_cross_national.pdf": _export_fig10_cross_national,
}

GENERATORS: dict[str, str] = {
    "fig1_deployment.pdf": "build_paper2_figs.py::fig1_deployment",
    "fig4_lro_gap.pdf": "build_paper2_figs.py::fig4_lro_gap",
    "fig5_model_inversion.pdf": "build_paper2_figs.py::fig5_model_inversion",
    "fig11_leakage_controls.pdf": "build_paper2_figs.py::fig11_leakage_controls",
    "fig4_descriptor_families.pdf": "build_paper2_figs.py::fig4_descriptor_families",
    "fig5_fewshot_curve.pdf": "build_paper2_figs.py::fig5_fewshot_curve",
    "fig2_study_area.pdf": "build_paper2_figs.py::fig2_study_area",
    "fig3_llm_text_gain.pdf": "build_paper2_figs.py::fig3_llm_text_gain",
    "fig6_conformal_heatmap.pdf": "build_paper2_figs.py::fig6_conformal_heatmap",
    "fig_forest_content.pdf": "build_forest_plot.py::main",
    "fig9_tta_delta_rmse.pdf": "build_paper2_figs.py::fig9_tta_delta_rmse",
    "fig10_cross_national.pdf": "build_paper2_figs.py::fig10_cross_national",
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _readme(manifest: dict[str, Any]) -> str:
    lines = [
        "# Source data by figure",
        "",
        "Machine-readable source data for every figure of the manuscript, "
        "one directory per figure, keyed by the **printed figure number**.",
        "",
        f"Generated by `{manifest['generated_by']}`; figure numbers resolved "
        f"from `{manifest['figure_numbers_from']}`.",
        "",
        "Every CSV carries two provenance columns:",
        "",
        "| `source_kind` | meaning |",
        "| --- | --- |",
    ]
    for kind in PROVENANCE_KINDS:
        lines.append(f"| `{kind}` | {PROVENANCE_KIND_DOC[kind]} |")
    lines += [
        "",
        "`source` names the artefact (or the plotting-script constant) each "
        "row's numbers come from.",
        "",
        "| Figure | Label | Graphic | File | Rows | Provenance |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for fig in manifest["figures"]:
        for f in fig["files"]:
            lines.append(
                f"| {fig['figure']} | `{fig['label']}` | `{fig['graphic']}` | "
                f"`{fig['directory']}/{f['name']}` | {f['rows']} | "
                f"{', '.join(f['source_kinds'])} |")
    collated = [f["figure"] for f in manifest["figures"]
                if KIND_CONST in f["provenance_kinds"]]
    lines += [
        "",
        "## Figures that rest on a collated constant",
        "",
        ("Figures " + ", ".join(collated) + " include at least one value that "
         "exists only as a Python literal in the plotting script -- "
         "transcribed from a manuscript table or an off-repo run rather than "
         "read back out of a run artefact in the code release. The affected "
         "rows are the ones whose `source_kind` is `collated_constant`.")
        if collated else
        "None: every exported value is read from a released artefact.",
        "",
    ]
    return "\n".join(lines)


def export_all(out_dir: Path,
               ctx: ExportContext | None = None,
               figures: list[ManuscriptFigure] | None = None,
               only: list[str] | None = None,
               skip_missing: bool = False) -> dict[str, Any]:
    """Export source data for every figure the manuscript typesets.

    Parameters
    ----------
    out_dir
        The ``source_data/`` root. ``fig<N>/`` subdirectories are created
        under it, plus ``manifest.json`` and ``README.md``.
    ctx
        Where to read inputs from. Defaults to the monorepo layout.
    figures
        Pre-discovered figure list; discovered from the manuscript if omitted.
    only
        Restrict to these slugs (``fig3``, ``figS1``).
    skip_missing
        Record a figure as skipped instead of raising when one of its inputs
        is unavailable. Off by default so a release build cannot quietly ship
        an incomplete crosswalk.

    Returns the manifest dict (also written to ``out_dir/manifest.json``).
    """
    ctx = ctx or ExportContext()
    figures = figures if figures is not None else discover_manuscript_figures(
        ctx.paper_dir)
    if only:
        wanted = set(only)
        figures = [f for f in figures if f.slug in wanted]

    unregistered = [f.graphic for f in figures if f.graphic not in EXPORTERS]
    if unregistered:
        raise UnregisteredFigure(
            "the manuscript includes figures with no registered source-data "
            f"exporter: {', '.join(sorted(set(unregistered)))}. Add an entry "
            f"to EXPORTERS in {Path(__file__).name} -- "
            "data_availability.tex promises source data for every figure.")

    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for fig in figures:
        fig_dir = out_dir / fig.slug
        try:
            written = EXPORTERS[fig.graphic](ctx, fig_dir)
        except MissingInput as exc:
            if not skip_missing:
                raise
            LOG.warning("figure %s skipped: %s", fig.number, exc)
            entries.append({
                "figure": fig.number, "label": fig.label,
                "graphic": fig.graphic, "section": fig.section,
                "generator": GENERATORS.get(fig.graphic, ""),
                "directory": fig.slug, "provenance_kinds": [],
                "skipped": str(exc), "files": []})
            continue
        kinds = sorted({k for f in written for k in f.source_kinds})
        entries.append({
            "figure": fig.number, "label": fig.label, "graphic": fig.graphic,
            "section": fig.section,
            "generator": GENERATORS.get(fig.graphic, ""),
            "directory": fig.slug, "provenance_kinds": kinds,
            "files": [f.to_json() for f in written]})

    manifest = {
        "generated_by": "backend/scripts/export_figure_source_data.py",
        "paper": "docs/paper/paper_2_national",
        "figure_numbers_from": "docs/paper/paper_2_national/main.aux",
        "layout": "source_data/fig<N>/ -- one directory per printed figure number",
        "provenance_kinds": {k: PROVENANCE_KIND_DOC[k] for k in PROVENANCE_KINDS},
        "n_figures": len(entries),
        "figures": entries,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "README.md").write_text(_readme(manifest))
    LOG.info("wrote %s and README.md (%d figures)",
             out_dir / "manifest.json", len(entries))
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path,
                   default=REPO / "data/release/source_data",
                   help="source_data/ root to write (default: %(default)s)")
    p.add_argument("--paper-dir", type=Path, default=PAPER_DIR_DEFAULT)
    p.add_argument("--runs-dir", type=Path, default=REPO / "data/runs")
    p.add_argument("--research-dir", type=Path, default=REPO / "docs/research")
    p.add_argument("--parquet", type=Path,
                   default=REPO / "data/features/borings_japan_v4id.parquet")
    p.add_argument("--layers-csv", type=Path,
                   default=REPO / "data/features/derived/soil_text_layers.csv")
    p.add_argument("--figures", nargs="+", default=None,
                   help="restrict to these slugs, e.g. fig3 figS1")
    p.add_argument("--skip-missing", action="store_true",
                   help=("record figures whose inputs are unavailable as "
                         "skipped instead of failing"))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    ctx = ExportContext(paper_dir=args.paper_dir, runs_dir=args.runs_dir,
                        research_dir=args.research_dir, parquet=args.parquet,
                        layers_csv=args.layers_csv)
    manifest = export_all(args.out, ctx=ctx, only=args.figures,
                          skip_missing=args.skip_missing)
    n_files = sum(len(f["files"]) for f in manifest["figures"])
    n_rows = sum(x["rows"] for f in manifest["figures"] for x in f["files"])
    LOG.info("%d figures, %d files, %d rows -> %s",
             manifest["n_figures"], n_files, n_rows, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
