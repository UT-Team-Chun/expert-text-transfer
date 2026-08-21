"""Tests for ``scripts/export_figure_source_data``.

``data_availability.tex`` promises machine-readable source data for *every*
figure, cross-referenced by figure number. These tests hold that promise to
account:

- the list of figures is re-derived here, independently, from the
  ``\\includegraphics`` calls in ``docs/paper/paper_2_national/sections/*.tex``,
  so a figure added to the manuscript without an exporter fails the suite
  rather than shipping a broken crosswalk;
- every exported file parses as CSV and carries the provenance columns;
- for the figures whose numbers live as Python literals in the plotting
  script (Fig 3 model inversion, Fig 8 held-out 3-fold, Fig S3 TTA) the
  exported values are compared against those literals, and for a
  JSON-backed figure (Fig 4 leakage controls) against the result JSON.

Run artefacts (``data/runs/``) are not in git, so they are fabricated in
``tmp_path``; the ``docs/research/`` result JSONs are committed and are read
for real.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # noqa: E402  # headless before any pyplot import

import pytest  # noqa: E402

_BACKEND = Path(__file__).resolve().parents[2]
_REPO = _BACKEND.parent
_PAPER = _REPO / "docs/paper/paper_2_national"
_SRC = _BACKEND / "scripts" / "export_figure_source_data.py"

# The figure crosswalk is derived from the manuscript's own sources, so these
# tests need `sections/*.tex` and `main.aux`. The companion CODE repository
# deliberately does not ship the manuscript, and its build runs this suite as
# its gate -- so skip cleanly there rather than failing a build for an input
# that is correctly absent.
_HAVE_PAPER = (_PAPER / "sections").is_dir() and (_PAPER / "main.aux").exists()
pytestmark = pytest.mark.skipif(
    not _HAVE_PAPER,
    reason="manuscript sources are not present (expected in the code-only "
           "companion repository)")


def _load(name: str, src: Path):
    spec = importlib.util.spec_from_file_location(name, src)
    assert spec is not None and spec.loader is not None, f"cannot load {src}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


efsd = _load("export_figure_source_data_under_test", _SRC)
bpf = efsd._load_sibling("build_paper2_figs")


# ============================================================
# An independent reading of the manuscript
# ============================================================

_FIG_ENV = re.compile(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.S)
_GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")
_LABEL = re.compile(r"\\label\{([^}]*)\}")


def manuscript_graphics() -> dict[str, str]:
    """``{graphic file name: figure label}`` straight from the .tex sources.

    Deliberately re-implemented here instead of importing the exporter's own
    discovery, so the test fails if the exporter ever stops seeing a figure
    the manuscript typesets.
    """
    found: dict[str, str] = {}
    for tex in sorted((_PAPER / "sections").glob("*.tex")):
        body_all = tex.read_text(encoding="utf-8", errors="replace")
        for env in _FIG_ENV.finditer(body_all):
            body = env.group(1)
            labels = _LABEL.findall(body)
            for g in _GRAPHIC.findall(body):
                found[g] = labels[0] if labels else ""
    return found


# ============================================================
# Fixtures: fabricated run artefacts + feature tables
# ============================================================

_REGIME_N = len(bpf.REGIME_NAMES)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _conformal(seed: float) -> dict[str, Any]:
    alphas = [0.5, 0.8, 0.95]
    return {
        "alphas": alphas,
        "per_regime": {
            str(a): {
                str(c): {"name": bpf.REGIME_NAMES[c],
                         "n_eval": 1000 + c, "n_cal": 900 + c,
                         "coverage": a + 0.001 * (c + 1) + seed,
                         "uses_marginal_fallback": False}
                for c in range(_REGIME_N)
            } for a in alphas
        },
        "marginal": {str(a): {"gap_mondrian": -0.0004 + seed,
                              "gap_marginal": 0.0002} for a in alphas},
        "n_cal_per_regime": {str(c): 900 + c for c in range(_REGIME_N)},
    }


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("runs")
    for i, region in enumerate(bpf.LRO_REGIONS):
        _write_json(d / f"dkl_national_lro_{region}" / "summary.json",
                    {"spatial_kfold": [{"rmse": 13.0 + i, "mae": 9.0 + i}]})
    _write_json(d / "dkl_national_full_v2" / "summary.json",
                {"spatial_kfold": [{"rmse": 7.5}, {"rmse": 7.6}]})
    for i, cell in enumerate(("full_v2", "matern52_v2", "censored_v2")):
        _write_json(d / f"dkl_national_{cell}" / "conformal_mondrian.json",
                    _conformal(0.0005 * i))
    return d


@pytest.fixture(scope="module")
def parquet(tmp_path_factory) -> Path:
    import pandas as pd
    d = tmp_path_factory.mktemp("features")
    n = 40
    df = pd.DataFrame({
        "latitude_deg": [34.0 + 0.01 * i for i in range(n)],
        "longitude_deg": [135.0 + 0.01 * i for i in range(n)],
        "regime_code": [i % _REGIME_N for i in range(n)],
    })
    p = d / "borings.parquet"
    df.to_parquet(p)
    return p


@pytest.fixture(scope="module")
def layers_csv(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("layers")
    p = d / "soil_text_layers.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["file_path", "layer_idx", "observation_text", "char_length"])
        for b in range(12):
            for layer in range(3 + b % 5):
                w.writerow([f"/x/{b}.html", layer, "sand", 10 + 3 * layer + b])
    return p


@pytest.fixture(scope="module")
def ctx(runs_dir, parquet, layers_csv):
    return efsd.ExportContext(
        paper_dir=_PAPER, runs_dir=runs_dir, research_dir=_REPO / "docs/research",
        parquet=parquet, layers_csv=layers_csv, max_scatter_points=25)


@pytest.fixture(scope="module")
def exported(tmp_path_factory, ctx) -> tuple[Path, dict[str, Any]]:
    out = tmp_path_factory.mktemp("source_data")
    manifest = efsd.export_all(out, ctx=ctx)
    return out, manifest


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _rows_for(out: Path, manifest: dict[str, Any], number: str,
              name: str) -> list[dict[str, str]]:
    entry = next(f for f in manifest["figures"] if f["figure"] == number)
    return _read_csv(out / entry["directory"] / name)


# ============================================================
# Coverage of the manuscript
# ============================================================

def test_discovery_sees_exactly_the_manuscript_figures():
    graphics = manuscript_graphics()
    assert graphics, "no \\includegraphics found -- test is looking in the wrong place"
    found = efsd.discover_manuscript_figures(_PAPER)
    assert {f.graphic for f in found} == set(graphics)
    for f in found:
        assert graphics[f.graphic] == f.label
        assert f.number, f"{f.graphic} has no printed figure number"


def test_every_manuscript_figure_has_a_registered_exporter():
    missing = sorted(set(manuscript_graphics()) - set(efsd.EXPORTERS))
    assert not missing, (
        "these figures are typeset by the manuscript but have no source-data "
        f"exporter: {missing}")


def test_exporter_writes_a_file_for_every_manuscript_figure(exported):
    out, manifest = exported
    graphics = manuscript_graphics()
    by_graphic = {f["graphic"]: f for f in manifest["figures"]}
    assert set(by_graphic) == set(graphics)
    for graphic, entry in by_graphic.items():
        assert entry["files"], f"{graphic}: no source-data file written"
        assert "skipped" not in entry, f"{graphic}: {entry.get('skipped')}"
        fig_dir = out / entry["directory"]
        assert fig_dir.is_dir()
        assert entry["directory"] == f"fig{entry['figure']}"
        for f in entry["files"]:
            assert (fig_dir / f["name"]).exists()


def test_figure_numbers_come_from_the_aux(exported):
    _, manifest = exported
    numbers = {f["label"]: f["figure"] for f in manifest["figures"]}
    # spot-check against main.aux directly
    aux = efsd._aux_figure_numbers(_PAPER / "main.aux")
    for label, number in numbers.items():
        assert aux[label] == number


# ============================================================
# Every exported file is well-formed
# ============================================================

def test_every_exported_file_parses_and_matches_the_manifest(exported):
    out, manifest = exported
    for entry in manifest["figures"]:
        for f in entry["files"]:
            path = out / entry["directory"] / f["name"]
            rows = _read_csv(path)
            assert rows, f"{path} has no data rows"
            assert len(rows) == f["rows"]
            assert list(rows[0]) == f["columns"]


def test_every_row_carries_usable_provenance(exported):
    out, manifest = exported
    for entry in manifest["figures"]:
        for f in entry["files"]:
            for row in _read_csv(out / entry["directory"] / f["name"]):
                assert row["source_kind"] in efsd.PROVENANCE_KINDS
                assert row["source"].strip(), f"{f['name']}: empty source"


def test_manifest_and_readme_are_written(exported):
    out, manifest = exported
    assert json.loads((out / "manifest.json").read_text())["n_figures"] == \
        len(manifest["figures"])
    readme = (out / "README.md").read_text()
    assert "collated_constant" in readme
    for entry in manifest["figures"]:
        assert entry["directory"] in readme


def test_collated_constant_figures_are_flagged_as_such(exported):
    """The distinction between a run artefact and a hand-collated constant
    must survive into the release."""
    _, manifest = exported
    kinds = {f["figure"]: f["provenance_kinds"] for f in manifest["figures"]}
    # Fig 3 (model inversion), Fig 8 (held-out 3-fold) and Fig S3 (TTA) rest
    # on Python literals; Fig 2 and Fig S1 are read back from run artefacts.
    assert efsd.KIND_CONST in kinds["3"]
    assert efsd.KIND_CONST in kinds["8"]
    assert efsd.KIND_CONST in kinds["S3"]
    assert kinds["S1"] == [efsd.KIND_RUN]
    assert efsd.KIND_RUN in kinds["2"]


# ============================================================
# Exported values == plotted values
# ============================================================

def test_fig3_matches_the_model_inversion_constants(exported):
    """Fig 3 is one of the four figures that had no source data at all."""
    out, manifest = exported
    a = _rows_for(out, manifest, "3", "fig3_panel_a_kanto_in_region.csv")
    assert [r["model"] for r in a] == [r["model"] for r in bpf.KANTO_INREGION_BARS]
    for row, const in zip(a, bpf.KANTO_INREGION_BARS):
        assert float(row["heldout_rmse_spt_n"]) == pytest.approx(const["rmse"])
        assert row["bar_color"] == const["color"]
        assert row["source_kind"] == efsd.KIND_CONST
        assert "KANTO_INREGION_BARS" in row["source"]

    b = _rows_for(out, manifest, "3", "fig3_panel_b_lro_average.csv")
    assert [r["model"] for r in b] == [r["model"] for r in bpf.LRO_AVERAGE_BARS]
    for row, const in zip(b, bpf.LRO_AVERAGE_BARS):
        assert float(row["mean_rmse_spt_n"]) == pytest.approx(const["rmse"])
        assert float(row["std_rmse_spt_n"]) == pytest.approx(const["std"])
        assert "LRO_AVERAGE_BARS" in row["source"]


def test_figS3_matches_the_tta_records(exported):
    out, manifest = exported
    rows = _rows_for(out, manifest, "S3", "figS3_tta_delta_rmse.csv")
    assert len(rows) == len(bpf.TTA_RECORDS)
    by_key = {(r["region"], r["strategy"]): r for r in bpf.TTA_RECORDS}
    for row in rows:
        const = by_key[(row["region"], row["strategy"])]
        assert float(row["rmse_before"]) == pytest.approx(const["rmse_before"])
        assert float(row["rmse_after"]) == pytest.approx(const["rmse_after"])
        # the plotted bar height
        assert float(row["delta_rmse"]) == pytest.approx(
            const["rmse_after"] - const["rmse_before"])
        # the inset point
        assert float(row["mean_std_after"]) == pytest.approx(
            const["mean_std_after"])
        assert row["source_kind"] == efsd.KIND_CONST
    # plotted x order
    assert [r["region"] for r in rows[::len(bpf.FIG9_STRATEGIES)]] == \
        list(bpf.FIG9_REGION_ORDER)


def test_fig8_matches_the_heldout_3fold_constant(exported):
    out, manifest = exported
    rows = _rows_for(out, manifest, "8", "fig8_panel_ab_heldout_3fold.csv")
    assert len(rows) == len(bpf.HELDOUT_3FOLD)
    for row, const in zip(rows, bpf.HELDOUT_3FOLD):
        assert row["variant"] == const["variant"]
        assert int(row["ell"]) == const["ell"]
        assert int(row["m_inducing"]) == const["m"]
        assert float(row["heldout_rmse_n"]) == pytest.approx(const["rmse_n"])
        assert float(row["heldout_mae_n"]) == pytest.approx(const["mae_n"])
        assert float(row["heldout_rmse_gw"]) == pytest.approx(const["rmse_gw"])
    # the -22.5% headline the figure annotates must be recoverable
    by = {(r["variant"], r["ell"]): r for r in rows}
    v4 = float(by[("v4 (baseline)", "2")]["heldout_rmse_n"])
    v5 = float(by[("v5 Sarashina", "2")]["heldout_rmse_n"])
    assert 100.0 * (v5 - v4) / v4 == pytest.approx(-22.5, abs=0.1)


def test_fig4_matches_the_leakage_result_json(exported):
    """A JSON-backed figure: the export must equal the artefact, so the two
    cannot drift apart."""
    out, manifest = exported
    rows = _rows_for(out, manifest, "4", "fig4_leakage_controls.csv")
    jp = json.loads((_REPO / "docs/research"
                     / "2026-06-21_text_leakage_japan.json").read_text())["variants"]
    uk = json.loads((_REPO / "docs/research"
                     / "2026-06-21_text_leakage_uk.json").read_text())["variants"]
    assert len(rows) == 2 * len(bpf._LEAK_ORDER)
    for row in rows:
        src = jp if row["domain"] == "Japan" else uk
        assert float(row["content_pct"]) == pytest.approx(
            src[row["variant"]]["content_pct"])
        assert row["source_kind"] == efsd.KIND_RESEARCH
    order = [r["variant"] for r in rows if r["domain"] == "Japan"]
    assert order == [k for k, _ in bpf._LEAK_ORDER]


def test_fig2_matches_the_run_summaries(exported, runs_dir):
    out, manifest = exported
    rows = _rows_for(out, manifest, "2", "fig2_lro_heldout_rmse.csv")
    assert len(rows) == len(bpf.LRO_REGIONS)
    for row in rows:
        summary = json.loads(
            (runs_dir / f"dkl_national_lro_{row['region']}"
             / "summary.json").read_text())
        assert float(row["heldout_rmse_spt_n"]) == pytest.approx(
            summary["spatial_kfold"][0]["rmse"])
    # drawn in ascending-RMSE order
    values = [float(r["heldout_rmse_spt_n"]) for r in rows]
    assert values == sorted(values)
    refs = _rows_for(out, manifest, "2", "fig2_reference_lines.csv")
    national = next(r for r in refs if r["reference"] == "national_random_kfold")
    assert float(national["heldout_rmse_spt_n"]) == pytest.approx(7.55)


def test_fig7_scatter_holds_only_the_plotted_subsample(exported, ctx):
    out, manifest = exported
    rows = _rows_for(out, manifest, "7", "fig7_borehole_scatter.csv")
    assert len(rows) == ctx.max_scatter_points
    assert {r["regime_name"] for r in rows} <= set(bpf.REGIME_NAMES)


# ============================================================
# Guard rails
# ============================================================

def _fake_paper(tmp_path: Path, graphic: str, label: str,
                number: str = "9", with_aux: bool = True) -> Path:
    paper = tmp_path / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "sections" / "99_new.tex").write_text(
        "\\begin{figure}\n"
        f"\\includegraphics[width=\\textwidth]{{{graphic}}}\n"
        f"\\caption{{A brand new figure}}\\label{{{label}}}\n"
        "\\end{figure}\n")
    aux = f"\\newlabel{{{label}}}{{{{{number}}}{{1}}}}\n" if with_aux else ""
    (paper / "main.aux").write_text(aux)
    return paper


def test_a_new_manuscript_figure_cannot_be_silently_missed(tmp_path, ctx):
    paper = _fake_paper(tmp_path, "brand_new_figure.pdf", "fig:brandnew")
    new_ctx = efsd.ExportContext(
        paper_dir=paper, runs_dir=ctx.runs_dir, research_dir=ctx.research_dir,
        parquet=ctx.parquet, layers_csv=ctx.layers_csv)
    with pytest.raises(efsd.UnregisteredFigure) as exc:
        efsd.export_all(tmp_path / "out", ctx=new_ctx)
    assert "brand_new_figure.pdf" in str(exc.value)


def test_a_figure_missing_from_the_aux_is_an_error(tmp_path):
    paper = _fake_paper(tmp_path, "brand_new_figure.pdf", "fig:brandnew",
                        with_aux=False)
    with pytest.raises(efsd.MissingInput) as exc:
        efsd.discover_manuscript_figures(paper)
    assert "main.aux" in str(exc.value)


def test_missing_inputs_raise_by_default_and_can_be_skipped(tmp_path, ctx):
    empty_runs = tmp_path / "empty_runs"
    empty_runs.mkdir()
    broken = efsd.ExportContext(
        paper_dir=_PAPER, runs_dir=empty_runs, research_dir=ctx.research_dir,
        parquet=ctx.parquet, layers_csv=ctx.layers_csv,
        max_scatter_points=ctx.max_scatter_points)
    with pytest.raises(efsd.MissingInput):
        efsd.export_all(tmp_path / "out_strict", ctx=broken, only=["fig2"])
    manifest = efsd.export_all(tmp_path / "out_lax", ctx=broken, only=["fig2"],
                               skip_missing=True)
    assert manifest["figures"][0]["skipped"]


def test_cli_runs_and_writes_the_bundle_layout(tmp_path, ctx):
    out = tmp_path / "source_data"
    rc = efsd.main([
        "--out", str(out),
        "--paper-dir", str(_PAPER),
        "--runs-dir", str(ctx.runs_dir),
        "--research-dir", str(ctx.research_dir),
        "--parquet", str(ctx.parquet),
        "--layers-csv", str(ctx.layers_csv),
        "--figures", "fig3", "figS3",
        "--log-level", "WARNING",
    ])
    assert rc == 0
    assert (out / "fig3").is_dir() and (out / "figS3").is_dir()
    manifest = json.loads((out / "manifest.json").read_text())
    assert [f["figure"] for f in manifest["figures"]] == ["3", "S3"]


# ---------------------------------------------------------------------------
# Manuscript descriptors of the layer corpus (Fig 8c + Methods).
#
# These two numbers were typed by hand in two places in
# sections/02_national_data_stack.tex and had drifted to a median of 38, which
# no version of the corpus produces -- the plotted panel printed 40 all along,
# so figure and caption disagreed in the submitted PDF. Both sites now quote
# macros from headline_numbers.tex; these tests hold those macros to the CSV.
# ---------------------------------------------------------------------------

_LAYERS_CSV = _REPO / "data/features/derived/soil_text_layers.csv"


def _headline_macros() -> dict[str, str]:
    text = (_PAPER / "headline_numbers.tex").read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*?)\}\s*$", text,
                           flags=re.MULTILINE))


def test_layer_corpus_macros_are_defined_and_quoted_not_typed():
    """The manuscript must not carry a second hand-typed copy of these."""
    macros = _headline_macros()
    for name in ("layerCharMedian", "layerCharMean", "layerCharClipPct",
                 "layerCharMeanClipped"):
        assert name in macros, f"{name} missing from headline_numbers.tex"

    section = (_PAPER / "sections/02_national_data_stack.tex").read_text(
        encoding="utf-8")
    assert "median 38" not in section and "38 characters" not in section, (
        "the stale hand-typed char_length median is back in the section")
    assert section.count(r"\layerCharMedian") == 2, (
        "both char_length sites should quote the macro")


@pytest.mark.skipif(not _LAYERS_CSV.exists(),
                    reason="soil_text_layers.csv is not in git (250 MB)")
def test_layer_corpus_macros_match_the_real_csv():
    """Recompute every layer-corpus macro from the CSV it claims to describe."""
    import pandas as pd

    cl = pd.read_csv(_LAYERS_CSV, usecols=["char_length"])["char_length"]
    cl = cl.astype(float)
    macros = _headline_macros()

    # The median is invariant to the panel's 200-character clip; the mean is
    # not, which is exactly why two mean macros exist.
    assert float(macros["layerCharMedian"]) == cl.median()
    assert round(cl.mean()) == float(macros["layerCharMean"])
    assert round(cl.clip(upper=200).mean(), 1) == float(
        macros["layerCharMeanClipped"])
    clip_pct = float(macros["layerCharClipPct"].rstrip("\\%"))
    assert round(float((cl > 200).mean()) * 100, 2) == clip_pct


def test_source_paths_resolve_inside_the_deposit(tmp_path):
    """`source` is written for a reader of the Zenodo bundle, not of this repo.

    ZENODO.md declares `runs/<cell>/<file>` and `provenance/<name>.json`. Repo
    paths (`data/runs/...`, `docs/research/...`) do not exist in the deposit --
    its `data/` holds three files -- so a reader following the column would
    find nothing.
    """
    exp = _load("export_figure_source_data", _SRC)
    # Only lines that build a `source` value; ExportContext's runs_dir /
    # research_dir defaults are input locations and are correctly repo paths.
    for i, line in enumerate(_SRC.read_text(encoding="utf-8").splitlines(), 1):
        if '"source"' not in line:
            continue
        for bad in ("data/runs/", "docs/research/"):
            assert bad not in line, (
                f"{_SRC.name}:{i} emits a repo-relative source path: {line.strip()}")

    ctx = exp.ExportContext(paper_dir=_PAPER, repo=_REPO)
    assert ctx.rel(_REPO / "data/runs/cell_x/summary.json") == \
        "runs/cell_x/summary.json"
    assert ctx.rel(_REPO / "docs/research/2026-08-11_join_audit.json") == \
        "provenance/2026-08-11_join_audit.json"
    # anything else keeps its repo-relative form
    assert ctx.rel(_REPO / "data/features/derived/soil_text_layers.csv") == \
        "data/features/derived/soil_text_layers.csv"
