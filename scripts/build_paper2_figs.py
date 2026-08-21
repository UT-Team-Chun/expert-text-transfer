#!/usr/bin/env python
"""Auto-generate publication-quality figures for Paper B' (national pivot,
target venue: *Nature Communications Earth & Environment*).

Closes the figure_render slice of the NCE&E pivot plan. The eight figures
follow the broad-significance narrative the audit demands -- a non-geotech
reader can parse the pipeline + three headline claims in ~30 seconds:

- **fig1_concept**          - hand-drawn pipeline schematic; this module
                              emits a *placeholder* PDF reminding the
                              author to swap in the final Inkscape file.
- **fig2_study_area.pdf**   - national scatter of 123,787 borings coloured
                              by 8-way AIST regime + layers-per-boring inset.
- **fig3_llm_text_gain.pdf**- v4 vs v5-Sarashina vs v5-Ruri held-out
                              RMSE_N (l=1, l=2) and RMSE_GW; char_length
                              histogram of 1.15M observation_text strings.
                              Headline:  -22.5 % RMSE / -29.7 % MAE.
- **fig4_lro_gap.pdf**      - cross-province generalisation gap; bar chart
                              of held-out RMSE per LRO region, with
                              kyushu_okinawa flagged in red (volcanic
                              terrane outlier). Reference lines: Kanto
                              in-region (5.875) and national K-fold (7.546).
- **fig5_model_inversion.pdf**- in-region vs cross-region model ranking
                              flip: DKL wins Kanto, HGB/GPBoost win the
                              LRO average. Same colour per model so the
                              inversion is visually instant.
- **fig6_conformal_heatmap.pdf**- 3x8 per-regime conformal-coverage gap
                              heatmap on full_v2 + marginal-gap strip
                              showing |gap|<0.002 across all 21 runs.
- **fig7_cube_slices.pdf**  - vertical (depth x distance) cross-section
                              of posterior-mean SPT-N along a const-lat
                              line (Tokyo Bay -> Boso, 35.6 N) with a
                              locator inset. Replaces the former filled
                              depth-slice maps, which carried the
                              source-level random-Fourier crosshatch on
                              every 2-D panel; a 1-D path along one lat row
                              never exposes that lateral mesh. Reads the
                              tiled zarr cube under
                              ``<cube-dir>/cube/tile_*.zarr``; accepts a
                              local directory or an NFS path. Placeholder
                              PDF when no tile zarrs are found.
- **fig8_uncertainty.pdf**  - 2-up composite: three 1-D posterior-mean
                              depth profiles (Tokyo Bay / Japan Alps /
                              Osaka), N vs depth with a regime-driven
                              conformal band, over the clean LPI national
                              hazard map (``<cube-dir>/maps/lpi_pga30.nc``,
                              variable ``lpi_pga30``). Replaces the former
                              filled p95 / half-width map panels, which all
                              carried the crosshatch; only the
                              groundwater-masked LPI map survives as a 2-D
                              product.
- **fig4_descriptor_families.pdf** - descriptor mechanism (``--figures
                              fig4_descriptors``): (a) the purified parser
                              rung, text-derived features only (66 feat.,
                              -17.531 %) vs the same rung with the AIST
                              archive codes mixed in (88 feat., -16.216 %,
                              the previously published number) -- dropping
                              the codes makes the text effect STRONGER;
                              (b) leave-one-descriptor-family-out, each bar
                              spanning the full -17.531 % effect with the
                              attenuation (pp) as a coloured tip. Largest
                              single family (lithology class) costs only
                              1.82 pp, so no one vocabulary carries the
                              signal. Reads
                              docs/research/2026-08-12_descriptor_families_japan.json.
- **fig5_fewshot_curve.pdf** - borehole-budgeted few-shot curve
                              (``--figures fig5_fewshot``): Spearman rho vs
                              {0, 10, 25, 50, 100, 300} target boreholes for
                              depth-only and depth+text, both directions,
                              with the zero-shot shuffled-embedding null and
                              the target-trained depth-only reference drawn
                              in both panels. Reads
                              docs/research/2026-08-12_fewshot_borehole_curve.json.
- **fig9_tta_delta_rmse.pdf** - per-region Delta-RMSE for the three TTA
                              strategies (BN-stats, TENT, self-training)
                              relative to the source-only DKL+SVGP LRO
                              baseline. Only Kanto and Kyushu-Okinawa
                              land below the zero-reference line; the
                              inset shows the ~335-680x predictive-sigma
                              collapse pathology under TENT and
                              self-training (mean post-TTA sigma
                              ~0.011 in every region).

Each figure is rendered to ``.pdf`` (vector) into
``docs/paper/paper_2_national/figures/``. A short markdown caption is
written alongside each PDF as ``<figure>.caption.md`` so the LaTeX
``\\caption{...}`` blocks can be cross-referenced.

CLI:

.. code-block:: bash

    python -m scripts.build_paper2_figs --figures all
    python -m scripts.build_paper2_figs --figures fig3 fig4 --out-dir /tmp/out

Mirroring ``backend/scripts/build_paper_figures.py`` for matplotlib
style, argparse layout, and Path-based I/O.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

LOG = logging.getLogger("scripts.build_paper2_figs")

# ---------------------------------------------------------------------------
# Constants -- canonical 8-way AIST regime palette (matches Paper 1 Fig 8)
# ---------------------------------------------------------------------------

REGIME_NAMES: tuple[str, ...] = (
    "ALLUVIAL",
    "DILUVIAL",
    "VOLCANIC_ASH",
    "SEDIMENTARY",
    "IGNEOUS",
    "METAMORPHIC",
    "LIMESTONE",
    "UNKNOWN",
)

REGIME_COLORS: tuple[str, ...] = (
    "#1f77b4",  # ALLUVIAL
    "#ff7f0e",  # DILUVIAL
    "#2ca02c",  # VOLCANIC_ASH
    "#d62728",  # SEDIMENTARY
    "#9467bd",  # IGNEOUS
    "#8c564b",  # METAMORPHIC
    "#e377c2",  # LIMESTONE
    "#7f7f7f",  # UNKNOWN
)

# 8 LRO regions, ordered so the audit's headline gap reads naturally.
LRO_REGIONS: tuple[str, ...] = (
    "kanto",
    "kansai",
    "chubu",
    "chugoku",
    "tohoku",
    "shikoku",
    "hokkaido",
    "kyushu_okinawa",
)

# Reader-facing English tick labels for the LRO codes. Module level (rather
# than a local in ``fig4_lro_gap``) so the source-data exporter can label its
# rows with the same strings the axis prints.
LRO_REGION_LABELS: dict[str, str] = {
    "kanto": "Kanto",
    "kansai": "Kansai",
    "chubu": "Chubu",
    "chugoku": "Chugoku",
    "tohoku": "Tohoku",
    "shikoku": "Shikoku",
    "hokkaido": "Hokkaido",
    "kyushu_okinawa": "Kyushu / Okinawa",
}

# The only two numeric quantities drawn on Fig 1 (an otherwise schematic
# figure). Both are read off ``docs/research/2026-08-11_join_audit.json``:
# ``join_delta.n_rows`` and ``join_delta.file_join_match_rate_pct`` (60.78,
# printed to one decimal). They are named here rather than inlined in the
# ``ax.text`` call so that ``scripts/export_figure_source_data.py`` exports
# exactly the values the figure prints instead of a second copy of them.
FIG1_CORPUS_RECORDS: int = 2_663_955
FIG1_TEXT_BEARING_PCT: float = 60.8
FIG1_ANNOTATION_SOURCE_JSON: str = "2026-08-11_join_audit.json"


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

def _set_paper_style() -> None:
    """Global matplotlib style. Uses DejaVu Sans (sans-serif) so the figures
    are readable in both digital + print, and pins axes label / tick sizes
    to a consistent 10 pt across Figs 3-6 per the camera-ready polish pass.
    ``constrained_layout`` is forced on every figure (rcParam) so individual
    callers do not need to remember the kwarg.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,
    })


def _write_caption(pdf_path: Path, caption_md: str) -> None:
    """Emit a sibling ``.caption.md`` so the LaTeX ``\\caption`` block can be
    cross-referenced. The paper author copies the body into
    ``main.tex`` once the figure is locked.
    """
    cap_path = pdf_path.with_suffix(".caption.md")
    cap_path.write_text(caption_md)
    LOG.info("Wrote caption %s", cap_path)


def _save_pdf(fig: plt.Figure, out: Path, caption_md: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote %s", out)
    _write_caption(out, caption_md)


def _placeholder(out: Path, message: str, caption_md: str) -> None:
    """Render a one-page PDF whose body is just ``message``. Used for Fig 1
    (concept diagram, hand-drawn) and Figs 7/8 when the national cube has
    not been NFS-synced locally.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11,
            transform=ax.transAxes, wrap=True)
    ax.set_axis_off()
    _save_pdf(fig, out, caption_md)


# ---------------------------------------------------------------------------
# Fig 1 -- concept diagram (placeholder)
# ---------------------------------------------------------------------------

def fig1_concept(out: Path) -> None:
    """Concept diagram: Japan map -> 1.15M layer narratives -> Japanese LLM
    (Ruri/Sarashina) -> 64-D embedding joined to boring rows -> DKL+SVGP ->
    calibrated SPT-N cube.

    Hand-drawn in Inkscape/Keynote. This entry point emits a placeholder
    PDF so the build pipeline succeeds end-to-end; swap the file in by hand.
    """
    msg = (
        "Fig 1 -- concept diagram (manual)\n\n"
        "Hand-drawn pipeline schematic; render in Inkscape/Keynote and\n"
        "save as fig1_concept.pdf to overwrite this placeholder."
    )
    caption = (
        "**Fig. 1** Pipeline schematic. National boring corpus (1.15M layer "
        "narratives across 123 k boreholes) is encoded by a Japanese LLM "
        "(Ruri / Sarashina) into a 64-D sentence embedding that augments "
        "each layer row before the DKL+SVGP foundation model produces a "
        "regime-conditioned, conformally-calibrated SPT-N cube. "
        "(Drawn manually; overwrite the placeholder PDF.)"
    )
    _placeholder(out, msg, caption)


# ---------------------------------------------------------------------------
# Fig 1 -- programmatic architecture schematic
# ---------------------------------------------------------------------------

def fig1_deployment(out: Path) -> None:
    """Fig 1: the deployment setting and the data-generating ORDER.

    NC pre-review P0-1: the previous schematic ran archive -> extraction ->
    encoder -> model -> 3-D cube -> engineering maps, which implies the text
    channel feeds a map product. It does not: the cube model has no text
    input, and at an unvisited location there is no description to read. The
    estimand this paper measures is prediction for a layer that HAS a
    description but no test, so the figure now shows that timeline and the
    held-out boundaries the evaluation crosses.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.6, 5.2), constrained_layout=False)
    ax.set_xlim(0, 116)
    ax.set_ylim(0, 56)
    ax.axis("off")

    STEP = {
        "obs": ("#e8eef7", "#2c5aa0"),
        "lock": ("#eae7f5", "#5b4b9a"),
        "hide": ("#f7e9e9", "#b03a3a"),
        "pred": ("#e6f2ea", "#2e7d4f"),
    }

    def _step(x, y, w, h, n, title, body, kind, title_size=10.0, body_size=8.0):
        fill, edge = STEP[kind]
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.9",
            linewidth=1.3, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h - 1.5, f"{n}. {title}", ha="center", va="top",
                fontsize=title_size, fontweight="bold", color=edge)
        for i, ln in enumerate(body):
            ax.text(x + w / 2, y + h - 4.2 - i * 1.5, ln, ha="center",
                    va="top", fontsize=body_size, color="#222222")
        return (x + w, y + h / 2)

    def _arrow(p0, p1, label=None):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=15, linewidth=1.6,
            color="#333333", shrinkA=2, shrinkB=2))
        if label:
            ax.text((p0[0] + p1[0]) / 2, p0[1] + 1.6, label, ha="center",
                    va="bottom", fontsize=7.6, style="italic", color="#444444")

    y0, h, w = 25.0, 19.0, 24.0
    gap = 4.6
    xs = [2.0 + i * (w + gap) for i in range(4)]

    a = _step(xs[0], y0, w, h, 1, "Layer logged",
              ["borehole drilled;", "geologist describes", "each recovered layer",
               "(grain size, weathering,", "angularity, water, colour)"], "obs")
    b = _step(xs[1], y0, w, h, 2, "Description fixed",
              ["narrative stored in the", "archive record with its",
               "depth interval", "[z_top, z_bottom)"], "lock")
    c = _step(xs[2], y0, w, h, 3, "Test not run",
              ["SPT performed at only", "some depths, or none;",
               "N is the missing", "quantity"], "hide")
    _step(xs[3], y0, w, h, 4, "N predicted",
          ["frozen LM embedding", "of the description +",
           "structured covariates", "-> posterior N"], "pred")

    _arrow(a, (xs[1], y0 + h / 2))
    _arrow(b, (xs[2], y0 + h / 2))
    _arrow(c, (xs[3], y0 + h / 2))

    ax.text(58.0, 55.4,
            "Estimand: predictive value of the layer description for layers "
            "that carry one",
            ha="center", va="top", fontsize=11.2, fontweight="bold",
            color="#1a1a1a")
    ax.text(58.0, 51.4,
            f"{FIG1_TEXT_BEARING_PCT}% of the "
            f"{FIG1_CORPUS_RECORDS / 1e6:.2f}M-record corpus is a "
            f"logged-but-untested depth",
            ha="center", va="top", fontsize=8.8, style="italic",
            color="#444444")

    # held-out boundaries the evaluation crosses
    ax.add_patch(FancyBboxPatch(
        (2.0, 3.0), 112.0, 18.0, boxstyle="round,pad=0.3,rounding_size=0.9",
        linewidth=1.2, edgecolor="#777777", facecolor="#fafafa",
        linestyle="--"))
    ax.text(58.0, 19.6, "Evaluated across held-out boundaries",
            ha="center", va="top", fontsize=9.6, fontweight="bold",
            color="#333333")
    bounds = [
        ("geographic region", "8 JP blocks / 5 UK regions"),
        ("geological class", "lithology, era"),
        ("logging provenance", "client, contractor, year, schema"),
        ("national archive", "Japan <-> UK, zero-shot"),
    ]
    bw = 26.0
    for i, (t, sub) in enumerate(bounds):
        bx = 4.0 + i * (bw + 1.6)
        ax.add_patch(FancyBboxPatch(
            (bx, 5.2), bw, 9.4, boxstyle="round,pad=0.2,rounding_size=0.7",
            linewidth=1.0, edgecolor="#8a8a8a", facecolor="#ffffff"))
        ax.text(bx + bw / 2, 12.6, t, ha="center", va="top", fontsize=8.6,
                fontweight="bold", color="#333333")
        ax.text(bx + bw / 2, 9.4, sub, ha="center", va="top", fontsize=7.6,
                color="#555555")

    caption = (
        "**Fig. 1** The prediction setting and the order in which the data "
        "are generated. A layer is described by the logging geologist (1) and "
        "the description is fixed in the archive record with its depth "
        "interval (2); the standard penetration test is run at only some "
        "depths, or not at all (3); the missing N is predicted from the "
        "description together with the structured covariates (4). The "
        "estimand is therefore the predictive value of a description for "
        f"layers that have one -- {FIG1_TEXT_BEARING_PCT}% of the "
        f"{FIG1_CORPUS_RECORDS / 1e6:.2f}-million-record corpus -- "
        "not prediction at unvisited locations, where no description exists. "
        "The evaluation crosses four held-out boundaries: geographic region, "
        "geological class, logging provenance (commissioning client, "
        "contractor, survey year, archive schema) and the national archive "
        "boundary."
    )
    _save_pdf(fig, out, caption)


def fig1_schematic(out: Path) -> None:
    """Architecture schematic for the national SPT-N foundation pipeline.

    Five-stage flow drawn with matplotlib primitives (``Rectangle`` blocks,
    ``FancyArrowPatch`` connectors, text annotations):

    1. **Archive**     - KuniJiban corpus (150,557 unique borehole locations, 2.66M SPT
                         records, 191,572 XML files across 6 DTD versions).
    2. **Extraction**  - per-layer structured columns (lat, lon, depth,
                         regime, ...) joined with free-text observation
                         narrative.
    3. **LM encoder**  - Japanese BERT (Sarashina v2-1B or Ruri v3-310m)
                         -> 64-D PCA features.
    4. **Foundation**  - DKL + SVGP joint LMC head (SPT-N, groundwater).
    5. **Products**    - 3-D probabilistic cube + 5 engineering map
                         deliverables (LPI, Vs30, NEHRP class, bearing
                         depth, q_a).

    Replaces the legacy ``fig1_concept.pdf`` placeholder. Kept side-by-side
    so the build pipeline can be A/B'd from the LaTeX side before the
    placeholder is retired.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(13.2, 6.8), constrained_layout=False)
    ax.set_xlim(0, 114)
    ax.set_ylim(0, 56)
    ax.set_axis_off()
    ax.set_aspect("equal")

    # Per-stage palette: muted, print-safe; complements Paper 1 Fig 8 family.
    PAL = {
        "archive":    ("#dfe9f3", "#1f3a5f"),  # fill, edge
        "extract":    ("#fde8d0", "#a44a00"),
        "lm":         ("#e3dcf2", "#4a2d7d"),
        "foundation": ("#d6ecdd", "#1f6b3a"),
        "products":   ("#f7d6d6", "#8a1c1c"),
    }

    def _block(x: float, y: float, w: float, h: float,
               title: str, body: list[str], kind: str,
               title_size: float = 10.5,
               body_size: float = 8.2) -> tuple[float, float]:
        """Rounded rectangle with bold title and bulleted body lines.
        Returns the (right_x, center_y) anchor for arrow plumbing.
        """
        fill, edge = PAL[kind]
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.25,rounding_size=0.9",
            linewidth=1.3, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 1.6, title,
                ha="center", va="top",
                fontsize=title_size, fontweight="bold", color=edge)
        if body:
            line_h = 1.55
            top = y + h - 4.0
            for i, ln in enumerate(body):
                ax.text(x + w / 2, top - i * line_h, ln,
                        ha="center", va="top",
                        fontsize=body_size, color="#222222")
        return (x + w, y + h / 2)

    def _arrow(p0: tuple[float, float], p1: tuple[float, float],
               label: str | None = None,
               label_y: float | None = None,
               color: str = "#333333") -> None:
        a = FancyArrowPatch(
            p0, p1,
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.4, color=color,
            shrinkA=2.0, shrinkB=2.0,
        )
        ax.add_patch(a)
        if label is not None:
            # Lift the connector label off the box bodies and into clear
            # whitespace (a caller-supplied absolute ``label_y``, defaulting
            # to just above the arrow). A faint white halo keeps it legible
            # where it crosses the inter-block gutter.
            mx = (p0[0] + p1[0]) / 2
            my = label_y if label_y is not None else (p0[1] + p1[1]) / 2 + 1.2
            ax.text(mx, my, label, ha="center", va="bottom",
                    fontsize=7.4, style="italic", color="#444444",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.85))

    # ------------------------------------------------------------------
    # Top row: 5 stages laid out horizontally
    # ------------------------------------------------------------------
    row_y = 29.0
    row_h = 16.0
    block_w = 16.0
    gap = 6.0                        # wider inter-block gutter so connector
                                     # arrow labels sit fully in whitespace
    x0 = 1.6
    row_top = row_y + row_h          # 45.0 -- top edge of the stage blocks
    conn_label_y = row_top + 1.6     # connector labels sit in the gutter above

    a_right = _block(
        x0, row_y, block_w, row_h,
        "1. Archive",
        [
            "150,557 locations",
            "2.66M SPT records",
            "191,572 XML files",
            "6 DTD versions",
            "(2.10: 56%, 4.00: 23%,",
            " 3.00: 18%, oth.: 3%)",
        ],
        "archive",
    )

    x1 = x0 + block_w + gap
    b_left = (x1, row_y + row_h / 2)
    b_right = _block(
        x1, row_y, block_w, row_h,
        "2. Extraction",
        [
            "Structured cols:",
            " lat, lon, depth,",
            " regime, thick., GW",
            "+ free text:",
            " observation_text",
            " (1.15M layers)",
        ],
        "extract",
    )

    x2 = x1 + block_w + gap
    c_left = (x2, row_y + row_h / 2)
    c_right = _block(
        x2, row_y, block_w, row_h,
        "3. LM encoder",
        [
            "Sarashina v2-1B",
            " or Ruri v3-310m",
            "(frozen, mean-pool)",
            r"$\rightarrow$ 2048/768-D",
            r"$\rightarrow$ 64-D PCA",
            r"($\sim 78\%$ var.)",
        ],
        "lm",
    )

    x3 = x2 + block_w + gap
    d_left = (x3, row_y + row_h / 2)
    d_right = _block(
        x3, row_y, block_w, row_h,
        "4. Foundation",
        [
            "Deep kernel:",
            " 3-layer MLP + RBF",
            "SVGP: 8,000 ind. pts",
            "Joint LMC head:",
            " N + groundwater",
            "Conformal calib.",
        ],
        "foundation",
    )

    x4 = x3 + block_w + gap
    e_left = (x4, row_y + row_h / 2)
    _block(
        x4, row_y, block_w, row_h,
        "5. Products",
        [
            "3-D prob. cube",
            " (mean, std, quant.)",
            "5 eng. maps:",
            " LPI, Vs30, NEHRP,",
            " bearing depth, q_a",
        ],
        "products",
    )

    # Top-row connectors -- labels lifted into the gutter above the blocks
    # so they no longer print across the box bodies.
    _arrow(a_right, b_left, label="parse XML", label_y=conn_label_y)
    _arrow(b_right, c_left, label="narratives", label_y=conn_label_y)
    _arrow(c_right, d_left, label="64-D feat.", label_y=conn_label_y)
    _arrow(d_right, e_left, label="posterior", label_y=conn_label_y)

    # ------------------------------------------------------------------
    # Bottom row: data-detail callouts under stages 2 / 3 / 4 / 5
    # ------------------------------------------------------------------
    bot_y = 4.0
    bot_h = 14.0

    # Callout A under "extraction": stylised per-layer row.
    cx0 = x1
    cw = block_w
    ax.add_patch(FancyBboxPatch(
        (cx0, bot_y), cw, bot_h,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        linewidth=0.9, edgecolor="#a44a00", facecolor="#fff6ec",
    ))
    ax.text(cx0 + cw / 2, bot_y + bot_h - 1.4, "Layer row",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color="#a44a00")
    # Abbreviated column headers / values so each cell label stays inside its
    # own rectangle (the prior "regime"/"text"/"loose sandy..." labels spilled
    # across cell boundaries).
    cols = ["lat", "lon", "z", "reg", "txt"]
    col_w = (cw - 1.6) / len(cols)
    hdr_y = bot_y + bot_h - 5.2
    for i, c in enumerate(cols):
        ax.add_patch(plt.Rectangle(
            (cx0 + 0.8 + i * col_w, hdr_y), col_w * 0.9, 1.6,
            linewidth=0.6, edgecolor="#a44a00", facecolor="#fde8d0"))
        ax.text(cx0 + 0.8 + i * col_w + col_w * 0.45, hdr_y + 0.8, c,
                ha="center", va="center", fontsize=6.8, color="#a44a00")
    body_vals = ["35.6", "139.7", "4.2", "ALL", "sand"]
    for i, v in enumerate(body_vals):
        ax.add_patch(plt.Rectangle(
            (cx0 + 0.8 + i * col_w, hdr_y - 2.2), col_w * 0.9, 1.6,
            linewidth=0.4, edgecolor="#a44a00", facecolor="white"))
        ax.text(cx0 + 0.8 + i * col_w + col_w * 0.45, hdr_y - 1.4, v,
                ha="center", va="center", fontsize=6.2, color="#222")
    ax.text(cx0 + cw / 2, bot_y + 1.4,
            "one row per layer",
            ha="center", va="bottom", fontsize=7.2, style="italic",
            color="#444")

    # Callout B under "LM encoder": PCA scree bar strip.
    cx1 = x2
    ax.add_patch(FancyBboxPatch(
        (cx1, bot_y), cw, bot_h,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        linewidth=0.9, edgecolor="#4a2d7d", facecolor="#f3eefb",
    ))
    ax.text(cx1 + cw / 2, bot_y + bot_h - 1.4,
            r"Embedding $\rightarrow$ PCA",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color="#4a2d7d")
    rng = np.random.default_rng(0)
    evr = np.sort(rng.dirichlet(np.ones(64) * 0.5))[::-1]
    evr = evr / evr.max()
    bar_x0 = cx1 + 1.0
    bar_y0 = bot_y + 2.6
    bar_w_tot = cw - 2.0
    bar_h_max = 6.0
    for i, v in enumerate(evr):
        ax.add_patch(plt.Rectangle(
            (bar_x0 + i * (bar_w_tot / 64), bar_y0),
            (bar_w_tot / 64) * 0.85, v * bar_h_max,
            linewidth=0.0, facecolor="#7b5cb8"))
    ax.plot([bar_x0, bar_x0 + bar_w_tot],
            [bar_y0, bar_y0], color="#4a2d7d", linewidth=0.6)
    ax.text(cx1 + cw / 2, bot_y + bot_h - 4.0,
            "64 components", ha="center", va="top",
            fontsize=7.2, color="#4a2d7d")
    ax.text(cx1 + cw / 2, bot_y + 1.3,
            "explained-var. ratio",
            ha="center", va="bottom", fontsize=7.0, style="italic",
            color="#444")

    # Callout C under "foundation": joint LMC two-task icon.
    cx2 = x3
    ax.add_patch(FancyBboxPatch(
        (cx2, bot_y), cw, bot_h,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        linewidth=0.9, edgecolor="#1f6b3a", facecolor="#eaf5ee",
    ))
    ax.text(cx2 + cw / 2, bot_y + bot_h - 1.4, "Joint LMC head",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color="#1f6b3a")
    tw = (cw - 2.4) / 2
    th = 5.0
    ty = bot_y + 3.6
    for i, (lbl, sub) in enumerate([("N", "SPT"),
                                     ("GW", "g.water")]):
        tx = cx2 + 0.8 + i * (tw + 0.8)
        ax.add_patch(plt.Rectangle(
            (tx, ty), tw, th,
            linewidth=0.8, edgecolor="#1f6b3a", facecolor="#cfe7d7"))
        ax.text(tx + tw / 2, ty + th - 1.2, lbl,
                ha="center", va="top", fontsize=10, fontweight="bold",
                color="#1f6b3a")
        ax.text(tx + tw / 2, ty + 1.1, sub,
                ha="center", va="bottom", fontsize=6.8, color="#1f6b3a")
    ax.text(cx2 + cw / 2, bot_y + 1.3,
            "shared latent + 2 tasks",
            ha="center", va="bottom", fontsize=7.0, style="italic",
            color="#444")

    # Callout D under "products": 5 deliverable maps.
    cx3 = x4
    ax.add_patch(FancyBboxPatch(
        (cx3, bot_y), cw, bot_h,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        linewidth=0.9, edgecolor="#8a1c1c", facecolor="#fdeeee",
    ))
    ax.text(cx3 + cw / 2, bot_y + bot_h - 1.4, "5 engineering maps",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color="#8a1c1c")
    # Lay the 5 deliverables out as a 3x2 chip grid (one empty slot) instead
    # of 5 cramped side-by-side tiles -- the wider chips let labels such as
    # "NEHRP" and "bearing" sit fully inside their own rectangle.
    map_names = ["LPI", r"$V_{s30}$", "NEHRP", "bearing", r"$q_a$"]
    n_cols, n_rows = 2, 3
    grid_w = cw - 2.4
    grid_h = 7.4
    chip_w = (grid_w - 0.8) / n_cols
    chip_h = (grid_h - 0.6 * (n_rows - 1)) / n_rows
    gx0 = cx3 + 1.2
    gy_top = bot_y + bot_h - 3.6
    for i, nm in enumerate(map_names):
        col = i % n_cols
        row = i // n_cols
        mx = gx0 + col * (chip_w + 0.8)
        myc = gy_top - row * (chip_h + 0.6) - chip_h
        ax.add_patch(plt.Rectangle(
            (mx, myc), chip_w, chip_h,
            linewidth=0.6, edgecolor="#8a1c1c", facecolor="#f6caca"))
        ax.text(mx + chip_w / 2, myc + chip_h / 2, nm,
                ha="center", va="center", fontsize=6.8,
                color="#8a1c1c")
    ax.text(cx3 + cw / 2, bot_y + 1.3,
            "from posterior cube",
            ha="center", va="bottom", fontsize=7.0, style="italic",
            color="#444")

    # Vertical "details" arrows from top blocks down to callouts.
    for cx in (x1, x2, x3, x4):
        _arrow(
            (cx + block_w / 2, row_y - 0.2),
            (cx + block_w / 2, bot_y + bot_h + 0.4),
            color="#888888",
        )

    # ------------------------------------------------------------------
    # Title strip
    # ------------------------------------------------------------------
    ax.text(50.0, 53.0,
            "Borehole free-text as a transferable covariate for subsurface prediction",
            ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="#222222")
    ax.text(50.0, 50.4,
            (r"KuniJiban $\rightarrow$ per-layer extraction $\rightarrow$"
             r" frozen LM (PCA-64) $\rightarrow$ regressor"
             r" $\rightarrow$ out-of-region prediction (cube + maps: SI)"),
            ha="center", va="center",
            fontsize=10, style="italic", color="#444444")

    caption = (
        "**Fig. 1** Schematic of the national SPT-$N$ modelling pipeline, "
        "centred on the free-text layer description as a transferable "
        "covariate. (1) The KuniJiban archive contributes "
        "150,557 unique borehole locations / 2.66M SPT records / 191,572 XML files "
        "spanning six DTD versions. (2) Per-layer extraction emits "
        "structured columns (latitude, longitude, depth, AIST regime, "
        "thickness, groundwater depth) alongside the free-text "
        "observation narrative (1.15M layer-level strings). (3) A frozen "
        "Japanese sentence encoder (Sarashina v2-1B, 2048-D; or Ruri "
        "v3-310m, 768-D; mean pooling) maps each narrative to its "
        "embedding, reduced to 64 principal components ($\\sim$78\\% "
        "variance retained). (4) The 64-D text feature is concatenated "
        "to the structured layer vector and consumed by a DKL + SVGP "
        "regressor with a joint linear model of coregionalisation "
        "(LMC) head over SPT-$N$ and groundwater depth, 8,000 inducing "
        "points, and conformal post-hoc calibration. (5) The trained "
        "model emits a 3-D probabilistic cube and the derived "
        "engineering maps reported in the Supplementary Information. "
        "This is the in-distribution national pipeline; the cross-archive "
        "transfer experiments instead use a model-agnostic gradient-boosted "
        "regressor on a multilingual embedding, so the transfer claim does "
        "not depend on this architecture."
    )
    _save_pdf(fig, out, caption)


# ---------------------------------------------------------------------------
# Fig 2 -- study area + national data stack
# ---------------------------------------------------------------------------

def fig2_study_area(out: Path, parquet_path: Path, layers_csv: Path | None,
                    max_points: int = 30_000) -> None:
    """National scatter of all KuniJiban borings coloured by AIST regime,
    with an inset histogram of layer count per boring.

    Falls back to a placeholder if pyarrow / parquet is not importable.
    """
    try:
        import pandas as pd  # noqa: F401 (used below)
    except Exception as exc:  # pragma: no cover - pandas is a hard dep
        LOG.warning("pandas import failed: %s", exc)
        _placeholder(out, f"Fig 2 placeholder (pandas missing: {exc})",
                     "**Fig. 2** Study area (placeholder).")
        return

    import pandas as pd
    try:
        df = pd.read_parquet(parquet_path,
                             columns=["latitude_deg", "longitude_deg",
                                      "regime_code"])
    except Exception as exc:
        LOG.warning("read_parquet failed for %s: %s", parquet_path, exc)
        _placeholder(out, f"Fig 2 placeholder (parquet read failed: {exc})",
                     "**Fig. 2** Study area (placeholder).")
        return

    # Subsample to keep PDF size reasonable.
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    fig, ax = plt.subplots(figsize=(6.0, 6.6), constrained_layout=True)
    for code in range(len(REGIME_NAMES)):
        mask = df["regime_code"].astype("Int64") == code
        sub = df.loc[mask]
        if len(sub) == 0:
            continue
        ax.scatter(sub["longitude_deg"], sub["latitude_deg"],
                   s=2.0, alpha=0.45, color=REGIME_COLORS[code],
                   label=f"{REGIME_NAMES[code]} (n={len(sub):,})",
                   linewidths=0)
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title("KuniJiban national boring corpus, coloured by AIST regime",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=6.5, markerscale=2.5,
              labelspacing=0.25)

    # Inset: layers-per-boring histogram (from soil_text_layers.csv).
    if layers_csv is not None and layers_csv.exists():
        try:
            counts = pd.read_csv(layers_csv, usecols=["file_path"],
                                 ).groupby("file_path").size()
            # Report the full-corpus mean (matches the caption's
            # 1.15M-layer corpus); only the *histogram bins* clip the long
            # tail for readability, not the annotated statistic.
            corpus_mean = counts.mean()
            counts_clipped = counts[counts < 60]
            inset = fig.add_axes([0.18, 0.18, 0.27, 0.16])
            inset.hist(counts_clipped.values, bins=30, color="#7f7f7f",
                       edgecolor="black", linewidth=0.3)
            inset.set_xlabel("Layers per boring", fontsize=7)
            inset.set_ylabel("Count", fontsize=7)
            inset.tick_params(axis="both", labelsize=6)
            inset.set_title(f"mean={corpus_mean:.2f}", fontsize=7)
        except Exception as exc:
            LOG.warning("inset histogram skipped: %s", exc)

    caption = (
        "**Fig. 2** National study area. Each point is one borehole "
        "(subsampled to 30 k for legibility), coloured by AIST 8-way "
        "lithological regime. Inset: distribution of layers per boring "
        "across the 1.15M-layer corpus (mean = 9.33)."
    )
    _save_pdf(fig, out, caption)


# ---------------------------------------------------------------------------
# Fig 3 -- LLM text gain (v4 vs v5 Sarashina / Ruri)
# ---------------------------------------------------------------------------

# Held-out 3-fold RMSE / MAE / gw-RMSE from
# docs/paper/paper_2_national/sections/02_national_data_stack.tex
# Table ``tab:pillar3_holdout_3fold`` (commit 314c7f8, 2026-05-31).
# Numbers below were aggregated from the 15 completed K-fold runs at
# /mnt/nas/runs/dkl_national_lmc_v{4,5}_*_kfold{0,1,2}/summary.json.
# v5 Sarashina l=1 is now a full 3-fold mean: the kfold1 retry
# (rmse_n=9.494) landed under the per-epoch-checkpoint path from
# commit efa0a31, replacing the prior 2-fold provisional mean of 9.711.
# v5 Ruri l=2 / m=8k was completed for camera-ready (mean rmse_n=8.972
# across folds [8.63, 9.365, 8.921]); the bar chart now renders all six
# (variant x level) combinations.
HELDOUT_3FOLD: list[dict[str, Any]] = [
    # (variant, ell, m, RMSE_N mean, MAE_N mean, RMSE_GW mean)
    # The ``folds`` field stores the per-fold (rmse_n, mae_n, rmse_gw)
    # triples used to compute the means; populated where the raw 3-fold
    # breakdown is available from the authoritative results dict.
    {"variant": "v4 (baseline)",     "ell": 1, "m": 12_000,
     "rmse_n": 11.194, "mae_n": 8.974, "rmse_gw": 4.958},
    {"variant": "v5 Sarashina",      "ell": 1, "m": 12_000,
     "rmse_n":  9.639, "mae_n": 5.989, "rmse_gw": 4.835,
     "folds": [(9.636, None, None),
               (9.494, None, None),
               (9.786, None, None)]},
    {"variant": "v5 Ruri",           "ell": 1, "m": 12_000,
     "rmse_n":  9.592, "mae_n": 6.116, "rmse_gw": 4.658},
    {"variant": "v4 (baseline)",     "ell": 2, "m":  8_000,
     "rmse_n": 11.333, "mae_n": 8.021, "rmse_gw": 4.670},
    {"variant": "v5 Sarashina",      "ell": 2, "m":  8_000,
     "rmse_n":  8.786, "mae_n": 5.638, "rmse_gw": 4.463,
     "folds": [(8.591, 5.458, 5.265),
               (8.931, 5.738, 4.046),
               (8.835, 5.717, 4.078)]},
    {"variant": "v5 Ruri",           "ell": 2, "m":  8_000,
     "rmse_n":  8.972, "mae_n": 5.792, "rmse_gw": 4.443,
     "folds": [(8.630, 5.428, 5.284),
               (9.365, 6.295, 4.031),
               (8.921, 5.653, 4.015)]},
]


def _grouped_bars(ax: plt.Axes, levels: list[str], variants: list[str],
                  values: dict[tuple[str, str], float],
                  colors: dict[str, str], width: float = 0.27,
                  ylabel: str = "", label_fontsize: float = 7.5,
                  label_offset_frac: float = 0.015) -> None:
    """Draw a grouped bar chart: x positions per level, sub-bars per variant.

    Bar-top value labels are placed at ``y + label_offset_frac * y_range`` so
    they clear the bar without colliding with the next bar's value (the prior
    fixed +0.05 offset overlapped at the l=2 panel where v5-Sarashina and
    v5-Ruri are within ~0.2 of each other). Missing entries (e.g. v5 Ruri at
    l=2 was not run for compute-budget reasons) are silently skipped instead
    of crashing.
    """
    x = np.arange(len(levels))
    drawn_values: list[float] = []
    for i, var in enumerate(variants):
        offsets = (i - (len(variants) - 1) / 2) * width
        for xi_idx, lvl in enumerate(levels):
            y = values.get((lvl, var))
            if y is None:
                continue
            xi = x[xi_idx] + offsets
            ax.bar(xi, y, width=width, color=colors[var],
                   edgecolor="black", linewidth=0.4,
                   label=var if xi_idx == 0 else None)
            drawn_values.append(y)
    # Compute a single offset based on the drawn value range so the labels
    # all sit a uniform distance above the bar tops.
    if drawn_values:
        y_range = max(drawn_values) - min(0.0, min(drawn_values))
        offset = max(label_offset_frac * y_range, 0.08)
    else:
        offset = 0.08
    # Keep each value label centred on its own bar (so it never drifts into a
    # neighbouring group's label) but apply a small per-variant vertical
    # stagger: the middle sub-bar's label is lifted a notch so it clears its
    # left/right neighbours when their bar tops are within ~0.1 of each other
    # (e.g. 9.64 vs 9.59 at l=1). This avoids both the within-group overlap
    # and the cross-group collision a horizontal nudge introduced.
    n_var = len(variants)
    for i, var in enumerate(variants):
        offsets = (i - (n_var - 1) / 2) * width
        # Centre bar lifted by one extra offset; outer bars sit at the base.
        is_centre = (n_var >= 3 and i == (n_var - 1) // 2)
        dy = offset * (2.4 if is_centre else 1.0)
        for xi_idx, lvl in enumerate(levels):
            y = values.get((lvl, var))
            if y is None:
                continue
            xi = x[xi_idx] + offsets
            ax.text(xi, y + dy, f"{y:.2f}", ha="center", va="bottom",
                    fontsize=label_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    # Add headroom for the value labels.
    if drawn_values:
        ymax = max(drawn_values)
        ax.set_ylim(top=ymax + 5.0 * offset)


def fig3_llm_text_gain(out: Path, layers_csv: Path | None) -> None:
    """3-panel figure showing per-layer Japanese sentence embeddings lift
    held-out RMSE / MAE / RMSE_GW, plus the char_length histogram of the
    1.15M observation_text strings the LLM ingests.
    """
    variants = ["v4 (baseline)", "v5 Sarashina", "v5 Ruri"]
    colors = {
        "v4 (baseline)": "#7f7f7f",
        "v5 Sarashina": "#1f77b4",
        "v5 Ruri":      "#ff7f0e",
    }
    levels = ["l=1, m=12k", "l=2, m=8k"]

    def lookup(level: str, var: str, key: str) -> float | None:
        """Return the held-out metric for (level, variant), or ``None``
        when the combination was never run. As of the v5 Ruri @ l=2 / m=8k
        completion (camera-ready), all six (variant x level) cells are
        populated; the function still tolerates a missing entry so the
        figure renders gracefully if the table is pared back later."""
        ell, m = (1, 12_000) if level.startswith("l=1") else (2, 8_000)
        for row in HELDOUT_3FOLD:
            if (row["variant"] == var and row["ell"] == ell
                    and row["m"] == m):
                return float(row[key])
        return None

    rmse_n = {(lvl, v): lookup(lvl, v, "rmse_n")
              for lvl in levels for v in variants
              if lookup(lvl, v, "rmse_n") is not None}
    rmse_gw = {(lvl, v): lookup(lvl, v, "rmse_gw")
               for lvl in levels for v in variants
               if lookup(lvl, v, "rmse_gw") is not None}

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6),
                             constrained_layout=True)

    # Panel (a): RMSE_N
    ax = axes[0]
    _grouped_bars(ax, levels, variants, rmse_n, colors,
                  ylabel="Held-out RMSE  [SPT $N$ units]")
    # Title raised (pad) so the legend strip can sit between title and axes
    # without either colliding with the v4 bar-top value label.
    ax.set_title("(a) SPT $N$ held-out RMSE", pad=24)
    # Legend lifted fully out of the top-left plotting region (it previously
    # printed over the v4 bar-top value label) and anchored above the axes
    # as a single horizontal strip.
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3,
              fontsize=7.0, frameon=False, columnspacing=1.0,
              handlelength=1.1, handletextpad=0.4)
    # Annotate the -22.5 % / -29.7 % headline (v5-Sarashina @ l=2 vs v4 @ l=2).
    # The box goes in the free wedge ABOVE the two v5 bars of the l=2 group and
    # to the right of the v4 bar: at the previous (1.35, 7.0) it was drawn
    # straight over the v5-Ruri bar.
    delta_rmse = (lookup("l=2, m=8k", "v5 Sarashina", "rmse_n")
                  - lookup("l=2, m=8k", "v4 (baseline)", "rmse_n"))
    delta_mae = (lookup("l=2, m=8k", "v5 Sarashina", "mae_n")
                 - lookup("l=2, m=8k", "v4 (baseline)", "mae_n"))
    pct_rmse = 100.0 * delta_rmse / lookup("l=2, m=8k", "v4 (baseline)",
                                           "rmse_n")
    pct_mae = 100.0 * delta_mae / lookup("l=2, m=8k", "v4 (baseline)",
                                         "mae_n")
    ax.annotate(f"{pct_rmse:+.1f} % RMSE\n{pct_mae:+.1f} % MAE",
                xy=(1.0, lookup("l=2, m=8k", "v5 Sarashina", "rmse_n")),
                xytext=(1.22, 10.7), ha="center", va="center",
                arrowprops=dict(arrowstyle="->", linewidth=0.7,
                                color="#1f77b4",
                                connectionstyle="arc3,rad=-0.2"),
                fontsize=8, color="#1f77b4",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#1f77b4", lw=0.6, alpha=0.9))

    # Panel (b): RMSE_GW
    ax = axes[1]
    _grouped_bars(ax, levels, variants, rmse_gw, colors,
                  ylabel="Held-out RMSE  [m]",
                  label_offset_frac=0.020)
    ax.set_title("(b) Groundwater depth held-out RMSE", pad=24)
    # No per-panel legend here: the variant key in panel (a) covers it, and a
    # legend in (b)'s upper-left collided with the v4 bar-top value label.

    # Panel (c): char_length histogram
    ax = axes[2]
    drew_hist = False
    if layers_csv is not None and layers_csv.exists():
        try:
            import pandas as pd
            cl = pd.read_csv(layers_csv, usecols=["char_length"])
            cl = cl["char_length"].clip(upper=200)
            ax.hist(cl.values, bins=40, color="#2ca02c",
                    edgecolor="black", linewidth=0.3)
            ax.set_xlabel("char_length (clipped @ 200)")
            ax.set_ylabel("Layers")
            ax.set_title(f"(c) observation_text length\n"
                         f"(median={int(cl.median())}, mean={cl.mean():.1f})",
                         fontsize=9)
            ax.grid(True, axis="y", linestyle=":", alpha=0.4)
            drew_hist = True
        except Exception as exc:
            LOG.warning("char_length histogram skipped: %s", exc)
    if not drew_hist:
        ax.text(0.5, 0.5,
                "soil_text_layers.csv\nnot available locally",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8)
        ax.set_axis_off()

    caption = (
        "**Fig. 3** Per-layer Japanese sentence embeddings lift held-out "
        "performance. (a) Spatial 3-fold RMSE for SPT-$N$ at "
        "two inducing-point settings; the v5-Sarashina embedding at l=2 "
        f"shaves {pct_rmse:+.1f}% RMSE and {pct_mae:+.1f}% MAE "
        "vs the geometry-only v4 baseline. "
        "(b) Same comparison for groundwater depth. "
        "(c) Distribution of observation_text character lengths across the "
        "1.15M-layer corpus the LLM ingests."
    )
    _save_pdf(fig, out, caption)


# ---------------------------------------------------------------------------
# Fig 4 -- LRO cross-province gap
# ---------------------------------------------------------------------------

def _load_lro_rmse(runs_dir: Path) -> dict[str, dict[str, float]]:
    """Pull spatial_kfold[0].{rmse,mae} for each of the 8 LRO regions."""
    out: dict[str, dict[str, float]] = {}
    for region in LRO_REGIONS:
        sj = runs_dir / f"dkl_national_lro_{region}" / "summary.json"
        if not sj.exists():
            LOG.warning("missing LRO summary: %s", sj)
            continue
        s = json.loads(sj.read_text())
        sk = s.get("spatial_kfold")
        if not sk:
            LOG.warning("no spatial_kfold in %s", sj)
            continue
        row = sk[0]
        out[region] = {"rmse": float(row["rmse"]),
                       "mae": float(row["mae"])}
    return out


def national_kfold_rmse(runs_dir: Path,
                        full_v2_run: str = "dkl_national_full_v2",
                        fallback: float = 7.546) -> float:
    """Mean spatial-K-fold RMSE of the national hero cell.

    This is the dotted "national random K-fold" reference line of Fig 2
    (``fig4_lro_gap``). Factored out of the figure body so the source-data
    exporter reports the same number from the same artefact rather than
    recomputing it from a second copy of this logic. Falls back to the
    committed value when the run summary is unavailable.
    """
    fv2 = runs_dir / full_v2_run / "summary.json"
    if not fv2.exists():
        return fallback
    try:
        s = json.loads(fv2.read_text())
        folds = s["spatial_kfold"]
        return float(sum(f["rmse"] for f in folds) / len(folds))
    except Exception as exc:
        LOG.warning("full_v2 RMSE not available: %s", exc)
        return fallback


def fig4_lro_gap(out: Path, runs_dir: Path,
                 kanto_ref_rmse: float = 5.875,
                 national_ref_rmse: float | None = None,
                 full_v2_run: str = "dkl_national_full_v2") -> None:
    """Horizontal bar chart of held-out RMSE per LRO region with reference
    lines for Kanto in-region and national K-fold. Kyushu/Okinawa flagged
    in red as the volcanic-terrane outlier (this is the NCE&E headline).
    """
    lro = _load_lro_rmse(runs_dir)
    if not lro:
        _placeholder(out, "Fig 4 placeholder (no LRO runs found)",
                     "**Fig. 4** Cross-province gap (placeholder).")
        return

    # Optional: pull the national K-fold RMSE from full_v2 summary if no
    # explicit override.
    if national_ref_rmse is None:
        national_ref_rmse = national_kfold_rmse(runs_dir, full_v2_run)

    # Sort regions ascending by RMSE so kyushu_okinawa lands at the bottom.
    sorted_regions = sorted(lro.keys(), key=lambda r: lro[r]["rmse"])
    rmses = [lro[r]["rmse"] for r in sorted_regions]
    colors = ["#d62728" if r == "kyushu_okinawa" else "#1f77b4"
              for r in sorted_regions]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    y = np.arange(len(sorted_regions))
    ax.barh(y, rmses, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    # Render region names with title-case + slash separators so the LRO
    # codes remain readable as English labels at 10 pt sans-serif.
    pretty = LRO_REGION_LABELS
    ax.set_yticklabels([pretty.get(r, r.replace("_", " / "))
                        for r in sorted_regions])
    ax.set_xlabel("Held-out RMSE  [SPT $N$ units]")
    ax.invert_yaxis()
    # Extend the visual top edge so the inline reference-line tags sit in
    # clear whitespace above the first bar (post-invert, smaller y = top).
    ax.set_ylim(len(sorted_regions) - 0.5, -1.5)
    # Reserve headroom on the right for the bar-tip value labels, the legend,
    # and the Kyushu/Okinawa callout. 1.22 left no free strip beyond the
    # longest bar, so the callout had to be placed at x ~ 0.55*xmax -- i.e.
    # squarely on top of the Shikoku and Hokkaido bars.
    xmax = max(rmses)
    ax.set_xlim(0, xmax * 1.52)
    # Reference lines (no legend handles: labelled inline at the lines).
    ax.axvline(kanto_ref_rmse, color="#2ca02c", linestyle="--",
               linewidth=1.0,
               label=f"Kanto in-region (companion study) = {kanto_ref_rmse:.2f}")
    if national_ref_rmse is not None:
        ax.axvline(national_ref_rmse, color="#7f7f7f", linestyle=":",
                   linewidth=1.0,
                   label=f"National random K-fold = {national_ref_rmse:.3f}")
    # Inline reference-line tags placed *at* the lines (top of the plot, in
    # the whitespace left of the shortest bar) so a reader does not have to
    # round-trip to the legend to know which line is which.
    # The two reference lines are only ~1.7 N apart, so centred tags collide
    # ("Kantonational" / "in-regionK-fold"). Hang them off opposite sides of
    # their own line instead, into the empty band above the first bar.
    y_top = -0.62  # just above the first (inverted) row
    ax.text(kanto_ref_rmse - xmax * 0.01, y_top, "Kanto\nin-region",
            ha="right", va="bottom", fontsize=7, color="#2ca02c",
            rotation=0)
    if national_ref_rmse is not None:
        ax.text(national_ref_rmse + xmax * 0.01, y_top, "national\nK-fold",
                ha="left", va="bottom", fontsize=7, color="#5f5f5f",
                rotation=0)
    # Legend relocated to the upper-right whitespace (right of the shortest
    # bars), clear of both the bar-tip value labels and the Kyushu/Okinawa
    # volcanic-terrane callout at the bottom.
    ax.legend(loc="upper right", fontsize=7.5,
              bbox_to_anchor=(0.995, 0.995), frameon=True,
              framealpha=0.9, edgecolor="#cccccc")
    # Annotate each bar with its value.
    for yi, ri in zip(y, rmses):
        ax.text(ri + xmax * 0.012, yi, f"{ri:.2f}", va="center", fontsize=8)
    # Annotate kyushu_okinawa with the volcanic-terrane note.
    if "kyushu_okinawa" in sorted_regions:
        kx = lro["kyushu_okinawa"]["rmse"]
        ref = national_ref_rmse if national_ref_rmse is not None else kanto_ref_rmse
        delta = kx - ref
        idx = sorted_regions.index("kyushu_okinawa")
        # Place the callout in the free strip to the RIGHT of every bar, so it
        # clears both the bar bodies and their tip value labels.
        ax.annotate(
            f"volcanic island terrane:\n+{delta:.1f} RMSE vs national",
            # Land the arrow inside the bar, not on its tip, where the
            # "18.33" value label sits.
            xy=(kx * 0.92, idx),
            xytext=(xmax * 1.30, max(idx - 1.6, 0.2)),
            arrowprops=dict(arrowstyle="->", linewidth=0.8,
                            color="#d62728",
                            connectionstyle="arc3,rad=0.15"),
            fontsize=8, color="#d62728",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#d62728", lw=0.6, alpha=0.9),
        )
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.set_title("Cross-province generalisation gap "
                 "(leave-region-out spatial 3-fold)")

    caption = (
        "**Fig. 4** Cross-province generalisation gap. Each bar reports "
        "spatial-3-fold held-out RMSE when the DKL+SVGP foundation model "
        "is trained on the other seven LRO regions and evaluated on the "
        "named region. Dashed reference: Kanto in-region (companion study, "
        f"RMSE={kanto_ref_rmse:.2f}). Dotted reference: national random "
        f"K-fold (RMSE={national_ref_rmse:.3f}). The Kyushu / Okinawa "
        "bar (red) is the volcanic-island outlier the NCE&E pivot calls out."
    )
    _save_pdf(fig, out, caption)


# ---------------------------------------------------------------------------
# Fig 5 -- in-region vs cross-region model ranking inversion
# ---------------------------------------------------------------------------

# Numbers hard-coded from sections/07 + Paper 1 results_table.md.
KANTO_INREGION_BARS: list[dict[str, Any]] = [
    {"model": "DKL+SVGP (ours)", "rmse": 5.88, "color": "#1f77b4"},
    {"model": "GPBoost",         "rmse": 6.40, "color": "#ff7f0e"},
    {"model": "LightGBM",        "rmse": 6.80, "color": "#2ca02c"},
    {"model": "HGB",             "rmse": 7.00, "color": "#d62728"},
]

LRO_AVERAGE_BARS: list[dict[str, Any]] = [
    {"model": "HGB",             "rmse": 11.23, "std": 0.534, "color": "#d62728"},
    {"model": "GPBoost",         "rmse": 11.37, "std": 0.649, "color": "#ff7f0e"},
    {"model": "DKL+SVGP (ours)", "rmse": 14.508, "std": 1.55,  "color": "#1f77b4"},
]


def fig5_model_inversion(out: Path) -> None:
    """Two-panel grouped bar chart showing the in-region vs cross-region
    ranking flip (broad-significance claim from the audit)."""

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)

    # Panel (a): Kanto in-region.
    ax = axes[0]
    labels = [r["model"] for r in KANTO_INREGION_BARS]
    rmses = [r["rmse"] for r in KANTO_INREGION_BARS]
    cols = [r["color"] for r in KANTO_INREGION_BARS]
    x = np.arange(len(labels))
    ax.bar(x, rmses, color=cols, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Held-out RMSE  [SPT $N$]")
    ax.set_title("(a) Kanto in-region", fontsize=10)
    a_offset = max(rmses) * 0.02
    for xi, yi in zip(x, rmses):
        ax.text(xi, yi + a_offset, f"{yi:.2f}", ha="center", va="bottom",
                fontsize=8)
    ax.set_ylim(top=max(rmses) * 1.12)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    # Panel (b): LRO 8-region average.
    ax = axes[1]
    labels = [r["model"] for r in LRO_AVERAGE_BARS]
    rmses = [r["rmse"] for r in LRO_AVERAGE_BARS]
    stds = [r["std"] for r in LRO_AVERAGE_BARS]
    cols = [r["color"] for r in LRO_AVERAGE_BARS]
    x = np.arange(len(labels))
    ax.bar(x, rmses, yerr=stds, color=cols, edgecolor="black", linewidth=0.5,
           capsize=4, error_kw={"elinewidth": 0.7})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean RMSE  [SPT $N$]")
    ax.set_title("(b) LRO 8-region cross-province average", fontsize=10)
    b_offset = (max(r + s for r, s in zip(rmses, stds))) * 0.02
    for xi, yi, si in zip(x, rmses, stds):
        ax.text(xi, yi + si + b_offset, f"{yi:.2f} $\\pm$ {si:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(top=(max(r + s for r, s in zip(rmses, stds))) * 1.18)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        "In-region best is not cross-region best:\n"
        "foundation-model encoders learn spatial lookup tables; "
        "tree baselines generalise",
        fontsize=10.5)

    caption = (
        "**Fig. 5** Model-ranking inversion across spatial regimes. "
        "(a) On the in-region Kanto benchmark of the companion study the "
        "DKL+SVGP "
        "foundation model is best (RMSE 5.88). "
        "(b) Under leave-region-out cross-province transfer the ranking "
        "flips: tree-based baselines (HGB, GPBoost) generalise better "
        "(RMSE 11.23 / 11.37) while the same DKL foundation model "
        "degrades to 14.508 (mean of 8 held-out regions, error bars = std). "
        "Colours match per model across panels."
    )
    _save_pdf(fig, out, caption)


# ---------------------------------------------------------------------------
# Fig 6 -- per-regime conformal coverage heatmap
# ---------------------------------------------------------------------------

def _load_conformal(json_path: Path) -> dict[str, Any]:
    return json.loads(json_path.read_text())


def fig6_conformal_heatmap(out: Path, runs_dir: Path,
                           primary_run: str = "dkl_national_full_v2") -> None:
    """3x8 heatmap of per-regime calibration gap on the primary run, plus a
    right-side strip showing marginal-gap |.|<0.002 across all available
    conformal_mondrian.json files under ``runs_dir``.
    """
    primary = runs_dir / primary_run / "conformal_mondrian.json"
    if not primary.exists():
        _placeholder(out, f"Fig 6 placeholder ({primary} missing)",
                     "**Fig. 6** Conformal coverage heatmap (placeholder).")
        return
    c = _load_conformal(primary)
    alphas = c["alphas"]
    n_cal_per_regime = c.get("n_cal_per_regime", {})

    # Build the gap matrix (rows = alpha, cols = regime).
    gap = np.full((len(alphas), len(REGIME_NAMES)), np.nan)
    for ai, a in enumerate(alphas):
        per_reg = c["per_regime"].get(str(a), {})
        for ri, _name in enumerate(REGIME_NAMES):
            entry = per_reg.get(str(ri))
            if entry is None:
                continue
            gap[ai, ri] = float(entry["coverage"]) - float(a)

    # constrained_layout is forced on globally; disable it here so the manual
    # subplots_adjust(top=0.88) headroom for panel (a)'s title (which must
    # clear the per-column n= annotation row) is respected.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2),
                             gridspec_kw={"width_ratios": [3.4, 1.0]},
                             constrained_layout=False)
    fig.subplots_adjust(top=0.88, bottom=0.18, left=0.06, right=0.97,
                        wspace=0.32)

    ax = axes[0]
    im = ax.imshow(gap, cmap="RdBu_r", vmin=-0.05, vmax=0.05, aspect="auto")
    ax.set_xticks(np.arange(len(REGIME_NAMES)))
    ax.set_xticklabels(REGIME_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(alphas)))
    ax.set_yticklabels([f"$\\alpha$={a}" for a in alphas])
    # Pad the title clear of the per-column n= annotation row that hangs above
    # the heatmap (the two collided at the default pad). The extra pad=20 sits
    # on top of the subplots_adjust(top=0.88) headroom reserved above.
    ax.set_title(f"(a) Per-regime calibration gap  [{primary_run}]",
                 fontsize=10, pad=20)
    for ai in range(len(alphas)):
        for ri in range(len(REGIME_NAMES)):
            v = gap[ai, ri]
            if np.isnan(v):
                continue
            ax.text(ri, ai, f"{v:+.3f}", ha="center", va="center",
                    fontsize=8,
                    color="black" if abs(v) < 0.03 else "white")
    # Annotate n_cal_per_regime above the heatmap as a secondary axis-tick
    # row, well clear of the rotated regime labels. Anchoring above (-0.55)
    # avoids the visual collision the prior (-0.65) placement produced with
    # the bottom-axis tick labels.
    for ri in range(len(REGIME_NAMES)):
        n = n_cal_per_regime.get(str(ri), "?")
        ax.text(ri, -0.55, f"n={n}", ha="center", va="bottom", fontsize=7,
                color="#444444")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("empirical $-$ nominal", fontsize=9)
    cbar.ax.tick_params(labelsize=9)

    # Panel (b): marginal-gap strip across all conformal_mondrian.json
    # files under runs_dir.
    ax = axes[1]
    marginal_gaps: list[float] = []
    labels: list[str] = []
    for cj in sorted(runs_dir.glob("dkl_national_*/conformal_mondrian.json")):
        try:
            cc = _load_conformal(cj)
            mg = cc.get("marginal", {}).get("0.95", {}).get("gap_mondrian")
            if mg is None:
                continue
            marginal_gaps.append(abs(float(mg)))
            labels.append(cj.parent.name.replace("dkl_national_", ""))
        except Exception as exc:
            LOG.warning("skip %s: %s", cj, exc)
    if marginal_gaps:
        x = np.arange(len(marginal_gaps))
        ax.axhspan(0.0, 0.002, color="#2ca02c", alpha=0.15,
                   label="|gap| < 0.002")
        ax.scatter(x, marginal_gaps, color="#1f77b4", s=22)
        ax.set_xticks([])
        ax.set_xlabel(f"21 runs ({len(marginal_gaps)} found)", fontsize=9)
        ax.set_ylabel(r"$|\mathrm{gap}_{\mathrm{Mondrian}}|$ @ $\alpha$=0.95",
                      fontsize=9)
        ax.set_title("(b) Marginal coverage gap", fontsize=10)
        ax.set_ylim(0, max(0.005, max(marginal_gaps) * 1.15))
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    else:
        ax.text(0.5, 0.5, "no conformal\nartefacts found",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    caption = (
        "**Fig. 6** Per-regime conformal calibration. (a) Signed coverage "
        "gap (empirical − nominal) for the primary national run "
        f"({primary_run}) at α∈{{0.5, 0.8, 0.95}} across the 8 AIST "
        "regimes; rare regimes (VOLCANIC_ASH n=2 260, LIMESTONE n=2 317) "
        "are explicitly retained rather than absorbed into a marginal "
        "average. (b) Marginal coverage gap at α=0.95 across all "
        "available DKL_national runs; the shaded band marks the |gap| "
        "< 0.002 zone."
    )
    _save_pdf(fig, out, caption)


# ---------------------------------------------------------------------------
# Fig 7 -- national cube depth slices (deferred / placeholder)
# ---------------------------------------------------------------------------

def _resolve_tile_dirs(cube_dir: Path) -> list[Path]:
    """Locate ``tile_*.zarr`` directories under ``cube_dir``.

    The canonical layout produced by ``predict_national_cube.py`` is::

        <cube_dir>/
            cube/
                tile_*.zarr
            maps/
                *.nc
            manifest.json

    so the primary search is ``cube_dir/cube/tile_*.zarr``. For
    back-compat with callers that already point ``--cube-dir`` directly
    at the ``cube/`` subdirectory, we fall back to a bare
    ``cube_dir/tile_*.zarr`` glob.
    """
    if not cube_dir.exists():
        return []
    primary = sorted((cube_dir / "cube").glob("tile_*.zarr"))
    if primary:
        return primary
    return sorted(cube_dir.glob("tile_*.zarr"))


def _import_cube_snap_helpers():
    """Return ``(build_global_lat_lon_axes, _snap_tile_coords)`` from the
    canonical cube assembler in ``predict_national_cube``.

    Prefer the ``scripts.<name>`` package path (works when the backend
    dir is on ``sys.path``, i.e. the production ``python -m`` entrypoint
    and the in-repo pytest run). Fall back to loading the module
    directly from the sibling source file when this module was itself
    spec-loaded under a synthetic name (the test harness does this to
    avoid stale ``scripts.*`` pickups across worktrees), in which case
    the relative ``scripts.*`` import may not resolve.
    """
    try:
        from scripts.predict_national_cube import (  # type: ignore
            build_global_lat_lon_axes,
            _snap_tile_coords,
        )
        return build_global_lat_lon_axes, _snap_tile_coords
    except Exception:
        import importlib.util as _ilu
        src = Path(__file__).resolve().parent / "predict_national_cube.py"
        spec = _ilu.spec_from_file_location(
            "predict_national_cube_under_test", src)
        if spec is None or spec.loader is None:
            raise
        mod = _ilu.module_from_spec(spec)
        sys.modules.setdefault("predict_national_cube_under_test", mod)
        spec.loader.exec_module(mod)
        return mod.build_global_lat_lon_axes, mod._snap_tile_coords


def _fill_thin_nan_stripes(
    combined: np.ndarray,
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
) -> np.ndarray:
    """Fill the periodic 1-cell NaN stripes left by tile snap-assignment.

    Root cause (see ``predict_national_cube.build_global_lat_lon_axes``):
    the shared global lon axis is built at ``d_lon = res/(R*cos(lat_min))``
    -- the LOWEST latitude in the bbox -- so the global grid is strictly
    FINER than every tile's native lon spacing (each tile uses
    ``cos(mid_lat) <= cos(lat_min)``). When each tile's nearest-neighbour-
    snapped cells are written via ``combined[np.ix_(lat_idx, lon_idx)]``,
    consecutive tile cells land on global indices spaced slightly more than
    1 apart, so interleaved global columns are never assigned and stay NaN.
    This paints periodic 1-cell-wide NaN columns (~1 every ~30 cells in the
    south, ~1 every ~5 cells in the north) and the analogous (smaller)
    horizontal seams in lat. ``_transparent_cmap`` renders those as the
    white vertical/horizontal lines in fig7/fig8.

    The fill is STRIPE-TARGETED: only NaN cells bounded on BOTH immediate
    sides (along the axis) by finite data are filled, with the nearest
    finite neighbour's value (so the filled cell inherits the adjacent
    borehole-driven value rather than smearing across a seam). Genuine
    no-data regions (offshore, no nearby boreholes / no valid groundwater)
    are wider than 1 cell and so are left NaN -- the caption's transparency
    contract is preserved. We sweep lon then lat over the trailing
    ``(lat, lon)`` axes, looping any leading non-geo dims (statistic/depth).

    Operates in place on ``combined`` and returns it.
    """
    if combined.ndim < 2:
        return combined
    n_lat = combined.shape[-2]
    n_lon = combined.shape[-1]
    leading = combined.shape[:-2]

    def _fill_axis_1cell(plane: np.ndarray, axis: int) -> None:
        """Fill width-1 NaN gaps along ``axis`` (0=lat, 1=lon) of a 2D plane."""
        n = plane.shape[axis]
        if n < 3:
            return
        finite = np.isfinite(plane)
        nan_mask = ~finite
        # A gap cell is fillable iff it is NaN AND both neighbours along the
        # axis are finite (so the NaN run is exactly 1 cell wide).
        if axis == 0:
            prev_finite = np.zeros_like(finite)
            next_finite = np.zeros_like(finite)
            prev_finite[1:, :] = finite[:-1, :]
            next_finite[:-1, :] = finite[1:, :]
            fillable = nan_mask & prev_finite & next_finite
            # Inherit the previous (lower-index) finite neighbour's value.
            src = np.empty_like(plane)
            src[1:, :] = plane[:-1, :]
            plane[fillable] = src[fillable]
        else:
            prev_finite = np.zeros_like(finite)
            next_finite = np.zeros_like(finite)
            prev_finite[:, 1:] = finite[:, :-1]
            next_finite[:, :-1] = finite[:, 1:]
            fillable = nan_mask & prev_finite & next_finite
            src = np.empty_like(plane)
            src[:, 1:] = plane[:, :-1]
            plane[fillable] = src[fillable]

    # Iterate every (lat, lon) plane in the (possibly multi-dim) array.
    for idx in np.ndindex(*leading) if leading else [()]:
        plane = combined[idx] if leading else combined
        # lon sweep first (the dominant stripe direction), then lat.
        _fill_axis_1cell(plane, axis=1)
        _fill_axis_1cell(plane, axis=0)
        if leading:
            combined[idx] = plane
    return combined


# ---------------------------------------------------------------------------
# Display-only block-mean downsample (file-size trim, NOT stripe removal)
# ---------------------------------------------------------------------------
#
# HISTORY / why this is no longer the stripe fix (2026-06-09 -> 2026-06-09b):
# the crosshatch ripple in fig7 + fig8(a)(c) was diagnosed as a grid-resampling
# artifact of the OLD loader's snap-then-assign: the shared global lon axis was
# built at ``d_lon = res/(R*cos(lat_min))`` (FINER than every tile's native
# spacing), so a ``np.searchsorted`` nearest-neighbour snap periodically
# DUPLICATED a source column/row, leaving a periodic ripple (rel amplitude
# ~0.088, dominant period ~9 cells in lon). The block-mean was originally bumped
# 4 -> 6 to *attenuate* that ripple, but a block-mean can only blur it -- it
# cannot remove it, and the ripple period scales with the block factor.
#
# ``_load_cube_dataarray`` now assembles the cube by LINEAR INTERPOLATION onto a
# uniform target grid at the MEDIAN native tile spacing (never finer), which
# eliminates the duplication at the source (see that function's docstring +
# ``test_load_cube_dataarray_linear_interp_kills_duplication_ripple``). The
# block-mean is therefore NO LONGER load-bearing for stripe removal; it is kept
# at a small factor purely to trim the on-disk PDF size (each display cell
# averages a 2x2 native block). Display-only: the on-disk cube is never
# modified. Set to 1 to render at full native resolution.
_DISPLAY_BLOCK_FACTOR = 2


def _block_mean_2d(arr: np.ndarray, factor: int) -> np.ndarray:
    """NaN-aware block-mean downsample of a 2D array by an integer ``factor``.

    The array is padded (on the trailing high-index edge of each axis) with
    NaN up to a multiple of ``factor``, reshaped to
    ``(H // f, f, W // f, f)``, and reduced with ``np.nanmean`` over the two
    block axes ``(1, 3)``. Each output cell is therefore the mean of the (up
    to) ``factor * factor`` finite native cells in its block; isolated NaN
    seam cells inside an otherwise-finite block are ignored and so vanish.

    Guards:

    - ``factor <= 1`` (or an array too small to tile) returns ``arr`` unchanged.
    - A block whose every cell is NaN stays NaN (``np.nanmean`` of an all-NaN
      slice is NaN); callers render those as transparent via ``set_bad``.
    """
    if factor is None or factor <= 1:
        return arr
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        return arr
    h, w = a.shape
    if h < factor or w < factor:
        # Too small to tile a single full block in some axis; leave as-is.
        return arr
    pad_h = (-h) % factor
    pad_w = (-w) % factor
    if pad_h or pad_w:
        a = np.pad(a, ((0, pad_h), (0, pad_w)),
                   mode="constant", constant_values=np.nan)
    hh, ww = a.shape
    blocks = a.reshape(hh // factor, factor, ww // factor, factor)
    # All-NaN blocks legitimately produce NaN; silence the RuntimeWarning.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanmean(blocks, axis=(1, 3))
    return out


# Gaussian display-smoothing sigma in DISPLAY cells. The cube values carry
# a baked-in crosshatch: the GP samples coarse-raster covariates (AIST
# regime mesh, DEM) on the 1 km grid, so the predicted field steps at every
# raster-cell boundary -> a periodic mesh texture. Block-mean DOES NOT remove
# it (subsampling aliases the ripple to a new period that scales with the
# block factor); a true low-pass Gaussian does.
#
# NOTE (2026-06-16): a display-only Gaussian low-pass was NOT sufficient to
# clean the figures. Pushing sigma up to the FFT-measured mesh period
# (sigma=3.73, P=13.03 display cells) attenuated the mesh ~5x on the transfer
# function yet the visual verdict still showed (i) a strong, regular
# crosshatch surviving across all fig7/fig8(a,c) panels and (ii) the regional
# basin/archipelago structure WASHED to only dim perceptibility -- the
# smoothing degraded real geography before it removed the seam grid. The
# crosshatch is not band-limited at a single period (it is the per-tile-seam
# stepping of the coarse covariate raster), so no single isotropic Gaussian
# both kills it and keeps the regional field. The honest fix is SOURCE-LEVEL:
# regenerate the national cube without the coarse-raster covariate stepping
# (e.g. bilinearly resample AIST/DEM covariates to the 1 km predict grid
# before the GP forward pass, or predict on the native covariate grid and
# resample the OUTPUT). Until that cube exists we keep the modest legacy
# sigma=2.5 as a cosmetic touch only and do NOT claim the figures are mesh-
# free. Display-only; the on-disk cube is never modified.
_DISPLAY_SMOOTH_SIGMA = 2.5


def _smooth_nan_aware(arr: np.ndarray, sigma: float = _DISPLAY_SMOOTH_SIGMA) -> np.ndarray:
    """NaN-aware Gaussian low-pass of a 2D array (normalized convolution).

    A plain ``gaussian_filter`` propagates NaN across the whole kernel
    footprint, eroding the map. Normalized convolution instead smooths the
    NaN-zeroed data and the finite-mask with the SAME kernel and divides:
    ``out = G(nan->0 data) / G(mask)``, restoring NaN where the local
    finite weight is negligible. This blurs the baked-in raster crosshatch
    (a true low-pass, so -- unlike block-mean subsampling -- it does not
    alias the ripple to a new period) while leaving NaN no-data regions and
    the coastline punch-out intact.

    ``sigma <= 0`` (or a non-2D / too-small array) returns ``arr`` unchanged.
    """
    if sigma is None or sigma <= 0:
        return arr
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.size == 0:
        return arr
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:  # pragma: no cover - scipy is a hard dep, defensive only
        return arr
    finite = np.isfinite(a)
    if not finite.any():
        return arr
    data0 = np.where(finite, a, 0.0)
    w = finite.astype(np.float64)
    num = gaussian_filter(data0, sigma=sigma, mode="nearest")
    den = gaussian_filter(w, sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    # Restore NaN where the local finite weight is negligible (interior of
    # no-data regions / outside the data footprint).
    out[den < 1e-3] = np.nan
    return out


# Fig 7 transect: modest 1-D smoothing along the DISTANCE (lon / column) axis
# only. The depth x distance section carries full-depth vertical striping from
# (i) discrete single-borehole IDW columns and (ii) the band-5 ~72 km Fourier
# lon component showing as a couple of broad undulations. The transect spans
# ~80-120 columns over ~120 km, so 1 column ~= 1-1.5 km; sigma=3.0 columns
# ~= 4-5 km is a standard geological cross-section smoothing that removes
# single-borehole-width spikes while preserving the broad bay -> stiff-ridge
# -> peninsula trend. The DEPTH axis is NEVER smoothed (the soft-surface /
# stiffening structure must stay sharp). Display-only; the cube is unmodified.
_TRANSECT_SMOOTH_SIGMA = 3.0


def _smooth_along_distance(section: np.ndarray,
                           sigma: float = _TRANSECT_SMOOTH_SIGMA) -> np.ndarray:
    """NaN-aware 1-D Gaussian low-pass along the DISTANCE axis (axis=1) only.

    ``section`` is the fig7 cross-section shaped ``(n_depth, n_lon)``; smoothing
    is applied PER ROW along the distance (column / lon) axis and never across
    depth, so the vertical depth structure stays sharp while single-borehole
    IDW columns and the broad lon-periodic undulation are damped.

    Implemented as a normalized convolution along axis=1: the NaN-zeroed data
    and the finite mask are each smoothed with the same 1-D Gaussian kernel and
    divided (``out = G(nan->0 data) / G(mask)``). NaN is restored where the
    local finite weight is negligible, so no-data gaps are preserved rather
    than bled across by the kernel.

    ``sigma <= 0`` (or a non-2D / too-small array) returns ``section``
    unchanged.
    """
    if sigma is None or sigma <= 0:
        return section
    a = np.asarray(section, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 2:
        return section
    try:
        from scipy.ndimage import gaussian_filter1d
    except Exception:  # pragma: no cover - scipy is a hard dep, defensive only
        return section
    finite = np.isfinite(a)
    if not finite.any():
        return section
    data0 = np.where(finite, a, 0.0)
    w = finite.astype(np.float64)
    # axis=1 == distance / lon; depth (axis=0) is left untouched.
    num = gaussian_filter1d(data0, sigma=sigma, axis=1, mode="nearest")
    den = gaussian_filter1d(w, sigma=sigma, axis=1, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    # Keep NaN where the local along-distance finite weight is negligible.
    out[den < 1e-3] = np.nan
    return out


def _downsample_axis(axis: np.ndarray, factor: int) -> np.ndarray:
    """Return the block-centre coordinate for each downsampled cell.

    Pads the coordinate axis to a multiple of ``factor`` with NaN, reshapes
    to ``(N // f, f)`` and takes the NaN-aware mean of each block so the
    display-cell centres line up with ``_block_mean_2d``'s blocks (and the
    image ``extent`` derived from min/max stays correct).
    """
    if factor is None or factor <= 1:
        return axis
    a = np.asarray(axis, dtype=np.float64)
    if a.ndim != 1 or a.size < factor:
        return axis
    pad = (-a.size) % factor
    if pad:
        a = np.pad(a, (0, pad), mode="constant", constant_values=np.nan)
    blocks = a.reshape(a.size // factor, factor)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(blocks, axis=1)


def _block_subsample_codes_2d(codes: np.ndarray, factor: int) -> np.ndarray:
    """Down-sample a CATEGORICAL 2D grid (e.g. regime codes) to the display
    grid by sampling the centre cell of each block.

    A NaN-aware mean is wrong for categorical codes (it would invent
    fractional regimes), so we instead pick the value at each block's centre
    offset ``factor // 2`` -- cheap, shape-consistent with ``_block_mean_2d``,
    and adequate for the per-regime conformal-multiplier lookup that drives
    fig8 panel (b)'s spatial structure. Returns ``codes`` unchanged when
    ``factor <= 1`` or the grid is too small to tile.
    """
    if factor is None or factor <= 1:
        return codes
    c = np.asarray(codes)
    if c.ndim != 2:
        return codes
    h, w = c.shape
    if h < factor or w < factor:
        return codes
    off = factor // 2
    rows = np.arange(off, h, factor)
    cols = np.arange(off, w, factor)
    # Match _block_mean_2d's block count (ceil division of padded extent).
    n_rows = (h + factor - 1) // factor
    n_cols = (w + factor - 1) // factor
    rows = rows[:n_rows]
    cols = cols[:n_cols]
    return c[np.ix_(rows, cols)]


def _median_native_spacing(coords_list: list[np.ndarray]) -> float:
    """Median per-tile native cell spacing along one axis.

    Each tile's axis is regular (``predict_cube`` builds it with a single
    ``cos(mid_lat)``-derived step), so the per-tile step is just
    ``diff[0]``. We take the MEDIAN across tiles -- NOT the minimum -- so
    the assembled target grid is no finer than a representative tile; a
    finer-than-native target is exactly what produced the nearest-snap
    duplication ripple. Falls back to a robust median of all pairwise
    diffs if a tile is degenerate (< 2 cells).
    """
    steps: list[float] = []
    for c in coords_list:
        c = np.asarray(c, dtype=np.float64)
        if c.size >= 2:
            steps.append(float(np.median(np.diff(c))))
    if not steps:
        raise ValueError("no tile has >= 2 cells along axis; cannot derive "
                         "a native spacing for the target grid")
    return float(np.median(steps))


def _tile_lat_lon_bounds(tile) -> tuple[float, float, float, float]:
    """Return ``(lat_min, lat_max, lon_min, lon_max)`` for a lazily-opened
    tile DataArray.

    Reads only the (tiny) ``lat`` / ``lon`` coordinate vectors -- NOT the
    chunked data payload -- so the bounds check costs one small metadata
    read per tile rather than a deep chunk fetch. This is what makes the
    sparse loader ESTALE-resistant on the flaky aries NFS: a tile whose
    bbox does not intersect the request is never touched beyond its coords.
    """
    t_lat = np.asarray(tile.coords["lat"].values, dtype=np.float64)
    t_lon = np.asarray(tile.coords["lon"].values, dtype=np.float64)
    return (float(t_lat.min()), float(t_lat.max()),
            float(t_lon.min()), float(t_lon.max()))


def _ranges_intersect(lo_a: float, hi_a: float,
                      lo_b: float, hi_b: float) -> bool:
    """Closed-interval intersection test (inclusive of touching edges)."""
    return (lo_a <= hi_b) and (lo_b <= hi_a)


def _combine_tile_cubes(cubes: list, resolution_m: float = 1000.0):
    """Assemble a list of (already lazily-opened) tile DataArrays into one
    ``xarray.DataArray`` via LINEAR interpolation onto a uniform target
    grid at the median native tile spacing.

    Factored out of :func:`_load_cube_dataarray` so both the full-cube
    loader and the sparse :func:`_load_tiles_covering` share the same
    snap-free linear-interp regrid (see the historical note in
    ``_load_cube_dataarray``'s docstring for *why* linear-interp, not
    nearest-snap). ``cubes`` must be non-empty.
    """
    import xarray as xr  # type: ignore

    if not cubes:
        raise ValueError("_combine_tile_cubes called with no tiles")
    if len(cubes) == 1:
        return cubes[0]

    # ------------------------------------------------------------------
    # 1. Uniform global target grid at the MEDIAN native tile spacing.
    # ------------------------------------------------------------------
    all_lats = np.concatenate([c.coords["lat"].values for c in cubes])
    all_lons = np.concatenate([c.coords["lon"].values for c in cubes])
    lat_min, lat_max = float(all_lats.min()), float(all_lats.max())
    lon_min, lon_max = float(all_lons.min()), float(all_lons.max())
    d_lat = _median_native_spacing([c.coords["lat"].values for c in cubes])
    d_lon = _median_native_spacing([c.coords["lon"].values for c in cubes])
    n_lat = max(int(round((lat_max - lat_min) / d_lat)) + 1, 1)
    n_lon = max(int(round((lon_max - lon_min) / d_lon)) + 1, 1)
    lat_axis_global = np.linspace(lat_min, lat_max, num=n_lat, endpoint=True)
    lon_axis_global = np.linspace(lon_min, lon_max, num=n_lon, endpoint=True)
    LOG.info(
        "Linear-interp target grid: lat=%d cells (d=%.6g), "
        "lon=%d cells (d=%.6g)",
        n_lat, d_lat, n_lon, d_lon,
    )

    template = cubes[0]
    non_geo_dims = tuple(d for d in template.dims if d not in ("lat", "lon"))
    non_geo_shape = tuple(template.sizes[d] for d in non_geo_dims)
    combined_shape = non_geo_shape + (n_lat, n_lon)
    combined = np.full(combined_shape, np.nan, dtype=np.float64)
    non_geo_slices = (slice(None),) * len(non_geo_dims)

    # ------------------------------------------------------------------
    # 2-3. Resample each tile onto the in-bounds target points and assign.
    # ------------------------------------------------------------------
    for tile in cubes:
        t_lat = np.asarray(tile.coords["lat"].values, dtype=np.float64)
        t_lon = np.asarray(tile.coords["lon"].values, dtype=np.float64)
        lat_sel = (lat_axis_global >= t_lat.min()) & (
            lat_axis_global <= t_lat.max())
        lon_sel = (lon_axis_global >= t_lon.min()) & (
            lon_axis_global <= t_lon.max())
        if not lat_sel.any() or not lon_sel.any():
            continue
        tgt_lat = lat_axis_global[lat_sel]
        tgt_lon = lon_axis_global[lon_sel]
        interp = tile.interp(lat=tgt_lat, lon=tgt_lon, method="linear")
        interp = interp.transpose(*non_geo_dims, "lat", "lon")
        lat_idx = np.flatnonzero(lat_sel)
        lon_idx = np.flatnonzero(lon_sel)
        combined[non_geo_slices + np.ix_(lat_idx, lon_idx)] = (
            interp.values.astype(np.float64))

    combined = _fill_thin_nan_stripes(
        combined, lat_axis_global, lon_axis_global)

    coords = {d: template.coords[d].values for d in non_geo_dims}
    coords["lat"] = lat_axis_global
    coords["lon"] = lon_axis_global
    cube = xr.DataArray(
        combined,
        dims=non_geo_dims + ("lat", "lon"),
        coords=coords,
        name=template.name or "prediction",
    )
    LOG.info(
        "Combined cube shape (linear-interp regrid) = %s; "
        "target grid lat=%d lon=%d",
        dict(cube.sizes), n_lat, n_lon,
    )
    return cube


def _load_tiles_covering(cube_dir: Path,
                         lat_range: tuple[float, float],
                         lon_range: tuple[float, float],
                         resolution_m: float = 1000.0):
    """Aggregate ONLY the tile zarrs whose lat/lon footprint intersects the
    requested ``lat_range`` / ``lon_range`` rectangle.

    This is the ESTALE-resistant alternative to :func:`_load_cube_dataarray`
    for the 1-D figure renderers (fig7 transect, fig8 depth profiles). The
    full national cube is ~825 tiles; a deep chunk read across all of them
    on the flaky aries NFS periodically throws ``[Errno 116] Stale file
    handle``. By globbing the tiles, opening each LAZILY (``xr.open_dataarray``
    does not read the data payload), inspecting only the small ``lat`` /
    ``lon`` coordinate vectors, and keeping ONLY the handful of tiles that
    intersect the request, we cut the heavy chunk reads by ~100x (a const-lat
    transect band touches ~1 row of tiles; three point bboxes touch ~3 tiles).

    Robustness contract: each tile open is wrapped so a single bad tile (a
    truncated zarr, an ESTALE during the metadata read) is SKIPPED with a
    warning rather than aborting the whole figure. The selected tiles are
    combined via the same linear-interp regrid as the full loader
    (:func:`_combine_tile_cubes`).

    ``lat_range`` / ``lon_range`` are ``(min, max)`` pairs; pass
    ``(-inf, inf)`` on an axis to keep every tile along it. Raises
    ``FileNotFoundError`` when no tile zarrs exist at all, and
    ``ValueError`` when tiles exist but none intersect the request (the
    callers translate both into a placeholder / per-figure fallback).
    """
    import xarray as xr  # type: ignore

    tile_dirs = _resolve_tile_dirs(cube_dir)
    if not tile_dirs:
        raise FileNotFoundError(
            f"no tile_*.zarr found under {cube_dir} (looked at "
            f"{cube_dir / 'cube'} and {cube_dir}); ensure the cube is "
            "synced or pass a path that contains cube/ + maps/")

    lat_lo, lat_hi = float(lat_range[0]), float(lat_range[1])
    lon_lo, lon_hi = float(lon_range[0]), float(lon_range[1])

    selected: list = []
    n_skipped = 0
    for t in tile_dirs:
        try:
            tile = xr.open_dataarray(t, engine="zarr")
        except Exception as exc:
            n_skipped += 1
            LOG.warning("skipping unreadable tile %s: %s", t, exc)
            continue
        try:
            t_lat_lo, t_lat_hi, t_lon_lo, t_lon_hi = _tile_lat_lon_bounds(tile)
        except Exception as exc:
            n_skipped += 1
            LOG.warning("skipping tile with unreadable coords %s: %s", t, exc)
            continue
        if (_ranges_intersect(t_lat_lo, t_lat_hi, lat_lo, lat_hi)
                and _ranges_intersect(t_lon_lo, t_lon_hi, lon_lo, lon_hi)):
            selected.append(tile)

    LOG.info(
        "Sparse tile load: %d/%d tiles intersect lat[%.4g,%.4g] "
        "lon[%.4g,%.4g] (%d skipped as unreadable)",
        len(selected), len(tile_dirs), lat_lo, lat_hi, lon_lo, lon_hi,
        n_skipped,
    )
    if not selected:
        raise ValueError(
            f"no tile intersects lat={lat_range} lon={lon_range} "
            f"among {len(tile_dirs)} tiles under {cube_dir}")
    return _combine_tile_cubes(selected, resolution_m=resolution_m)


def _load_cube_dataarray(cube_dir: Path, resolution_m: float = 1000.0):
    """Aggregate tile zarrs into a single ``xarray.DataArray`` via LINEAR
    interpolation onto a uniform target grid at the median native tile
    spacing.

    The on-disk ``tile_*.zarr`` files are written by
    ``engine.predict_cube()`` BEFORE any coordinate snapping, so each
    tile's lon axis is built at its own mid-latitude
    (``d_lon = res / (111320 * cos(lat))``). Two tiles in the same lon
    column but different lat-bands therefore have *different* lon
    vectors (verified divergence ~5e-3 deg; no two lat-bands share any
    lon coordinate). A bare ``xr.combine_by_coords(..., fill_value=NaN)``
    either raises "Resulting object does not have monotonic global
    indexes along dimension lon" (newer xarray) or outer-joins ~74,800
    distinct lon values into a giant grid that is ~97% NaN.

    The previous fix built a shared global axis *finer* than every tile
    (``d_lon = res/(R*cos(lat_min))``) and snapped each tile via nearest
    neighbour (``np.searchsorted``). Because the target was ~0.6% finer
    than the native spacing, the snap periodically DUPLICATED source
    columns/rows, producing a crosshatch ripple (rel amplitude ~0.088)
    whose period scaled with any downstream block factor -- so a
    display block-mean could not remove it (autocorrelation-confirmed).

    The correct fix (this implementation):

    1. Build a uniform global target grid (regular lat/lon) at the
       MEDIAN native tile spacing -- NOT finer than a representative
       tile, so there is no systematic over-sampling to duplicate from.
    2. Resample each tile onto the target points that fall within that
       tile's bounds via ``DataArray.interp(lat=..., lon=...,
       method="linear")``. Linear interpolation BLENDS neighbouring
       source cells smoothly instead of nearest-picking a single cell,
       which eliminates the duplication ripple.
    3. Assign each interpolated tile into a pre-allocated global array
       (last-tile-wins on shared boundary cells -- the same contract a
       successful ``combine_by_coords`` would honour).

    NaN handling is preserved: ``interp`` over a tile region with NaN
    yields NaN there, and target cells covered by no tile stay NaN
    (rendered transparent by ``_transparent_cmap``). Because the target
    grid is no finer than native, no interior width-1 NaN stripes are
    introduced, so the snap-era ``_fill_thin_nan_stripes`` pass is no
    longer needed on this path (it would otherwise smear across genuine
    disjoint-tile gaps).

    Returns the DataArray, which still carries the ``statistic``
    dimension when the cube was written with posterior ``mean`` and
    ``std`` slices.

    Raises if no tile zarrs are found or the load itself fails; the
    callers translate that into a placeholder PDF.

    .. note::

       This loader reads EVERY tile (~825 across the national cube). The
       1-D figure renderers (fig7 transect, fig8 depth profiles) instead
       use :func:`_load_tiles_covering`, which keeps only the handful of
       tiles intersecting the requested lat/lon window -- ~100x fewer deep
       chunk reads, so it does not trip the aries-NFS stale-handle. This
       function stays the canonical full-cube path (still used by the
       regression tests asserting the full assembled grid).
    """
    import xarray as xr  # type: ignore

    tile_dirs = _resolve_tile_dirs(cube_dir)
    if not tile_dirs:
        raise FileNotFoundError(
            f"no tile_*.zarr found under {cube_dir} (looked at "
            f"{cube_dir / 'cube'} and {cube_dir}); ensure the cube is "
            "synced or pass a path that contains cube/ + maps/")
    LOG.info("Aggregating %d tile cubes from %s (linear-interp regrid)",
             len(tile_dirs), tile_dirs[0].parent)
    cubes = [xr.open_dataarray(t, engine="zarr") for t in tile_dirs]
    return _combine_tile_cubes(cubes, resolution_m=resolution_m)


# ---------------------------------------------------------------------------
# Japan basemap helpers
# ---------------------------------------------------------------------------
#
# A *very* coarse simplified Japan coastline polygon, used as a basemap
# overlay when ``cartopy`` is not available in the rendering image.
# Each tuple is one closed ring of (lon, lat) pairs in EPSG:4326. The
# four rings trace the four main islands at ~50 km vertex spacing -- the
# goal is not cartographic accuracy but a recognisable "Japan-shaped"
# silhouette so the cube slice is geographically anchored at a glance.
# Manually digitised from the GMT global coastline (gshhg) reduced to
# a per-island convex-ish hull; reviewed against AIST 2026 base map.
_JAPAN_RINGS: tuple[tuple[tuple[float, float], ...], ...] = (
    # Hokkaido
    (
        (140.0, 41.4), (140.5, 42.0), (140.3, 42.6), (140.7, 43.2),
        (141.4, 43.4), (141.8, 43.2), (142.6, 42.8), (143.4, 42.3),
        (143.9, 42.5), (144.4, 42.9), (144.8, 43.4), (145.4, 43.6),
        (145.3, 44.1), (144.8, 44.4), (144.1, 44.6), (143.2, 44.8),
        (142.4, 44.9), (141.6, 45.0), (141.2, 45.3), (141.5, 45.5),
        (141.0, 45.5), (140.4, 45.4), (139.9, 44.7), (139.7, 44.0),
        (140.0, 43.4), (140.4, 42.7), (140.3, 42.1), (140.0, 41.4),
    ),
    # Honshu
    (
        (140.3, 41.2), (141.1, 40.8), (141.5, 40.3), (141.6, 39.6),
        (141.9, 39.0), (142.1, 38.4), (141.6, 37.8), (141.0, 37.1),
        (140.8, 36.4), (140.6, 35.7), (140.2, 35.2), (139.8, 34.9),
        (139.2, 35.2), (138.7, 35.0), (138.2, 34.6), (137.5, 34.7),
        (136.9, 34.5), (136.4, 34.3), (135.9, 33.7), (135.4, 33.5),
        (135.0, 33.9), (134.6, 34.3), (134.2, 34.6), (133.8, 34.4),
        (133.0, 34.3), (132.4, 34.2), (131.7, 34.0), (131.0, 34.4),
        (130.9, 35.0), (131.6, 35.5), (132.4, 35.5), (133.0, 35.4),
        (133.7, 35.6), (134.3, 35.6), (135.0, 35.7), (135.7, 35.5),
        (136.3, 36.2), (136.9, 36.8), (137.4, 37.4), (138.1, 37.9),
        (138.9, 38.0), (139.4, 38.4), (139.7, 39.1), (139.8, 39.7),
        (139.7, 40.3), (140.0, 40.8), (140.3, 41.2),
    ),
    # Shikoku
    (
        (132.5, 33.5), (133.0, 33.3), (133.7, 33.1), (134.3, 33.5),
        (134.7, 33.8), (134.6, 34.2), (134.2, 34.4), (133.5, 34.4),
        (132.9, 34.3), (132.5, 33.9), (132.5, 33.5),
    ),
    # Kyushu
    (
        (130.2, 31.0), (130.7, 31.3), (131.4, 31.4), (131.7, 31.9),
        (131.8, 32.6), (132.1, 33.1), (131.5, 33.6), (130.8, 33.5),
        (130.2, 33.3), (129.6, 33.0), (129.5, 32.5), (129.8, 31.9),
        (130.0, 31.4), (130.2, 31.0),
    ),
)


#: Simplified MLIT C23 national coastline, 247 parts / 5,346 vertices, built by
#: ``scripts.build_japan_coastline``. Loaded once and memoised.
_COASTLINE_ASSET = (Path(__file__).resolve().parents[1]
                    / "national/data/assets/japan_coastline.json")
_COASTLINE_CACHE: list[list[list[float]]] | None = None


def _japan_coastline() -> tuple[tuple[tuple[float, float], ...], ...]:
    """Return the coastline polylines, preferring the MLIT-derived asset.

    Falls back to the 100-vertex ``_JAPAN_RINGS`` hull when the asset is
    absent (e.g. a checkout without it). The hull does not read as Japan at
    locator-inset scale, so the asset is strongly preferred.
    """
    global _COASTLINE_CACHE
    if _COASTLINE_CACHE is None:
        try:
            _COASTLINE_CACHE = json.loads(_COASTLINE_ASSET.read_text())
        except Exception as exc:  # noqa: BLE001
            LOG.warning("coastline asset unavailable (%s); falling back to "
                        "the simplified hull. Rebuild with "
                        "`python -m scripts.build_japan_coastline`.", exc)
            _COASTLINE_CACHE = [[list(p) for p in ring]
                                for ring in _JAPAN_RINGS]
    return _COASTLINE_CACHE


def _geographic_aspect(lat_extent: tuple[float, float]) -> float:
    """Display aspect that makes one degree of latitude and one degree of
    longitude cover the same ground distance at the extent's mid-latitude.

    Plotting lon/lat with ``aspect="equal"`` stretches the map horizontally by
    ``1/cos(lat)`` -- about 22 % at 35 N -- which is a large part of why the
    basemaps did not read as Japan.
    """
    mid = math.radians(0.5 * (lat_extent[0] + lat_extent[1]))
    return 1.0 / max(math.cos(mid), 1e-6)


def _draw_japan_basemap(ax: plt.Axes, lon_extent: tuple[float, float],
                        lat_extent: tuple[float, float],
                        linewidth: float = 0.6,
                        color: str = "#222222",
                        alpha: float = 0.85,
                        set_limits: bool = True,
                        set_aspect: bool = True) -> None:
    """Overlay the Japan coastline on ``ax`` in data coordinates.

    The polylines are drawn in EPSG:4326, so the caller must use
    ``ax.imshow(..., extent=[lon0, lon1, lat0, lat1])`` rather than the
    pixel-coordinate default.
    """
    for ring in _japan_coastline():
        if len(ring) < 2:
            continue
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha,
                solid_capstyle="round", solid_joinstyle="round")
    if set_limits:
        ax.set_xlim(lon_extent)
        ax.set_ylim(lat_extent)
    if set_aspect:
        ax.set_aspect(_geographic_aspect(lat_extent))


def _cube_extent(da) -> tuple[float, float, float, float] | None:
    """Return ``(lon0, lon1, lat0, lat1)`` for an xarray DataArray, or
    ``None`` if the spatial coordinates cannot be inferred.

    Looks for the canonical coordinate names produced by
    ``predict_national_cube.py``: ``lon`` / ``lat`` (preferred) or
    ``longitude`` / ``latitude``.
    """
    lon_key = None
    lat_key = None
    for name in ("lon", "longitude", "x"):
        if name in da.coords:
            lon_key = name; break
    for name in ("lat", "latitude", "y"):
        if name in da.coords:
            lat_key = name; break
    if lon_key is None or lat_key is None:
        return None
    lon = da.coords[lon_key].values
    lat = da.coords[lat_key].values
    return (float(lon.min()), float(lon.max()),
            float(lat.min()), float(lat.max()))


def _quantile_vlim(arr: np.ndarray, q_lo: float = 0.02,
                   q_hi: float = 0.98) -> tuple[float, float]:
    """Compute (vmin, vmax) from finite-only quantiles. Returns
    ``(0.0, 1.0)`` if no finite values are available so callers do not
    crash.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.nanquantile(finite, q_lo))
    hi = float(np.nanquantile(finite, q_hi))
    if hi <= lo:
        # Degenerate slice (constant or one-sided); widen so imshow does
        # not collapse the colormap to a single value.
        hi = lo + max(abs(lo), 1.0) * 0.05
    return lo, hi


def _transparent_cmap(name: str = "viridis"):
    """Return a copy of ``name`` colormap with NaN cells set to
    transparent (RGBA alpha=0). Keeps the basemap visible behind
    no-data pixels.
    """
    import matplotlib.cm as mcm
    cmap = mcm.get_cmap(name).copy()
    cmap.set_bad(color=(0, 0, 0, 0))
    return cmap


# ---------------------------------------------------------------------------
# Cross-section / depth-profile sampling helpers
# ---------------------------------------------------------------------------
#
# These extract 1-D views (a single (lat,lon) column or a single const-lat
# lon row) from the cube. A 1-D path never exposes the lateral 2-D
# random-Fourier positional-encoding crosshatch baked into the FILLED maps,
# so the transect / profile figures are clean by construction (probe
# 2026-06-16: transect along-lon lag-1 autocorr +0.38, POSITIVE => smooth
# geological structure, not the alternating-sign signature of a residual
# Fourier mesh).

# The default const-lat transect: Tokyo metropolitan lowlands westward across
# Tokyo Bay onto the Boso Peninsula. Probe nearest-grid row at lat 35.596 N;
# 80 lon samples, all finite, smooth along-line structure.
_TRANSECT_LAT = 35.6
_TRANSECT_LON0 = 139.0
_TRANSECT_LON1 = 140.3
_TRANSECT_N = 80

# Three depth-profile sites (probe site_coords, regime codes annotated).
# Each entry is (label, short_type, lat, lon).
_PROFILE_SITES: tuple[tuple[str, str, float, float], ...] = (
    ("Tokyo Bay", "soft alluvium", 35.6, 139.8),
    ("Japan Alps", "stiff bedrock", 36.2, 138.2),
    ("Osaka plain", "soft alluvium", 34.7, 135.5),
)


def _extract_depth_column(mean_da, std_da, lat: float, lon: float
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                     float, float]:
    """Sample a single posterior-mean (+ std) depth column at the cube cell
    nearest ``(lat, lon)``.

    Returns ``(depths, mean_col, std_col, used_lat, used_lon)`` where
    ``depths`` is the cube's depth axis (ascending) and the two value
    arrays are the per-depth posterior-mean / posterior-std at the nearest
    grid cell. ``used_lat`` / ``used_lon`` are the actual grid coordinates
    selected (so the caller can annotate the snap). A 1-D column samples
    one (lat,lon) cell at every depth, so it never carries the lateral 2-D
    crosshatch.
    """
    sel_mean = mean_da.sel(lat=lat, lon=lon, method="nearest")
    depths = np.asarray(sel_mean.coords["depth"].values, dtype=np.float64)
    order = np.argsort(depths)
    depths = depths[order]
    mean_col = np.asarray(sel_mean.values, dtype=np.float64)[order]
    used_lat = float(sel_mean.coords["lat"].values)
    used_lon = float(sel_mean.coords["lon"].values)
    if std_da is not None:
        sel_std = std_da.sel(lat=lat, lon=lon, method="nearest")
        std_col = np.asarray(sel_std.values, dtype=np.float64)[order]
    else:
        std_col = np.full_like(mean_col, np.nan)
    return depths, mean_col, std_col, used_lat, used_lon


def _extract_const_lat_transect(mean_da, lat: float,
                                lon0: float, lon1: float, n: int
                                ) -> tuple[np.ndarray, np.ndarray,
                                           np.ndarray, float]:
    """Sample the posterior-mean cube along a constant-latitude line.

    Walks ``n`` evenly spaced longitudes in ``[lon0, lon1]`` at the cube
    row nearest ``lat`` and stacks the per-depth columns into a
    ``(n_depth, n)`` cross-section. Because every sample lies on one lat
    row, the result is a 1-D spatial path and does NOT reproduce the 2-D
    lateral random-Fourier crosshatch the filled depth-slice maps carried.

    Returns ``(lon_samples, depths, section, used_lat)`` with ``section``
    shaped ``(n_depth, n_lon)`` (depth ascending). GP posteriors can return
    small negative N at depth (probe value range min -4.3); the caller
    clips the DISPLAY floor at 0 and annotates -- the underlying array is
    returned unclipped.
    """
    lon_samples = np.linspace(lon0, lon1, num=n)
    # Interpolate to EXACTLY the requested latitude rather than snapping to
    # the nearest regrid row. The sparse-tile target grid does not always
    # carry a row at ``lat`` (tile origins fall on 1/3-degree multiples, so
    # method="nearest" could land ~0.4 deg away -- e.g. 35.6 N snapping to
    # 36.0 N, which would mislabel a "Tokyo Bay" transect as inland Kanto).
    # Linear interpolation keeps the line on the geographically-intended
    # parallel so the title, caption, and locator all agree.
    row = mean_da.interp(lat=lat)
    used_lat = float(lat)
    sel = row.sel(lon=lon_samples, method="nearest")
    # Order by ascending depth so the imshow y-axis is monotone.
    depths = np.asarray(sel.coords["depth"].values, dtype=np.float64)
    order = np.argsort(depths)
    depths = depths[order]
    sel = sel.transpose("depth", "lon")
    section = np.asarray(sel.values, dtype=np.float64)[order, :]
    return lon_samples, depths, section, used_lat


def _draw_transect_locator_inset(ax, lat: float, lon0: float,
                                 lon1: float) -> None:
    """Draw a small Japan locator map with the transect line overlaid.

    Uses the MLIT-derived coastline at a true geographic aspect; the
    const-lat line is drawn in red so the reader can place the cross-section
    geographically. A wider window than the transect itself is deliberate --
    the point of a locator is to show WHERE in Japan the line sits.
    """
    lon_lo, lon_hi = lon0 - 3.2, lon1 + 3.2
    lat_lo, lat_hi = lat - 3.4, lat + 3.4
    _draw_japan_basemap(ax, lon_extent=(lon_lo, lon_hi),
                        lat_extent=(lat_lo, lat_hi),
                        linewidth=0.5)
    # Halo under the transect so it stays visible where it crosses coastline.
    ax.plot([lon0, lon1], [lat, lat], color="white", linewidth=3.2,
            solid_capstyle="round", zorder=4)
    ax.plot([lon0, lon1], [lat, lat], color="#d62728", linewidth=1.8,
            solid_capstyle="round", zorder=5)
    ax.plot([lon0, lon1], [lat, lat], color="#d62728", marker="o",
            markersize=2.5, linestyle="none", zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("locator", fontsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


def fig7_cube_transect(out: Path, cube_dir: Path,
                       lat: float = _TRANSECT_LAT,
                       lon0: float = _TRANSECT_LON0,
                       lon1: float = _TRANSECT_LON1,
                       n_samples: int = _TRANSECT_N) -> None:
    """Vertical (depth x distance) cross-section of posterior-mean SPT-$N$
    along a constant-latitude line, with a locator inset.

    Replaces the former 2-D filled depth-slice maps (``fig7_cube_slices``),
    which carried the source-level random-Fourier positional-encoding
    crosshatch on every (lat,lon) FILLED panel. The transect samples ONE
    lat row, so it is a 1-D spatial path that never exposes the lateral 2-D
    mesh (probe 2026-06-16: along-lon lag-1 autocorrelation +0.38 => smooth
    geological structure, the opposite of an oscillatory Fourier residual).

    The default line is the Tokyo metropolitan lowlands at lat 35.6 N from
    lon 139.0 E across Tokyo Bay onto the Boso Peninsula (140.3 E), sampled
    over the cube's depth levels. GP posteriors can return small negative N
    at depth, so the colour floor is clipped at 0 (and annotated); the
    underlying cube is unmodified.

    Reads the tiled-zarr cube produced by ``predict_national_cube.py``
    (``<cube_dir>/cube/tile_*.zarr``; for back-compat ``<cube_dir>`` may
    point directly at the ``cube/`` subdirectory). ``cube_dir`` may be a
    local path or an NFS mount. A placeholder PDF is emitted when no tile
    zarrs are found.
    """
    tile_dirs = _resolve_tile_dirs(cube_dir)
    if not tile_dirs:
        _placeholder(
            out,
            (f"Fig 7 -- DEFERRED\n\n"
             f"National cube tiles not found.\n"
             f"Expected at: {cube_dir}/cube/tile_*.zarr\n\n"
             "Point --cube-dir at a directory containing cube/ + maps/, "
             "either a local sync target or an NFS mount such as "
             "/mnt/nas/products/national_cube_japan_1km_v2hero."),
            ("**Fig. 7** National cube vertical transect "
             "(deferred; no tile zarrs found)."),
        )
        return

    # Sparse load: a const-lat transect only needs the ~1 row of tiles along
    # the line. We open a thin lat band around ``lat`` (a small pad covers the
    # nearest-row snap + any tile that abuts the line) over the full lon span
    # of the transect, instead of the full 825-tile cube whose deep chunk
    # reads trip the aries-NFS [Errno 116] stale-handle.
    _LAT_PAD = 0.25  # deg; ~1 tile worth either side of the line
    _LON_PAD = 0.10
    lat_band = (lat - _LAT_PAD, lat + _LAT_PAD)
    lon_band = (min(lon0, lon1) - _LON_PAD, max(lon0, lon1) + _LON_PAD)
    try:
        cube = _load_tiles_covering(cube_dir, lat_band, lon_band)
        if "statistic" in cube.dims:
            mean_da = cube.sel(statistic="mean")
        else:
            mean_da = cube
        lon_samples, depths, section, used_lat = _extract_const_lat_transect(
            mean_da, lat, lon0, lon1, n_samples)
    except Exception as exc:
        LOG.warning("transect extraction failed: %s", exc)
        _placeholder(
            out,
            f"Fig 7 placeholder (cube transect failed: {exc})",
            "**Fig. 7** National cube vertical transect (load failed).",
        )
        return

    # Distance along the line in km (great-circle approx at this latitude).
    km_per_deg_lon = 111.320 * float(np.cos(np.deg2rad(used_lat)))
    dist_km = (lon_samples - lon_samples[0]) * km_per_deg_lon

    # Modest along-DISTANCE smoothing for display (sigma ~ 3 columns ~= 4-5 km;
    # standard geological cross-section rendering): suppresses full-depth
    # vertical striping from single-borehole IDW columns + the broad lon-
    # periodic undulation, while the DEPTH axis stays UNSMOOTHED so the soft-
    # surface / stiffening structure remains sharp. Display-only; the on-disk
    # cube is never modified.
    section = _smooth_along_distance(section)

    # Display floor at 0: GP can return small negative N at depth; clip the
    # colour mapping (NOT the data) so the soft-basin lows do not read as a
    # second hot band, and annotate the clip in the caption.
    section_disp = np.where(np.isfinite(section), np.maximum(section, 0.0),
                            np.nan)
    vmin, vmax = _quantile_vlim(section_disp, 0.02, 0.98)
    vmin = max(vmin, 0.0)

    # The cross-section is a full-bleed pcolormesh with no whitespace, so an
    # overlaid locator inset necessarily hides data -- at its previous
    # ax.inset_axes([0.66, 0.62, 0.32, 0.36]) an opaque white box covered the
    # shallow 0-7 m band over the Boso half of the line, and its "locator"
    # title was clipped by the axes top edge. Give the locator its own column
    # instead so nothing is occluded.
    fig = plt.figure(figsize=(10.2, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1.0], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    cmap = _transparent_cmap("viridis")
    ma = np.ma.masked_invalid(section_disp)
    # pcolormesh over (distance, depth): cell edges from the sample +
    # depth axes so the depth-down orientation is exact (y inverted).
    dist_edges = _edges_from_centres(dist_km)
    depth_edges = _edges_from_centres(depths)
    mesh = ax.pcolormesh(dist_edges, depth_edges, ma, cmap=cmap,
                         vmin=vmin, vmax=vmax, shading="flat")
    ax.invert_yaxis()  # depth increases downward
    ax.set_xlabel(f"Distance along {used_lat:.2f}$^\\circ$N "
                  f"({lon0:.1f}$^\\circ$E $\\to$ {lon1:.1f}$^\\circ$E)  [km]")
    ax.set_ylabel("Depth  [m]")
    ax.set_title(
        f"Posterior-mean SPT-$N$ cross-section along {used_lat:.2f}"
        "$^\\circ$N (Tokyo Bay $\\to$ Boso)")
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Posterior mean SPT-$N$  [blow count]")

    # Locator panel (own column): Japan coastline + the transect line.
    loc_ax = fig.add_subplot(gs[0, 1])
    _draw_transect_locator_inset(loc_ax, used_lat, lon0, lon1)

    caption = (
        "**Fig. 7** Vertical cross-section of posterior-mean SPT-$N$ along "
        f"a constant-latitude line at {used_lat:.2f}$^\\circ$N, from "
        f"{lon0:.1f}$^\\circ$E across Tokyo Bay onto the Boso Peninsula "
        f"({lon1:.1f}$^\\circ$E), sampled at "
        f"{n_samples} longitudes over the cube's "
        "depth levels (depth increasing downward). Because every sample "
        "lies on a single latitude row, this is a 1-D spatial path that does "
        "NOT reproduce the lateral 2-D random-Fourier positional-encoding "
        "texture \\citep{tancik2020fourier} that contaminated the earlier "
        "filled depth-slice maps; along-line lag-1 autocorrelation is "
        "positive (smooth geological structure, not an oscillatory mesh). "
        "The colour mapping is floored at 0 because the Gaussian-process "
        "posterior can return small negative $N$ at depth; the underlying "
        "cube is unmodified. A modest 1-D Gaussian smoothing "
        "($\\sigma\\approx4$--$5$ km) is applied ALONG the distance axis only "
        "for display -- consistent with standard geological cross-section "
        "rendering -- to suppress fine vertical striping from discrete "
        "single-borehole inverse-distance columns and the broad longitudinal "
        "undulation; the depth structure is left unsmoothed and the on-disk "
        "cube is unchanged. The right-hand locator panel shows the line "
        "(red) over the MLIT C23 coastline at true geographic aspect. "
        "Only the per-tile zarrs under "
        "``<cube-dir>/cube/`` whose footprint intersects the transect band "
        "are read (a single tile row), then linearly interpolated onto a "
        "uniform median-native-spacing grid -- a sparse load that avoids "
        "deep reads across the full national cube."
    )
    _save_pdf(fig, out, caption)


def _edges_from_centres(centres: np.ndarray) -> np.ndarray:
    """Return ``len(centres)+1`` cell edges from monotone centre coords so a
    ``pcolormesh(edges_x, edges_y, C)`` places each value cell-centred.

    Midpoints between successive centres become interior edges; the two
    outer edges extrapolate the first/last half-step. Falls back to a unit
    span when ``centres`` has a single element.
    """
    c = np.asarray(centres, dtype=np.float64)
    if c.size == 0:
        return np.array([0.0, 1.0])
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mids = 0.5 * (c[:-1] + c[1:])
    first = c[0] - (mids[0] - c[0])
    last = c[-1] + (c[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


# ---------------------------------------------------------------------------
# Fig 8 -- uncertainty + LPI (deferred / placeholder)
# ---------------------------------------------------------------------------

def _load_lpi_map(cube_dir: Path):
    """Load the LPI map from ``<cube_dir>/maps/lpi_pga30.nc``.

    The variable name inside the NetCDF is ``lpi_pga30`` (produced by
    ``predict_national_cube.py`` as
    ``f"lpi_pga{int(scenario_pga * 100):02d}"`` for the default
    ``scenario_pga=0.30``). Returns the ``xr.DataArray`` on success or
    ``None`` if the file is missing or unreadable.
    """
    import time

    import xarray as xr  # type: ignore

    nc_path = cube_dir / "maps" / "lpi_pga30.nc"
    # NFS attribute-cache misses under the concurrent zarr tile-read load can
    # make a single ``os.stat`` raise a TRANSIENT OSError (PermissionError /
    # ESTALE) on a file that is in fact readable -- the same failure mode
    # ``_resolve_conformal_json`` already guards. ``Path.exists()`` is not
    # wrapped, so an unguarded transient stat used to abort the whole fig8
    # render. Retry the stat (and the open) a few times before giving up so a
    # one-off NFS hiccup degrades to the placeholder instead of crashing.
    exists = False
    for attempt in range(5):
        try:
            exists = nc_path.exists()
            break
        except OSError as exc:  # transient NFS stat error (ESTALE/EACCES)
            LOG.warning("LPI stat attempt %d failed (%s); retrying", attempt, exc)
            time.sleep(0.5 * (attempt + 1))
    if not exists:
        LOG.info("LPI map not found / unreadable at %s", nc_path)
        return None
    ds = None
    for attempt in range(5):
        try:
            ds = xr.open_dataset(nc_path)
            break
        except OSError as exc:  # transient NFS read error
            LOG.warning("LPI open attempt %d failed (%s); retrying", attempt, exc)
            time.sleep(0.5 * (attempt + 1))
        except Exception as exc:
            LOG.warning("LPI map load failed from %s: %s", nc_path, exc)
            return None
    if ds is None:
        LOG.warning("LPI map unreadable after retries at %s", nc_path)
        return None
    var = "lpi_pga30" if "lpi_pga30" in ds.data_vars else next(
        iter(ds.data_vars))
    return ds[var]


# Gaussian one-sided 95th-percentile multiplier (mean + z * std with
# z = 1.6449 for alpha = 0.95). The on-disk cube stores posterior
# ``mean`` and ``std`` slices only -- no quantile slices -- so the p95
# surface is reconstructed under a Gaussian-posterior approximation.
_GAUSSIAN_Z_95 = 1.6449

# Default nominal level for the panel-(b) conformal interval half-width.
_CONFORMAL_HALFWIDTH_ALPHA = 0.90


def _resolve_conformal_json(
    runs_dir: Path | None,
    primary_run: str,
    *,
    conformal_json: Path | None = None,
    cube_dir: Path | None = None,
) -> Path | None:
    """Locate ``conformal_mondrian.json`` across local + NFS layouts.

    On the render pod the conformal table is NOT under the figure-build
    ``--runs-dir`` (``/workspace/data/runs/<primary_run>/``). It lives on
    NFS at ``/mnt/nas/runs/<primary_run>/conformal_mondrian.json`` -- a
    sibling of the cube product directory, not of the local runs dir --
    so ``fig8`` previously fell back to the flat Gaussian-z multiplier and
    panel (b) showed no per-regime structure.

    Search order (first existing wins):

    1. ``conformal_json`` -- an explicit ``--conformal-json`` override
       (may point at the file directly OR at a run directory containing
       ``conformal_mondrian.json``).
    2. ``<runs_dir>/<primary_run>/conformal_mondrian.json`` -- the local
       in-repo layout used by the pytest suite + a developer's checkout.
    3. Cube-dir-adjacent NFS candidates derived from ``cube_dir``: the
       products tree (``/mnt/nas/geo-estimation/products/...``) and the
       runs tree typically share a mount root, so we probe
       ``<root>/runs/<primary_run>/`` for each ancestor ``<root>`` of the
       cube dir, plus the canonical ``/mnt/nas/runs/<primary_run>/``.

    Returns the resolved ``Path`` to the JSON, or ``None`` when no
    candidate exists (the caller then tries ``predictions.npz`` recompute
    and finally the Gaussian fallback).
    """
    candidates: list[Path] = []
    if conformal_json is not None:
        cj = Path(conformal_json)
        # Accept either the file itself or a directory holding it.
        candidates.append(cj if cj.suffix == ".json"
                          else cj / "conformal_mondrian.json")
    if runs_dir is not None:
        candidates.append(runs_dir / primary_run / "conformal_mondrian.json")
    if cube_dir is not None:
        cube_dir = Path(cube_dir)
        # Walk up the cube path; for each ancestor probe a sibling runs/
        # tree (covers /mnt/nas/geo-estimation/products/<cube> ->
        # /mnt/nas/geo-estimation/runs/<run> AND /mnt/nas/products/<cube>
        # -> /mnt/nas/runs/<run>).
        for ancestor in cube_dir.resolve().parents:
            candidates.append(
                ancestor / "runs" / primary_run / "conformal_mondrian.json")
    # Canonical pod mount last (always probed even without a cube dir).
    candidates.append(
        Path("/mnt/nas/runs") / primary_run / "conformal_mondrian.json")
    seen: set[Path] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            if cand.exists():
                LOG.info("Resolved conformal table at %s", cand)
                return cand
        except OSError:  # pragma: no cover - unreadable mount
            continue
    return None


def _resolve_predictions_npz(
    runs_dir: Path | None,
    primary_run: str,
    *,
    json_dir: Path | None = None,
    cube_dir: Path | None = None,
) -> Path | None:
    """Locate ``predictions.npz`` across the local + NFS run layouts.

    The per-regime conformal multipliers are recomputed from this file
    when no persisted ``quantiles_per_group`` table is available. On the
    render pod the standard command passes ONLY ``--cube-dir`` (no
    ``--runs-dir``), so the recompute must be able to find the npz on the
    cube-dir-adjacent NFS runs tree -- otherwise panel (b) regresses to
    the flat Gaussian-z fallback and shows no per-regime structure.

    Search order (first existing wins), mirroring
    :func:`_resolve_conformal_json`:

    1. ``<json_dir>/predictions.npz`` -- a sibling of an already-resolved
       ``conformal_mondrian.json`` (the JSON and the npz live together).
    2. ``<runs_dir>/<primary_run>/predictions.npz`` -- the local in-repo
       layout used by the pytest suite + a developer's checkout.
    3. Cube-dir-adjacent NFS candidates: for each ancestor ``<root>`` of
       the cube dir, probe ``<root>/runs/<primary_run>/predictions.npz``
       (covers ``/mnt/nas/geo-estimation/products/<cube>`` ->
       ``.../runs/<run>`` AND ``/mnt/nas/products/<cube>`` ->
       ``/mnt/nas/runs/<run>``), plus the canonical
       ``/mnt/nas/runs/<primary_run>/predictions.npz``.

    Returns the resolved ``Path`` or ``None`` when no candidate exists.
    """
    candidates: list[Path] = []
    if json_dir is not None:
        candidates.append(Path(json_dir) / "predictions.npz")
    if runs_dir is not None:
        candidates.append(runs_dir / primary_run / "predictions.npz")
    if cube_dir is not None:
        cube_dir = Path(cube_dir)
        for ancestor in cube_dir.resolve().parents:
            candidates.append(
                ancestor / "runs" / primary_run / "predictions.npz")
    candidates.append(
        Path("/mnt/nas/runs") / primary_run / "predictions.npz")
    seen: set[Path] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            if cand.exists():
                LOG.info("Resolved predictions.npz at %s", cand)
                return cand
        except OSError:  # pragma: no cover - unreadable mount
            continue
    return None


def _load_conformal_multipliers(
    runs_dir: Path | None,
    primary_run: str = "dkl_national_full_v2",
    alpha: float = _CONFORMAL_HALFWIDTH_ALPHA,
    *,
    conformal_json: Path | None = None,
    cube_dir: Path | None = None,
) -> tuple[dict[int, float], float | None]:
    """Return ``({regime_code: q_group(alpha)}, q_marginal(alpha))``.

    The per-regime Mondrian conformal multiplier ``q_group`` is applied as
    ``half_width = base_sigma * q_group[regime]`` (see
    ``national.evaluation.calibration.ConformalCalibrator.interval_mondrian``).

    Two sources, in priority order:

    1. A persisted ``conformal_mondrian.json`` carrying a
       ``quantiles_per_group`` block (added by
       ``run_mondrian_recal_national.py``), located via
       :func:`_resolve_conformal_json` across the local ``--runs-dir``
       layout, an explicit ``--conformal-json`` override, and the
       cube-dir-adjacent NFS mount (``/mnt/nas/runs/<primary_run>/...``).
       Keyed by regime code (string or int) -> {alpha-as-string: q}. The
       marginal fallback ``q`` is read from a ``quantiles_marginal`` block
       if present.
    2. Recompute from ``predictions.npz`` (a sibling of the resolved JSON,
       else ``<runs_dir>/<primary_run>/predictions.npz``) via
       ``ConformalCalibrator.fit_mondrian`` with the run's split
       (seed=42, cal_fraction=0.5) -- exactly what
       ``run_mondrian_recal_national.py`` does -- when the JSON lacks the
       multiplier table.

    Returns ``({}, None)`` if neither source is available; the caller then
    falls back to a Gaussian z multiplier so the panel still renders.
    """
    a_key = str(float(alpha))
    cj = _resolve_conformal_json(
        runs_dir, primary_run,
        conformal_json=conformal_json, cube_dir=cube_dir)
    # Directory holding the resolved JSON (the run dir on NFS or locally);
    # the predictions.npz recompute reads its sibling. Falls back to the
    # local run dir when no JSON resolved at all.
    run_dir = (cj.parent if cj is not None
               else (runs_dir / primary_run if runs_dir is not None else None))

    # ---- Source 1: persisted multiplier table in the JSON ----------------
    if cj is not None and cj.exists():
        try:
            payload = _load_conformal(cj)
        except Exception as exc:  # pragma: no cover - corrupt json
            LOG.warning("conformal json parse failed (%s): %s", cj, exc)
            payload = {}
        qpg = payload.get("quantiles_per_group")
        if qpg:
            table: dict[int, float] = {}
            for g_key, per_alpha in qpg.items():
                if a_key in per_alpha:
                    table[int(g_key)] = float(per_alpha[a_key])
                elif str(alpha) in per_alpha:
                    table[int(g_key)] = float(per_alpha[str(alpha)])
            q_marg = None
            qm = payload.get("quantiles_marginal", {})
            if a_key in qm:
                q_marg = float(qm[a_key])
            if table:
                LOG.info("Loaded %d per-regime conformal multipliers from %s "
                         "at alpha=%s", len(table), cj, a_key)
                return table, q_marg

    # ---- Source 2: recompute from predictions.npz ------------------------
    # Resolve the npz across the local + NFS run layouts. Critically this
    # walks the cube-dir-adjacent ``runs/<primary_run>/`` tree, so the
    # standard render command (``--cube-dir`` only, no ``--runs-dir``)
    # still finds ``predictions.npz`` on NFS and recomputes the 8-regime
    # multipliers rather than regressing to the flat Gaussian fallback.
    npz_path = _resolve_predictions_npz(
        runs_dir, primary_run, json_dir=run_dir, cube_dir=cube_dir)
    if npz_path is None:
        LOG.info("No conformal multiplier table available "
                 "(no JSON table, no predictions.npz on local/NFS runs tree "
                 "for run=%s)", primary_run)
        return {}, None
    try:
        from national.evaluation.calibration import ConformalCalibrator
        with np.load(npz_path) as d:
            y_true = np.asarray(d["y_true"], dtype=np.float64)
            pred_mean = np.asarray(d["pred_mean"], dtype=np.float64)
            pred_std = np.maximum(
                np.asarray(d["pred_std"], dtype=np.float64), 1e-6)
            regime = np.asarray(d["regime"])
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(y_true))
        n_cal = int(0.5 * len(y_true))
        cal_idx = idx[:n_cal]
        cal = ConformalCalibrator().fit_mondrian(
            y_true[cal_idx], pred_mean[cal_idx], pred_std[cal_idx],
            groups=regime[cal_idx], alphas=(alpha,),
        )
        table = {
            int(g): float(qd[float(alpha)])
            for g, qd in (cal.quantiles_per_group or {}).items()
            if float(alpha) in qd
        }
        q_marg = None
        if cal.quantiles is not None and float(alpha) in cal.quantiles:
            q_marg = float(cal.quantiles[float(alpha)])
        LOG.info("Recomputed %d per-regime conformal multipliers from %s "
                 "at alpha=%s (marginal q=%s)", len(table), npz_path,
                 alpha, q_marg)
        return table, q_marg
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("conformal multiplier recompute failed (%s): %s",
                    npz_path, exc)
        return {}, None


def _load_regime_grid(cube_dir: Path):
    """Load the per-cell regime-code grid from ``maps/regime_code.nc``.

    ``predict_national_cube.py`` may optionally emit ``regime_code.nc``
    alongside the other maps (same lat/lon axes as the cube). Returns the
    integer regime-code ``xr.DataArray`` on success, else ``None`` so the
    caller can degrade gracefully.
    """
    import xarray as xr  # type: ignore

    nc_path = cube_dir / "maps" / "regime_code.nc"
    if not nc_path.exists():
        LOG.info("regime grid not found at %s", nc_path)
        return None
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as exc:
        LOG.warning("regime grid load failed from %s: %s", nc_path, exc)
        return None
    var = "regime_code" if "regime_code" in ds.data_vars else next(
        iter(ds.data_vars))
    return ds[var]


def _import_build_regime_grid():
    """Return ``predict_national_cube.build_regime_grid`` (or ``None``).

    Mirrors :func:`_import_cube_snap_helpers`' dual-path resolution so the
    registry-backed regime sampler is available both under the production
    ``python -m`` entrypoint and the spec-loaded pytest harness. Returns
    ``None`` (rather than raising) so the caller degrades gracefully when
    the prediction stack is not importable.
    """
    try:
        from scripts.predict_national_cube import build_regime_grid  # type: ignore
        return build_regime_grid
    except Exception:
        try:
            import importlib.util as _ilu
            src = Path(__file__).resolve().parent / "predict_national_cube.py"
            spec = _ilu.spec_from_file_location(
                "predict_national_cube_under_test", src)
            if spec is None or spec.loader is None:
                return None
            mod = _ilu.module_from_spec(spec)
            sys.modules.setdefault("predict_national_cube_under_test", mod)
            spec.loader.exec_module(mod)
            return mod.build_regime_grid
        except Exception:  # pragma: no cover - defensive
            return None


def _regime_grid_from_registry(
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
    boring_parquet: Path | None,
    *,
    knn_k: int = 4,
    knn_max_distance_km: float = 20.0,
) -> np.ndarray | None:
    """Sample the AIST 8-regime code per display cell via the covariate
    registry, when no precomputed ``maps/regime_code.nc`` is available.

    Reuses ``predict_national_cube.build_regime_grid`` -- the SAME
    ``boring:knn`` ``CovariateRegistry`` loader the prediction engine uses
    -- so the figure's per-cell regime assignment is bit-for-bit the same
    plurality-vote-over-k-nearest-borings lookup the model saw, rather than
    a figure-only re-derivation. ``lat_axis`` / ``lon_axis`` are the
    DISPLAY (block-mean) axes, so the returned grid already aligns
    cell-for-cell with the downsampled ``std`` field that panel (b)
    multiplies -- no further subsampling needed.

    Returns the ``int16`` regime grid on success, else ``None`` (missing
    parquet, registry import failure, or KNN build failure) so panel (b)
    degrades to the marginal/Gaussian fallback multiplier.
    """
    if boring_parquet is None:
        LOG.info("no boring parquet for registry regime sampling; "
                 "panel (b) will use the fallback multiplier")
        return None
    boring_parquet = Path(boring_parquet)
    if not boring_parquet.exists():
        LOG.info("boring parquet %s missing; skipping registry regime "
                 "sampling", boring_parquet)
        return None
    build_regime_grid = _import_build_regime_grid()
    if build_regime_grid is None:
        LOG.warning("could not import build_regime_grid; skipping registry "
                    "regime sampling")
        return None
    lat_f = lat_axis[np.isfinite(lat_axis)] if lat_axis is not None else None
    lon_f = lon_axis[np.isfinite(lon_axis)] if lon_axis is not None else None
    if lat_f is None or lon_f is None or lat_f.size == 0 or lon_f.size == 0:
        return None
    try:
        LOG.info("Sampling AIST regime per display cell via covariate "
                 "registry (knn boring:knn) from %s onto a %dx%d display "
                 "grid", boring_parquet, lat_f.size, lon_f.size)
        return build_regime_grid(
            boring_parquet, lat_f, lon_f,
            knn_k=knn_k, knn_max_distance_km=knn_max_distance_km)
    except Exception as exc:  # pragma: no cover - defensive (heavy IO path)
        LOG.warning("registry regime sampling failed (%s); panel (b) will "
                    "use the fallback multiplier", exc)
        return None


def _build_conformal_halfwidth(
    std_5m: np.ndarray | None,
    regime_grid: np.ndarray | None,
    q_table: dict[int, float],
    q_marginal: float | None,
    alpha: float = _CONFORMAL_HALFWIDTH_ALPHA,
) -> np.ndarray | None:
    """Compute the Mondrian conformal interval half-width at 5 m.

    ``half_width(lat, lon) = std_5m * q_group[regime_grid(lat, lon)]``,
    with the marginal ``q`` used for regimes absent from the table and
    UNKNOWN (code 7). When no conformal table is available at all, falls
    back to the Gaussian one-sided ``z`` so the panel still renders a
    meaningful (if uncalibrated) uncertainty surface rather than the
    degenerate near-flat ``std`` field.

    NaN cells in ``std_5m`` (no-data) propagate to NaN in the half-width.
    """
    if std_5m is None:
        return None
    # Fallback multiplier for cells whose regime has no calibrated q.
    fallback = q_marginal if q_marginal is not None else _GAUSSIAN_Z_95
    if regime_grid is None or not q_table:
        # No spatially-varying regime info: scale by the single fallback.
        return std_5m * float(fallback)
    reg = np.asarray(regime_grid)
    mult = np.full(reg.shape, float(fallback), dtype=np.float64)
    for code, q in q_table.items():
        mult[reg == code] = float(q)
    # Regime grid may be smaller/larger than std grid if axes differ; only
    # multiply where shapes align (they share the cube lat/lon axes).
    if mult.shape != std_5m.shape:
        LOG.warning("regime grid shape %s != std shape %s; using fallback "
                    "scalar multiplier", mult.shape, std_5m.shape)
        return std_5m * float(fallback)
    return std_5m * mult


def _lookup_site_regime(boring_parquet: Path | None, lat: float,
                        lon: float) -> int | None:
    """Sample the AIST regime code at a single ``(lat, lon)`` site via the
    covariate registry KNN loader. Returns ``None`` (not raises) when the
    parquet is absent or the registry is unimportable, so the caller widens
    the band with the marginal multiplier instead.
    """
    grid = _regime_grid_from_registry(
        np.array([lat], dtype=np.float64),
        np.array([lon], dtype=np.float64),
        boring_parquet,
        knn_max_distance_km=200.0)
    if grid is None or grid.size == 0:
        return None
    return int(np.asarray(grid).ravel()[0])


def fig8_depth_profiles_and_lpi(
        out: Path, cube_dir: Path, main_only: bool = False,
        supp_out: Path | None = None,
        runs_dir: Path | None = None,
        primary_run: str = "dkl_national_full_v2",
        conformal_alpha: float = _CONFORMAL_HALFWIDTH_ALPHA,
        conformal_json: Path | None = None,
        boring_parquet: Path | None = None,
        ) -> None:
    """2-up composite: three posterior-mean depth profiles (top) over the
    clean LPI national hazard map (bottom).

    Replaces the former 2x2 uncertainty/p95 MAP panels (``fig8_uncertainty``
    panels a/b/c), every one of which was a FILLED 2-D (lat,lon) surface
    that carried the source-level random-Fourier positional-encoding
    crosshatch. The surviving honest products are:

    - **Top -- FIG A**: a 3-column row of 1-D depth profiles, one per
      reference site (Tokyo Bay soft alluvium / Japan Alps stiff bedrock /
      Osaka plain), posterior-mean SPT-$N$ vs depth (depth axis inverted,
      down) with a conformal/std uncertainty band. Each profile samples a
      single (lat,lon) column, so it never exposes the lateral 2-D mesh
      (probe 2026-06-16: profiles clean, 10 finite levels each). Each site
      column is read via a SPARSE per-site tile load (a tiny bbox around the
      site touches only its containing tile via
      :func:`_load_tiles_covering`), not the full 825-tile cube -- so the
      deep chunk reads that trip the aries-NFS stale-handle are avoided.
    - **Bottom -- FIG C**: the existing clean LPI map
      (``<cube_dir>/maps/lpi_pga30.nc``), the sparse groundwater-masked
      product that is free of the crosshatch (probe: range [0, 122.4],
      0.34% NaN).

    Uncertainty band honesty (probe ``per_site_conformal_nondegenerate =
    false``): the cube posterior $\\sigma$ is depth-flat (~7.58 at every
    level, CV ~1e-4), so the band is rendered as an essentially
    CONSTANT-width ribbon and the caption states it is regime-driven, NOT
    depth-resolved. Its width is $\\sigma \\cdot q_{\\mathrm{group}}
    [\\mathrm{site\\ regime}]$ with the per-AIST-regime Mondrian conformal
    multiplier (the only real spatial signal: ~1.4x dynamic range ACROSS
    regimes); the marginal $q$ (else Gaussian $z$) is used when the
    per-regime table or the site regime is unavailable. Pass ``runs_dir``
    pointing at the runs tree (e.g. ``/mnt/nas/runs``) so the per-regime
    ``q_group`` multipliers load instead of the flat Gaussian fallback.

    ``main_only`` / ``supp_out`` are accepted for call-signature
    compatibility with the previous renderer (the LPI panel always stays in
    the main composite here, so ``supp_out`` is only written a placeholder
    when ``main_only`` is set, keeping downstream callers that expect the
    file present happy).

    ``cube_dir`` may be a local path or an NFS mount. A placeholder PDF is
    emitted when no tile zarrs are found under ``cube_dir/cube``.
    """
    tile_dirs = _resolve_tile_dirs(cube_dir)
    if not tile_dirs:
        _placeholder(
            out,
            (f"Fig 8 -- DEFERRED\n\n"
             f"Cube tiles not found.\n"
             f"Expected at: {cube_dir}/cube/tile_*.zarr\n"
             f"and maps under: {cube_dir}/maps/lpi_pga30.nc\n\n"
             "Point --cube-dir at a directory containing cube/ + maps/, "
             "either a local sync target or an NFS mount."),
            ("**Fig. 8** Depth profiles + LPI hazard map "
             "(deferred; no tile zarrs found)."),
        )
        if main_only and supp_out is not None:
            _placeholder(
                supp_out,
                "Fig 8b (Supplementary) -- DEFERRED",
                ("**Supplementary Fig. 8b** LPI scenario "
                 "(deferred; no tile zarrs found)."),
            )
        return

    # Per-regime conformal multiplier table (q_group) + marginal fallback.
    # Resolved across the local --runs-dir, an explicit --conformal-json
    # override, AND the cube-dir-adjacent NFS runs tree (so the standard
    # render command finds the table on NFS rather than falling back to the
    # flat Gaussian z). The band width is regime-driven; this is the only
    # real spatial signal in the (depth-flat) sigma channel.
    q_table, q_marginal = _load_conformal_multipliers(
        runs_dir, primary_run=primary_run, alpha=conformal_alpha,
        conformal_json=conformal_json, cube_dir=cube_dir)
    fallback_q = q_marginal if q_marginal is not None else _GAUSSIAN_Z_95

    # ---- Per-site depth columns via SPARSE per-site tile loads ----------
    # A depth profile only needs the cube column at one (lat,lon); that
    # column lives in the single tile containing the site. We open a tiny
    # bbox around each site so only its containing tile (~1 zarr) is read
    # -- not the full 825-tile cube whose deep chunk reads trip the
    # aries-NFS [Errno 116] stale-handle. Each site's load is wrapped so a
    # missing/unreadable site tile is skipped with a warning, never fatal.
    _SITE_BBOX = 0.10  # deg; small box guarantees we grab the site's tile
    site_profiles: list[dict[str, Any]] = []
    for label, stype, slat, slon in _PROFILE_SITES:
        try:
            site_cube = _load_tiles_covering(
                cube_dir,
                (slat - _SITE_BBOX, slat + _SITE_BBOX),
                (slon - _SITE_BBOX, slon + _SITE_BBOX))
            if "statistic" in site_cube.dims:
                stat_codes = [
                    str(s) for s in
                    np.asarray(site_cube.coords["statistic"].values).ravel()]
                mean_da = site_cube.sel(statistic="mean")
                std_da = (site_cube.sel(statistic="std")
                          if "std" in stat_codes else None)
            else:
                mean_da, std_da = site_cube, None
            depths, mean_col, std_col, ulat, ulon = _extract_depth_column(
                mean_da, std_da, slat, slon)
        except Exception as exc:
            LOG.warning("profile load/extraction failed at %s (%s, %s): %s",
                        label, slat, slon, exc)
            continue
        # Band multiplier: per-regime q_group at the site's regime (the only
        # real spatial signal). Degrade to marginal q / Gaussian z.
        site_regime = _lookup_site_regime(boring_parquet, slat, slon)
        if site_regime is not None and site_regime in q_table:
            q = float(q_table[site_regime])
        else:
            q = float(fallback_q)
        site_profiles.append({
            "label": label, "type": stype,
            "lat": ulat, "lon": ulon,
            "depths": depths, "mean": mean_col, "std": std_col,
            "q": q, "regime": site_regime,
        })

    # ---- LPI national hazard map (clean, sparse groundwater-masked) -----
    lpi_da = _load_lpi_map(cube_dir)
    lpi_arr: np.ndarray | None = None
    lpi_extent: tuple[float, float, float, float] | None = None
    if lpi_da is not None:
        lpi_arr = np.asarray(lpi_da.values, dtype=np.float64)
        lpi_extent = _cube_extent(lpi_da)

    # ---- Compose: 3 profiles (top row) + LPI map (bottom, wide) ---------
    pct = int(round(conformal_alpha * 100))
    fig = plt.figure(figsize=(11.0, 9.0), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1.0, 1.5],
        hspace=0.12, wspace=0.18)
    prof_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    panel_letters = "abc"
    for i, ax in enumerate(prof_axes):
        if i >= len(site_profiles):
            ax.text(0.5, 0.5, "site unavailable", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        p = site_profiles[i]
        depths = p["depths"]
        mean_col = p["mean"]
        std_col = p["std"]
        # Display floor at 0: GP can return small negative N at depth.
        mean_disp = np.where(np.isfinite(mean_col),
                             np.maximum(mean_col, 0.0), np.nan)
        # Conformal/std band: mean +/- std * q_group[regime]. The sigma
        # channel is depth-flat, so this is an essentially constant-width
        # ribbon (regime-driven, NOT depth-resolved -- stated in caption).
        if np.isfinite(std_col).any():
            half = std_col * p["q"]
            lo = np.maximum(mean_col - half, 0.0)
            hi = mean_col + half
            ax.fill_betweenx(depths, lo, hi, color="#1f77b4", alpha=0.20,
                             linewidth=0,
                             label=f"conformal {pct}% band")
        ax.plot(mean_disp, depths, color="#1f77b4", linewidth=1.8,
                marker="o", markersize=3.0, label="posterior mean")
        ax.invert_yaxis()  # depth increases downward
        ax.set_xlim(left=0.0)
        ax.set_xlabel("SPT-$N$  [blow count]")
        if i == 0:
            ax.set_ylabel("Depth  [m]")
        ax.set_title(f"({panel_letters[i]}) {p['label']}\n{p['type']}",
                     fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        ax.annotate(
            f"{p['lat']:.2f}$^\\circ$N, {p['lon']:.2f}$^\\circ$E",
            xy=(0.96, 0.04), xycoords="axes fraction", ha="right",
            va="bottom", fontsize=6.5, color="#555555")
        if i == 0:
            # Upper right, not lower right: the site coordinate annotation
            # above is anchored at (0.96, 0.04), so a lower-right legend was
            # printed straight on top of it and both became unreadable. The
            # profiles run down the left of each panel, so the upper right is
            # the free corner.
            ax.legend(loc="upper right", fontsize=6.5, framealpha=0.85)

    # Bottom: clean LPI national hazard map. Japan is taller than it is wide,
    # so a map_ax spanning all three columns can never fill the row -- the
    # fixed aspect shrinks the axes inside an 11 in cell and the leftover
    # width becomes whitespace on one side or a stranded colorbar on the
    # other. Give the map a snug, centred cell so the colorbar sits beside it.
    # Middle ratio tuned so the cell is just wide enough for map + colorbar:
    # any wider and the height-limited map leaves intra-cell slack, which the
    # colorbar anchoring turns into a visible off-centre shift.
    map_gs = gs[1, :].subgridspec(1, 3, width_ratios=[1.0, 1.7, 1.0])
    map_ax = fig.add_subplot(map_gs[0, 1])
    if lpi_arr is not None:
        vmin, vmax = _quantile_vlim(lpi_arr, 0.02, 0.98)
        vmin = max(vmin, 0.0)
        cmap = _transparent_cmap("plasma")
        ma = np.ma.masked_invalid(lpi_arr)
        if lpi_extent is not None:
            im = map_ax.imshow(
                ma, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap,
                # aspect="equal" treats a degree of longitude and a degree of
                # latitude as the same length, stretching Japan ~22 % wide at
                # 35 N. Use the true mid-latitude ratio instead.
                aspect=_geographic_aspect((lpi_extent[2], lpi_extent[3])),
                extent=[lpi_extent[0], lpi_extent[1],
                        lpi_extent[2], lpi_extent[3]])
            _draw_japan_basemap(
                map_ax, lon_extent=(lpi_extent[0], lpi_extent[1]),
                lat_extent=(lpi_extent[2], lpi_extent[3]))
            map_ax.set_xlabel("Longitude (deg E)")
            map_ax.set_ylabel("Latitude (deg N)")
        else:
            im = map_ax.imshow(ma, origin="lower", vmin=vmin, vmax=vmax,
                               cmap=cmap, aspect="auto")
            map_ax.set_xticks([]); map_ax.set_yticks([])
        cbar = fig.colorbar(im, ax=map_ax, fraction=0.025, pad=0.02)
        cbar.set_label("LPI  [unitless]", fontsize=9)
        # Leave the parent anchored towards the colorbar so the bar stays
        # beside the map: the cell above is already snug and centred, so the
        # map lands centred in the figure without stranding the colorbar.
        map_ax.set_title("(d) Liquefaction-potential index (LPI), "
                         "PGA = 0.30 g, $M_w$ = 7.5", fontsize=9)
    else:
        map_ax.text(0.5, 0.5, "LPI map unavailable\n"
                    f"(expected {cube_dir / 'maps' / 'lpi_pga30.nc'})",
                    ha="center", va="center", transform=map_ax.transAxes,
                    fontsize=9)
        map_ax.set_xticks([]); map_ax.set_yticks([])

    caption = (
        "**Fig. 8** Site-scale depth profiles and national liquefaction "
        "hazard. (a-c) Posterior-mean SPT-$N$ versus depth (depth "
        "increasing downward) at three reference sites -- Tokyo Bay soft "
        "alluvium, Japan Alps stiff bedrock, and the Osaka plain -- each "
        "sampled as a single (lat, lon) cube column, so the profiles never "
        "carry the lateral 2-D random-Fourier positional-encoding texture "
        "\\citep{tancik2020fourier} that contaminated the earlier filled "
        f"map panels. The shaded ribbon is the conformal {pct}% interval, "
        "$\\mu \\pm \\sigma \\cdot q_{\\mathrm{group}}[\\mathrm{regime}]$, "
        "where $q_{\\mathrm{group}}$ is the per-AIST-regime Mondrian "
        "conformal multiplier. The posterior $\\sigma$ channel is "
        "depth-flat (it saturates to ~7.58 at every level), so this ribbon "
        "is essentially CONSTANT-width within a site; its width is "
        "regime-driven (the per-regime multiplier gives ~1.4x dynamic range "
        "ACROSS regimes, the only spatial signal in the calibrated "
        "uncertainty) and does NOT track depth. Colour/value floor is 0 "
        "because the Gaussian-process posterior can return small negative "
        "$N$ at depth; the underlying cube is unmodified. (d) "
        "Liquefaction-potential index under a uniform PGA = 0.30 g, "
        "$M_w$ = 7.5 scenario, read from the cube's groundwater-conditioned "
        "``maps/lpi_pga30.nc``; high LPI concentrates in the saturated "
        "alluvial lowlands (Tokyo Bay, Nobi, Osaka, Niigata, Ishikari) and "
        "is near-zero over bedrock terrain. This is the sparse "
        "groundwater-masked product, free of the sub-grid crosshatch, so it "
        "is shown unfiltered."
    )
    _save_pdf(fig, out, caption)

    # Back-compat: when main_only is requested, still ensure supp_out
    # exists (downstream callers expect the file present). The LPI map now
    # lives in the main composite, so the supplementary is a short note.
    if main_only and supp_out is not None:
        _placeholder(
            supp_out,
            ("Fig 8b (Supplementary) -- LPI now shown in the main Fig. 8 "
             "composite (bottom panel). No separate supplementary LPI map "
             "is produced under the restructured layout."),
            "**Supplementary Fig. 8b** LPI now in main Fig. 8 (bottom).",
        )


# ---------------------------------------------------------------------------
# Fig 9 -- TTA delta-RMSE per region/strategy (null result with one exception)
# ---------------------------------------------------------------------------

# Hard-coded TTA records: per-region x per-strategy before/after metrics from
# the TTA sweep described in
# docs/paper/paper_2_national/sections/07_cross_region_transfer.tex
# (sub-section ``tta`` introduced 2026-06-01). Each row is one of 24 cells
# (8 LRO regions x 3 strategies). ``rmse_before`` matches the source-only
# DKL+SVGP baseline of tab:lro_per_region. ``mean_std_after`` collapses
# from ~7.4 to ~0.011 under TENT/self-training (variance pathology).
TTA_RECORDS: list[dict[str, Any]] = [
    {"region": "chubu", "strategy": "bn_stats",
     "rmse_before": 13.8897203448472, "rmse_after": 14.341520906259957,
     "mae_before": 9.769990715723853, "mae_after": 9.633783560386933,
     "mean_std_before": 7.288400629512824, "mean_std_after": 7.288602080438784,
     "n_below_lpi_threshold_before": 384708, "n_below_lpi_threshold_after": 402726,
     "wall_clock_s": 86.49813961982727, "n_adapted_rows": 457691},
    {"region": "chubu", "strategy": "tent",
     "rmse_before": 13.8897203448472, "rmse_after": 16.474599864977616,
     "mae_before": 9.769990715723853, "mae_after": 12.64108729979096,
     "mean_std_before": 7.288400629512824, "mean_std_after": 0.0112613917044105,
     "n_below_lpi_threshold_before": 384708, "n_below_lpi_threshold_after": 369155,
     "wall_clock_s": 143.31929302215576, "n_adapted_rows": 457691},
    {"region": "chubu", "strategy": "self_training",
     "rmse_before": 13.8897203448472, "rmse_after": 36.38769233648159,
     "mae_before": 9.769990715723853, "mae_after": 32.04419485091109,
     "mean_std_before": 7.288400629512824, "mean_std_after": 0.0114073494902817,
     "n_below_lpi_threshold_before": 384708, "n_below_lpi_threshold_after": 149077,
     "wall_clock_s": 68.25460505485535, "n_adapted_rows": 457691},
    {"region": "chugoku", "strategy": "bn_stats",
     "rmse_before": 13.964465867587434, "rmse_after": 15.231718072186471,
     "mae_before": 9.968026079612198, "mae_after": 10.789829826135092,
     "mean_std_before": 7.281441255771879, "mean_std_after": 7.2829981256796525,
     "n_below_lpi_threshold_before": 346116, "n_below_lpi_threshold_after": 362500,
     "wall_clock_s": 165.68508076667786, "n_adapted_rows": 417227},
    {"region": "chugoku", "strategy": "tent",
     "rmse_before": 13.964465867587434, "rmse_after": 34.24294811036865,
     "mae_before": 9.968026079612198, "mae_after": 29.64992221805091,
     "mean_std_before": 7.281441255771879, "mean_std_after": 0.01124623657921544,
     "n_below_lpi_threshold_before": 346116, "n_below_lpi_threshold_after": 229275,
     "wall_clock_s": 64.26360273361206, "n_adapted_rows": 417227},
    {"region": "chugoku", "strategy": "self_training",
     "rmse_before": 13.964465867587434, "rmse_after": 64.6246575287374,
     "mae_before": 9.968026079612198, "mae_after": 59.02105986075149,
     "mean_std_before": 7.281441255771879, "mean_std_after": 0.011382882459053226,
     "n_below_lpi_threshold_before": 346116, "n_below_lpi_threshold_after": 333402,
     "wall_clock_s": 63.5197811126709, "n_adapted_rows": 417227},
    {"region": "hokkaido", "strategy": "bn_stats",
     "rmse_before": 15.16265845333872, "rmse_after": 15.4964733830225,
     "mae_before": 11.614636637355034, "mae_after": 11.59919687847076,
     "mean_std_before": 7.327343795262253, "mean_std_after": 7.3270191138876815,
     "n_below_lpi_threshold_before": 162500, "n_below_lpi_threshold_after": 177582,
     "wall_clock_s": 86.31645154953003, "n_adapted_rows": 216692},
    {"region": "hokkaido", "strategy": "tent",
     "rmse_before": 15.16265845333872, "rmse_after": 31.777470340958462,
     "mae_before": 11.614636637355034, "mae_after": 26.75426634190946,
     "mean_std_before": 7.327343795262253, "mean_std_after": 0.0113614482642053,
     "n_below_lpi_threshold_before": 162500, "n_below_lpi_threshold_after": 169946,
     "wall_clock_s": 38.68611407279968, "n_adapted_rows": 216692},
    {"region": "hokkaido", "strategy": "self_training",
     "rmse_before": 15.16265845333872, "rmse_after": 25.90711013733775,
     "mae_before": 11.614636637355034, "mae_after": 21.25238428781747,
     "mean_std_before": 7.327343795262253, "mean_std_after": 0.01150842088950994,
     "n_below_lpi_threshold_before": 162500, "n_below_lpi_threshold_after": 174434,
     "wall_clock_s": 37.907084465026855, "n_adapted_rows": 216692},
    {"region": "kansai", "strategy": "bn_stats",
     "rmse_before": 13.301542647548722, "rmse_after": 14.398061352515876,
     "mae_before": 9.00709641342626, "mae_after": 9.746460533381082,
     "mean_std_before": 7.672441815103022, "mean_std_after": 7.672854963203194,
     "n_below_lpi_threshold_before": 413019, "n_below_lpi_threshold_after": 440585,
     "wall_clock_s": 196.06695246696472, "n_adapted_rows": 493613},
    {"region": "kansai", "strategy": "tent",
     "rmse_before": 13.301542647548722, "rmse_after": 15.65039073092592,
     "mae_before": 9.00709641342626, "mae_after": 13.091404735524119,
     "mean_std_before": 7.672441815103022, "mean_std_after": 0.011380862896284797,
     "n_below_lpi_threshold_before": 413019, "n_below_lpi_threshold_after": 294309,
     "wall_clock_s": 73.7350594997406, "n_adapted_rows": 493613},
    {"region": "kansai", "strategy": "self_training",
     "rmse_before": 13.301542647548722, "rmse_after": 48.51619308875934,
     "mae_before": 9.00709641342626, "mae_after": 41.149868622870784,
     "mean_std_before": 7.672441815103022, "mean_std_after": 0.011402966004017249,
     "n_below_lpi_threshold_before": 413019, "n_below_lpi_threshold_after": 209635,
     "wall_clock_s": 73.18015766143799, "n_adapted_rows": 493613},
    {"region": "kanto", "strategy": "bn_stats",
     "rmse_before": 13.159674432466444, "rmse_after": 12.67820170020209,
     "mae_before": 9.336102378116962, "mae_after": 8.661827033933289,
     "mean_std_before": 7.791442466681781, "mean_std_after": 7.789988317921997,
     "n_below_lpi_threshold_before": 385504, "n_below_lpi_threshold_after": 395178,
     "wall_clock_s": 94.71063780784607, "n_adapted_rows": 495725},
    {"region": "kanto", "strategy": "tent",
     "rmse_before": 13.159674432466444, "rmse_after": 28.254217594805322,
     "mae_before": 9.336102378116962, "mae_after": 24.480918516811418,
     "mean_std_before": 7.791442466681781, "mean_std_after": 0.01147673592808174,
     "n_below_lpi_threshold_before": 385504, "n_below_lpi_threshold_after": 214560,
     "wall_clock_s": 153.5374698638916, "n_adapted_rows": 495725},
    {"region": "kanto", "strategy": "self_training",
     "rmse_before": 13.159674432466444, "rmse_after": 37.842427429042495,
     "mae_before": 9.336102378116962, "mae_after": 33.28092169983279,
     "mean_std_before": 7.791442466681781, "mean_std_after": 0.011473752715159841,
     "n_below_lpi_threshold_before": 385504, "n_below_lpi_threshold_after": 173185,
     "wall_clock_s": 73.38342523574829, "n_adapted_rows": 495725},
    {"region": "kyushu_okinawa", "strategy": "bn_stats",
     "rmse_before": 18.32602284309858, "rmse_after": 13.106258493181635,
     "mae_before": 12.13558811584711, "mae_after": 8.435557038345996,
     "mean_std_before": 7.350690090104146, "mean_std_after": 7.311454926074201,
     "n_below_lpi_threshold_before": 335652, "n_below_lpi_threshold_after": 375113,
     "wall_clock_s": 82.83241510391235, "n_adapted_rows": 431229},
    {"region": "kyushu_okinawa", "strategy": "tent",
     "rmse_before": 18.32602284309858, "rmse_after": 12.179583562939932,
     "mae_before": 12.13558811584711, "mae_after": 7.9490347867990305,
     "mean_std_before": 7.350690090104146, "mean_std_after": 0.02192685507514905,
     "n_below_lpi_threshold_before": 335652, "n_below_lpi_threshold_after": 418405,
     "wall_clock_s": 136.55995106697083, "n_adapted_rows": 431229},
    {"region": "kyushu_okinawa", "strategy": "self_training",
     "rmse_before": 18.32602284309858, "rmse_after": 46.164517873521326,
     "mae_before": 12.13558811584711, "mae_after": 40.142234219241296,
     "mean_std_before": 7.350690090104146, "mean_std_after": 0.021387576169544353,
     "n_below_lpi_threshold_before": 335652, "n_below_lpi_threshold_after": 381019,
     "wall_clock_s": 65.35194492340088, "n_adapted_rows": 431229},
    {"region": "shikoku", "strategy": "bn_stats",
     "rmse_before": 14.199232669940747, "rmse_after": 15.639115373207167,
     "mae_before": 10.521390723963265, "mae_after": 11.953112652297383,
     "mean_std_before": 7.404526252312944, "mean_std_after": 7.40799132549815,
     "n_below_lpi_threshold_before": 196634, "n_below_lpi_threshold_after": 197172,
     "wall_clock_s": 96.5782241821289, "n_adapted_rows": 242488},
    {"region": "shikoku", "strategy": "tent",
     "rmse_before": 14.199232669940747, "rmse_after": 20.87680708623615,
     "mae_before": 10.521390723963265, "mae_after": 17.937045705912904,
     "mean_std_before": 7.404526252312944, "mean_std_after": 0.011315995268588216,
     "n_below_lpi_threshold_before": 196634, "n_below_lpi_threshold_after": 89585,
     "wall_clock_s": 41.773096561431885, "n_adapted_rows": 242488},
    {"region": "shikoku", "strategy": "self_training",
     "rmse_before": 14.199232669940747, "rmse_after": 24.163048156099475,
     "mae_before": 10.521390723963265, "mae_after": 19.817079810811492,
     "mean_std_before": 7.404526252312944, "mean_std_after": 0.011568500249297899,
     "n_below_lpi_threshold_before": 196634, "n_below_lpi_threshold_after": 194526,
     "wall_clock_s": 41.368542194366455, "n_adapted_rows": 242488},
    {"region": "tohoku", "strategy": "bn_stats",
     "rmse_before": 14.055708895549195, "rmse_after": 14.402904799439257,
     "mae_before": 10.753462244014587, "mae_after": 11.213946976776656,
     "mean_std_before": 7.647896082125552, "mean_std_after": 7.645103444923598,
     "n_below_lpi_threshold_before": 204464, "n_below_lpi_threshold_after": 197825,
     "wall_clock_s": 114.47129368782043, "n_adapted_rows": 287496},
    {"region": "tohoku", "strategy": "tent",
     "rmse_before": 14.055708895549195, "rmse_after": 50.98751347270584,
     "mae_before": 10.753462244014587, "mae_after": 43.54019384773409,
     "mean_std_before": 7.647896082125552, "mean_std_after": 0.01146113477367431,
     "n_below_lpi_threshold_before": 204464, "n_below_lpi_threshold_after": 83440,
     "wall_clock_s": 47.59789276123047, "n_adapted_rows": 287496},
    {"region": "tohoku", "strategy": "self_training",
     "rmse_before": 14.055708895549195, "rmse_after": 26.877421566175393,
     "mae_before": 10.753462244014587, "mae_after": 22.142141598328546,
     "mean_std_before": 7.647896082125552, "mean_std_after": 0.01159660777187906,
     "n_below_lpi_threshold_before": 204464, "n_below_lpi_threshold_after": 146710,
     "wall_clock_s": 47.10991191864014, "n_adapted_rows": 287496},
]

# Fig 9 region order matches tab:lro_per_region (alphabetical) so the
# per-region comparison aligns one-to-one between the table and the figure.
FIG9_REGION_ORDER: tuple[str, ...] = (
    "chubu", "chugoku", "hokkaido", "kansai",
    "kanto", "kyushu_okinawa", "shikoku", "tohoku",
)

FIG9_STRATEGIES: tuple[str, ...] = ("bn_stats", "tent", "self_training")
FIG9_STRATEGY_LABELS: dict[str, str] = {
    "bn_stats": "BN-stats",
    "tent": "TENT",
    "self_training": "Self-train",
}
FIG9_STRATEGY_COLORS: dict[str, str] = {
    "bn_stats": "#4C72B0",       # steel blue
    "tent": "#DD8452",           # orange
    "self_training": "#C44E52",  # crimson
}


def fig9_tta_delta_rmse(out: Path) -> None:
    """Grouped bar chart of per-region delta-RMSE for three TTA strategies
    (BN-stats, TENT, self-training) relative to the source-only DKL+SVGP
    baseline of tab:lro_per_region, with an inset showing the
    predictive-sigma collapse pathology for TENT / self-training.

    Records are hard-coded above (``TTA_RECORDS``) so the figure is fully
    reproducible from this script alone; the underlying numbers come from
    the 8 LRO held-out folds (n_adapted_rows = 216,692 -- 495,725 per
    region, per strategy).
    """
    # Index records by (region, strategy) for O(1) lookup.
    by_key: dict[tuple[str, str], dict[str, Any]] = {
        (r["region"], r["strategy"]): r for r in TTA_RECORDS
    }

    # Build the delta-RMSE matrix [n_regions x n_strategies].
    n_reg = len(FIG9_REGION_ORDER)
    n_strat = len(FIG9_STRATEGIES)
    deltas = np.full((n_reg, n_strat), np.nan)
    for ri, region in enumerate(FIG9_REGION_ORDER):
        for si, strat in enumerate(FIG9_STRATEGIES):
            row = by_key.get((region, strat))
            if row is None:
                continue
            deltas[ri, si] = row["rmse_after"] - row["rmse_before"]

    fig, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)

    x = np.arange(n_reg)
    width = 0.27
    for si, strat in enumerate(FIG9_STRATEGIES):
        offsets = (si - (n_strat - 1) / 2.0) * width
        xi = x + offsets
        y = deltas[:, si]
        ax.bar(
            xi, y, width=width,
            color=FIG9_STRATEGY_COLORS[strat],
            edgecolor="black", linewidth=0.4,
            label=FIG9_STRATEGY_LABELS[strat],
        )

    # Reference: zero line.
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, zorder=0)

    # Annotate (i) every below-zero bar with its delta, and (ii) catastrophic
    # self-training bars (delta > 30) with their values so the truncation is
    # explicit.
    for ri, region in enumerate(FIG9_REGION_ORDER):
        for si, strat in enumerate(FIG9_STRATEGIES):
            d = deltas[ri, si]
            if np.isnan(d):
                continue
            offsets = (si - (n_strat - 1) / 2.0) * width
            xi = x[ri] + offsets
            if d < 0:
                ax.text(xi, d - 1.2, f"{d:+.2f}", ha="center", va="top",
                        fontsize=7, color=FIG9_STRATEGY_COLORS[strat],
                        fontweight="bold")
            elif d > 30:
                ax.text(xi, d + 0.6, f"{d:+.1f}", ha="center", va="bottom",
                        fontsize=7, color=FIG9_STRATEGY_COLORS[strat])

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in FIG9_REGION_ORDER],
                       fontsize=9)
    ax.set_ylabel(r"$\Delta$ RMSE  [SPT $N$]  (after $-$ before)")
    ax.set_title(
        "Test-time adaptation deltas vs source-only DKL+SVGP baseline\n"
        "(8 leave-region-out folds; negative = improvement)",
        fontsize=10,
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", ncol=3, fontsize=8,
              bbox_to_anchor=(0.0, 1.02), frameon=False)

    # Inset (top right): predictive-sigma collapse on log scale, one point
    # per (region, strategy). This single inset communicates the variance
    # pathology that the main panel's delta-RMSE bars cannot.
    #
    # It is placed in AXES fractions rather than figure fractions: at the
    # previous fig.add_axes([0.68, 0.55, ...]) its lower edge landed at about
    # y = 26 in data space, which cut through the kyushu self-train and tohoku
    # TENT bars. The y limit below is then solved so the inset's lower edge
    # sits clear of the tallest bar that lies underneath it, whatever the
    # data happens to be.
    ins_x0, ins_y0, ins_w, ins_h = 0.63, 0.58, 0.35, 0.36
    ymin = -10.0
    # x span covered by the inset, in data coordinates of the categorical axis.
    x_lo, x_hi = ax.get_xlim()
    inset_x_lo = x_lo + ins_x0 * (x_hi - x_lo)
    under = [deltas[ri, si]
             for ri in range(n_reg) for si in range(n_strat)
             if not np.isnan(deltas[ri, si])
             and x[ri] + (si - (n_strat - 1) / 2.0) * width + width / 2
             >= inset_x_lo]
    local_max = max(under) if under else 0.0
    global_max = float(np.nanmax(deltas))
    # Lower edge of the inset must clear local_max (plus room for its value
    # label); the axis must still show the global maximum.
    ymax = max(ymin + (local_max * 1.15 + 2.0 - ymin) / ins_y0,
               global_max * 1.12)
    ax.set_ylim(ymin, ymax)

    inset = ax.inset_axes([ins_x0, ins_y0, ins_w, ins_h])
    strat_x = np.arange(n_strat)
    for region in FIG9_REGION_ORDER:
        sigmas = []
        for strat in FIG9_STRATEGIES:
            row = by_key.get((region, strat))
            sigmas.append(np.nan if row is None else row["mean_std_after"])
        inset.plot(
            strat_x, sigmas, marker="o", markersize=3.5,
            linewidth=0.6, alpha=0.75,
            color="#1f77b4" if region == "kanto"
            else ("#d62728" if region == "kyushu_okinawa" else "#888888"),
            label=region if region in {"kanto", "kyushu_okinawa"} else None,
        )
    inset.set_yscale("log")
    inset.set_xticks(strat_x)
    inset.set_xticklabels(
        [FIG9_STRATEGY_LABELS[s] for s in FIG9_STRATEGIES],
        fontsize=7,
    )
    inset.tick_params(axis="y", labelsize=7)
    inset.set_ylabel(r"post-TTA mean $\sigma$", fontsize=7)
    inset.set_title("variance collapse", fontsize=8)
    inset.axhline(7.4, color="black", linestyle=":", linewidth=0.6)
    # Nudged right of the BN-stats column: at x=0.05 this label was printed on
    # top of the source-sigma marker it annotates.
    inset.text(0.38, 4.0, "source ~7.4", fontsize=6, color="black",
               va="center", transform=inset.get_yaxis_transform())
    inset.grid(True, which="both", linestyle=":", alpha=0.3)
    # The sigma traces fall steeply from BN-stats to TENT, leaving the lower
    # left of the inset empty; "center left" put the key on the traces.
    inset.legend(fontsize=6, loc="lower left", frameon=False)

    # Aggregate mean deltas across the 8 regions (matches the
    # ``mean (8 reg.)`` row of tab:tta_per_region_strategy).
    mean_delta = {
        FIG9_STRATEGY_LABELS[s]: float(np.nanmean(deltas[:, si]))
        for si, s in enumerate(FIG9_STRATEGIES)
    }

    caption = (
        "**Fig. 9** Per-region $\\Delta$RMSE for three test-time adaptation "
        "strategies (BN-stats, TENT, self-training) applied to the same 8 "
        "leave-region-out (LRO) DKL+SVGP checkpoints of "
        "tab:lro_per_region; n_adapted rows per region range from 216,692 "
        "(Hokkaido) to 495,725 (Kanto). Bars below the dashed zero line "
        "indicate held-out RMSE improvement; only Kanto (BN-stats, "
        f"{deltas[FIG9_REGION_ORDER.index('kanto'), 0]:+.2f}) and "
        "Kyushu/Okinawa (BN-stats "
        f"{deltas[FIG9_REGION_ORDER.index('kyushu_okinawa'), 0]:+.2f}, "
        f"TENT {deltas[FIG9_REGION_ORDER.index('kyushu_okinawa'), 1]:+.2f}) "
        "land below zero. Mean $\\Delta$RMSE across the 8 regions: "
        f"BN-stats {mean_delta['BN-stats']:+.2f}, "
        f"TENT {mean_delta['TENT']:+.2f}, "
        f"self-train {mean_delta['Self-train']:+.2f}. Strategy definitions: "
        "**BN-stats** re-estimates batch-norm running mean/variance from "
        "the target features (no gradient steps); **TENT** additionally "
        "updates batch-norm affine parameters by entropy minimisation on "
        "the predictive distribution; **self-training** freezes the encoder "
        "and re-fits the SVGP variational distribution against its own "
        "pseudo-labels for one epoch. Inset (log scale): post-adaptation "
        "mean predictive standard deviation per (region, strategy); TENT "
        "and self-training collapse $\\sigma$ from $\\sim$7.4 to "
        "$\\sim$0.011 in every region ($\\sim$335--680$\\times$ shrinkage), "
        "which destroys the conformal calibration of "
        "sections:cross_region / per_regime_calibration."
    )
    _save_pdf(fig, out, caption)


# ---------------------------------------------------------------------------
# Fig 10 -- cross-national content-vs-capacity transfer (the make-or-break)
# ---------------------------------------------------------------------------

# Japan HGB 8-region leave-region-out decomposition (docs/research/
# 2026-06-19_lro_text_transfer_b1_decomposition.md, F0/F4/F3):
#   baseline 11.23 -> +shuffled-null 9.70 (provenance+capacity) -> +text 8.99 (content)
JP_DECOMP = {"baseline": 11.23, "shuffled": 9.70, "text": 8.99}


def _content_pct(d: dict) -> float:
    return 100 * (d["text"]["mean_rmse"] - d["shuffled"]["mean_rmse"]) / d["shuffled"]["mean_rmse"]


def _capacity_pct(d: dict) -> float:
    return 100 * (d["shuffled"]["mean_rmse"] - d["no_text"]["mean_rmse"]) / d["no_text"]["mean_rmse"]


def _per_region_content(d: dict) -> list[float]:
    tpr, spr = d["text"].get("per_region", {}), d["shuffled"].get("per_region", {})
    return [100 * (tpr[r] - spr[r]) / spr[r] for r in tpr if r in spr]


def fig10_cross_national(out: Path, uk_json: Path | None,
                         storm_json: Path | None = None,
                         storm_nosize_json: Path | None = None,
                         japan_json: Path | None = None) -> None:
    """Three-domain centerpiece for ``words generalize, coordinates memorize``.

    (a) The genuine-content effect (text vs shuffled null, %) vs the
        capacity/provenance null effect (shuffled vs no-text, %) for THREE
        domains -- Japanese boreholes, UK boreholes, and US storm reports --
        on a common percentage axis (raw RMSE units differ: SPT-N vs hail-in).
        Content is large and negative everywhere; the null is ~0.
    (b) Per-region/state content effects, nearly all negative, with per-domain
        means -- the principle replicates across regions in every domain and
        spans subsurface geology to atmospheric hazards. All three use the
        identical leak-proof per-fold-PCA, multi-seed protocol.
    """
    jp = json.loads(japan_json.read_text()) if (japan_json and japan_json.exists()) else None
    uk = json.loads(uk_json.read_text()) if (uk_json and uk_json.exists()) else None
    storm = json.loads(storm_json.read_text()) if (storm_json and storm_json.exists()) else None
    storm_ns = (json.loads(storm_nosize_json.read_text())
                if (storm_nosize_json and storm_nosize_json.exists()) else None)
    if uk is None or storm is None or jp is None:
        LOG.warning("fig10: missing a domain json; using committed fallbacks")
        jp = jp or {"no_text": {"mean_rmse": 12.936}, "shuffled": {"mean_rmse": 11.922},
                    "text": {"mean_rmse": 9.393}}
        uk = uk or {"no_text": {"mean_rmse": 18.53}, "shuffled": {"mean_rmse": 18.03},
                    "text": {"mean_rmse": 14.64}}
        storm = storm or {"no_text": {"mean_rmse": 0.512}, "shuffled": {"mean_rmse": 0.508},
                          "text": {"mean_rmse": 0.329}}

    domains = [("Japanese\nboreholes", jp, "8 regions", "$p{=}0.004$"),
               ("UK\nboreholes", uk, "5 regions", "$p{=}0.031$"),
               ("US storm\nreports", storm, "30 states", "$p{\\approx}0$")]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1), constrained_layout=True)
    c_null, c_content = "#f0a35e", "#2a7ab0"

    # Panel (a): % decomposition, common axis across domains.
    ax = axes[0]
    x = np.arange(len(domains)); w = 0.36
    caps = [_capacity_pct(d) for _, d, _, _ in domains]
    cons = [_content_pct(d) for _, d, _, _ in domains]
    ax.bar(x - w / 2, caps, w, label="capacity + provenance null\n(shuffled vs no-text)",
           color=c_null, edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, cons, w, label="genuine content\n(real text vs shuffled)",
           color=c_content, edgecolor="black", linewidth=0.5)
    for xi, cap, con, (_, _, _, sig) in zip(x, caps, cons, domains):
        ax.text(xi - w / 2, cap + 0.4, f"{cap:+.1f}%", ha="center", va="bottom", fontsize=8)
        lbl = f"{con:+.1f}%" + (f"\n{sig}" if sig else "")
        ax.text(xi + w / 2, con - 0.6, lbl, ha="center", va="top", fontsize=8.5,
                color="white", fontweight="bold")
    # stripped-storm robustness marker on the storm content bar
    if storm_ns is not None:
        c_ns = _content_pct(storm_ns)
        ax.plot([x[2] + w / 2 - w / 2.2, x[2] + w / 2 + w / 2.2], [c_ns, c_ns],
                color="#0b3d61", lw=1.6, zorder=5)
        # Label to the LEFT of the marker: anchored to its right end the text
        # ran off the right edge of the axes. Nothing is plotted in the wedge
        # left of the storm bars at this depth.
        ax.text(x[2] + w / 2 - w / 2.2 - 0.04, c_ns,
                f"size words\nstripped\n{c_ns:+.1f}%",
                ha="right", va="center", fontsize=7.2, color="#0b3d61")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in domains], fontsize=9)
    ax.set_ylabel("Effect on leave-region-out RMSE  [%]")
    ax.set_title("(a) Content transfers; capacity/provenance null does not", fontsize=10)
    ax.set_ylim(bottom=min(cons) * 1.22, top=8)
    ax.legend(fontsize=7.4, loc="lower left", ncol=1)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    # Panel (b): per-region/state content effect distribution per domain.
    ax = axes[1]
    series = [("Japanese\nboreholes", _per_region_content(jp), "#7a7f87"),
              ("UK\nboreholes", _per_region_content(uk), c_content),
              ("US storm\nreports", _per_region_content(storm), "#1f9e6b")]
    rng = np.random.default_rng(0)
    for i, (name, vals, col) in enumerate(series):
        if not vals:
            continue
        jit = (rng.random(len(vals)) - 0.5) * 0.34 if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full(len(vals), i) + jit, vals, s=22, color=col, alpha=0.8,
                   edgecolor="black", linewidth=0.3, zorder=3)
        m = float(np.mean(vals))
        ax.plot([i - 0.25, i + 0.25], [m, m], color="black", lw=2.0, zorder=4)
        n_neg = sum(v < 0 for v in vals)
        ax.text(i, max(vals) + 2.0, f"mean {m:.1f}%\n{n_neg}/{len(vals)} neg", ha="center",
                va="bottom", fontsize=7.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(series))); ax.set_xticklabels([s[0] for s in series], fontsize=9)
    ax.set_ylabel("Per-region genuine content effect  [% RMSE]")
    ax.set_title("(b) Negative in nearly every region, every domain", fontsize=10)
    ax.set_ylim(top=14)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("Words generalize, coordinates memorize: free-text content transfers "
                 "out-of-distribution\nacross three domains --- subsurface geology to "
                 "atmospheric hazards", fontsize=10.5)

    ns_txt = ""
    if storm_ns is not None:
        ns_txt = (f" Stripping explicit hail-size descriptors from the storm narratives "
                  f"leaves a content effect of ${_content_pct(storm_ns):+.1f}\\%$ (panel a, "
                  f"dark line), so the signal is not merely the literally-stated size.")
    caption = (
        "**Fig. 10** The text-content effect transfers out-of-distribution across three "
        "domains, under one leak-proof per-fold-PCA, multi-seed protocol. (a) On a common "
        "percentage axis (raw RMSE units differ --- SPT-$N$ vs hail diameter), the "
        "genuine-content effect (real text vs the shuffled-embedding null) is large and "
        "negative in all three domains ($-21.2\\%$ Japanese boreholes, $-18.8\\%$ UK "
        "boreholes, $-35.2\\%$ US storm reports), while the capacity+provenance null "
        "(shuffled vs no-text) is small. Significance from a per-region sign test."
        + ns_txt +
        " (b) Per-region/state content effects: negative in 8/8 Japanese regions, 5/5 UK "
        "regions and 29/30 US states; black bars are per-domain means. The principle holds "
        "from subsurface "
        "geology to atmospheric hazards, where coordinates --- meaningless across these "
        "boundaries --- cannot."
    )
    _save_pdf(fig, out, caption)


_LEAK_ORDER: tuple[tuple[str, str], ...] = (
    ("lm_full", "full\ntext"),
    ("lm_lithology_only", "lithology-only\n(strength/N\nstripped)"),
    ("lm_hardness_only", "hardness-only\n(lithology\nstripped)"),
    ("tfidf_char", "char n-gram\nTF-IDF"),
    ("structured_litho", "structured\nlithology\nparser"),
    ("dictionary_onehot", "dictionary\none-hot"),
)


def fig11_leakage_controls(out: Path, japan_json: Path, uk_json: Path) -> None:
    """Leakage-control decomposition (the figure that answers the circularity
    objection): is the text a geological prior, or a paraphrase of the SPT-$N$
    label? Grouped bars of the genuine-content effect (%, real text vs shuffled
    null, leak-proof per-fold PCA, 5 seeds) for Japan and UK across five text
    representations -- frozen-LM full / lithology-only / hardness-only, and the
    two language-model-free baselines (char n-gram TF-IDF, dictionary one-hot).
    Lithology-only (all strength/N vocabulary stripped) still transfers -> not a
    label paraphrase; dictionary ~ null -> the carrier is narrative detail, not
    the lithology class; TF-IDF recovers most -> the frozen LM is a convenient
    high-coverage representation, not semantic magic.
    """
    jp = json.loads(japan_json.read_text())["variants"]
    uk = json.loads(uk_json.read_text())["variants"]
    labels = [lab for _, lab in _LEAK_ORDER]
    keys = [k for k, _ in _LEAK_ORDER]
    jp_vals = [jp[k]["content_pct"] for k in keys]
    uk_vals = [uk[k]["content_pct"] for k in keys]

    fig, ax = plt.subplots(figsize=(9.4, 4.7), constrained_layout=True)
    c_jp, c_uk = "#2a7ab0", "#f0a35e"
    x = np.arange(len(labels))
    w = 0.38
    bjp = ax.bar(x - w / 2, jp_vals, w, label="Japan (8 regions)",
                 color=c_jp, edgecolor="black", linewidth=0.5)
    buk = ax.bar(x + w / 2, uk_vals, w, label="UK (5 regions)",
                 color=c_uk, edgecolor="black", linewidth=0.5)

    def _annot(bars, vals, data):
        for b, v, k in zip(bars, vals, keys):
            xi = b.get_x() + b.get_width() / 2
            n_neg = str(data[k].get("n_neg", ""))
            sig_ns = (data[k].get("sign_p") or 1.0) > 0.05
            lab = f"{v:+.1f}%"
            ax.text(xi, v - 0.4 if v <= 0 else v + 0.4, lab, ha="center",
                    va="top" if v <= 0 else "bottom", fontsize=8.2,
                    fontweight="bold" if k == "lm_lithology_only" else "normal")
            tag = "n.s." if sig_ns else n_neg
            ax.text(xi, 0.35, tag, ha="center", va="bottom", fontsize=7.0,
                    color="0.35")

    _annot(bjp, jp_vals, jp)
    _annot(buk, uk_vals, uk)

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(2.5, color="0.6", lw=1.0, linestyle="--")
    ymin = min(min(jp_vals), min(uk_vals))
    ax.text(1.0, 3.0, "frozen language model", ha="center", fontsize=8.5, color="0.3")
    ax.text(4.0, 3.0, "no language model", ha="center", fontsize=8.5, color="0.3")
    # headline marker on lithology-only (the conservative, non-circular number)
    ax.annotate("conservative,\nnon-circular headline",
                xy=(1.0, min(jp_vals[1], uk_vals[1]) - 0.3),
                xytext=(1.0, ymin * 1.18), ha="center", va="top",
                fontsize=7.6, color="#0b3d61",
                arrowprops=dict(arrowstyle="->", color="#0b3d61", lw=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.0)
    ax.set_ylabel("Genuine-content effect on LRO RMSE  [%]\n"
                  "(real text vs shuffled null; more negative = better)")
    ax.set_ylim(bottom=ymin * 1.32, top=5)
    ax.set_title("Text leakage controls: a geological prior, not a paraphrase of the "
                 "SPT-$N$ label", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    caption = (
        "**Fig. (leakage)** Text leakage controls under the identical leak-proof, "
        "per-fold-PCA, multi-seed, leave-region-out protocol; bars are the "
        "genuine-content effect (real text vs the shuffled-embedding null) and the "
        "tag under each bar is the number of regions with a helpful (negative) effect "
        "or `n.s.`. Stripping all strength/consistency/N-value vocabulary "
        "(*lithology-only*, bold) leaves the effect essentially intact in Japan "
        f"({jp['lm_lithology_only']['content_pct']:+.1f}% vs "
        f"{jp['lm_full']['content_pct']:+.1f}% full) and still significant in the UK "
        f"({uk['lm_lithology_only']['content_pct']:+.1f}%), so the signal is **not** a "
        "paraphrase of the SPT-$N$ label. A coarse lithology dictionary transfers "
        f"nothing (Japan {jp['dictionary_onehot']['content_pct']:+.1f}%, UK "
        f"{uk['dictionary_onehot']['content_pct']:+.1f}%, both n.s.), so the carrier is "
        "the fine-grained free text, not the lithology *class*; a character n-gram "
        "TF-IDF on the same strings recovers most of the effect, so the frozen language "
        "model is a convenient high-coverage representation rather than a source of "
        "semantic magic."
    )
    _save_pdf(fig, out, caption)


# ---------------------------------------------------------------------------
# Fig 4 (descriptor mechanism) -- which words carry the transferable signal?
# ---------------------------------------------------------------------------

# Leave-one-descriptor-family-out arms of
# docs/research/2026-08-12_descriptor_families_japan.json (P-T10). Keys are the
# JSON arm names; values are the reader-facing English labels.
_DESCRIPTOR_FAMILIES: tuple[tuple[str, str], ...] = (
    ("minus_lith_class", "lithology class"),
    ("minus_grain_size", "grain size"),
    ("minus_water_state", "water state"),
    ("minus_dictionary", "strength dictionary"),
    ("minus_sorting", "sorting / grading"),
    ("minus_angularity", "angularity"),
    ("minus_composition_pct", "composition %"),
    ("minus_colour", "colour"),
    ("minus_weathering", "weathering"),
)


def fig4_descriptor_families(out: Path, families_json: Path) -> None:
    """Mechanism figure: *which* descriptors carry the transferable content?

    (a) The purified parser rung. The published parser number mixed
        text-derived features with AIST archive codes (88 features,
        -16.216%); restricting the rung to features actually derived from the
        layer narrative (66 features) makes the content effect *stronger*
        (-17.531%), so the archive codes are not what carries the transfer.
    (b) Leave-one-descriptor-family-out. Each bar spans the full text-only
        effect; the coloured tip is the attenuation (percentage points) caused
        by deleting that family, the grey remainder is the effect that
        survives. The largest single family (lithology class) removes
        1.82 pp of 17.53 pp, i.e. ~90% of the effect survives the deletion of
        any one family -- the signal is distributed across the fine-grained
        description, not carried by one vocabulary.

    Both panels report the genuine-content effect (real text vs the
    shuffled-embedding null) under the leave-region-out protocol; every arm is
    negative in 8/8 held-out Japanese regions.
    """
    data = json.loads(families_json.read_text())
    arms = data["arms"]

    text_only = arms["parser_text_only"]
    with_codes = arms["parser_with_codes"]
    full_effect = abs(float(text_only["content_pct"]))       # 17.531 pp

    fams = [(lab, arms[k]) for k, lab in _DESCRIPTOR_FAMILIES if k in arms]
    fams.sort(key=lambda kv: float(kv[1]["attenuation_pp"]), reverse=True)

    c_text, c_codes = "#2a7ab0", "#f0a35e"
    c_keep = "#d7dce2"

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5),
                             width_ratios=[1.0, 1.32],
                             constrained_layout=True)

    # ------------------------------------------------------------------
    # Panel (a): purified parser rung (2 bars, negative -> leftwards)
    # ------------------------------------------------------------------
    ax = axes[0]
    # (y-tick label, arm, bar colour, in-bar text colour). The in-bar colour is
    # picked per bar so the "n features" tag keeps contrast on both the dark
    # blue and the light orange fill.
    rows = [
        ("text-derived\nfeatures only", text_only, c_text, "white"),
        ("+ AIST archive\ncodes", with_codes, c_codes, "#1a1a1a"),
    ]
    ys = np.arange(len(rows), dtype=float)
    v_text = float(text_only["content_pct"])
    v_codes = float(with_codes["content_pct"])
    d_pp = abs(v_text) - abs(v_codes)

    # Shaded gap between the two bar tips: the panel's whole point.
    ax.fill_betweenx([-0.62, 1.78], v_text, v_codes, color="#0b3d61",
                     alpha=0.10, linewidth=0, zorder=0)

    for y, (_, arm, col, tcol) in zip(ys, rows):
        v = float(arm["content_pct"])
        ax.barh(y, v, height=0.42, color=col, edgecolor="black", linewidth=0.5,
                zorder=2)
        ax.text(v - 0.45, y, f"{v:+.2f}%", ha="right", va="center",
                fontsize=9.5, fontweight="bold")
        ax.text(-0.5, y, f"{int(arm['n_features'])} features",
                ha="right", va="center", fontsize=8.0, color=tcol,
                zorder=3)
        ax.text(v / 2.0, y + 0.31, f"{arm['regions_negative']} regions negative",
                ha="center", va="center", fontsize=7.4, color="0.25")
    ax.text(v_codes / 2.0, 1.53, "= the previously published rung",
            ha="center", va="center", fontsize=7.6, style="italic",
            color="0.30")

    # The take-home of the panel, under the shaded gap.
    ax.text((v_text + v_codes) / 2.0, 2.00, f"+{d_pp:.2f} pp",
            ha="center", va="center", fontsize=8.4, fontweight="bold",
            color="#0b3d61")
    ax.text(-12.0, 2.52,
            "dropping the archive codes\nstrengthens the text effect",
            ha="center", va="center", fontsize=8.2, color="#0b3d61")

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_ylim(3.05, -0.75)
    ax.set_xlim(-24.5, 0.0)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Genuine-content effect on LRO RMSE  [%]\n"
                  "(real text vs shuffled null; more negative = stronger)")
    ax.set_title("(a) The archive codes are not the carrier", fontsize=10)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    # ------------------------------------------------------------------
    # Panel (b): leave-one-descriptor-family-out
    # ------------------------------------------------------------------
    ax = axes[1]
    yb = np.arange(len(fams), dtype=float)
    for y, (lab, arm) in zip(yb, fams):
        att = float(arm["attenuation_pp"])
        kept = full_effect - att
        ax.barh(y, kept, height=0.62, color=c_keep, edgecolor="black",
                linewidth=0.4,
                label="effect that survives the deletion" if y == 0 else None)
        ax.barh(y, att, left=kept, height=0.62, color=c_text,
                edgecolor="black", linewidth=0.4,
                label="effect lost (attenuation)" if y == 0 else None)
        # A hairline at the split so families whose attenuation is a fraction
        # of a pp (weathering, 0.075) still show a visible boundary.
        ax.plot([kept, kept], [y - 0.31, y + 0.31], color="#0b3d61", lw=1.0,
                solid_capstyle="butt", zorder=4)
        ax.text(full_effect + 0.45, y, f"{att:.2f}", ha="left", va="center",
                fontsize=8.4)

    ax.axvline(full_effect, color="black", lw=1.0, linestyle="--", zorder=5)
    ax.text(full_effect - 0.25, -0.78,
            f"full text-only effect  {full_effect:.2f} pp",
            ha="right", va="center", fontsize=7.8)
    ax.text(full_effect + 0.45, -0.78, "lost (pp)", ha="left", va="center",
            fontsize=7.8, fontweight="bold")

    ax.set_yticks(yb)
    ax.set_yticklabels([lab for lab, _ in fams], fontsize=9)
    ax.set_ylim(len(fams) - 0.35, -1.15)
    ax.set_xlim(0.0, full_effect + 2.6)
    ax.set_xlabel("Magnitude of the genuine-content effect  "
                  "[% of LRO RMSE]")
    ax.set_title("(b) Leave-one-descriptor-family-out", fontsize=10)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.11), ncol=2,
              fontsize=8.0, frameon=False, handlelength=1.3,
              handletextpad=0.5, columnspacing=1.4)

    fig.suptitle(
        "No single descriptor family carries the effect: the signal is "
        "distributed across the fine-grained description\n"
        "(every arm shown stays negative in all 8 held-out Japanese regions)",
        fontsize=10.0)

    top = fams[0]
    caption = (
        "**Fig. 4** Which descriptors carry the transferable content "
        "(Japan, leave-region-out, "
        f"n = {int(data['config']['n_rows']):,} layers, seeds "
        f"{'/'.join(str(s) for s in data['config']['seeds'])}). "
        "(a) The structured-parser rung purified: restricting the parser to "
        "features derived from the layer narrative itself "
        f"({int(text_only['n_features'])} features) gives a genuine-content "
        f"effect of {float(text_only['content_pct']):+.2f}%, *stronger* than "
        "the previously published rung that also carried the AIST archive "
        f"codes ({int(with_codes['n_features'])} features, "
        f"{float(with_codes['content_pct']):+.2f}%); the archive codes "
        f"therefore dilute rather than create the effect ({d_pp:.2f} pp). "
        "(b) Leave-one-descriptor-family-out. Each bar spans the full "
        f"text-only effect ({full_effect:.2f} pp, dashed line); the coloured "
        "tip is the attenuation caused by deleting that family and the grey "
        "remainder is what survives. The most influential family "
        f"({top[0]}) accounts for only "
        f"{float(top[1]['attenuation_pp']):.2f} pp "
        f"({100 * float(top[1]['attenuation_pp']) / full_effect:.0f}% of the "
        "effect), and every single-family ablation remains negative in 8/8 "
        "held-out regions -- no single vocabulary carries the transfer."
    )
    _save_pdf(fig, out, caption)


# ---------------------------------------------------------------------------
# Fig 5 (few-shot) -- borehole-budgeted adaptation curve, both directions
# ---------------------------------------------------------------------------

FEWSHOT_BUDGETS: tuple[int, ...] = (0, 10, 25, 50, 100, 300)

FEWSHOT_DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("japan_to_uk", r"(a) Japan $\rightarrow$ UK"),
    ("uk_to_japan", r"(b) UK $\rightarrow$ Japan"),
)


def fig5_fewshot_curve(out: Path, fewshot_json: Path) -> None:
    """Borehole-budgeted few-shot adaptation curve for both transfer
    directions (P-T6).

    x is the number of target-region boreholes released for adaptation.
    Because a budget of 0 cannot sit on a log axis and a linear count axis
    crushes {0, 10, 25, 50}, the budgets are drawn at **evenly spaced
    categorical positions** labelled with the true counts; the caption says so
    explicitly. Bands are +/- 1 s.d. over seeds 42/43/44.

    Two horizontal references per panel: the zero-shot shuffled-embedding null
    arm, and a depth-only model trained on the target archive's own boreholes.
    Both are drawn in both panels, including the direction where the null is
    embarrassingly high (Japan -> UK, 0.285 vs depth-only 0.116).
    """
    d = json.loads(fewshot_json.read_text())

    c_text, c_depth = "#2a7ab0", "#f0a35e"
    c_null, c_ref = "#6f6f6f", "#1a1a1a"
    x = np.arange(len(FEWSHOT_BUDGETS), dtype=float)
    x_pad_lo, x_gutter, x_lab = -0.42, 1.50, 0.28

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True,
                             constrained_layout=True)

    n_cells = n_text_ahead = 0
    for ax, (key, title) in zip(axes, FEWSHOT_DIRECTIONS):
        blk = d[key]
        curve = blk["fewshot_curve"]

        def _arm(arm: str) -> tuple[np.ndarray, np.ndarray]:
            m, s = [], []
            for b in FEWSHOT_BUDGETS:
                cell = curve[f"boreholes={b}"][arm]
                m.append(float(cell["spearman_rho_mean"]))
                s.append(float(cell.get("spearman_rho_std", 0.0)))
            return np.asarray(m), np.asarray(s)

        m_dep, s_dep = _arm("depth_only")
        m_txt, s_txt = _arm("depth_text")
        n_cells += len(FEWSHOT_BUDGETS)
        n_text_ahead += int(np.sum(m_txt > m_dep))

        shuffled = float(blk["zero_shot_holdout"]["depth_shuffled"]["spearman_rho"])
        tgt_ref = float(blk["reference_target_trained_depth_only"]["spearman_rho"])

        # References first, clipped to the data region so their labels sit in
        # a clean right-hand gutter rather than on top of the lines.
        x_ref = [x_pad_lo, x[-1] + 0.12]
        ax.plot(x_ref, [shuffled] * 2, color=c_null, lw=1.3, linestyle=":",
                zorder=2,
                label="zero-shot shuffled-embedding null" if key == "japan_to_uk" else None)
        ax.plot(x_ref, [tgt_ref] * 2, color=c_ref, lw=1.3, linestyle="-.",
                zorder=2,
                label="target-trained depth-only reference" if key == "japan_to_uk" else None)
        ax.text(x[-1] + x_lab, shuffled, f"shuffled null\n{shuffled:.3f}",
                ha="left", va="center", fontsize=7.0, color=c_null)
        ax.text(x[-1] + x_lab, tgt_ref,
                f"target-trained\ndepth-only\n{tgt_ref:.3f}",
                ha="left", va="center", fontsize=7.0, color=c_ref)

        for m, s, col, mk, ls, lab in (
                (m_dep, s_dep, c_depth, "s", "--", "depth only"),
                (m_txt, s_txt, c_text, "o", "-", "depth + text")):
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.20,
                            linewidth=0, zorder=3)
            ax.plot(x, m, color=col, marker=mk, markersize=4.6, lw=1.9,
                    linestyle=ls, markeredgecolor="black",
                    markeredgewidth=0.4, zorder=4,
                    label=lab if key == "japan_to_uk" else None)

        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in FEWSHOT_BUDGETS])
        ax.set_xlim(x_pad_lo, x[-1] + x_gutter)
        ax.set_xlabel("Target-region boreholes used for adaptation\n"
                      "(evenly spaced categories; 0 = zero-shot)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

        if key == "japan_to_uk":
            ax.set_ylabel(r"Spearman $\rho$ on the shared held-out boreholes")
            ceiling = float(np.max(m_dep))
            zs_text = float(m_txt[0])
            ax.text(x_pad_lo + 0.12, 0.575,
                    "text with no local boreholes "
                    f"($\\rho={zs_text:.3f}$)\nbeats depth-only at every "
                    f"budget\n(depth-only ceiling $\\rho={ceiling:.3f}$ "
                    "up to 300)",
                    ha="left", va="top", fontsize=7.6, color="#0b3d61")
            ax.text(x_pad_lo + 0.12, 0.012,
                    "here the shuffled null sits ABOVE depth-only: much of this\n"
                    "direction's zero-shot transfer is generic embedding structure",
                    ha="left", va="bottom", fontsize=7.0, color="0.30")
        else:
            ax.text(x_pad_lo + 0.12, 0.012,
                    "here the shuffled null sits BELOW depth-only, so the\n"
                    "ordering of the two nulls is direction-dependent",
                    ha="left", va="bottom", fontsize=7.0, color="0.30")

    axes[0].set_ylim(0.0, 0.60)

    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(l) for l in
             ("depth + text", "depth only",
              "zero-shot shuffled-embedding null",
              "target-trained depth-only reference") if l in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="outside lower center", ncol=4, fontsize=8.4,
               frameon=False, handlelength=2.2, columnspacing=1.8)

    fig.suptitle(
        # Deliberately NOT "text beats local drilling": the reference the text
        # arm overtakes is a depth-only local model, not a fully-featured one,
        # and the panel annotation states that scope. The headline claim here
        # is the one the 12/12 count actually supports.
        "Adding the layer description beats depth alone at every borehole "
        f"budget,\nin both directions ({n_text_ahead}/{n_cells} cells); bands "
        "are $\\pm$1 s.d. over 3 seeds",
        fontsize=10.0)

    jp = d["japan_to_uk"]
    uk = d["uk_to_japan"]
    jp_curve = jp["fewshot_curve"]
    jp_dep_max = max(float(jp_curve[f"boreholes={b}"]["depth_only"]
                           ["spearman_rho_mean"]) for b in FEWSHOT_BUDGETS)
    jp_zs_text = float(jp_curve["boreholes=0"]["depth_text"]["spearman_rho_mean"])
    caption = (
        "**Fig. 5** Borehole-budgeted few-shot adaptation. Spearman $\\rho$ "
        "on a fixed 50% held-out set of *target boreholes* (shared across "
        "budgets, arms and seeds, so no sibling layer of a training borehole "
        "can leak into the test set) as a function of the number of target "
        "boreholes released for adaptation, for (a) Japan $\\rightarrow$ UK "
        f"({jp['n_holdout_boreholes']:,} held-out boreholes / "
        f"{jp['n_holdout_rows']:,} layers) and (b) UK $\\rightarrow$ Japan "
        f"({uk['n_holdout_boreholes']:,} / {uk['n_holdout_rows']:,}). "
        "Budgets are plotted at evenly spaced categorical positions labelled "
        "with the true borehole counts, because a zero budget has no place on "
        "a count/log axis; bands are $\\pm$1 s.d. over seeds 42/43/44. The "
        "depth + text arm is above depth-only in "
        f"{n_text_ahead}/{n_cells} cells. "
        "In (a) the depth-only arm never exceeds "
        f"$\\rho = {jp_dep_max:.3f}$ at any budget up to 300 boreholes, while "
        f"text with no local boreholes at all already reaches "
        f"$\\rho = {jp_zs_text:.3f}$ -- the description channel is worth more "
        "than 300 locally drilled and logged boreholes without it. Two "
        "references are drawn in both panels: the zero-shot "
        "shuffled-embedding null arm (dotted) and a depth-only model fitted "
        "on the target archive's own boreholes (dash-dot). Their ordering is "
        "direction-dependent and we do not hide it: Japan $\\rightarrow$ UK "
        f"the shuffled null ({float(jp['zero_shot_holdout']['depth_shuffled']['spearman_rho']):.3f}) "
        "sits well above depth-only "
        f"({float(jp['zero_shot_holdout']['depth_only']['spearman_rho']):.3f}), so "
        "much of that direction's apparent zero-shot transfer is generic "
        "embedding structure rather than content, whereas UK "
        f"$\\rightarrow$ Japan the null "
        f"({float(uk['zero_shot_holdout']['depth_shuffled']['spearman_rho']):.3f}) "
        "falls below depth-only "
        f"({float(uk['zero_shot_holdout']['depth_only']['spearman_rho']):.3f})."
    )
    _save_pdf(fig, out, caption)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ALL_FIGURES: tuple[str, ...] = (
    "fig1", "fig1_schematic", "fig1_deployment",
    "fig2", "fig3", "fig4", "fig4_descriptors", "fig5", "fig5_fewshot",
    "fig6", "fig7", "fig8", "fig9", "fig10", "fig11",
)


def _artefact(name: str, repo: Path) -> Path:
    """Default path of a result-provenance JSON, layout-independent.

    In the development monorepo these live at ``<repo>/docs/research/<name>``
    (``repo`` = ``parents[2]`` of this file, i.e. the directory above
    ``backend/``). In the companion reproduction repository assembled by
    ``scripts/build_paper2_companion_repo.py`` the tree root corresponds to
    ``backend/``, so ``parents[2]`` points OUTSIDE the tree and that default
    can never resolve; the same artefacts ship there in ``results/``. Prefer
    the monorepo location when it exists, else the sibling ``results/`` dir,
    else the monorepo location again so the error message names the canonical
    path.
    """
    monorepo = repo / "docs/research" / name
    if monorepo.exists():
        return monorepo
    companion = Path(__file__).resolve().parents[1] / "results" / name
    return companion if companion.exists() else monorepo


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path,
                        default=repo / "data/runs")
    parser.add_argument("--parquet", type=Path,
                        default=repo / "data/features/borings_japan_v4.parquet")
    parser.add_argument("--layers-csv", type=Path,
                        default=repo / "data/features/derived/soil_text_layers.csv")
    parser.add_argument(
        "--cube-dir", type=Path,
        default=repo / "data/products/national_cube_kanto_v2hero",
        help=(
            "Directory containing the national-cube product, expected "
            "layout ``<cube-dir>/{cube/tile_*.zarr, maps/*.nc, "
            "manifest.json}``. May be either a local path or an "
            "NFS-mounted path (e.g. /mnt/nas/geo-estimation/products/"
            "national_cube_japan_1km_v2hero); no local rsync is "
            "required. For back-compat we also accept --cube-dir "
            "pointed directly at the cube/ subdirectory."),
    )
    parser.add_argument("--out-dir", type=Path,
                        default=repo / "docs/paper/paper_2_national/figures")
    parser.add_argument(
        "--uk-json", type=Path,
        default=_artefact("2026-06-21_uk_transfer_leakproof.json", repo),
        help="UK BGS transfer result JSON for Fig 10 (cross-national, leak-proof per-fold PCA).")
    parser.add_argument(
        "--storm-json", type=Path,
        default=_artefact("2026-06-21_storm_transfer_3rd_domain.json", repo),
        help="NOAA Storm Events transfer result JSON for Fig 10 (3rd domain).")
    parser.add_argument(
        "--storm-nosize-json", type=Path,
        default=_artefact("2026-06-21_storm_transfer_nosize.json", repo),
        help="Storm size-stripped robustness JSON for Fig 10 (optional annotation).")
    parser.add_argument(
        "--japan-json", type=Path,
        default=_artefact("2026-06-21_japan_transfer_leakproof.json", repo),
        help="Japan leak-proof per-fold transfer JSON for Fig 10.")
    parser.add_argument(
        "--japan-leak-json", type=Path,
        default=_artefact("2026-06-21_text_leakage_japan.json", repo),
        help="Japan text-leakage-controls JSON for the leakage figure (fig11).")
    parser.add_argument(
        "--uk-leak-json", type=Path,
        default=_artefact("2026-06-21_text_leakage_uk.json", repo),
        help="UK text-leakage-controls JSON for the leakage figure (fig11).")
    parser.add_argument(
        "--descriptor-families-json", type=Path,
        default=_artefact("2026-08-12_descriptor_families_japan.json", repo),
        help=("P-T10 descriptor-family ablation JSON for the descriptor "
              "mechanism figure (fig4_descriptors)."))
    parser.add_argument(
        "--fewshot-json", type=Path,
        default=_artefact("2026-08-12_fewshot_borehole_curve.json", repo),
        help=("P-T6 borehole-budgeted few-shot curve JSON for the few-shot "
              "figure (fig5_fewshot)."))
    parser.add_argument("--figures", nargs="+",
                        default=["all"],
                        help=("Subset of " + ",".join(ALL_FIGURES)
                              + " or 'all'."))
    parser.add_argument("--main-only", action="store_true",
                        help=("Fig 8 mode: emit only panel (a) and write "
                              "the LPI panel to fig8b_lpi_supp.pdf."))
    parser.add_argument(
        "--conformal-json", type=Path, default=None,
        help=(
            "Explicit path to the Mondrian conformal table "
            "(conformal_mondrian.json, or a directory containing it) for "
            "Fig 8 panel (b). Overrides the --runs-dir lookup. Needed on "
            "the render pod, where the table lives on NFS at "
            "/mnt/nas/runs/dkl_national_full_v2/conformal_mondrian.json "
            "rather than under the figure-build runs dir. When omitted, "
            "Fig 8 also probes the cube-dir-adjacent NFS runs tree and the "
            "canonical /mnt/nas/runs/<run>/ mount automatically."),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _set_paper_style()
    figs = list(ALL_FIGURES) if "all" in args.figures else list(args.figures)

    if "fig1" in figs:
        fig1_concept(args.out_dir / "fig1_concept.pdf")
    if "fig1_schematic" in figs:
        fig1_schematic(args.out_dir / "fig1_schematic.pdf")
    if "fig1_deployment" in figs:
        fig1_deployment(args.out_dir / "fig1_deployment.pdf")
    if "fig2" in figs:
        fig2_study_area(args.out_dir / "fig2_study_area.pdf",
                        args.parquet, args.layers_csv)
    if "fig3" in figs:
        fig3_llm_text_gain(args.out_dir / "fig3_llm_text_gain.pdf",
                           args.layers_csv)
    if "fig4" in figs:
        fig4_lro_gap(args.out_dir / "fig4_lro_gap.pdf", args.runs_dir)
    if "fig4_descriptors" in figs:
        fig4_descriptor_families(args.out_dir / "fig4_descriptor_families.pdf",
                                 args.descriptor_families_json)
    if "fig5" in figs:
        fig5_model_inversion(args.out_dir / "fig5_model_inversion.pdf")
    if "fig5_fewshot" in figs:
        fig5_fewshot_curve(args.out_dir / "fig5_fewshot_curve.pdf",
                           args.fewshot_json)
    if "fig6" in figs:
        fig6_conformal_heatmap(args.out_dir / "fig6_conformal_heatmap.pdf",
                               args.runs_dir)
    if "fig7" in figs:
        fig7_cube_transect(args.out_dir / "fig7_cube_slices.pdf",
                           args.cube_dir)
    if "fig8" in figs:
        fig8_depth_profiles_and_lpi(
            args.out_dir / "fig8_uncertainty.pdf",
            args.cube_dir,
            main_only=args.main_only,
            supp_out=args.out_dir / "fig8b_lpi_supp.pdf",
            runs_dir=args.runs_dir,
            conformal_json=args.conformal_json,
            boring_parquet=args.parquet)
    if "fig9" in figs:
        fig9_tta_delta_rmse(args.out_dir / "fig9_tta_delta_rmse.pdf")
    if "fig10" in figs:
        fig10_cross_national(args.out_dir / "fig10_cross_national.pdf",
                             args.uk_json, args.storm_json, args.storm_nosize_json,
                             args.japan_json)
    if "fig11" in figs:
        fig11_leakage_controls(args.out_dir / "fig11_leakage_controls.pdf",
                               args.japan_leak_json, args.uk_leak_json)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
