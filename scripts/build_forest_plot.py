#!/usr/bin/env python
"""NC round-3 [E5] — forest plot of per-region content effects across fold families.

The strongest visual answer to "effective n is only 8+5 regions": every held-out
unit's content effect, across four fold families (JP administrative, UK
administrative, JP geological lithology/era folds) plus the 30-state storm control,
with the region-bootstrap 95% CI for the two headline archives. All negative points
tell the story at a glance.

Reads the v2 result JSONs (run AFTER the v2 pipeline lands):
  within_null_v2_{japan,uk}.json   -> per-region within_class content
  geo_fold_v2.json                 -> per-group leave-litho / leave-era content
  storm_transfer_nosize.json       -> per-state content (size-stripped)
  region_bootstrap CIs recomputed inline (10^4 percentile) from per-region values.

Output: figures/fig_forest_content.pdf (+ .png)

Run: cd backend && .venv/bin/python -m scripts.build_forest_plot \
        --out ../docs/paper/paper_2_national/figures/fig_forest_content.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _default_research_dir() -> Path:
    """Result-JSON directory, resolved for both source layouts.

    In the research monorepo the four inputs live in ``docs/research/``; in
    the companion repository (``expert-text-transfer``, assembled by
    ``build_paper2_companion_repo.py``) the same files are flattened into
    ``results/`` at the repo root. Probe both so the script runs unchanged
    from either tree; ``--research-dir`` overrides.
    """
    for cand in (REPO / "docs/research",
                 Path(__file__).resolve().parents[1] / "results"):
        if cand.is_dir():
            return cand
    return REPO / "docs/research"



def _ci(vals: list[float], n: int = 10_000, seed: int = 42) -> tuple[float, float]:
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def collect(research_dir: Path | None = None) -> list[dict]:
    R = research_dir if research_dir is not None else _default_research_dir()
    fam = []
    # Headers are drawn inside the panels, so they must stay short enough not
    # to run into the neighbouring column. The within-class-null protocol and
    # the full family definitions are spelled out in the figure caption.
    # The two administrative families are the PRE-REGISTERED primary estimand and
    # must come from the borehole-block null, not from the row-shuffled
    # within_null_v2 artefacts this function used to read. Those are a different
    # null on a different baseline over a near-disjoint subsample; plotting them
    # under a headline label made the figure disagree with the main text.
    for f, label in (("2026-08-18_grouped_null_japan_s42.json",
                      "Japan — leave-region-out (8 regions)"),
                     ("2026-08-18_grouped_null_uk.json",
                      "UK — leave-region-out (5 regions)")):
        d = json.loads((R / f).read_text())
        per = {r: v["content_pct_block"] for r, v in d["per_region"].items()}
        # Refuse to plot a null that was not a permutation. The artefact carries
        # the evidence; if a future re-run loses that property the figure should
        # fail loudly rather than render a misleading point.
        for r, v in d["per_region"].items():
            npm = v.get("null_permutation") or {}
            if not npm.get("all_draws_bijective"):
                raise SystemExit(
                    f"{f}:{r} does not certify all_draws_bijective; refusing to "
                    f"plot it as the primary estimand")
        vals = list(per.values())
        lo, hi = _ci(vals)
        fam.append({"label": label, "units": per, "mean": float(np.mean(vals)),
                    "ci": (lo, hi)})
    g = json.loads((R / "2026-07-04_geo_fold_v2.json").read_text())
    for k, lbl in (("leave_litho_macro_out", "Japan — leave-lithology-class-out (LM)"),
                   ("leave_era_out", "Japan — leave-geological-era-out (LM)")):
        per = g["geological_fold_lm"][k]["per_group_content"]
        vals = list(per.values())
        lo, hi = _ci(vals)
        fam.append({"label": lbl, "units": per, "mean": float(np.mean(vals)), "ci": (lo, hi)})
    s = json.loads((R / "2026-06-21_storm_transfer_nosize.json").read_text())
    per_s = s.get("per_region_content_pct") or s.get("per_state_content_pct") or {}
    if per_s:
        vals = list(per_s.values())
        lo, hi = _ci(vals)
        fam.append({"label": "US storms — leave-state-out (30 states)",
                    "units": per_s, "mean": float(np.mean(vals)), "ci": (lo, hi)})
    return fam


def _split_families(fam: list[dict]) -> tuple[list[dict], list[dict]]:
    """Partition the families into two columns of roughly equal row count.

    Families are kept whole and in order; the split point is the one that
    minimises the difference in rendered rows between the two columns.
    """
    rows = [len(f["units"]) + 2 for f in fam]
    total = sum(rows)
    best_i, best_gap = 1, None
    for i in range(1, len(fam)):
        gap = abs(sum(rows[:i]) - (total - sum(rows[:i])))
        if best_gap is None or gap < best_gap:
            best_i, best_gap = i, gap
    return fam[:best_i], fam[best_i:]


def _draw_column(ax, fam: list[dict], n_rows: int) -> None:
    """Draw one column of families and label its rows."""
    y = 0
    yticks, ylabels = [], []
    for f in fam:
        y -= 1
        # Family headers are drawn INSIDE the axes rather than as y-tick
        # labels: as tick labels their ~50-character text sets the width of
        # the label gutter, which at this figure size leaves almost nothing
        # for the data. Full family definitions are in the caption.
        ax.text(0.012, y, f["label"], transform=ax.get_yaxis_transform(),
                fontsize=6.2, va="center", ha="left", color="#222222")
        for name, v in sorted(f["units"].items(), key=lambda kv: kv[1]):
            y -= 1
            ax.plot(v, y, "o", ms=3.0, color="#1f6fb2" if v < 0 else "#c0392b")
            yticks.append(y); ylabels.append(name)
        y -= 1
        lo, hi = f["ci"]
        ax.plot([lo, hi], [y, y], "-", lw=1.8, color="#333")
        ax.plot(f["mean"], y, "D", ms=4.5, color="#333")
        yticks.append(y); ylabels.append("family mean (95% CI)")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=6.0)
    # Pad both columns to the same row count so the two panels share a pitch
    # and the shorter one does not stretch its rows apart.
    ax.set_ylim(-n_rows - 0.5, 0.5)
    ax.grid(True, axis="x", linestyle=":", alpha=0.35)
    ax.tick_params(axis="x", labelsize=7, length=2.5)
    ax.tick_params(axis="y", length=2.0, pad=1.5)


def plot(fam: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One column of 79 rows at the old 0.34 in pitch is 26.9 in tall (aspect
    # 3.3). At \textwidth that wants 17.7 in of vertical space on a 9.7 in
    # text block, so LaTeX truncated the figure at the bottom of the page and
    # dropped its caption entirely. Two columns halve the height and land the
    # figure at a page-shaped aspect that fits a single float page.
    left_fam, right_fam = _split_families(fam)
    n_rows = max(sum(len(f["units"]) + 2 for f in half)
                 for half in (left_fam, right_fam))

    # Sized to the LaTeX \textwidth of the paper (6.27 in) so that
    # \includegraphics[width=\textwidth] scales by ~1 and the 6 pt row labels
    # stay 6 pt on the page instead of shrinking to an unreadable 4 pt.
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 6.2), sharex=True,
                             constrained_layout=True)
    for ax, half in zip(axes, (left_fam, right_fam)):
        _draw_column(ax, half, n_rows)

    fig.supxlabel(
        "Genuine text-content effect on held-out-unit RMSE (%); "
        "negative = text helps", fontsize=7.5)
    fig.suptitle("Per-held-out-unit content effect across fold families",
                 fontsize=8.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("wrote", out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/paper/paper_2_national/figures/fig_forest_content.pdf")
    ap.add_argument("--research-dir", type=Path, default=None,
                    help="Directory of the four result JSONs (default: "
                         "docs/research/ in the monorepo, results/ in the "
                         "companion repository).")
    a = ap.parse_args(argv)
    plot(collect(a.research_dir), a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
