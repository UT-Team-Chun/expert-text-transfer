#!/usr/bin/env python
"""Auto-generate the Appendix C Mondrian per-regime coverage table for
Paper B' (national pivot, target venue: *Nature Communications Earth &
Environment*).

Globs ``data/runs/dkl_national_*/conformal_mondrian.json`` and emits a
self-contained LaTeX fragment containing one tabular environment per
$\\alpha\\in\\{0.5, 0.8, 0.95\\}$ summarising the per-regime conformal
calibration gap (empirical minus nominal) across every available run.

The output is intended to be ``\\input{}``-ed from
``appendix_c_mondrian_recal_full.tex``.

CLI:

.. code-block:: bash

    cd backend && .venv/bin/python -m scripts.build_mondrian_recal_appendix_table \\
        --out ../docs/paper/paper_2_national/sections/appendix_c_mondrian_recal_full_data.tex

Style mirrors ``backend/scripts/build_paper2_figs.py``: argparse + Path
I/O, ``REGIME_NAMES`` canonical ordering, robust to missing entries.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

LOG = logging.getLogger("scripts.build_mondrian_recal_appendix_table")

# Canonical 8-way AIST regime ordering (matches build_paper2_figs.py).
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

ALPHAS: tuple[float, ...] = (0.5, 0.8, 0.95)


def _load_conformal(json_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(json_path.read_text())
    except Exception as exc:
        LOG.warning("skip %s: %s", json_path, exc)
        return None


def _gap_row(c: dict[str, Any], alpha: float) -> list[str]:
    """Return 8 string cells with signed coverage gap for ``alpha``."""
    per_reg = c.get("per_regime", {}).get(str(alpha), {})
    cells: list[str] = []
    for ri in range(len(REGIME_NAMES)):
        entry = per_reg.get(str(ri))
        if entry is None:
            cells.append("--")
            continue
        try:
            gap = float(entry["coverage"]) - float(alpha)
            cells.append(f"${gap:+.4f}$")
        except (KeyError, TypeError, ValueError):
            cells.append("--")
    return cells


#: Every run in this table shares this prefix. Spelling it out in all 21 row
#: labels plus spelling out all eight regime names in the header made the
#: tabular 264 pt wider than \textwidth, so pdflatex silently truncated the
#: last three regime columns at the paper edge. Both are abbreviated instead:
#: the prefix is stated once in the caption, and the regime columns use the
#: numeric codes already defined by ``tab:per_regime_full_v2``.
RUN_PREFIX = "dkl_national_"


def _escape_run(name: str) -> str:
    return name.replace("_", r"\_")


def _short_run(name: str) -> str:
    """Row label with the shared ``dkl_national_`` prefix stripped."""
    return _escape_run(name.removeprefix(RUN_PREFIX))


def _emit_subtable(rows: list[tuple[str, list[str]]], alpha: float) -> str:
    """Emit one tabular environment for a single $\\alpha$ level."""
    header = " & ".join(str(i) for i in range(len(REGIME_NAMES)))
    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(
        rf"\caption{{Per-regime conformal coverage gap (empirical $-$ nominal) "
        rf"at $\alpha={alpha}$ across all available "
        rf"\texttt{{dkl\_national\_*}} runs. Row labels omit the shared "
        rf"\texttt{{dkl\_national\_}} prefix; column headings $0$--$7$ are the "
        rf"AIST regime codes of Table~\ref{{tab:per_regime_full_v2}} "
        rf"({_regime_code_key()}). Cells marked ``--'' had no "
        rf"calibration entry for that regime (typically $n_{{\mathrm{{cal}}}} "
        rf"< 30$).}}"
    )
    lines.append(
        rf"\label{{tab:mondrian_recal_full_alpha_{str(alpha).replace('.', '_')}}}"
    )
    # @{} on both ends drops the outer \tabcolsep pads; without them the nine
    # columns still ran 4 pt past \textwidth.
    col_spec = "@{}l" + "r" * len(REGIME_NAMES) + "@{}"
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append("Run & " + header + r" \\")
    lines.append(r"\midrule")
    for run_name, cells in rows:
        lines.append(_short_run(run_name) + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _regime_code_key() -> str:
    """``0 = ALLUVIAL, 1 = DILUVIAL, ...`` for the caption, LaTeX-escaped."""
    return ", ".join(
        rf"{i} = \textsc{{{name.replace('_', r'\_').lower()}}}"
        for i, name in enumerate(REGIME_NAMES)
    )


def build_table(runs_dir: Path, out_path: Path) -> int:
    """Glob conformal artefacts, build per-$\\alpha$ subtables, write file."""
    artefacts = sorted(runs_dir.glob("dkl_national_*/conformal_mondrian.json"))
    if not artefacts:
        LOG.warning("no conformal_mondrian.json found under %s", runs_dir)
        out_path.write_text(
            "% Auto-generated by scripts/build_mondrian_recal_appendix_table.py\n"
            "% TODO: no conformal_mondrian.json artefacts found locally; "
            "re-run after syncing data/runs/dkl_national_*/.\n"
        )
        return 1

    per_alpha_rows: dict[float, list[tuple[str, list[str]]]] = {
        a: [] for a in ALPHAS
    }
    for cj in artefacts:
        c = _load_conformal(cj)
        if c is None:
            continue
        run_name = cj.parent.name
        for a in ALPHAS:
            per_alpha_rows[a].append((run_name, _gap_row(c, a)))

    # Header comment: keep the source path repo-relative. The absolute form
    # leaked the author's local home-directory path into a committed .tex.
    try:
        runs_note = runs_dir.resolve().relative_to(Path(__file__).resolve().parents[2])
    except ValueError:
        runs_note = runs_dir
    body: list[str] = [
        "% Auto-generated by scripts/build_mondrian_recal_appendix_table.py",
        f"% Source: {runs_note}/dkl_national_*/conformal_mondrian.json "
        f"({len(artefacts)} runs)",
        "",
    ]
    for a in ALPHAS:
        body.append(_emit_subtable(per_alpha_rows[a], a))
        body.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(body))
    LOG.info("Wrote %s (%d runs x %d alphas)", out_path, len(artefacts),
             len(ALPHAS))
    return 0


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    default_out = (repo / "docs/paper/paper_2_national/sections"
                   / "appendix_c_mondrian_recal_full_data.tex")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path,
                        default=repo / "data/runs")
    parser.add_argument("--out", type=Path, default=default_out,
                        help="Output .tex fragment path.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    return build_table(args.runs_dir, args.out)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
