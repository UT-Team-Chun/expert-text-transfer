#!/usr/bin/env python
"""Build the v4id boring parquet: v4 + borehole identity, provably unchanged.

NC pre-review response R0-1 (2026-08-11). The v4 parquet
(``borings_japan_v4.parquet``) has no borehole identity column, so every
downstream join and resampling unit fell back to rounded (lat, lon) --- which
the join audit showed is NOT a valid borehole key (9.0% of 10 m keys hold >1
borehole; ~15% of boreholes share byte-identical coordinates).

This script rebuilds the row set from the identity-bearing upstream CSV
(``data/outputs/location_n_values.csv``: 2,703,566 SPT rows keyed by
``file_path`` + ``dtd_version``) and re-attaches v4's per-location covariates
by exact float32 (lat, lon) lookup --- valid because every covariate in v4 is
per-location by construction (``enrich()`` computes them on ``unique_locs``
and merges back). No geometry is recomputed and no XML is re-parsed.

The output is accepted ONLY if it reproduces v4 exactly:

- row count == v4 row count, and
- every v4 column is byte-identical after aligning on the sort order.

That makes v4id a strict superset of v4 (10 columns identical + ``boring_file``
+ ``dtd_version``), so every number computed on v4 remains valid on v4id. (The name 'v5' is
NOT used: in the manuscript and the LMC pipeline it already denotes the
text-augmented parquet/model.)

CLI::

    cd backend
    .venv/bin/python -m scripts.attach_identity_to_parquet \
        --v4 ../data/features/borings_japan_v4.parquet \
        --csv ../data/outputs/location_n_values.csv \
        --out ../data/features/borings_japan_v4id.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from national.data.enrich import _normalize_input_columns

LOG = logging.getLogger("scripts.attach_identity_to_parquet")

REPO = Path(__file__).resolve().parents[2]

#: v4 columns that are functions of the location only (attached per unique
#: (lat, lon) in enrich() and merged back), i.e. safe to re-attach by
#: location lookup. absolute_elevation is per-row (mouth_elevation - depth),
#: so it is recomputed from the CSV instead and checked by the equality gate.
_PER_LOCATION = [
    "river_distance_km",
    "coast_distance_km",
    "regime_code",
    "aist_era_code",
    "aist_litho_macro_code",
    "groundwater_depth_m",
]


def build(v4_path: Path, csv_path: Path, out_path: Path) -> int:
    LOG.info("reading v4 parquet %s", v4_path)
    v4 = pd.read_parquet(v4_path)
    LOG.info("v4: %d rows x %d cols", len(v4), v4.shape[1])

    LOG.info("reading identity CSV %s", csv_path)
    # dtd_version as string ("2.10" must not collapse to float 2.1)
    csv = pd.read_csv(csv_path, dtype={"dtd_version": str, "file_path": str})
    if "file_path" not in csv.columns or "dtd_version" not in csv.columns:
        raise SystemExit("identity CSV lacks file_path/dtd_version")
    csv = _normalize_input_columns(csv)
    LOG.info("CSV after the enrich row filters: %d rows", len(csv))
    if len(csv) != len(v4):
        # The original v2 build may have applied the 'japan' bbox preset.
        japan = (24.0, 122.0, 46.0, 146.5)
        m = (
            (csv["latitude_deg"] >= japan[0]) & (csv["latitude_deg"] <= japan[2])
            & (csv["longitude_deg"] >= japan[1]) & (csv["longitude_deg"] <= japan[3])
        )
        if int(m.sum()) == len(v4):
            LOG.info("row count matches after the 'japan' bbox preset; applying it")
            csv = csv[m].reset_index(drop=True)
        else:
            raise SystemExit(
                f"row-count mismatch after filters: CSV {len(csv)} "
                f"(japan-bbox {int(m.sum())}) vs v4 {len(v4)} "
                "-- the CSV is not the parquet's upstream; aborting."
            )

    # Reproduce the per-row derivations of enrich() on the CSV side, in
    # float32 so the alignment keys are exact.
    csv = csv.copy()
    csv["depth_from_surface"] = csv["spt_start_depth"]
    csv["absolute_elevation"] = csv["mouth_elevation"] - csv["spt_start_depth"]
    for c in ("latitude_deg", "longitude_deg", "depth_from_surface",
              "absolute_elevation", "n_value"):
        csv[c] = csv[c].astype(np.float32)
    csv["boring_file"] = (
        csv["file_path"].astype(str).str.replace("\\", "/", regex=False)
        .str.rsplit("/", n=1).str[-1]
    )

    # ---- 1:1 row alignment on the per-row measured values -----------------
    # v4's covariates are NOT unique per float32 (lat, lon): enrich() computed
    # them per float64 location and cast to float32 afterwards, so two nearby
    # float64 locations can collapse onto one float32 coordinate with
    # different river/coast distances (measured: ~1.1k rows). A location
    # lookup is therefore ill-defined. Instead we align v4 rows and CSV rows
    # one-to-one on the five per-row measured columns plus a within-group
    # cumcount, and take every v4 column verbatim from v4 -- which makes the
    # equality property structural rather than asserted.
    keys = ["latitude_deg", "longitude_deg", "depth_from_surface",
            "absolute_elevation", "n_value"]
    v4k = v4.copy()
    v4k["_k"] = v4k.groupby(keys, sort=False).cumcount()
    csvk = csv[[*keys, "boring_file", "dtd_version", "file_path"]].copy()
    csvk["_k"] = csvk.groupby(keys, sort=False).cumcount()

    merged = v4k.merge(csvk, on=[*keys, "_k"], how="left", validate="1:1")
    n_unmatched = int(merged["boring_file"].isna().sum())
    if n_unmatched:
        raise SystemExit(
            f"{n_unmatched} v4 rows found no identity row -- the CSV is not "
            "the parquet's upstream (multiset mismatch on the measured "
            "columns); aborting."
        )

    # Identity-assignment ambiguity: within a duplicate-key group the pairing
    # between v4 rows and CSV rows is arbitrary. That is only material when
    # the group spans >1 distinct borehole file; report it honestly.
    grp = csvk.groupby(keys, sort=False)["boring_file"].nunique()
    dup_multi = grp[grp > 1]
    n_amb_groups = int(len(dup_multi))
    n_amb_rows = int(csvk.set_index(keys).index.isin(dup_multi.index).sum()) if n_amb_groups else 0
    LOG.info(
        "identity-ambiguous groups (identical measured values, >1 file): "
        "%d groups / %d rows (%.3f%% of corpus)",
        n_amb_groups, n_amb_rows, 100.0 * n_amb_rows / max(len(csvk), 1),
    )

    out_df = merged[[*v4.columns, "boring_file", "dtd_version"]].copy()
    out_df["boring_file"] = out_df["boring_file"].astype("category")
    out_df["dtd_version"] = out_df["dtd_version"].astype(str).astype("category")

    # ---- equality gate (now structural; assert as a regression guard) -----
    LOG.info("verifying v4id == v4 on all %d v4 columns ...", len(v4.columns))
    for c in v4.columns:
        av, bv = v4[c].to_numpy(), out_df[c].to_numpy()
        if np.issubdtype(av.dtype, np.floating):
            same = np.array_equal(av, bv, equal_nan=True)
        else:
            same = np.array_equal(av, bv)
        if not same:
            raise SystemExit(f"column {c!r} differs from v4; aborting.")
    LOG.info("equality gate PASSED: v4id is v4 + identity, nothing else changed")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    LOG.info(
        "wrote %s: %d rows x %d cols, %d distinct boreholes, dtd distribution %s",
        out_path, len(out_df), out_df.shape[1], out_df["boring_file"].nunique(),
        out_df["dtd_version"].value_counts().to_dict(),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v4", type=Path,
                    default=REPO / "data/features/borings_japan_v4.parquet")
    ap.add_argument("--csv", type=Path,
                    default=REPO / "data/outputs/location_n_values.csv")
    ap.add_argument("--out", type=Path,
                    default=REPO / "data/features/borings_japan_v4id.parquet")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    return build(args.v4, args.csv, args.out)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
