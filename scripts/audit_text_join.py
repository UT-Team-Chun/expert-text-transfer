#!/usr/bin/env python
"""Join audit: quantify the rounded-coordinate text-join defect (R0-2).

NC pre-review P0-6. Reproduces, from the repo's own artefacts, every number
behind the decision to replace the 4-decimal rounded-coordinate text join
with the exact borehole-identity join, and measures the data-level delta
between the two joins. No model is trained here.

Sections of the output JSON:

- ``corpus_keys``     -- how many 10 m keys exist, how many hold >1 borehole,
                         how many boreholes share byte-identical float32
                         coordinates (inseparable by any rounding).
- ``collision_mix``   -- severity: among multi-borehole keys, how many mix
                         different projects / contractors / survey years /
                         DTD versions (exhaustive, using the national
                         metadata parquet -- not a sample).
- ``join_delta``      -- the decisive numbers: both joins simulated on the
                         full v4id x layers-CSV cross; % of coordinate-join
                         matches whose description came from a DIFFERENT
                         borehole; agreement rate between the two joins.
- ``boundary_sensitivity`` -- fraction of ID-join matches within 0.1 / 0.25 /
                         0.5 m of a layer boundary (reviewer's robustness
                         ask).

CLI::

    cd backend
    .venv/bin/python -m scripts.audit_text_join \
        --out ../docs/research/2026-08-11_join_audit.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("scripts.audit_text_join")

REPO = Path(__file__).resolve().parents[2]


def _file_key(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.replace("\\", "/", regex=False)
        .str.rsplit("/", n=1).str[-1]
    )


def _load_inputs(v4id: Path, layers_csv: Path, metadata: Path):
    LOG.info("loading %s", v4id)
    boring = pd.read_parquet(
        v4id, columns=["latitude_deg", "longitude_deg", "depth_from_surface",
                       "boring_file"])
    boring["boring_file"] = _file_key(boring["boring_file"])
    LOG.info("loading %s", layers_csv)
    layers = pd.read_csv(
        layers_csv,
        usecols=["file_path", "latitude_deg", "longitude_deg", "layer_idx",
                 "depth_top_m", "depth_bottom_m"],
        dtype={"file_path": str},
    )
    layers["boring_file"] = _file_key(layers["file_path"])
    # Mirror the production pipeline: the embedding parquet stores the layer
    # depths as float32 (matching the boring parquet), so the audit joins in
    # float32 too.
    for c in ("depth_top_m", "depth_bottom_m"):
        layers[c] = layers[c].astype(np.float32)
    LOG.info("loading %s", metadata)
    meta = pd.read_parquet(metadata)
    return boring, layers, meta


def audit_corpus_keys(boring: pd.DataFrame) -> dict:
    """Rounded-key and byte-identical collision rates over the SPT corpus."""
    per_file = (
        boring.groupby("boring_file", observed=True)
        .agg(lat=("latitude_deg", "first"), lon=("longitude_deg", "first"),
             n_rows=("latitude_deg", "size"))
        .reset_index()
    )
    per_file["lat_r"] = per_file["lat"].astype("float64").round(4)
    per_file["lon_r"] = per_file["lon"].astype("float64").round(4)

    by_key = per_file.groupby(["lat_r", "lon_r"], observed=True)["boring_file"].nunique()
    multi = by_key[by_key > 1]
    files_under_multi = per_file.set_index(["lat_r", "lon_r"]).index.isin(multi.index)
    rows_under_multi = (
        per_file.loc[files_under_multi.nonzero()[0], "n_rows"].sum()
        if len(multi) else 0
    )

    by_exact = per_file.groupby(["lat", "lon"], observed=True)["boring_file"].nunique()
    exact_multi = by_exact[by_exact > 1]
    files_exact_multi = int(
        per_file.set_index(["lat", "lon"]).index.isin(exact_multi.index).sum()
    ) if len(exact_multi) else 0

    return {
        "n_boreholes": int(per_file.shape[0]),
        "n_rounded_keys": int(len(by_key)),
        "n_keys_multi_borehole": int(len(multi)),
        "pct_keys_multi_borehole": round(100.0 * len(multi) / len(by_key), 2),
        "n_boreholes_under_multi_key": int(files_under_multi.sum()),
        "pct_boreholes_under_multi_key": round(
            100.0 * files_under_multi.sum() / len(per_file), 2),
        "n_spt_rows_under_multi_key": int(rows_under_multi),
        "pct_spt_rows_under_multi_key": round(
            100.0 * rows_under_multi / boring.shape[0], 2),
        "n_exact_coord_groups_multi_borehole": int(len(exact_multi)),
        "n_boreholes_sharing_exact_coords": files_exact_multi,
        "pct_boreholes_sharing_exact_coords": round(
            100.0 * files_exact_multi / len(per_file), 2),
        "max_boreholes_one_key": int(by_key.max()),
    }


def audit_collision_mix(boring: pd.DataFrame, meta: pd.DataFrame) -> dict:
    """Among multi-borehole keys: what fraction mix provenance groups?

    Exhaustive over the corpus (an earlier pass sampled 1,200 keys).
    """
    per_file = (
        boring.groupby("boring_file", observed=True)
        .agg(lat=("latitude_deg", "first"), lon=("longitude_deg", "first"))
        .reset_index()
    )
    per_file["lat_r"] = per_file["lat"].astype("float64").round(4)
    per_file["lon_r"] = per_file["lon"].astype("float64").round(4)
    joined = per_file.merge(
        meta[["boring_file", "project_key", "surveyor_name", "survey_year",
              "dtd_version"]],
        on="boring_file", how="left",
    )
    grp = joined.groupby(["lat_r", "lon_r"], observed=True)
    sizes = grp["boring_file"].nunique()
    multi_idx = sizes[sizes > 1].index
    multi = joined.set_index(["lat_r", "lon_r"]).loc[multi_idx].reset_index()
    g = multi.groupby(["lat_r", "lon_r"], observed=True)

    def _mix_rate(col: str) -> float:
        nun = g[col].nunique(dropna=True)
        return round(100.0 * float((nun > 1).mean()), 2)

    return {
        "n_multi_borehole_keys": int(len(multi_idx)),
        "pct_keys_mixing_projects": _mix_rate("project_key"),
        "pct_keys_mixing_contractors": _mix_rate("surveyor_name"),
        "pct_keys_mixing_survey_years": _mix_rate("survey_year"),
        "pct_keys_mixing_dtd_versions": _mix_rate("dtd_version"),
    }


def _simulate_join(boring: pd.DataFrame, layers: pd.DataFrame,
                   mode: str) -> pd.Series:
    """Reproduce join_soil_text_to_parquet's merge for one mode, attaching a
    layer label `(boring_file, layer_idx)` instead of embeddings. Returns a
    Series of labels aligned to `boring`'s row order (NaN where unmatched).
    """
    b = boring.reset_index(drop=False)  # keep original row id
    la = layers.copy()
    la["_label"] = la["boring_file"] + "#" + la["layer_idx"].astype(str)
    if mode == "file":
        b["_grp"] = b["boring_file"]
        la["_grp"] = la["boring_file"]
        by = ["_grp"]
    elif mode == "coord":
        b["lat_r"] = b["latitude_deg"].astype("float64").round(4)
        b["lon_r"] = b["longitude_deg"].astype("float64").round(4)
        la["lat_r"] = la["latitude_deg"].astype("float64").round(4)
        la["lon_r"] = la["longitude_deg"].astype("float64").round(4)
        by = ["lat_r", "lon_r"]
    else:  # pragma: no cover
        raise ValueError(mode)

    b = b.sort_values("depth_from_surface")
    la = la.sort_values("depth_top_m")
    merged = pd.merge_asof(
        b, la[by + ["depth_top_m", "depth_bottom_m", "_label", "boring_file"]]
        .rename(columns={"boring_file": "_src_file"}),
        by=by, left_on="depth_from_surface", right_on="depth_top_m",
        direction="backward", allow_exact_matches=True,
    )
    bad = merged["depth_from_surface"] >= merged["depth_bottom_m"]
    merged.loc[bad.fillna(True), ["_label", "_src_file"]] = np.nan
    merged = merged.sort_values("index").reset_index(drop=True)
    return merged[["_label", "_src_file"]]


def audit_join_delta(boring: pd.DataFrame, layers: pd.DataFrame) -> dict:
    LOG.info("simulating legacy coordinate join ...")
    coord = _simulate_join(boring, layers, "coord")
    LOG.info("simulating exact identity join ...")
    fil = _simulate_join(boring, layers, "file")

    n = len(boring)
    c_matched = coord["_label"].notna()
    f_matched = fil["_label"].notna()
    # Mis-attachment: coordinate-join matches whose source borehole is not
    # the row's own borehole.
    own = boring["boring_file"].reset_index(drop=True)
    mis = c_matched & (coord["_src_file"] != own)
    both = c_matched & f_matched
    label_diff = both & (coord["_label"] != fil["_label"])

    return {
        "n_rows": int(n),
        "coord_join_matched": int(c_matched.sum()),
        "coord_join_match_rate_pct": round(100.0 * c_matched.mean(), 2),
        "file_join_matched": int(f_matched.sum()),
        "file_join_match_rate_pct": round(100.0 * f_matched.mean(), 2),
        "coord_matches_from_other_borehole": int(mis.sum()),
        "pct_coord_matches_from_other_borehole": round(
            100.0 * mis.sum() / max(int(c_matched.sum()), 1), 2),
        "rows_matched_by_both": int(both.sum()),
        "rows_label_changed_between_joins": int(label_diff.sum()),
        "pct_label_changed_among_both": round(
            100.0 * label_diff.sum() / max(int(both.sum()), 1), 2),
        "coord_only_matches": int((c_matched & ~f_matched).sum()),
        "file_only_matches": int((f_matched & ~c_matched).sum()),
    }


def audit_boundary_sensitivity(boring: pd.DataFrame,
                               layers: pd.DataFrame) -> dict:
    """Among ID-join matches, how close do test depths sit to layer
    boundaries? (Reviewer ask: +-0.1 / 0.25 / 0.5 m exclusion robustness.)"""
    b = boring.reset_index(drop=False)
    la = layers.sort_values("depth_top_m")
    b = b.sort_values("depth_from_surface")
    merged = pd.merge_asof(
        b, la[["boring_file", "depth_top_m", "depth_bottom_m"]],
        by="boring_file", left_on="depth_from_surface", right_on="depth_top_m",
        direction="backward", allow_exact_matches=True,
    )
    ok = merged["depth_from_surface"] < merged["depth_bottom_m"]
    m = merged[ok.fillna(False)]
    dist = np.minimum(
        (m["depth_from_surface"] - m["depth_top_m"]).to_numpy(),
        (m["depth_bottom_m"] - m["depth_from_surface"]).to_numpy(),
    )
    out = {"n_id_join_matches": int(len(m))}
    for eps in (0.1, 0.25, 0.5):
        out[f"pct_within_{eps}m_of_boundary"] = round(
            100.0 * float((dist < eps).mean()), 2)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v4id", type=Path,
                    default=REPO / "data/features/borings_japan_v4id.parquet")
    ap.add_argument("--layers-csv", type=Path,
                    default=REPO / "data/features/derived/soil_text_layers.csv")
    ap.add_argument("--metadata", type=Path,
                    default=REPO / "data/features/derived/kunijiban_metadata.parquet")
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/research/2026-08-11_join_audit.json")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")

    boring, layers, meta = _load_inputs(args.v4id, args.layers_csv, args.metadata)
    result = {
        "_provenance": {
            "purpose": (
                "NC pre-review P0-6: quantify the rounded-coordinate text-join "
                "defect and the delta to the exact borehole-identity join; "
                "basis for switching join_soil_text_to_parquet to "
                "--join-key file and for the SI join-audit section."),
            "inputs": {
                "v4id": str(args.v4id), "layers_csv": str(args.layers_csv),
                "metadata": str(args.metadata)},
            "script": "backend/scripts/audit_text_join.py",
            "no_model_trained": True,
        },
        "corpus_keys": audit_corpus_keys(boring),
        "collision_mix": audit_collision_mix(boring, meta),
        "join_delta": audit_join_delta(boring, layers),
        "boundary_sensitivity": audit_boundary_sensitivity(boring, layers),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    LOG.info("wrote %s", args.out)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
