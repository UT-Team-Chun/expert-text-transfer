# expert-text-transfer

**Companion repository for the paper**:

> Okauchi, R. and Chun, P.-J. (2026). _Fine-grained geological descriptions
> enable cross-region prediction of penetration resistance across national
> borehole archives._ Submitted to _Nature Communications_.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Zenodo code-snapshot DOI badge added at submission -->

This repository is the **canonical reproduction package** for every numerical
result, table and figure in the paper. The paper makes a single claim with a
cautionary contrast: the expert's free-text description of what was observed
— recorded in nearly every borehole, field and survey log yet discarded by
predictive models — carries information that **transfers to out-of-region
prediction of a physical quantity** (standard penetration resistance, SPT-N),
established on two independent national borehole archives (Japan KuniJiban,
2.66 M records; UK BGS) and reproduced in a maximally different domain (US
storm reports), **whereas learned spatial coordinates memorize location and
do not transfer**.

It is the sister repository of
[`kanto-calibrated-spt-prior`](https://github.com/UT-Team-Chun/kanto-calibrated-spt-prior)
(the single-region precursor paper); it contains only the code, tests and
result-provenance artefacts needed to reproduce this paper's results,
independent of the research monorepo in which they were developed.

## Quick reference: paper headline numbers ↔ provenance files

Every number below is quoted in the manuscript; each traces to a checked-in
JSON produced by a script in this repository (see `REPRODUCIBILITY.md` for the
full number → command → file map, and `prereg/` for the pre-registration the
confirmatory analyses were run against).

### Primary analysis — borehole-block permutation null (pre-registered)

| Headline number (paper) | Canonical file |
|---|---|
| **Japan content effect: region mean −12.7 %, pooled −11.3 %, BCa 95 % [−11.5, −11.2] %, 8 of 8 regions negative, per-region −8.0 % to −16.1 %** | `results/2026-08-18_grouped_null_japan_fullpop.json` |
| Japan permutation stage −14.0 %, Bonferroni-min p 0.008, Cauchy p 0.0010 | `results/2026-08-18_grouped_null_japan_s42.json` |
| Robustness to the subsample draw, −12.5 % to −14.0 % | `results/2026-08-18_grouped_null_japan_s{42,43,44}.json` |
| **UK replication: region mean −6.5 %, pooled −4.0 %, BCa [−5.0, −2.9] %, 5 of 5, Bonferroni-min p 0.005** | `results/2026-08-18_grouped_null_uk.json` |
| Unit-of-permutation sensitivity: block minus row −0.59 pp (JP) / −0.47 pp (UK) | same two files (`summary.delta_block_minus_row_pp`) |
| **Depersonalisation control** (proper nouns, headers, project templates stripped): −13.775 %, 8 of 8, attenuation +0.26 pp against a 5 pp bar | `results/2026-08-18_grouped_null_japan_pt5_deperson.json` |
| Provenance folds: client −14.33 % (7/7), year −12.80 % (6/6), contractor −10.52 % (8/8), schema −10.42 % (5/5) | `results/2026-08-18_provenance_folds_japan.json` |
| **Coordinate-free arm** (Fourier off *and* residual geo kernel off): mean −0.19 %, 4 regions | `results/2026-08-18_pt9_coordfree.json` |
| In-distribution identity-join effect −25.0 % (RMSE 8.505 ± 0.162 vs 11.333 ± 0.212) | `results/2026-08-14_pt7_identity_join_3fold.json` |
| Borehole-budgeted few-shot curve (text > no-text at every budget, both directions) | `results/2026-08-12_fewshot_borehole_curve.json` |
| Conformal coverage under row / borehole / site calibration splits, with width and Winkler | `results/2026-08-12_conformal_grouped_split{,_lro}.json` |
| Descriptor families, text-derived-only parser −17.5 % (8/8, 66 features) | `results/2026-08-12_descriptor_families_{japan,uk}.json` |
| Borehole-identity join audit (the coordinate join is not a valid key) | `results/2026-08-11_join_audit.json` |

### Earlier rounds (still reported; superseded as the headline)

| Headline number (paper) | Canonical file |
|---|---|
| Row-level within-class null −12.6 % (JP) / −9.3 % (UK); global −20.9 % / −9.6 % | `results/2026-06-24_within_region_null_{japan,uk}.json` |
| Lithology-only (5-seed leakage battery) −21.1 % / −10.6 %; region-bootstrap CIs | `results/2026-06-21_text_leakage_{japan,uk}.json`, `results/2026-06-21_region_bootstrap_ci.json` |
| Uniform-protocol full text −21.2 % (JP) / −18.8 % (UK); capacity nulls −7.8 % / −2.7 % | `results/2026-06-21_{japan,uk}_transfer_leakproof.json` |
| Structured lithology parser (no LM) −16.2 % / −9.2 % | `results/2026-06-23_structured_litho_baseline.json` |
| Frozen-LM under geological folds: leave-lithology −10.9 %, leave-era −13.5 %; regime ablation −21.1 % → −19.0 % | `results/2026-06-24_geo_fold_lm_and_regime_ablation.json` |
| Depth-band control −16.9 / −20.7 / −24.3 / −20.3 % (all 8/8) | `results/2026-06-24_depth_band_japan.json` |
| **National DKL coordinate 2×2: coords hurt OOR in 8/8 regions (mean +16.2 %)** | `results/2026-06-29_national_dkl_coord_ablation.json` |
| Model-agnostic HGB coordinate ablation +15.3 % (JP) / +4.3 % (UK) | `results/2026-06-23_coord_ablation_hgb.json` |
| Storm cross-domain control −35.2 % full / **−10.6 % size-stripped** (28/30 states) vs −0.6 % null | `results/2026-06-21_storm_transfer_{3rd_domain,nosize}.json` |
| Target harmonisation (z-score / quantile) robustness | `results/2026-06-23_target_harmonization.json` |
| Geological-split same-rows control −14.8 % | `results/2026-06-23_geological_split_subset_control.json` |
| Cross-experiment single source of truth | `results/2026-06-23_factorial_table.md` |

## What changed since the earlier draft

Two corrections to the statistical machinery; both are recorded in
`prereg/2026-08-14_nc_text_prereg_amendment_1.md` and the numbers above are
all post-correction.

1. **The borehole-block permutation null is now a bijection.** Its
   predecessor drew a donor borehole by free permutation and then *clipped*
   the donor's text block to the recipient's length, repeating the donor's
   deepest layer whenever the recipient was longer — 42.7 % of null-arm rows
   duplicated, 22,547 source rows never used. That is not a permutation of
   the embedding matrix, so the observed statistic was not exchangeable with
   the null draws, and every effect it touched was inflated by 0.6–1.4 pp.
   `national/evaluation/grouped_null.py` now performs a length-matched whole-
   block derangement with a rank-wise residual; every draw is checked and
   every artefact records `all_draws_bijective`. The defective routine is
   kept, unused, as `legacy_clipped_block_permutation` so the bias can be
   measured — `tests/national/test_grouped_null.py` asserts that the new
   routine returns a permutation and that the legacy one does not.
2. **Fold p-values are combined by Bonferroni-corrected minimum and Cauchy.**
   Leave-one-region-out folds share most of their training rows, so they are
   dependent and Stouffer (which assumes independence) is anti-conservative.
   Both reported combinations are valid under arbitrary dependence. Intervals
   are borehole-block BCa over paired per-borehole losses using the full
   leave-one-out jackknife, with the null accumulated over all draws.

## Layout

```
scripts/    entry points (python -m scripts.<name> from the repo root)
national/   library code (models, evaluation harnesses, calibration, data tools)
tests/      unit tests for the harness code (pytest tests/ -q)
results/    result-provenance JSONs for every manuscript number
prereg/     the pre-registration, its amendment, and the verdicts (verbatim)
manifest.json  sha256 of every file at assembly time
```

Key entry points:

- `scripts/nc_grouped_null.py` — the pre-registered primary analysis:
  borehole-block permutation null with the (1+r)/(1+n) correction, seed
  pairing, Bonferroni/Cauchy fold combination and borehole-block BCa
  intervals, on the primitive in `national/evaluation/grouped_null.py`.
  `--strip-mode lithology_only_depersonalised` is the depersonalisation
  control; `--regions/--shard/--combine` shard it by held-out region.
- `scripts/nc_provenance_folds.py` — leave-project-, leave-contractor-,
  leave-client-, leave-year- and leave-schema-out evaluation.
- `scripts/nc_fewshot_curve.py` — borehole-budgeted few-shot curve over a
  holdout shared by every budget, arm and seed.
- `scripts/nc_descriptor_families.py` — descriptor-family mechanism, on the
  parser in `scripts/text_leakage_controls.py` (vocabulary strips,
  dictionary, structured lithology parser, character n-grams).
- `scripts/attach_identity_to_parquet.py`,
  `scripts/extract_kunijiban_metadata.py`, `scripts/audit_text_join.py` —
  the borehole identity spine, the national archive-header pass across all
  six DTD generations, and the join audit.
- `scripts/nc_null_controls.py` — the **earlier** row-level
  shuffled-embedding nulls that the block null supersedes; retained because
  the paper reports the row-vs-block contrast.
- `scripts/uk_transfer_test.py`, `scripts/japan_transfer_test.py`,
  `scripts/storm_transfer_test.py` — the three-domain leave-region-out
  transfer harnesses (leak-proof per-fold PCA, multi-seed);
  `scripts/nc_geo_ablation.py`, `scripts/nc_depth_band.py` —
  geological-fold and depth-band robustness.
- `scripts/train_kanto_smoke.py --leave-region <r> [--zero-fourier]
  [--no-residual-geo]` — the from-scratch national DKL coordinate-ablation
  cells, including the genuinely coordinate-free arm;
  `scripts/train_lmc_national.py` — multi-task LMC training.
- `scripts/run_mondrian_recal_grouped.py` — per-regime split-conformal
  recalibration under row-, borehole- and site-grouped calibration splits,
  with interval width and Winkler score alongside coverage;
  `scripts/run_mondrian_recal_lmc.py` — the LMC counterpart.
  `national/evaluation/calibration.py` implements the Mondrian quantiles
  described in Methods.
- `scripts/build_paper2_figs.py`, `scripts/build_forest_plot.py`,
  `scripts/build_mondrian_recal_appendix_table.py` — figures and SI tables.

## Data availability

Raw KuniJiban borehole XML is **not redistributable** by us; it is publicly
downloadable from https://www.kunijiban.pref.hokkaido.lg.jp / the national
KuniJiban portal (see the paper's Data Availability). UK BGS AGS borehole
records are available under BGS terms from the BGS AGS download service.
NOAA storm reports are US public domain. Derived, redistributable artefacts
(embeddings, calibration tables, predictions, the trained national model)
are archived on Zenodo (DOI in the paper's Data Availability statement).
The `results/` artefacts in this repository let you verify the paper's
statistical results — every effect size, interval, p-value, fold count and
coverage figure — without downloading any raw data. The trained-model tables
(the model hierarchy, the per-region leave-region-out detail, the
test-time-adaptation sweep and the 23-cell Mondrian recalibration tables) are
read from per-run artefacts in the Zenodo bundle; `results/` carries the
collated written record of those, and `REPRODUCIBILITY.md` §14 states exactly
which numbers fall outside this repository and where they live.

## Environment

```bash
pip install -e .          # or: uv sync
pytest tests/ -q          # 294 tests, CPU-only, ~30 s
```

Five of those tests skip without the raw corpora (three leave-region-out
cases need the administrative-polygon lookup and the Kanto corpus, two
leakage cases need the real text corpus); the other 289 run on synthetic
fixtures and must pass for the build to succeed.

Transfer/leakage/null experiments are CPU-only (hours; the full-population
grouped-null shards are 1–3 h each); the national DKL coordinate-ablation
cells require a GPU (~12 h/cell at n_inducing=6000).

## License

MIT (code). See `LICENSE`. Cite via `CITATION.cff`.
