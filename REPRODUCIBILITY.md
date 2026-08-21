# REPRODUCIBILITY — manuscript number → command → provenance file

All commands run from the repository root (`python -m scripts.<name>`). CPU
unless a block says GPU. Data prerequisites are listed per block; the
checked-in `results/` JSONs are the canonical outputs those commands
regenerate, and every headline number in the paper is read out of one of them.

**Read this first.** The paper's primary content-vs-capacity test is the
**borehole-block permutation null** (`scripts/nc_grouped_null.py`, on
`national/evaluation/grouped_null.py`), pre-registered before the
computations were run. The **row-level** shuffled-embedding null
(`scripts/nc_null_controls.py`, §10 below) is the *earlier* work that it
supersedes; it still ships, and the paper still reports it, because the
row-vs-block contrast is itself a pre-registered sensitivity check (P-T3).
Do not read §10 as the headline result.

Two corrections that a reader comparing against an earlier draft will notice:

1. **The block null is now a bijection.** Its predecessor assigned a donor
   borehole by free permutation and then *clipped* the donor's text block to
   the recipient's length, so a recipient longer than its donor received the
   donor's deepest layer repeated — 42.7 % of null-arm rows duplicated on the
   P-T1 population. That is not a permutation of the embedding matrix, and it
   biased every effect it touched (by 0.6–1.4 pp). The repaired scheme is a
   length-matched whole-block derangement with a rank-wise residual;
   `block_permutation_indices` returns a permutation of `arange(n)` on every
   draw, and each artefact records `all_draws_bijective`. The defective
   routine is retained, unused, as `legacy_clipped_block_permutation` so the
   bias can be measured. **The `results/` directory contains only
   corrected-machinery artefacts**; the 2026-08-12/2026-08-13 grouped-null and
   provenance JSONs produced by the predecessor are deliberately not shipped.
2. **Fold p-values are combined by Bonferroni-corrected minimum and Cauchy,
   not Stouffer.** Leave-one-region-out folds are dependent — any two of eight
   share 6/8 of their training rows — and Stouffer assumes independence, so it
   is anti-conservative here. Bonferroni-min and Cauchy are valid under
   arbitrary dependence. Stouffer is still written to the artefacts as a
   secondary figure; it is not quoted in the paper. Intervals are
   borehole-block **BCa** over paired per-borehole losses, using the full
   leave-one-out jackknife, with the null accumulated over all draws.

The pre-registration, its amendment and the verdicts recorded against it are
in `prereg/`, verbatim. The verdicts document links three companion notes and
one JSON by bare filename; those live in `results/`
(`2026-08-12_pt{6,8,10}_*.md`, `2026-08-14_pt7_identity_join_3fold.json`).

---

## 0. Headline numbers at a glance

| Manuscript number | Artefact (`results/`) | JSON key |
|---|---|---|
| Japan content effect, region mean **−12.7 %** | `2026-08-18_grouped_null_japan_fullpop.json` | `summary.content_pct_block_mean` |
| Japan content effect, pooled **−11.3 %**, BCa 95 % **[−11.5, −11.2] %** | same | `summary.borehole_block_bootstrap.{theta_hat_pct,bca_95}` |
| Japan per-region range **−8.0 % to −16.1 %**, **8 of 8** negative | same | `summary.region_spread_block`, `summary.regions_negative_block` |
| Japan subsample effect **−14.0 %**, Bonferroni-min p **0.008**, Cauchy p **0.0010** | `2026-08-18_grouped_null_japan_s42.json` | `summary.{content_pct_block_mean,bonferroni_min_p_block,cauchy_p_block}` |
| Japan subsample-draw range **−12.5 % to −14.0 %** | `..._s42/_s43/_s44.json` | `summary.content_pct_block_mean` (−14.031 / −13.077 / −12.507) |
| UK region mean **−6.5 %**, pooled **−4.0 %**, BCa **[−5.0, −2.9] %**, **5 of 5**, Bonferroni-min p **0.005** | `2026-08-18_grouped_null_uk.json` | `summary.*` |
| Block-minus-row sensitivity **−0.59 pp** (JP) / **−0.47 pp** (UK) | `..._japan_s42.json`, `..._uk.json` | `summary.delta_block_minus_row_pp` |
| Depersonalisation control **−13.775 %** (8/8), attenuation **+0.26 pp** | `2026-08-18_grouped_null_japan_pt5_deperson.json` | `summary.content_pct_block_mean` |
| Provenance folds: client **−14.33 %** (7/7), year **−12.80 %** (6/6), contractor **−10.52 %** (8/8), DTD **−10.42 %** (5/5) | `2026-08-18_provenance_folds_japan.json` | `families.<fam>.{mean_content_pct_block,units_negative}` |
| Provenance folds: project **−9.77 %** (8/8), full population, 12,678 rows | `project__p00..p07-*.json` (8 files) | `families.project.per_fold.<label>.content_pct_block` |
| Coordinate-free arm: mean **−0.19 %**, max \|Δ\| **1.28 %**, 4 regions | `2026-08-18_pt9_coordfree.json` | `per_region.<r>.pct_B_minus_A` |
| In-distribution identity-join effect **−25.0 %** (RMSE 8.505 ± 0.162 vs 11.333 ± 0.212) | `2026-08-14_pt7_identity_join_3fold.json` | `effect.rmse_pct`, `text_arm.*`, `no_text_baseline.*` |
| Few-shot curve, text > no-text at every budget, both directions | `2026-08-12_fewshot_borehole_curve.json` | `{japan_to_uk,uk_to_japan}.fewshot_curve` |
| Conformal coverage under row / borehole / site calibration splits | `2026-08-12_conformal_grouped_split.json` | `splits.<unit>.alpha_0.95.{coverage,gap,mean_width,winkler}` |
| Descriptor families: text-only parser **−17.5 %** (8/8), 66 features | `2026-08-12_descriptor_families_japan.json` | `arms.parser_text_only` |
| Join audit (coordinate join vs identity join) | `2026-08-11_join_audit.json` | `corpus_keys`, `collision_mix`, `join_delta` |

Earlier-round numbers (transfer harnesses, leakage battery, geological folds,
depth bands, coordinate ablations, storm domain) are mapped in §7–§12.

---

## 1. Primary content-vs-capacity test — borehole-block permutation null (P-T1/P-T2/P-T3)

The pre-registered protocol is two-stage: the **permutation p-value** comes
from a balanced 500-files-per-region subsample (1000 block draws per fold);
the **point estimate and interval** come from the full text-bearing
population (1,298,728 Japan rows). Both stages are shipped.

Baseline = richest non-text ladder rung (depth, elevation, river/coast
distance, AIST regime/litho-macro/era one-hots, train-side KNN spatial prior,
own borehole excluded; no raw lat/lon). Text arm = strength-stripped (v2)
lithology-only embedding, per-fold PCA fit on training regions only. Null
stratum = region × lithology-macro (Japan) / region (UK — the UK archive
carries no AIST code; the artefact records which strata were actually used).

```bash
# --- Japan, permutation stage, three independent subsample draws
for S in 42 43 44; do
  python -m scripts.nc_grouped_null --domain japan \
      --per-region-files 500 --sample-seed $S \
      --n-perm-block 1000 --n-perm-row 200 \
      --out results/2026-08-18_grouped_null_japan_s${S}.json
done

# --- Japan, full text-bearing population (point estimate + BCa interval).
# Sharded one process per held-out region, then combined by the same code a
# single-process run uses.
for R in hokkaido tohoku kanto chubu kansai chugoku shikoku kyushu_okinawa; do
  python -m scripts.nc_grouped_null --domain japan --per-region-files 0 \
      --regions $R --shard --n-perm-block 10 --n-perm-row 10 \
      --out shards/japan_full_${R}.json
done
python -m scripts.nc_grouped_null --combine shards/ \
    --out results/2026-08-18_grouped_null_japan_fullpop.json

# --- UK replication (full UK text-bearing population)
python -m scripts.nc_grouped_null --domain uk \
    --n-perm-block 1000 --n-perm-row 200 \
    --out results/2026-08-18_grouped_null_uk.json
```

`--strata-col` overrides the stratification (default
`region aist_litho_macro_code`; columns absent from the frame are dropped and
the ones used are recorded in the artefact). A row-level null runs alongside
every block null, which is where `summary.delta_block_minus_row_pp` — the
P-T3 sensitivity number — comes from; no separate run is needed.

Prereqs: `borings_japan_v4id.parquet` (§5), `soil_text_layers.csv`,
`uk_bgs_spt_full.parquet`. Runtime: subsample stage ~3 h/seed on 8 cores;
the full-population shards are ~1–3 h each.

**Verifying the null is the corrected one.** This is the single check that
distinguishes the shipped code from its predecessor:

```python
import numpy as np
from national.evaluation.grouped_null import (
    block_permutation_indices, legacy_clipped_block_permutation)

# ragged frame: boreholes of unequal length across several strata
lengths = [1,2,2,3,3,3,4,5,5,7,9,11,2,3,4,6,6,8,1,12]
g = np.array([f"bh{i:03d}" for i, L in enumerate(lengths) for _ in range(L)])
d = np.array([float(k) for L in lengths for k in range(L)])
s = np.array([f"r{i % 3}" for i, L in enumerate(lengths) for _ in range(L)])

idx, diag = block_permutation_indices(g, d, s, np.random.default_rng(0),
                                      return_diagnostics=True)
assert np.array_equal(np.sort(idx), np.arange(len(g)))   # a permutation
assert diag["is_bijection"]

bad, _ = legacy_clipped_block_permutation(g, d, s, np.random.default_rng(0),
                                          return_diagnostics=True)
assert not np.array_equal(np.sort(bad), np.arange(len(g)))  # it is NOT
```

`tests/national/test_grouped_null.py` asserts the same invariants over many
seeds and shapes.

## 2. Provenance-transfer folds (P-T4)

Leave-contractor-, leave-client-, leave-year- and leave-schema(DTD)-out. Same
engine, same arms; the fold family changes, the null's strata stay geographic.
Grouping metadata comes from the national archive-header pass (§5).

```bash
python -m scripts.nc_provenance_folds --list-folds        # enumerate units
python -m scripts.nc_provenance_folds \
    --per-region-files 500 --n-perm-block 500 --n-perm-row 200 --top-n 8 \
    --out results/2026-08-18_provenance_folds_japan.json

# sharded: one process per (family, fold), then combine
python -m scripts.nc_provenance_folds --families client \
    --fold-labels "<one label from --list-folds>" --shard --out shards/
python -m scripts.nc_provenance_folds --combine shards/ \
    --out results/2026-08-18_provenance_folds_japan.json
```

The combined artefact holds **four** families (client, year, contractor,
dtd), 26 shards, all with `all_draws_bijective: true`. The fifth family,
leave-project-out, has only one project clearing the 300-row floor on the
500-files-per-region subsample, so it is not in the combined file and is not
measurable at subsample scale at all. It runs against the **full text-bearing
population** instead, sharded one artefact per held-out project, at ten
permutation draws per fold (a point-estimate budget, so the family carries no
permutation `p`):

```bash
python -m scripts.nc_provenance_folds --families project \
    --per-region-files 0 --n-perm-block 10 --top-n 8 --shard --out shards/
```

All **eight** folds of the paper's leave-project-out table ship, one file
each, so every row of that table can be checked individually:

| Artefact | Held-out project (archive label) | `n_te` | `content_pct_block` |
| --- | --- | ---: | ---: |
| `results/project__p03-41f8797b.json` | 滝川河川事務所管内地質調査業務 | 1,727 | −21.313 |
| `results/project__p02-482f081d.json` | 札幌河川事務所管内地質調査業務 | 1,580 | −16.436 |
| `results/project__p00-1cd5faba.json` | 地質調査業務 | 3,768 | −12.403 |
| `results/project__p04-553e5e57.json` | 大野油坂道路大野・大野東区間地質調査業務 | 1,212 | −9.199 |
| `results/project__p07-cfa26457.json` | 石狩川上流美瑛川外堤防質的整備詳細点検業務 | 800 | −6.262 |
| `results/project__p05-f4e89d17.json` | 平成30年度板橋免地区外地質調査業務 | 830 | −5.514 |
| `results/project__p01-e3827f1b.json` | 斐伊川下流堤防浸透調査検討業務 | 1,992 | −4.909 |
| `results/project__p06-4df448a6.json` | 平成16年度施行　常呂川外堤防詳細点検業務 | 769 | −2.087 |

Eight of eight negative; 12,678 held-out rows; family mean **−9.77 %**. Each
shard self-certifies `all_draws_bijective: true`.

## 3. Depersonalisation control (P-T5)

Same protocol, different text arm: removed from each layer description are
(a) 2–16-character substrings that also occur in that borehole's own header
(project name, ordering agency, survey contractor), (b) 2–8-character
kanji/katakana runs immediately preceding an administrative or geographic
suffix, and (c) the maximal substring shared by ≥90 % of the layers of one
project (the template). Then the P-T1 protocol runs unchanged. The embedding
cache key is content-hashed, so the control gets its own cache entry
automatically.

```bash
python -m scripts.nc_grouped_null --domain japan \
    --strip-mode lithology_only_depersonalised \
    --per-region-files 500 --sample-seed 42 \
    --n-perm-block 1000 --n-perm-row 200 \
    --out results/2026-08-18_grouped_null_japan_pt5_deperson.json
```

Region mean **−13.775 %** against the frozen arm's −14.031 %, i.e. an
attenuation of **+0.26 pp** on average and **+0.75 pp** at worst, against a
pre-registered bar of ≤5 pp; **8 of 8** regions negative; Bonferroni-min
p 0.0080; pooled θ̂ −11.545 % with BCa 95 % [−12.27, −10.80]. The strip is a
control, not a deletion: it removes 5.68 % of characters and empties 71 of
1,298,783 layers (0.0055 %), with zero removals of the lithology terms
火山灰 / 安山 / 段丘 / 条線 / 区分.

The strip is implemented in `scripts/text_leakage_controls.py`
(`depersonalise_text` / `depersonalise_corpus`) and unit-tested in
`tests/national/test_text_leakage_controls.py`.

## 4. Borehole-budgeted few-shot curve (P-T6)

Evaluation holdout = a fixed seed-0 50 % split of target **boreholes**, shared
by every budget, arm and seed; adaptation budgets counted in boreholes
{0, 10, 25, 50, 100, 300}, drawn from the non-holdout pool only.

```bash
python -m scripts.nc_fewshot_curve \
    --out results/2026-08-12_fewshot_borehole_curve.json
```

## 5. Borehole identity spine, archive headers, join audit

The identity spine is a prerequisite for every analysis above: without it,
joins and resampling units fall back to rounded (lat, lon), which is not a
valid borehole key (9.0 % of 10 m keys hold more than one borehole).

```bash
# v4 + borehole identity, accepted only if it reproduces v4 byte-for-byte
python -m scripts.attach_identity_to_parquet \
    --v4 <borings_japan_v4.parquet> --csv <location_n_values.csv> \
    --out <borings_japan_v4id.parquet>

# archive headers across all six DTD generations (project/client/contractor/year/schema)
python -m scripts.extract_kunijiban_metadata \
    --xml-dir <kunijiban/xml> --out <kunijiban_metadata.parquet>

# how the rounded-coordinate join differs from the identity join (SI "Join audit")
python -m scripts.audit_text_join --out results/2026-08-11_join_audit.json
```

## 6. Conformal calibration under grouped splits (P-T8)

Re-evaluates the same prediction artefacts under three calibration split
units — `row` (the legacy split, which reproduces the published numbers),
`borehole`, and ~500 m `site` cells — reporting per-regime coverage, mean
interval width **and** Winkler score at α ∈ {0.5, 0.8, 0.95}. Coverage alone
can be bought with width, so both are shown.

```bash
python -m scripts.run_mondrian_recal_grouped --run dkl_national_full_v2 \
    --out results/2026-08-12_conformal_grouped_split.json
# the leave-region-out counterpart:
python -m scripts.run_mondrian_recal_grouped --run <lro run> \
    --out results/2026-08-12_conformal_grouped_split_lro.json
# the LMC per-regime recalibration of Methods:
python -m scripts.run_mondrian_recal_lmc --run-dir <trained cell with predictions.npz>
```

The calibration-budget curve (mean and worst-region coverage gap against the
number of calibration points, quoted in Results and the Conclusion) is
`results/2026-06-21_conformal_budget.json`; the per-regime cell tables of
Appendix C are collated in `results/2026-05-29_phase_c_mondrian_recal.md`.

## 7. Coordinates do not transfer (P-T9 + the national DKL 2×2) — GPU

`national/models/foundation.py` adds a raw-(lat, lon) Matérn-3/2 residual
kernel on the GP side independently of the encoder's Fourier path, so a
"Fourier OFF" arm was **not** coordinate-free. P-T9 adds the arm that is.

```bash
# P-T9: 4 regions x 2 arms; B differs from A only by --no-residual-geo
for R in chubu kansai kyushu_okinawa tohoku; do
  for ARM in "--zero-fourier" "--zero-fourier --no-residual-geo"; do
    python -m scripts.train_kanto_smoke --parquet <borings_japan.parquet> \
        --leave-region $R $ARM --kernel-type rbf --mean-type linear \
        --inducing-init random --n-inducing 6000 --n-epochs 50 --regime-one-hot \
        --output-dir runs/pt9_${R}_<arm>
  done
done
# held-out RMSE is computed from each run's predictions.npz (the trainer
# summary records only the in-region holdout); collated in
# results/2026-08-18_pt9_coordfree.json

# The earlier coordinate 2x2 over all 8 regions (coords hurt held-out RMSE in
# 8/8, mean +16.2 %), collated in
# results/2026-06-29_national_dkl_coord_ablation.json; the single-region
# from-scratch Kanto 2x2 that preceded it is
# results/2026-06-21_coord_ablation_trainfromscratch.json:
for R in hokkaido tohoku kanto chubu kansai chugoku shikoku kyushu_okinawa; do
  python -m scripts.train_kanto_smoke --parquet <borings_japan.parquet> \
      --leave-region $R --kernel-type rbf --mean-type linear \
      --inducing-init random --n-inducing 6000 --n-epochs 50 --regime-one-hot \
      --output-dir runs/coordabl_${R}_fourier_on
  # ... and the same with --zero-fourier into runs/coordabl_${R}_fourier_off
done
```

Both arms of a P-T9 pair must run on identical hardware: the same model,
config and split differed by 0.55 % between an H100 and an RTX 6000 Ada, and
the bar is a 2 % difference.

## 8. Descriptor-family mechanism (P-T10, exploratory)

Which observations carry the transferable signal, measured through the
language-model-free structured parser in its **text-derived-only**
configuration (`parser_text_only`, 66 features) alongside the legacy
configuration that mixed AIST archive codes in (`parser_with_codes`), so the
two are never conflated.

```bash
python -m scripts.nc_descriptor_families --domain japan \
    --out results/2026-08-12_descriptor_families_japan.json
python -m scripts.nc_descriptor_families --domain uk \
    --out results/2026-08-12_descriptor_families_uk.json
```

This arm deviates from the frozen protocol (thin baseline, raw lat/lon,
row-level null, un-stripped text); the pre-registration marks it exploratory
and the paper reports it as such.

## 9. In-distribution effect on the identity join (P-T7)

Spatially blocked 3-fold cross-validation, DKL + SVGP with a joint two-task
LMC head, full national corpus on the identity-join parquet. This is **not**
the transfer estimand of §1 — different model, different split, different
population; the two must never be quoted as one effect.

```bash
for F in 0 1 2; do
  python -m scripts.train_lmc_national \
      --parquet <borings_japan_v5id_sarashina_per_layer.parquet> \
      --kfold-test-fold $F --kfold-mesh-level 2 --num-latents 2 \
      --n-inducing 8000 --n-epochs 50 \
      --output-dir runs/dkl_national_lmc_v5id_sarashina_l2_kfold${F}
done
```

The shipped artefact was consolidated **by hand** from the three
`summary.json` files (no recomputation); its `_provenance.consolidated_from`
names them. Its `hardware_inhomogeneity` field records that fold 0 ran on an
RTX 6000 Ada and folds 1–2 on H100 NVL, and quantifies the resulting spread
(0.55 %, inside the 8.32–8.64 fold-to-fold range).

## 10. Earlier row-level nulls — superseded by §1

`scripts/nc_null_controls.py` shuffles embedding **rows** independently. That
over-fragments the within-borehole text block (layers of one borehole share a
logger, a template, a project and a stratigraphic context), which can make the
null too easy to beat. The paper reports these as the earlier round and as the
row arm of the P-T3 sensitivity contrast, not as the headline.

Row-level within-class / global nulls: −12.6 % (JP) / −9.3 % (UK) within-class;
−20.9 % / −9.6 % global; perm-p < 0.01 over 100 permutations.

```bash
python -m scripts.nc_null_controls --domain japan --representation lithology_only \
    --out results/2026-06-24_within_region_null_japan.json --n-perm 100
python -m scripts.nc_null_controls --domain uk --representation lithology_only \
    --out results/2026-06-24_within_region_null_uk.json --n-perm 100
# strip-vocabulary v2 reruns:
#   results/2026-07-04_within_null_v2_{japan,uk}.json
```
~1.5 h JP / ~25 min UK.

## 11. Leakage battery and the uniform three-domain transfer

Leakage battery (Table "Text-leakage controls"): lithology-only −21.1 % /
−10.6 %; hardness-only; dictionary ≈ 0; structured parser −16.2 % / −9.2 %;
char-TF-IDF −17.8 % / −14.2 %.

```bash
python -m scripts.text_leakage_controls --domain japan \
    --cache-dir <cache> --out results/2026-06-21_text_leakage_japan.json
python -m scripts.text_leakage_controls --domain uk \
    --cache-dir <cache> --out results/2026-06-21_text_leakage_uk.json
# v2 strip vocabulary: results/2026-07-04_leakage_v2_{japan,uk}.json
# strip audit:         results/2026-07-04_leakage_audit.json
```

Uniform three-domain transfer: JP −21.2 % (null −7.8 %); UK −18.8 % (null
−2.7 %); storm −35.2 % full / −10.6 % size-stripped vs −0.6 % null.

```bash
python -m scripts.japan_transfer_test --out results/2026-06-21_japan_transfer_leakproof.json
python -m scripts.uk_transfer_test --parquet <uk_bgs_spt_full.parquet> \
    --out results/2026-06-21_uk_transfer_leakproof.json --cache-dir <cache>
python -m scripts.storm_transfer_test --out results/2026-06-21_storm_transfer_3rd_domain.json
python -m scripts.storm_transfer_test --strip-size --out results/2026-06-21_storm_transfer_nosize.json
```

Region bootstrap CIs [−23.7, −18.3] % (JP) / [−14.7, −6.6] % (UK):
nonparametric bootstrap over held-out regions, 10⁴ resamples, percentile
intervals, of the per-region effects in `results/2026-06-21_text_leakage_*.json`;
collated in `results/2026-06-21_region_bootstrap_ci.json`.

## 12. Geological folds, depth bands, baseline ladder, missingness

Leave-lithology −10.9 % (12/13), leave-era −13.5 % (9/9); regime ablation
−21.1 → −19.0; depth bands −16.9 / −20.7 / −24.3 / −20.3 % (8/8 each).

```bash
python -m scripts.nc_geo_ablation --out results/2026-06-24_geo_fold_lm_and_regime_ablation.json
python -m scripts.nc_depth_band --domain japan --out results/2026-06-24_depth_band_japan.json
python -m scripts.nc_rich_baseline  --out results/2026-07-04_rich_baseline_ladder.json
python -m scripts.nc_knn_prior      --out results/2026-07-04_knn_prior_rung.json
python -m scripts.nc_cross_archive  --out results/2026-07-04_cross_archive_transfer.json
python -m scripts.nc_missingness    --out results/2026-07-04_missingness_ipw.json
# v2 reruns: results/2026-07-04_{geo_fold_v2,depth_band_v2_japan}.json
```

Model-agnostic HGB coordinate ablation, target harmonisation and the
same-rows control are collated in
`results/2026-06-23_{coord_ablation_hgb,target_harmonization,geological_split_subset_control,geological_province_split}.json`;
`results/2026-06-23_factorial_table.md` is the cross-experiment index.

## 13. Figures

```bash
# All figures except the forest plot (the flag is --out-dir, not --out).
# fig7/fig8 belong to a separate follow-on paper (the 3-D cube and the
# engineering decision maps) and are NOT part of this manuscript; do not pass
# --figures all, which would still try to build them.
python -m scripts.build_paper2_figs \
  --figures fig1_deployment fig2 fig3 fig4 fig5 fig6 fig9 fig10 fig11 \
            fig4_descriptors fig5_fewshot \
  --out-dir <figures dir>

# The per-unit forest plot (reads the result JSONs from results/).
python -m scripts.build_forest_plot --out <figures dir>/fig_forest_content.pdf

# The SI per-regime coverage tables (needs the per-run
# conformal_mondrian.json artefacts from the Zenodo bundle).
python -m scripts.build_mondrian_recal_appendix_table --runs-dir <runs dir> --out <tex path>
```

`build_paper2_figs` and `build_forest_plot` both look for their result JSONs
in `docs/research/` (monorepo layout) and then in `results/` (this
repository), so the commands above need no `--*-json` flags here.

## 14. What this repository cannot verify on its own

Stated plainly, because a reproduction package that overclaims is worse than
one with a known boundary.

- **Trained-model artefacts.** The three-tier model hierarchy, the
  three-model leave-region-out table, the per-region LRO detail, the
  test-time-adaptation sweep and the 23-cell Mondrian recalibration tables
  are *model outputs*: they are read from each run's `summary.json`,
  `predictions.npz` and `conformal_mondrian.json`. Those live in the Zenodo
  data bundle, not here. The result notes shipped in `results/`
  (`2026-05-27_phase_c_national_lro.md`,
  `2026-05-29_phase_c_dkl_lro_8way.md`,
  `2026-05-28_phase_c_dkl_v2_and_ablations.md`,
  `2026-05-29_phase_c_mondrian_recal.md`,
  `2026-06-01_phase_4_tta_24cell_sweep_results.md`,
  `2026-06-02_phase_3_ruri_l2_3fold.md`,
  `2026-06-19_lro_text_transfer_b1_decomposition.md`) record the collated
  values and the run directories they came from, so the tables can be
  checked against a written record even without the bundle.
- **`14.508` (GPBoost national LRO RMSE) and `11.194` (spatially blocked
  3-fold no-text RMSE)** are quoted in the manuscript but appear in no
  artefact shipped here; both are read from the monorepo's cross-project
  `results_table.md`, which is not part of this release.
- **The `97.06 %` text-coverage figure** of the identity-join population
  (1,314,638 of 1,354,412 rows) was recomputed directly from
  `borings_japan_v4id.parquet` × `soil_text_layers.csv` and was never
  written to an artefact. `results/2026-07-04_missingness_ipw.json` carries
  the earlier coordinate-join figure (`frac_text_bearing: 0.9746`) that it
  replaces.
- **Corpus, embedding and pipeline counts** in the Data section trace to the
  phase notes in `results/` rather than to machine-readable JSON.
- The binomial sign-test p-values in the storm SI are analytic; the storm
  artefacts store `sign_test_p_one_sided: 0.0` (below float resolution).

## 15. Integrity

```bash
pytest tests/ -q            # in-tree test run; must pass for the build to succeed
python - <<'PY'             # every shipped file matches manifest.json
import hashlib, json, pathlib
m = json.load(open("manifest.json"))
for rel, meta in m.items():
    h = hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest()
    assert h == meta["sha256"], rel
print(len(m), "files verified")
PY
```

`manifest.json` describes the tree exactly: every file in the repository is
listed, and every listed file is present (the builder asserts both directions
and removes its own `__pycache__`/`.pytest_cache` by-products before
finishing).
