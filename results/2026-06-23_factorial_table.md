# 2026-06-23 — Factorial table (SINGLE SOURCE OF TRUTH; v2 refresh 2026-07-06)

Every transfer number in the manuscript must trace to a cell here.
**2026-07-06 refresh:** all strip-dependent cells recomputed under the extended v2 strip
vocabulary (`STRIP_VOCAB_VERSION="v2"`; see `2026-07-04_strip_vocab_v2_audit.md`) and the
stratified-null headline re-estimated (`2026-07-04_within_null_v2_{japan,uk}.json`). v1 values
retained in brackets where they changed; strip-independent cells (full text, parser, TF-IDF,
dictionary, coordinate ablation, harmonization) are unchanged.
Sources: `2026-07-04_leakage_v2_{japan,uk}.json`, `2026-07-04_within_null_v2_{japan,uk}.json`,
`2026-07-04_geo_fold_v2.json`, `2026-07-04_depth_band_v2_japan.json`,
`2026-07-04_rich_baseline_ladder.json`, `2026-07-04_knn_prior_rung.json`,
`2026-07-04_cross_archive_transfer.json`, `2026-07-04_missingness_ipw.json` + round-1/2 JSONs.
All HGB, leak-proof per-fold PCA, 5 seeds (nulls: 100 permutations), content = real vs
shuffled-embedding null.

## A. Text-representation decomposition (administrative 8/5-region LRO)

| Text representation (content %) | Japan (8) | UK (5) |
|---|---:|---:|
| Frozen LM, full text | −21.2 | −18.8 |
| Frozen LM, lithology-only v2 (global-null upper bracket) | −19.8 [v1 −21.1] | −10.5 [v1 −10.6] |
| Frozen LM, hardness-only | −19.5 | −17.2 |
| char n-gram TF-IDF (no LM) | −17.8 | −14.2 |
| structured lithology parser (~61-D, no LM) | −16.2 | −9.2 |
| coarse dictionary (regime/macro class, no LM) | +0.0 | −3.3 |
| conservative full-corpus (has_text + AIST-regime controlled, Japan) | −7.3 | — |

Region-bootstrap 95% CI (lithology-only v2): Japan [−22.0, −17.4]; UK [−14.9, −6.4] (both exclude 0).

## A′. Stratified nulls (100 permutations; the HEADLINE row)

| null (lithology-only v2) | Japan (8) | UK (5) |
|---|---:|---:|
| global | −19.6 | −9.8 |
| within-region | −19.4 | −9.8 |
| **within-class (HEADLINE)** | **−11.3** [v1 −12.6] | **−9.5** [v1 −9.3] |

All 8/8 and 5/5 negative; every region beats all 100 permutations (p < 0.01).
within-region ≈ global in both archives → no region-cluster-matching.

## A″. Rich-baseline ladder + KNN rung (content on top of each baseline, Japan)

| baseline | content % | regions |
|---|---:|---|
| thin (depth, lat, lon) | −19.8 | 8/8 |
| + regime one-hot (8) | −17.3 | 8/8 |
| + granular geology (litho-macro 14 + era 9) | −16.7 | 8/8 |
| + 61-D text parser (in baseline) | −4.4 | 8/8 |
| + KNN spatial prior (thin base, leave-own-borehole-out) | −20.3 | 8/8 |
| richest (regime+geology+parser+KNN) | −4.9 | 8/8 |
| UK: thin → + parser | −10.5 → −5.9 | 5/5 |

## B. Coordinate ablation (HGB, model-agnostic, administrative LRO) — risk 4

| | Japan (8) | UK (5) |
|---|---:|---:|
| depth-only RMSE | 11.224 | 17.757 |
| depth + raw lat/lon RMSE | 12.936 | 18.529 |
| **coordinate effect (adding lat/lon)** | **+15.3% (HURTS)** | **+4.3% (HURTS)** |
| text-channel effect (structured, no coords) | −11.1% | −9.9% |

DKL from-scratch national 2×2 (uniform 6,000 inducing): coordinates raise held-out RMSE in
8/8 regions, mean **+16.2%** (`2026-06-29_national_dkl_coord_ablation.json`).

## C. Geological folds (v2, frozen-LM channel, 5 seeds)

| split | content % | groups |
|---|---:|---|
| leave-lithology-macro-out (13 groups, NN join cov 1.0) | −9.2 [v1 −10.9] | 12/13 |
| leave-era-out (9 groups) | −11.7 [v1 −13.5] | 9/9 |
| regime ablation on admin folds: thin → +regime | −19.8 → −17.3 [v1 −21.1 → −19.0] | 8/8 |

Subset control (M2, v1 structured channel): admin folds on the same AIST-joined subset −14.8%
(8/8) vs −16.2% full sample → subset costs ~1.4 pp; the further drop is the harder fold.

## C′. Depth bands (v2, overburden confound)

| band | content % | n |
|---|---:|---:|
| 0–5 m | −16.1 | 10,329 |
| 5–10 m | −19.8 | 9,854 |
| 10–20 m | −23.3 | 11,186 |
| >20 m | −17.4 | 9,777 |

All 8/8 [v1: −16.9/−20.7/−24.3/−20.3].

## D. Target-harmonization robustness (structured channel, administrative LRO) — risk 6

| target | Japan | UK |
|---|---:|---:|
| raw N | −16.2% | −9.2% |
| within-archive z-score | −16.2% | −9.2% |
| within-archive quantile-rank | −18.9% | −7.7% |

## E. Cross-archive zero-shot transfer (E1; Spearman ρ, no target rows)

| direction | depth only | +text | +shuffled | target-trained depth ref | few-shot n=1000 |
|---|---:|---:|---:|---:|---:|
| Japan → UK | 0.118 | **0.333** | 0.273 | 0.409 | 0.506 |
| UK → Japan | 0.257 | **0.297** | 0.157 | 0.266 | 0.461 |

## F. Missingness / IPW (E3; analysis population 0<N≤100, n=1,354,412)

Text-bearing 97.5%; all covariate SMDs |≤0.36|; content −19.85% unweighted vs −19.89% IPW
(8/8); effective sample fraction 1.0. Estimands: text-bearing (headline) vs population
(anchored by the −7.3% has_text-controlled decomposition).

## Notes
- The manuscript HEADLINE is the within-class v2 pair **−11.3 / −9.5** (A′), with the global-null
  −19.6/−9.8 as the upper bracket and −7.3% as the strictest full-corpus bound.
- Robustness experiments in B/D use the embedding-free structured channel (no GPU);
  C is the frozen-LM channel (v2).
- B's `depth+coords`=12.936 (JP) equals the round-1 leakage no-text baseline → consistent.
