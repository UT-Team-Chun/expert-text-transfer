# 2026-06-19 — LRO text-transfer, B1 (trees vs DKL), shuffle decomposition, Vs30 validation

NCE&E "ceiling push" gate experiments (all on the v5 Sarashina per-layer parquet,
joint-LMC l=2 / M=8k / 50ep, leave-region-out over the 8 geomorphic regions).
These resolve the prior adversarial panel's blockers — and redirect the paper.

## 1. Text-transfer survives leave-region-out (8/8 regions)

Joint-LMC l=2, train 7 regions / evaluate held-out region. Held-out RMSE-N:

| held-out region | v4 (no text) | v5sara (text) | Δ% |
|---|---|---|---|
| hokkaido | 13.747 | 11.045 | −19.7 |
| tohoku | 11.926 | 9.350 | −21.6 |
| kanto | 11.933 | 8.085 | −32.2 |
| chubu | 13.104 | 9.565 | −27.0 |
| kansai | 11.830 | 8.518 | −28.0 |
| chugoku | 12.679 | 9.816 | −22.6 |
| shikoku | 12.757 | 10.085 | −21.0 |
| kyushu_okinawa | 11.901 | 9.584 | −19.5 |
| **mean** | **12.48** | **9.75** | **−23.7** |

Per-layer geologist text improves cross-region transfer in **every** region. Cells:
`infra/utens/sweep_submit.py` label `lro_lmc_{v4,v5sara}_*`; runs at
`/mnt/nas/geo-estimation/runs/dkl_national_lro_lmc_*`.

## 2. B1 — fair tree baseline with the SAME text features (the pivotal test)

Gradient-boosted trees (HGB), same 8-region LRO split, feature set = baseline +
embed_0..63 + has_text (`scripts/run_leave_region_out` via inline job;
`/mnt/nas/.../runs/B1_text_hgb_lro.json`). Mean held-out RMSE-N:

| | no-text | + text |
|---|---|---|
| **HGB (trees)** | **11.23** | **8.99** |
| **LMC (DKL foundation model)** | 12.48 | 9.75 |

- Text helps **both** models cross-region (HGB −20%, LMC −24%) → **model-agnostic** feature.
- **text-HGB (8.99) BEATS text-LMC (9.75)**, and no-text HGB (11.23) beats no-text LMC (12.48).
  → **Trees out-predict the DKL "foundation model" cross-region, with and without text.**
  The "foundation-model-is-best" headline is **falsified**. (no-text HGB 11.23 reconfirms the
  prior baseline exactly → setup validated.)

## 3. Shuffled-embedding null — content vs capacity

Same config on the row-shuffled v5sara parquet (`scripts/shuffle_embeddings.py`;
embedding↔row link broken, feature count + has_text preserved):

| region | v4 | shuf | real | content share |
|---|---|---|---|---|
| chubu | 13.10 | 10.40 | 9.57 | ~23% |
| kansai | 11.83 | 9.68 | 8.52 | ~35% |

→ ~65–77% of the text gain is **capacity + the has_text indicator**; only **~25–35%
(~−8% net RMSE) is genuine narrative content**. The raw "−22.5%/−28%" oversells the
content effect. (kanto pending; 8-region completion is a Phase-A control.)

## 3b. Feature-set decomposition — why does shuffled-embed beat v4? (HGB, 8-region LRO mean)

| feature set | RMSE | Δ vs v4 |
|---|---|---|
| F0 v4 baseline (5) | 11.233 | — |
| F1 v4 + has_text only | 9.705 | −1.53 |
| F2 v4 + real embed64 (no flag) | 9.001 | −2.23 |
| F3 v4 + real embed64 + has_text | 8.990 | −2.24 |
| F4 v4 + SHUFFLED embed64 + has_text | 9.702 | −1.53 |
| F5 v4 + SHUFFLED embed64 (no flag) | 9.695 | −1.54 |

(`/mnt/nas/.../runs/decomp_hgb_lro.json`; in-memory shuffle, seed 42.)

**The shuffled-embed gain is ENTIRELY the `has_text` indicator** (F1≈F4≈F5≈9.70); the 64
shuffled-noise dimensions add ~0 (confirmed inert). "Has a detailed log" is a data-provenance /
site-type covariate (logged boreholes are deeper/engineered → correlate with N; cf. BGS `TP`
trial-pits 0.7–1.4 m vs `BHN` 11–13 m). **Clean genuine-content effect = F3(real) 8.99 vs
F4(shuf) 9.70 = −0.71 RMSE (~−7%)**, corroborated by F2-vs-F5 (−0.69). Decomposition of the raw
−20%: provenance indicator −14% + **genuine linguistic content −7%** + noise ~0. The ~30%-content
fraction REPLICATES the LMC shuffle estimate across a different model class → robust. (Real
embeddings subsume has_text: F2≈F3, so once content is present the flag adds nothing.)

## 4. has_text=1/0 split (`scripts/analyze_has_text_split.py`)

The v4→v5 gain appears in **zero-text rows too** (kanto: text −23% vs no-text −44%;
all regions show both groups improving) → it is a **global-model effect** (text improves
the shared encoder/GP and transfers spatially), **not** a per-row lookup. Refutes the
structured-missingness-indicator proxy.

## 5. Vs30 areal product fails independent validation (commit `b0c8978`)

Model Vs30 map (cube-derived, Imai&Tonouchi) vs J-SHIS AVS30 (52k pts,
`scripts/validate_vs30_jshis.py`): **Pearson R = −0.009, Spearman 0.005, 50km-bin R = −0.05**,
~2× low bias (median ratio 0.48). Zero areal skill at every scale — same Fourier
coordinate-memorization that killed the 2D N maps (fig7/8). Point predictions strong;
**areal engineering maps are not validated** → demote/cut as deliverables.

## Conclusion → paper redirect

The "national subsurface **foundation model**" + "validated **engineering decision maps**"
framings are both falsified by these experiments. A re-run 5-reviewer adversarial panel
scored the current paper **NCE&E ~9–12% (all 5 → specialist venue)**. What survives is
rigorous and novel: **geologist field text is a model-agnostic, cross-region-transferable
predictor, honestly decomposed (~30% content) and diagnosed against coordinate
memorization**. Redirect (approved plan): reframe to **"words generalize, coordinates
memorize"** + add a **second country (UK BGS)** as the breadth lever → NCE&E ~30–40%;
floor = strong C&G/GMD.

### Commits
- `b0c8978` Vs30-vs-J-SHIS validation + RED finding
- `5f41e79` has_text split analysis ; `fe3fc04` shuffle-embedding null
- `ea1b239` LRO sweep cells + Azure Files data path ; `5d549ef` LRO trainer support
- `dd21641` 64Gi OOM fix ; `3ea6092` Ruri wave + ruri parquet staging
