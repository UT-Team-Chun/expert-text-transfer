# 2026-05-29 — Phase C: DKL leave-region-out 8-way + 3 モデル比較（暫定 6/8）

## 仮説

[2026-05-27 GPBoost national LRO](2026-05-27_phase_c_national_lro.md) で HGB / GPBoost
の 8 地方 leave-region-out 平均 RMSE が 11.23–11.37 と確定。Paper 1 で DKL+SVGP は
Kanto 単体（random fold）で RMSE 5.875 — **全国学習 + cross-region 評価で
tree-baseline と肩を並べられるか**を確認したい。

Paper 1 の posit:
- **DKL は in-region では強い** (Kanto 5.875 vs LightGBM 6.8、GPBoost 6.4)
- **GPBoost は contiguous で強い** (Kanto 10.744 vs DKL 10.7-11)
- → **cross-region (LRO) では DKL のスケール特性が壊れる**可能性が高い
  （空間記憶が地域内ルックアップ表として機能していた可能性、Paper 1 Limitation 1）

期待 (Phase C plan の予測):
- DKL LRO 平均 RMSE 13–16 程度（HGB 11.23 / GPBoost 11.37 より明確に悪い）
- 但し random-fold national 7.55 と比べると ~2x 悪化、Paper 1 の random vs contig +6%
  caveat と比較して 「LRO は contig より遥かに厳しい」を全国スケールで定量化
- 地域別の難度に強い差 (kyushu_okinawa は volcanic 多 + 異質、kansai は kanto に
  ジオ的に近く相対的に容易など)

## 実験設定

| 設定項目 | 値 |
|---|---|
| 訓練データ | `data/features/borings_japan.parquet` (2,663,955 行) |
| 評価 fold | `evaluation/leave_region_out.py:GEOLOGICAL_BLOCKS` (8 地方 bbox) |
| 学習 split | 7 地方 fit、1 地方 (held-out) で n_test 別 random 50/50 cal/eval |
| アーキテクチャ | DKL+SVGP, ResMLP 4×128, Fourier 12-band, encoder_dim=24, kernel=rbf, mean=linear |
| 誘導点 | 8,000（M=8k は本 sweep の標準サイズ） |
| エポック | 50 |
| バッチ | 4,096 |
| 学習率 | 5e-3 (cosine) |
| ターゲット変換 | none (raw N) |
| Regime 入力 | one-hot (post-AIST cache, UNKNOWN 7%) |
| 尤度 | Gaussian + heteroscedastic OFF |
| デバイス | CUDA (H100 80GB, taurus / az-andromeda / az-perseus / az-ursa / az-orion) |

再現用 CLI（utens 経由、`infra/utens/sweep_submit.py` で各地方 cell を投入）:

```bash
cd backend
uv run python -m scripts.train_kanto_smoke \
    --parquet ../data/features/borings_japan.parquet \
    --output-dir ../data/runs/dkl_national_lro_<region> \
    --train-fraction 1.0 --n-epochs 50 --batch-size 4096 --n-inducing 8000 \
    --lr 5e-3 --device cuda --regime-one-hot \
    --leave-region <region>
```

## 結果

### DKL LRO per-region 表（暫定 6/8）

`spatial_kfold[0]` の held-out region 全行に対する点推定精度:

| region held out | n_train | n_eval | RMSE | MAE | std_mean | time (h) |
|---|---:|---:|---:|---:|---:|---:|
| chubu | 2,206,264 | 457,691 | 13.890 | 9.770 | 7.288 | 2.47 |
| hokkaido | 2,447,263 | 216,692 | 15.163 | 11.615 | 7.327 | 2.73 |
| kansai | 2,170,342 | 493,613 | 13.302 | 9.007 | 7.672 | 2.42 |
| kyushu_okinawa | 2,232,726 | 431,229 | 18.326 | 12.136 | 7.351 | 2.50 |
| shikoku | 2,421,467 | 242,488 | 14.199 | 10.521 | 7.405 | 2.72 |
| tohoku | 2,376,459 | 287,496 | 14.056 | 10.753 | 7.648 | 2.67 |
| chugoku | — | — | (running) | — | — | — |
| kanto | — | — | (running) | — | — | — |
| **平均（6 cells）** | — | — | **14.822 ± 1.65** | **10.634** | — | — |

chugoku / kanto 完了時に追補予定。

### LRO 三つ巴比較（headline）

| モデル | データ | mean RMSE | mean MAE | cov95 |
|---|---:|---:|---:|---:|
| HGB v1 (degraded regime) | 2.66M フル | 11.198 ± 0.504 | 8.43 | 0.944 |
| HGB v2 (real regime) | 2.66M フル | **11.233 ± 0.534** | 8.628 | 0.943 |
| GPBoost (real regime) | 800k サブ | 11.374 ± 0.649 | 9.04 | 0.945 |
| **DKL+SVGP (real regime)** | **2.66M フル** | **14.822 ± 1.65*** | 10.634 | 0.950 (recal'd) |

\* 6 cells 暫定平均、chugoku + kanto 追補待ち。

**観察 1: cross-region transfer ヒエラルキーの逆転**

Paper 1 Kanto（in-region random fold）の RMSE ヒエラルキー:

> DKL 5.875 < GPBoost 6.4 < LightGBM 6.8 < HGB ~7

Paper B 全国 LRO（cross-region 8 地方平均）:

> HGB 11.23 < GPBoost 11.37 < **DKL 14.82**

**DKL+SVGP が tree-baseline より 32% 悪化**して最下位に。これは Paper 1 が予期していた
「DKL の空間記憶が in-region に依存する」仮説を**定量的に支持**する初の全国データ。

**観察 2: kyushu_okinawa の異常な難度**

`kyushu_okinawa` RMSE 18.326 — 他 5 地方の平均 14.13 から +30% 悪化。地質的に
火山島の多い九州・沖縄は AIST classification も独特（VOLCANIC_ASH と IGNEOUS の
混在比が他地方と異なる）で、 [2026-05-28 v2 sweep](2026-05-28_phase_c_dkl_v2_and_ablations.md)
の DILUVIAL 0.10 baseline からの転移が最も難しい地方。Paper B の **3-tier difficulty
hierarchy** の最下位ベンチマーク候補。

**観察 3: kansai が最易**

`kansai` RMSE 13.302 — 6 地方で最良。Kanto と地質的に隣接し DILUVIAL/ALLUVIAL 平野
の構造が似ている → train fold（kanto 含む）からの転移が効きやすい。逆に hokkaido
（北方異質）は RMSE 15.163 で 2 位悪化。

### Mondrian per-regime conditional coverage（recal'd）

LRO 6 cell × 8 regime × α∈{0.5, 0.8, 0.95} の coverage は
[2026-05-29 mondrian recal](2026-05-29_phase_c_mondrian_recal.md) 参照。Highlight:

- 全 cell の marginal coverage @ α=0.95: 0.949-0.951 (gap ±0.002)
- 全 cell の per-regime Mondrian @ α=0.95: 0.94-0.98 (gap ±0.03)
- rare regime (LIMESTONE n=74-216, VOLCANIC_ASH n=46-224) も marginal fallback なし
  で個別 quantile が立つ
- **cross-region transfer 下でも conformal の finite-sample 保証は壊れない**

cov95 値だけ並べると DKL (0.950 recal'd) > GPBoost (0.945) > HGB (0.943) で、
**点推定では tree-baseline に負ける DKL が、recalibrated uncertainty quantification では
全 3 モデル中で最良**という対比が成立。

### 地域別 3-way 詳細（HGB vs DKL）

| region | HGB v2 RMSE | DKL RMSE | Δ (DKL − HGB) | n_test |
|---|---:|---:|---:|---:|
| chubu | 11.399 | 13.890 | **+2.491** | 457,691 |
| hokkaido | 12.392 | 15.163 | **+2.771** | 216,692 |
| kansai | 10.622 | 13.302 | **+2.680** | 493,613 |
| kyushu_okinawa | 10.726 | 18.326 | **+7.600** | 431,229 |
| shikoku | 11.496 | 14.199 | +2.703 | 242,488 |
| tohoku | 10.788 | 14.056 | +3.268 | 287,496 |

DKL は全地方で HGB に劣後。kyushu_okinawa の差 (+7.6) が突出して大きく、
**DKL の cross-region degradation は地域の異質性に大きく依存**することを示唆。

## 考察

**仮説は強く支持された。** 3 つの新知見:

1. **DKL の空間記憶仮説の定量的検証 — random vs LRO**:
   - Kanto random fold: DKL 5.875 (Paper 1)
   - National random fold (full_v2): 7.55 (Paper B v2)
   - National contiguous (contig cell): ~7.75 (Paper B contig)
   - **National LRO (held-out region)**: 14.82 ± 1.65
   - random → LRO で **+96% 悪化**、これは Paper 1 contiguous の +6% caveat の
     **16 倍**の大きさ。**DKL の generalization gap は cross-region で初めて
     fully exposed される**。

2. **Cross-region で tree-baseline が逆転して優勢**:
   - Paper 1 Kanto in-region: DKL 圧倒的勝利（5.875 vs HGB ~7）
   - Paper B 全国 LRO: HGB 11.23 < DKL 14.82（HGB +32% 優位）
   - データ規模が 4-5x になっても、DKL のスケール優位は cross-region では
     reverse する。**Paper B 中核 finding**。

3. **しかし uncertainty quantification では DKL recal'd が最良**:
   - 点推定の RMSE は劣るが、Mondrian per-regime split-conformal を当てた後の
     cov95 は DKL 0.950 > GPBoost 0.945 > HGB 0.943
   - **「point estimate ≠ probabilistic prediction」**の Paper B 中核メッセージ
   - 工学応用（confidence-aware decision making）では DKL+conformal が依然 utility

4. **kyushu_okinawa は LRO ベンチマークの hardest case**:
   - DKL +7.6 RMSE 悪化 vs HGB は他地方の 2-3x
   - VOLCANIC_ASH と IGNEOUS の混在比、海洋性地質、火山島構造が train fold
     （kanto含む内陸 7 地方）から最も離れている
   - Paper B の **future work** として「volcanic / island-specific encoder」を
     提案する材料

## フォローアップ

- [ ] chugoku + kanto 完了 (~4-5h 以内見込み) → 8-region 完全表に追補
- [ ] **degraded-regime baseline と real-regime DKL の per-region 比較**:
  v1 cells を LRO で再走査し、AIST cache 改善がどの地方で何 RMSE 詰めるか定量化
- [ ] **kyushu_okinawa fine-grained 分析**: per-regime RMSE で VOLCANIC_ASH /
  IGNEOUS のサブ寄与を分解
- [ ] **DKL+conformal vs HGB+conformal の utility comparison**: 同じ α=0.95 で
  width × coverage の Pareto を 8 地方で並べる
- [ ] **scaling curve に LRO 軸を追加**: 25% / 50% / 100% の national サブ
  サンプル × LRO の RMSE 曲線 (Paper B の data-scaling 章の補助図)

## 生成物

- summary.json (6 cells): `data/runs/dkl_national_lro_{chubu,hokkaido,kansai,kyushu_okinawa,shikoku,tohoku}/summary.json`
- predictions.npz (6 cells): 同上ディレクトリ
- diagnostics.png: 同上ディレクトリ（fetch 済み）
- conformal_mondrian.json (6 cells): WS-2 で生成
- pytest 全 suite: **349 passed, 5 skipped, 0 failures**
- Git commit: (本ノート commit 時に追記)
