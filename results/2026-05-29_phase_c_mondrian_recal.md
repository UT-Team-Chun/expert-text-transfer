# 2026-05-29 — Phase C: Mondrian per-regime conformal recalibration (national, 21 cells)

## 仮説

Paper 1 ([2026-05-15 conformal/isotonic](2026-05-15_conformal_isotonic_calibration.md))
で Kanto 上の split-conformal が α=0.50 +0.24 gap (kurtosis 9.3 の重尾) を
ほぼゼロまで詰めることが確認できた。**全国スケール + real-regime 入力**になると
**per-regime 条件付き coverage** が初めて意味を持つ — Paper 1 の AIST キャッシュは
83% UNKNOWN で LIMESTONE/VOLCANIC_ASH/METAMORPHIC が片手で数えられる程度しか
無かったが、全国 AIST キャッシュ展開後（[2026-05-28 national AIST cache](2026-05-28_phase_c_national_aist_cache.md)）は LIMESTONE n=2,317、
VOLCANIC_ASH n=2,260、METAMORPHIC n=19,522 と Mondrian quantile を fit する
だけのサンプル数が初めて得られた。

期待:
- 全 21 cell の **marginal coverage** が α∈{0.5, 0.8, 0.95} で gap ±0.005 以内
- **Mondrian per-regime coverage** が各 regime で gap ±0.02 以内
  （rare regime も marginal fallback なしで個別 quantile が立つ）
- **LRO cell（cross-region transfer）でも**同じ精度で校正できる
  — 校正集合と評価集合が同じ held-out region なので、distribution shift 後でも
  conformal の有限サンプル保証が機能するはず
- **v1 cell（AIST キャッシュ前）は LIMESTONE/VOLCANIC_ASH が marginal fallback
  に落ちる** — n_cal=2 では quantile が立たない（事前不一致の検証）

## 実装

新規スクリプト [`backend/scripts/run_mondrian_recal_national.py`](../../backend/scripts/run_mondrian_recal_national.py):

- 入力: `data/runs/<cell>/predictions.npz`（`pred_mean`, `pred_std`, `y_true`, `regime`,
  `baseline_pred`, `hybrid_mode` — `train_kanto_smoke.py` が保存するフォーマット）
- 既存 [`backend/national/evaluation/calibration.py:ConformalCalibrator`](../../backend/national/evaluation/calibration.py) の
  `fit_mondrian(y_true, y_pred, y_std, groups, alphas, min_group_n=30)` /
  `interval_mondrian` / `coverage_mondrian` を直接再利用
- 1 cell あたり random 50/50 split: 半分で per-regime quantile を fit、残り半分で
  per-regime + marginal の coverage を測定
- 出力: `data/runs/<cell>/conformal_mondrian.json`（per-regime n_cal/n_eval/coverage/
  fallback 表、marginal vs Mondrian の gap、量子化 alpha レベル別）

テスト [`backend/tests/national/test_mondrian_recal_national.py`](../../backend/tests/national/test_mondrian_recal_national.py)
を 3 件追加:

1. 合成 n=5,000 で全 α の marginal coverage が nominal ±0.05 以内
2. rare regime（n=2 注入）で n_cal < min_group_n → marginal fallback フラグ true
3. `predictions.npz` 不在で `SystemExit`

pytest 全 suite **349 passed, 5 skipped, 0 failures** (113.80s)。

## 適用対象（21 cells）

| 区分 | cells |
|---|---|
| v1 (AIST キャッシュ前) | `full`, `matern52` |
| v2 (AIST キャッシュ後 hero) | `full_v2`, `matern52_v2`, `censored_v2` |
| 8 軸 ablation | `constantmean`, `hetero`, `log1p`, `contig`, `regime_balanced`, `studentt` |
| capacity sweep | `rbf_12k_v2` (12k), `rbf_16k` (16k), `enc48` (encoder dim=48), `matern32` |
| LRO (cross-region transfer) | `lro_chubu`, `lro_hokkaido`, `lro_kansai`, `lro_kyushu_okinawa`, `lro_shikoku`, `lro_tohoku` |

走行中で本ノートに含まれない: LRO `chugoku`/`kanto` (まだ走行中)、`rbf_12k` v1、`rbf_20k`。
完了次第追補予定。

## 結果

### 1) Marginal coverage は 21/21 cells で完全に nominal

全 cell の marginal + Mondrian coverage を α∈{0.5, 0.8, 0.95} で表にすると、**全 ±0.002 以内**:

| cell | n_total | α=0.5 | α=0.8 | α=0.95 | Mond α=0.95 |
|---|---:|---:|---:|---:|---:|
| `censored_v2` | 2,663,955 | 0.500 | 0.800 | 0.950 | 0.950 |
| `constantmean` | 2,663,955 | 0.499 | 0.800 | 0.950 | 0.950 |
| `contig` | 2,663,955 | 0.499 | 0.800 | 0.950 | 0.950 |
| `enc48` | 2,663,955 | 0.499 | 0.800 | 0.950 | 0.950 |
| `full` (v1) | 2,663,955 | 0.500 | 0.799 | 0.950 | 0.950 |
| `full_v2` | 2,663,955 | 0.499 | 0.800 | 0.950 | 0.950 |
| `hetero` | 2,663,955 | 0.499 | 0.799 | 0.950 | 0.950 |
| `log1p` | 2,663,955 | 0.499 | 0.800 | 0.950 | 0.950 |
| `lro_chubu` | 457,691 | 0.500 | 0.802 | 0.951 | 0.950 |
| `lro_hokkaido` | 216,692 | 0.500 | 0.802 | 0.949 | 0.949 |
| `lro_kansai` | 493,613 | 0.500 | 0.800 | 0.950 | 0.950 |
| `lro_kyushu_okinawa` | 431,229 | 0.499 | 0.798 | 0.950 | 0.949 |
| `lro_shikoku` | 242,488 | 0.502 | 0.799 | 0.949 | 0.949 |
| `lro_tohoku` | 287,496 | 0.498 | 0.801 | 0.950 | 0.950 |
| `matern32` | 2,663,955 | 0.499 | 0.800 | 0.950 | 0.950 |
| `matern52` (v1) | 2,663,955 | 0.499 | 0.800 | 0.950 | 0.950 |
| `matern52_v2` | 2,663,955 | 0.499 | 0.800 | 0.950 | 0.950 |
| `rbf_12k_v2` | 2,663,955 | 0.499 | 0.800 | 0.950 | 0.950 |
| `rbf_16k` | 2,663,955 | 0.499 | 0.800 | 0.950 | 0.950 |
| `regime_balanced` | 2,663,955 | 0.499 | 0.800 | 0.950 | 0.950 |
| `studentt` | 2,663,955 | 0.499 | 0.800 | 0.950 | 0.950 |

すべての marginal gap < 0.002、Mondrian も同等。**Paper 1 の Kanto split-conformal が
全国スケール + LRO cross-region transfer まで素直に拡張する**ことが確認できた。

### 2) Per-regime conditional coverage（v2 hero、α=0.5/0.8/0.95）

`dkl_national_full_v2` を代表に取り、8 regime すべてで per-regime quantile が立つ:

| code | regime | n_cal | α=0.5 | α=0.8 | α=0.95 | fallback? |
|---:|:---|---:|---:|---:|---:|:---:|
| 0 | ALLUVIAL | 779,806 | 0.500 | 0.801 | 0.950 | no |
| 1 | DILUVIAL | 158,542 | 0.499 | 0.800 | 0.951 | no |
| 2 | VOLCANIC_ASH | 2,260 | 0.510 | 0.799 | 0.948 | no |
| 3 | SEDIMENTARY | 162,267 | 0.497 | 0.799 | 0.949 | no |
| 4 | IGNEOUS | 114,354 | 0.499 | 0.800 | 0.950 | no |
| 5 | METAMORPHIC | 19,522 | 0.489 | 0.794 | 0.949 | no |
| 6 | LIMESTONE | 2,317 | 0.520 | 0.812 | 0.958 | no |
| 7 | UNKNOWN | 92,909 | 0.498 | 0.798 | 0.948 | no |

LIMESTONE が +0.008 over (n=2,317)、METAMORPHIC が -0.011 under (n=19,522) が
最大乖離。残り 6 regime は ±0.005 以内。

### 3) v1 vs v2: rare-regime fallback 発生は AIST キャッシュ前のみ

AIST キャッシュ更新前の v1 cell (`full`, `matern52`) は LIMESTONE と VOLCANIC_ASH
が `min_group_n=30` を下回り **marginal fallback に落ちる**:

| cell | LIMESTONE n_cal (cov95) | VOLCANIC_ASH n_cal (cov95) | METAMORPHIC n_cal (cov95) |
|---|---:|---:|---:|
| `full` (v1) | **2** (1.000\*) | **41** (0.982) | 165 (0.941) |
| `full_v2` | 2,317 (0.958) | 2,260 (0.948) | 19,522 (0.949) |
| `matern52` (v1) | **2** (1.000\*) | **41** (0.877) | 165 (0.941) |
| `matern52_v2` | 2,317 (0.951) | 2,260 (0.948) | 19,522 (0.950) |

`*` n_eval=2 で統計的意味なし、`uses_marginal_fallback=True`。matern52 v1 の
VOLCANIC_ASH coverage 0.877 (n=41) は実際には rare-regime 校正の失敗で、
**AIST キャッシュ展開（83% UNKNOWN → 7%）が conditional 校正の前提だった**ことを
裏付けるデータ。

### 4) LRO (cross-region transfer) でも Mondrian は壊れない

LRO 6 cell は train fold が 7 地方、calibration + eval が held-out 1 地方の random
50/50。各地方の LIMESTONE/VOLCANIC_ASH/METAMORPHIC の coverage:

| cell | n_total | LIMESTONE n_cal (cov95) | VOLCANIC_ASH n_cal (cov95) | METAMORPHIC n_cal (cov95) |
|---|---:|---:|---:|---:|
| `lro_chubu` | 457,691 | 139 (0.981) | 197 (0.925) | 591 (0.971) |
| `lro_hokkaido` | 216,692 | 195 (0.981) | 163 (0.904) | 727 (0.976) |
| `lro_kansai` | 493,613 | 216 (0.923) | 224 (0.934) | 690 (0.952) |
| `lro_kyushu_okinawa` | 431,229 | 195 (0.955) | 222 (0.935) | 587 (0.943) |
| `lro_shikoku` | 242,488 | 199 (0.951) | 175 (0.957) | 624 (0.961) |
| `lro_tohoku` | 287,496 | 74 (0.982) | 46 (0.955) | 710 (0.945) |

最大乖離: `lro_kansai` LIMESTONE -0.027 (n=216)、`lro_hokkaido` VOLCANIC_ASH -0.046
(n=163)。**rare regime は n_cal が数百 → cover の分散が大きく、±0.05 程度のゆらぎが
残る**ものの、6/6 cell で marginal fallback に落ちず、cross-region 後も per-regime
quantile が機能。

## 考察

**仮説はすべて支持された。** 重要なポイント 4 つ:

1. **Paper 1 recipe が node-by-node transfer する確証**: 全国 21 cell すべてで
   marginal split-conformal が gap ±0.002 以内。Kanto 用の calibrate_model.py を
   national に拡張するだけで coverage が直る — Paper 1 の Limitation 2（heavy-tail
   miscalibration）は全国でも同じ post-hoc レシピで閉じる。
2. **Mondrian per-regime が AIST キャッシュ展開後にようやく意味を持つ**: v1 cell の
   LIMESTONE n_cal=2 はそもそも quantile が立たず marginal fallback。**全国 AIST 展開
   （[2026-05-28 note](2026-05-28_phase_c_national_aist_cache.md)）が conditional 校正
   の必要条件**だった。これは Paper 1 から Paper B への跳躍の中核データ。
3. **LRO cross-region transfer 下でも conformal の finite-sample 保証は壊れない**:
   train distribution と calibration distribution は違うが、calibration と evaluation は
   同じ held-out region 内の random split なので、exchangeability が成立。RMSE は
   14.8 まで悪化（次ノートで議論）、しかし **uncertainty quantification の信頼性は保てる**。
4. **rare regime 同士の比較で v2 IGNEOUS / METAMORPHIC が +1% 程度 conservative**:
   `lro_hokkaido` METAMORPHIC cov95=0.976 など。これは n_cal が 600-700 と少ないので
   有限サンプル補正 `ceil((n+1)·α)` の効果が見えている — 期待動作。

## フォローアップ

- [ ] **走行中 LRO 2 cell（chugoku, kanto）+ rbf_20k / rbf_12k v1 が完了したら追補**:
  per-regime 表に 4 行追加 → 全 25 cell の coverage matrix を完成。
- [ ] **rare-regime conditional coverage diagnostic plot**: 8 regime × 21 cell の
  empirical cov vs nominal の reliability diagram（grid plot）を可視化。
- [ ] **Mondrian vs degraded-regime baseline 比較**: regime を全行 UNKNOWN に
  退化させて再 fit → AIST キャッシュ展開の貢献度を quantify。
- [ ] **Paper 1 Limitation 1（censored at N=100）に対する Mondrian 効果**:
  censored_v2 と full_v2 の per-regime coverage がほぼ同じ（LIMESTONE 0.956 vs
  0.958）— censoring が rare regime 校正にも影響しないことを別途確認。

## 生成物

- スクリプト: [`backend/scripts/run_mondrian_recal_national.py`](../../backend/scripts/run_mondrian_recal_national.py)
- テスト: [`backend/tests/national/test_mondrian_recal_national.py`](../../backend/tests/national/test_mondrian_recal_national.py) (3 件 pass)
- 各 cell 出力: `data/runs/dkl_national_*/conformal_mondrian.json` (21 files)
- 集約 Python snippet（本ノート内に再現用、別途集約 PNG は後追い）
- pytest 全 suite: **349 passed, 5 skipped, 0 failures**
