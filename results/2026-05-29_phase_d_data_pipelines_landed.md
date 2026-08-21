# 2026-05-29 — Phase D 開幕: Paper B' データパイプライン + Pillar 1/2/3/6 starter

## 仮説

Paper B′ の全体計画に沿って、Engineering Geology
→ Nature Communications Earth & Environment への跳躍を目指す 6 pillar の
**データ層 + 基盤コード**を本フェーズでまとめて実装する:

- **Pillar 1**: AIST granular (era + lithology macro) + 多モーダル covariate 配線
- **Pillar 2**: 孔内水位 XML 抽出 + LMC joint multi-task head
- **Pillar 3**: 観察記事 (geologist narrative) + Japanese BERT 埋め込み
- **Pillar 6**: MINE memorization metric

最低限のクラスター trial を投入して、次フェーズで**訓練結果分析と Pillar 4/5
+ paper drafting に集中できる**状態に到達する。

## ランディング済 (9 commits)

| Commit | Pillar | 内容 |
|---|---|---|
| `2eb41d1` | 1 | AIST granular features (era 11-way + litho macro 15-way) + 50 tests |
| `13ffa36` | 1 | v3 parquet schema + BoringDataset multi-categorical + train CLI |
| `0e9009c` | 1 | utens `dkl_national_v3_aist_granular` trial cell config |
| `e86e21e` | 1 | **AVS30Loader bug fix** (silently 全 0 を返してた) + warmup script |
| `d0a59e4` | 2 | 孔内水位 XML extractor + 8 tests + CLI |
| `ed25d2b` | 2 | 孔内水位 sentinel filter (99.99 / 999.99 / 9999.99 / 99999) |
| `a308e6d` | 2 | enrich.py に groundwater 結合 + v4 parquet schema |
| `b722ea7` | 2 | **LMC head 実装** (joint multi-task GP) + 国土 trainer + NVLink cell |
| `45448d2` | 6 | MINE memorization metric (随時) |
| `3e419d2` | 3 | 観察記事 extractor + Japanese BERT embedding pipeline (genuinely novel) |
| `19c7b49` | 2 | LMC NotPSDError fix (lengthscale init + jitter) |

加えて utens 側に:
- `dfef7bb` — LMC ablation cell (M=12k, num_latents=1 max-coupling)
- `_AFFINITY_GPU_NVLINK` — az-canis/az-ursa 2×H100 (160GB pool) アフィニティ
  ([mmr-epf sweep_submit](../../`~/main_desk/UTokyo/from_202604/mmr-epf/infra/utens/sweep_submit.py`) を参考)

## データ生成済 artifact

### `data/features/derived/aist_codes.parquet`（既存、再利用）
171,049 行。Paper 1 から引き継ぎ。

### `data/features/borings_japan_v3.parquet`（commit `13ffa36` で生成）
2,663,955 行 × 10 列。v2 + aist_era_code (11-way) + aist_litho_macro_code (15-way)。
分布検証:
- Era: HOLOCENE 56% / DILUVIAL 9% / MIOCENE 7% / CRETACEOUS 7% / UNKNOWN 11%
- Litho: ALLUVIAL 49% / MARINE 9% / VOLCANIC_PYROCLASTIC 8% / TERRACE 5% / RECLAIMED 5% / UNKNOWN 11%
- Cross-tabulation 地質学的常識と整合（HOLOCENE+ALLUVIAL dominant, etc.）

### `data/features/borings_japan_v4.parquet`（commit `a308e6d` で生成）
2,663,955 行 × 11 列。v3 + groundwater_depth_m。
- カバレッジ: 81.0% (2,157,675 / 2,663,955 行)
- 深度統計: median 2.27 m, p90 7.60 m, max 200 m (sentinel filter 後)
- NaN-mask が LMC head の per-row 観測マスクに使われる

### `data/features/derived/groundwater_depth.csv`（commit `d0a59e4` + `ed25d2b`）
191,572 行（per-XML）。78.7% 非空。
- min 0.01 m, median 2.47 m, mean 3.93 m, max 200 m (sentinel-filtered)
- 抽出時間: 191k ファイル / 44 秒 (8 workers)

### `data/features/derived/soil_text.csv`（commit `3e419d2`）
191,572 行（per-XML）。64.6% 非空 = **123,794 records が観察記事ナラティブ持ち**。
- 平均 543 chars, p99 2,904 chars (BERT 512 token context に余裕で収まる)
- サンプル: 「火山灰質の細砂主体 || φ2〜5mmの軽石を混入...」「ほぼ均一な細砂が主体。深度2.75〜2.85mで...」
- 抽出時間: 191k ファイル / 58 秒 (8 workers)
- **構造化 AIST 8軸では捕まらない地質家の知見**: 風化状態、礫の角度、含水評価、色

### `data/features/derived/jshis_avs30.parquet`（warmup 中）
J-SHIS AVS30 (V_s30) を 175k 唯一 boring 地点で API 取得。~4 req/s ETA ~10h。
**バグ修正**: AVSLoader が `response.features[0].properties.AVS` ではなく
`response.avs30` を読んでて無音で 0 を返してた (commit `e86e21e`)。

## 投入済 cluster cells

| Cell | Status | Goal |
|---|---|---|
| `dkl_national_v3_aist_granular` | Running on az-perseus, ~30/50 epochs | v3 AIST granular で RMSE が v2 hero 7.546 から動くか |
| `dkl_national_lmc_v4_m8k` | Failed (NotPSDError) → 修正後 resubmit 予定 | 2-task LMC joint trial |
| `dkl_national_lmc_v4_m12k_l1` | Failed (NotPSDError) → 修正後 resubmit 予定 | LMC M=12k + max-coupling ablation |

## クリティカルなデータパス（常に参照する場所）

リポ側の作業規約に追記 — `data/sample_xml/` には全 6 DTD バージョン
(1.10/2.00/2.01/2.10/3.00/4.00) の代表 XML が 1 ファイルずつ入ってる。新規 XML
extractor を書くときは**必ず全 6 で検証**する parametric test を追加すること。

## 主要な技術的決断

1. **v3 → v4 schema を段階分け**: AIST granular だけで先行 trial、groundwater は
   v4 で別 trial。レビュアーが「データを増やしすぎて何が効いたか分からん」と
   ならないようにする。

2. **LMC は per-latent batch + 共有 encoder**: GPyTorch の標準パターン。Encoder
   入力が (latents, B, D) で broadcast されるので、`_LMCApproximateGP.forward`
   で flatten → encode once → reshape back の handling が必要 (commit `b722ea7`)。

3. **NaN-masked log-likelihood**: groundwater 19% 欠損を imputation せずに済ます
   ため、`masked_multitask_log_prob` で per-cell 重み 0/1 マスク。LMC trainer
   は KL を手動分離（標準 VariationalELBO は素 LL 専用）。

4. **NotPSDError 防御**: per-latent kernel の lengthscale 初期値を 0.6 → 3.0、
   cholesky_jitter を 1e-6 → 1e-4 (float32) に bump (commit `19c7b49`)。Paper 1
   の LinearMean zero-init の LMC 版。

5. **Soil text 観察記事の埋め込み戦略**: 各層を ` || ` で結合、xlm-roberta-base
   で mean-pool last hidden state、PCA で 64-D に削減。Sentence-BERT 流の堅実
   recipe。日本語専用モデルより multilingual の方が地質記号 (φ, 〜, 半角/全角
   混在) に強い。

## 検証

- pytest backend/tests/: **464 passed, 5 skipped, 0 failures** (Phase C 終了時
  349 → +115 で本フェーズ)
- 全 6 KuniJiban DTD バージョンに対する parametric test が両 XML extractor
  (groundwater + soil_text) で pass
- v4 parquet をローカルで生成 + NFS にアップロード byte-identical
- LMC trainer の smoke run (CPU, 10k rows, M=256, 2 epochs) で
  loss 1.45 → 1.43 + 全 artifact 保存

## フォローアップ

### 次フェーズ最優先
- [ ] LMC m8k + m12k_l1 resubmit (Docker rebuild 完了後)
- [ ] v3 trial 完走時の RMSE 取得 → vs v2 hero 7.546 で +/- 判定
- [ ] BERT embedding を national soil_text.csv 全件に適用 (~20 min on H100)

### Pillar 4 (TTA) starter
- LayerNorm/TENT/pseudo-label の 3 戦略を `evaluation/test_time_adapt.py` に
- 8 region × 3 strategy で LRO RMSE 改善測定

### Pillar 5 (3D cube + apps) starter
- `scripts/predict_national_cube.py` driver
- Liquefaction LPI (Iwasaki)、bearing-stratum 深度、Vs30 from N、settlement risk

### Pillar 6 driver
- `scripts/run_memorization_audit.py` — foundation_model.pt を読み込み MI 計算
- per-region + per-prefecture fairness audit

### 公開準備
- Zenodo upload (model + cube + predictions)
- Docker image (推論専用 minimal)
- GitHub tag `paper_b_v1.0`

## 生成物

本フェーズで commit したファイル群（11 commits、全て feature/national-foundation）:

- スクリプト: `add_aist_granular_to_parquet`, `warmup_avs30_cache`,
  `extract_groundwater_from_xml`, `extract_soil_text_from_xml`, `embed_soil_text`,
  `train_lmc_national`
- モジュール: `aist_granular`, `groundwater_xml`, `soil_text_xml`, `lmc`,
  `memorization_metric`
- テスト: 各モジュールに対応 (累計 +115 tests)
- 設定: `multimodal_v3` covariate stack 仕様, 3 utens cells (v3, lmc_m8k, lmc_m12k_l1)
- Docs: リポ作業規約の sample_xml セクション、本ノート
