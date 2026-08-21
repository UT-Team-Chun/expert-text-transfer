# 2026-05-28 — Phase C: 全国 AIST regime キャッシュ完成（最大の data blocker 解消）

## 仮説

Paper 1 で `regime` 寄与は <1% RMSE と分離されていたが、Paper B の **per-regime
metrics / Mondrian 条件付き conformal / rare-regime 専用 fold** は regime が
**全国で本物**でないと成立しない。前 run の `borings_japan.parquet` は AIST キャッシュ
が関東のみ（18,993 / 150,607 地点 = 12.6% カバー）で、関東外は regime=UNKNOWN(7)
だった。**全国 AIST キャッシュを構築**し、再 enrich すれば Paper B の regime 条件付き
解析が初めて意味を持つ。

## 実施

| 項目 | 値 |
|---|---|
| 取得方法 | `scripts.fetch_aist_geology` ローカル run（`gbank.gsj.jp/seamless/v2/api/1.3/legend.php` REST API、rate-limit-s=0.03） |
| 新規 lookup 数 | **147,905**（既存 23,144 + 新規 → 合計 171,049 cached） |
| 実時間 | ~3 時間（バックグラウンド・ネットワーク bound、Mac CPU は遊休） |
| 再 enrich | `scripts.enrich_borings --region japan` → `borings_japan.parquet`（2,663,955 行） |
| AIST → regime | 既存 `data/derived/lithology.regime_from_legend`（legend 文字列ベース、修正なし） |

## 結果 — regime カバレッジの劇的改善

| regime | old (関東のみ AIST) | new (全国 AIST) | × |
|---|---:|---:|---:|
| 0 ALLUVIAL | 331,845 | 1,559,664 | **4.7** |
| 1 DILUVIAL | 87,776 | 317,013 | 3.6 |
| 2 VOLCANIC_ASH | 98 | **4,515** | **46** |
| 3 SEDIMENTARY | 18,799 | 325,132 | 17 |
| 4 IGNEOUS | 15,068 | 227,980 | 15 |
| 5 METAMORPHIC | 318 | **39,248** | **123** |
| 6 LIMESTONE | 3 | **4,633** | **1,544** |
| 7 UNKNOWN | 2,210,048 | 185,770 | 0.08 |
| **UNKNOWN 割合** | **83.0%** | **7.0%** | **−12 倍** |

「rare regime」が**真に rare ではなくなる**: 火山灰 (×46)、変成岩 (×123)、石灰岩 (×1544)。
特に石灰岩は 3 → 4,633 で、九州・沖縄の carbonate がやっと表現可能に。`leave_region_out.GEOLOGICAL_BLOCKS`
で `kyushu_okinawa` を分離していたのは正にこのため。

## 含意 — Paper B への影響

- **走行中の DKL 4セル**（rbf-8k / matern52-8k / rbf-12k / censored、現 H100×3+taurus）は
  **古い parquet（83% UNKNOWN）で学習中** → 結果は「degraded-regime national baseline」として記録。
  Paper 1 知見（regime <1% RMSE）から、ヘッドライン RMSE への影響は小さいはずだが、
  per-regime 解析は意味薄。
- **次の H100 run round**（新パーケで再 train、image 再ビルド要）= **Paper B の確定版**。
  特に censored-likelihood + Mondrian per-regime conformal は新パーケで初めて全国評価できる。
- **HGB national LRO を新パーケで再走中**（このノートと並行）→ regime 特徴量の効きを
  RMSE で測る `cf` 比較（before: 11.198）。

## 残る data 拡張（後続）

- 全国 DEM ingest（`infra/utens/Dockerfile` への DEM raster COPY、または NFS）→
  `mountain_front` + `absolute_elevation_DEM`（現状は CSV `mouth_elevation` 由来）の P1 接続。
- P1 残: PGA/PGV、JMA 降水、land-use25 → enrich 拡張は別途。

## フォローアップ

- [x] 新パーケでの HGB LRO 再走 → 数値が出たら本表 + results_table を更新。
- [ ] 走行中 DKL 4 セルの結果回収（NFS）→ "degraded-regime baseline" として results_table 記録。
- [ ] 新パーケで DKL 4 セル + GPBoost LRO + scaling-curve を再投入（定義版 Paper B run）。
- [ ] aist_codes.parquet をコミット対象に含めるかは別議論（gitignore 維持 + provenance だけ記録、が妥当）。

## 生成物

- AIST キャッシュ: `data/features/derived/aist_codes.parquet`（171,049 行・gitignored）
- 再 enrich パーケ: `data/features/borings_japan.parquet`（2,663,955 行・gitignored）
- Git commit: 本エントリと同 commit
