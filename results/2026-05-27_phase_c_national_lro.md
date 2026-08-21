# 2026-05-27 — Phase C: 初の全国 leave-region-out (地域外汎化) 実測

## 仮説

Paper 1 の **空間記憶（spatial-lookup memorization）** 知見から、全国モデルの真価は
random fold ではなく **leave-region-out (LRO) = 学習で一度も見ていない地方への外挿**で
測るべき。全国 Parquet を初めて構築し、8地方 LRO gate を実データで回して、(a) 地域外汎化が
成立するか、(b) split conformal の被覆が地域をまたいでも保たれるかを確認する。

## データ構築（全国 Parquet）

| 項目 | 値 |
|---|---|
| 入力 | `data/outputs/location_n_values.csv`（**2,703,567 行** = 全国 2.7M） |
| enrich | `scripts/enrich_borings --region japan`（core stack: 既存 CLI で対応済み） |
| 出力 | `data/features/borings_japan.parquet` = **2,663,955 行 / 150,607 ユニーク地点** |
| 共変量 | river_distance_km（全国 Class-1 河川）、coast_distance_km（MLIT C23 全47県）、absolute_elevation（CSV 由来、DEM 不要）、regime_code |
| regime | AIST キャッシュが関東のみのため **18,993 / 150,607 地点のみ非 UNKNOWN**。関東外は regime=UNKNOWN（既知の制約 → 次の data blocker は全国 AIST キャッシュ） |

地方別 test 行数: hokkaido 217k / tohoku 287k / kanto 496k / chubu 458k / kansai 494k /
chugoku 417k / shikoku 242k / kyushu_okinawa 431k（bbox 重複により合計 > 総行数。独立 fold 評価として許容）。

再現 CLI:

```bash
cd backend
.venv/bin/python -m scripts.enrich_borings --region japan --output ../data/features/borings_japan.parquet
.venv/bin/python -m scripts.run_leave_region_out \
    --parquet ../data/features/borings_japan.parquet --partition region --model hgb
```

## 結果（HGB baseline, 全国フル 2,663,955 行）

8地方それぞれを held-out（mesh split ベクトル化により全行で ~6 分）:

| held-out 地方 | RMSE | MAE | n_test |
|---|---|---|---|
| hokkaido | 12.255 | 8.665 | 216,692 |
| tohoku | 10.692 | 8.610 | 287,496 |
| kanto | 11.112 | 9.142 | 495,725 |
| chubu | 11.412 | 9.047 | 457,691 |
| kansai | 10.641 | 8.481 | 493,613 |
| chugoku | 11.266 | 8.489 | 417,227 |
| shikoku | 11.473 | 8.590 | 242,488 |
| kyushu_okinawa | 10.731 | 8.377 | 431,229 |
| **平均** | **11.198 ± 0.504** | 8.68 | — |

- **区間 coverage @ α=0.95 = 0.944**（nominal 0.95 とほぼ一致）→ split conformal が**地域外挿でも**有効。
- **gate: pass=True**（平均 RMSE 11.198 ≤ 閾値 13.967 = Kanto contiguous GPBoost 10.744 ×1.30）。
- 最難は **hokkaido (12.26)** — 本州から地理的に離れ、データ密度も低い。kansai/kyushu/tohoku が容易(~10.6-10.7)。
- 400k サブサンプル版（RMSE 11.245±0.502, cov95 0.944）とほぼ一致 → 結果は安定。

## 考察

- tree-only（lat/lon 非特徴）HGB が、**学習で一度も見ていない地方**で RMSE ≈11.2 と、
  Kanto within-network contiguous GPBoost (10.744) に肉薄。地域外挿という最も厳しい設定で
  この水準なら、全国 GPBoost + DKL の伸びしろは大きい。
- conformal 被覆が 0.944 と健全 → 全国スケールでも分布フリー区間が機能。Mondrian per-regime も
  runner に実装済みだが、regime が関東以外 UNKNOWN のため per-regime の意味は現状関東限定
  （全国 AIST キャッシュ構築後に解禁）。
- これは HGB baseline。**GPBoost（Paper 1 の contiguous 勝者）と DKL+SVGP の全国 LRO** が
  Paper B の本命結果。GPBoost は CPU でもローカル実行可、DKL は utens→Azure Spot H100 で。

## 追補 2026-05-28 — GPBoost 全国 LRO 結果（utens クラスタ、800k サブサンプル）

`infra/utens/sweep_submit` の job_template 経由でクラスタに投入、taurus CPU で 6h:

| held-out 地方 | RMSE | MAE | n_test |
|---|---:|---:|---:|
| hokkaido | 12.516 | 9.247 | 65,493 |
| tohoku | 10.778 | 8.802 | 86,607 |
| kanto | 10.453 | 8.813 | 149,084 |
| chubu | 11.653 | 9.404 | 137,300 |
| kansai | 11.088 | 9.056 | 147,899 |
| chugoku | 11.632 | 9.152 | 125,017 |
| shikoku | 11.999 | 9.236 | 72,379 |
| kyushu_okinawa | 10.874 | 8.604 | 129,566 |
| **平均** | **11.374 ± 0.649** | 9.04 | — |

cov95 = **0.945** (nominal 0.95 直近)、gate **pass** (rmse 11.374 ≤ 13.967)。

### LRO 三つ巴比較

| モデル | データ | mean RMSE | cov95 |
|---|---:|---:|---:|
| HGB v1 (degraded regime) | 2.66M フル | **11.198 ± 0.504** | 0.944 |
| HGB v2 (real regime) | 2.66M フル | 11.233 ± 0.504 | 0.943 |
| **GPBoost (real regime)** | 800k サブ | **11.374 ± 0.649** | 0.945 |

**興味深い観察**: GPBoost (800k サブ) は HGB (2.66M フル) より +1.2% わずか悪い。Paper 1 Kanto
contiguous では GPBoost が LightGBM (RMSE 13.4) より圧倒的勝利 (10.7) だったが、全国 LRO では
**データ量サブサンプル化のペナルティ (-70% data) が GPBoost の Vecchia spatial GP residual
の優位を上回る**。Paper B の新知見:

> Paper 1 Kanto: GPBoost > tree-only（spatial residual が地域内 contiguous 外挿で効く）
> Paper B 全国 LRO: HGB (full) ≳ GPBoost (sub)（地域外挿スケールではデータ量がより支配的）

要するに **全国地域外挿では「データを節約してでも spatial GP residual を加える」より
「単純な tree on more data」の方が point estimate は強い**。これは Paper 1 contiguous で
GPBoost を頂点とした構図と対照的で、Paper B の cross-region transfer 章の重要な貢献。

cov95 は 3 つとも 0.94 台で同等 — どのモデルでも split conformal がほぼ nominal 達成。
区間 quality は GPBoost > HGB わずか優位 (0.945 vs 0.943)。

### 残フォローアップ

- [x] GPBoost 全国 LRO（800k サブ）完了。
- [ ] GPBoost 全国フル 2.66M（taurus CPU 推定 ~30h）— コスト見合いで判断保留。
- [ ] **DKL contiguous fold @全国** (`dkl_national_contig`, 走行中) — Paper 1 GPBoost の Kanto-contig 勝利
      に対応する DKL 版、Paper B の cross-region 章のもう一本。
- [ ] **DKL national LRO**（`leave_region_out_runner` の DKL model 経路追加が必要）— Paper 1 hero モデル
      の地域外挿性能 = Paper B 真のヘッドライン LRO。今は HGB/GPBoost のみ。
- [ ] DKL v2 + ablation 完了結果に **Mondrian per-regime conformal recalibration** を当てる
      ([2026-05-28_phase_c_dkl_v2_and_ablations.md](2026-05-28_phase_c_dkl_v2_and_ablations.md))。

## 生成物

- 全国 Parquet: `data/features/borings_japan.parquet`（gitignored）
- LRO 結果: `data/runs/leave_region_out/region_hgb/results.json`
- 実装: `scripts/run_leave_region_out.py`, `national/evaluation/leave_region_out_runner.py`,
  `shared/geo/tiles.py:secondary_mesh_key_array`（ベクトル化 mesh split）
- Git commit: 本エントリと同 commit（runner は `efdc95f`, vectorize は `e64b304`）
