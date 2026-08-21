# 2026-06-20 — UK BGS cross-national text-content transfer (the breadth lever)

## 仮説

「words generalize, coordinates memorize」が *国境を越える原理* なら、地質家の自由記述
(layer description) を多言語 LLM で埋め込んだ特徴は、**日本で学習せずとも英国の SPT-N を
leave-region-out で改善する** はず。座標は国を跨いで意味を持たない（英国の緯度経度は日本の
モデルにとって無意味）が、「firm grey slightly sandy CLAY」のような岩相ナラティブは普遍的な
地盤情報を運ぶ、という主張の決定的検証。これが通れば「日本の地域パターン」から「地球科学的
原理」へ格上げされ、NCE&E が breadth の非交渉ラインと呼ぶバーを満たす。

## 実験設定

| 設定項目 | 値 | 真の設定ファイル |
|---|---|---|
| データ | UK BGS AGS4 全 GB 取得 | `scripts/ingest_uk_bgs.py` |
| 取得経路 | OGC borehole index → per-id `ags_export` (polygon 経路は 500 多発で不採用) | `national/data/derived/uk_bgs_ags.py` |
| 規模 | 9,148 project zip → **18,748 SPT 行 / 2,604 孔 / 5 region** | -- |
| 標高被覆 | 95.3% (LOCA_GL, AGS から無料、日本と comparable) | -- |
| テキスト | BS5930 岩相記述 (GEOL_DESC)、98% 被覆 | -- |
| 埋め込み | `intfloat/multilingual-e5-base` → PCA-64 | `scripts/uk_transfer_test.py` |
| baseline (fair) | depth + ground_level + lat + lon | -- |
| モデル | HistGradientBoostingRegressor (model-agnostic) | -- |
| 評価 | leave-region-out (5 UK macro-region) | -- |
| null | shuffled = 同一 PCA を行置換 (content 破壊、capacity+has_text 保存) | -- |

再現用 CLI:

```bash
cd backend
# 全 GB 取得（数時間、CPU+ネットワーク）
.venv/bin/python -m scripts.ingest_uk_bgs --out ../data/features/uk_bgs_spt_full.parquet \
    --cache-dir <cache> --min-depth 5 --per-region 4000
# fair transfer test（埋め込み + 3 feature-set の LRO）
.venv/bin/python -m scripts.uk_transfer_test --parquet ../data/features/uk_bgs_spt_full.parquet \
    --out ../docs/research/2026-06-20_uk_transfer_full.json --cache-dir <cache>
```

## 結果（決定的・全データ）

| feature set | UK LRO 平均 RMSE | vs no-text |
|---|---|---|
| no-text | 18.37 | — |
| shuffled (capacity null) | 18.00 | −2.0% |
| **text** | **14.88** | **−19.0%** |
| **content (text vs shuffled, leak-proof)** | | **−17.3%** |

per-region content (全て負): wales_west −29.3%、north_england −18.5%、scotland −14.4%、
midlands −11.6%、south_england −11.2%。

JSON: [2026-06-20_uk_transfer_full.json](2026-06-20_uk_transfer_full.json)。

## 考察

- **make-or-break は決定的に陽性**。全データで content effect は −17.3%、capacity (shuffle) は
  わずか −2.0% → テキスト利得の約 9 割が *純粋な地質ナラティブ内容*。これは leak-proof:
  text と shuffled は同一 PCA basis を共有するので、PCA leakage も capacity も has_text 指標も
  両者で相殺され、残るのは内容そのものだけ。
- **pilot (−11.8%) → 全データ (−17.3%) で効果が拡大**。データが増えると no-text/shuffled
  baseline は飽和するが text は信号を引き続き抽出。region を 4→5 に増やしても（wales_west
  −29%）一貫。
- **日英で coherent**: 日本の raw 利得は `has_text` provenance 指標に交絡されていた（content は
  decomposition 後 ~−7%）。英国は ~100% text-bearing → 交絡なし → 内容がクリーンに出る。
  2 国・2 言語（日本語 Sarashina/Ruri、英語 multilingual-e5）・2 モデル族（HGB と日本側 DKL+LMC）、
  全て shuffle-null 分離済。
- **座標は記憶、言葉は汎化**: 英国座標は日本モデルに無意味（cross-national では座標経路は
  定義上ゼロ転移）。転移したのは言葉だけ。これが本論文の中心命題の cross-national 証拠。

## 落とし穴・学び

- BGS polygon export (`/v7/ags_export_by_polygon`) は HTTP 500 頻発かつ count-on-error が
  silently 0 を返す → 取得漏れ。**OGC index + per-id export が rock-solid**。`_get` は 4xx+500 を
  即 fail（指数バックオフだと 1 孔 ~15s、即 fail なら ~1s）。→ [[lessons]] 候補。
- 最初の cut は thin baseline (depth+lat/lon) で効果を過大評価（−15.0%）。`ground_level`
  (LOCA_GL) を baseline に足すと公平になり、それでも content は強固に残る。**baseline の
  公平性を必ず確認**（ユーザ指摘「小規模で学習したの？」由来）。

## 次

- UK 側でも DKL（単一タスク SVGP）の LRO で text vs no-text を回し、FM もテキスト内容に
  乗ることを示す（2×2: country × model 完成）。on-prem 16GB GPU で可。
- MINE `--random-init` null（座標記憶の learned 証明）— code+test 済 (commit fd052d7)。

## 追記 (2026-06-21): LEAK-PROOF re-run — content −18.8% (panel 指摘対応)

敵対的パネル (ML) が指摘: −17.3% は PCA を全データ (hold-out 地域含む) で fit していた
(cross-region representation leak)。**per-fold PCA** (train 地域のみで fit、hold-out へ射影)
+ **5 seed** + 有意性検定で再実行。JSON: [2026-06-21_uk_transfer_leakproof.json](2026-06-21_uk_transfer_leakproof.json)。

| feature set | UK LRO RMSE (mean ± across-region std) |
|---|---|
| no-text | 18.53 ± 1.58 |
| shuffled (capacity null) | 18.03 ± 2.00 |
| **text** | **14.64 ± 1.53** |
| **content (text vs shuffled)** | **−18.8%** |

リーク修正で効果は **強まった** (−17.3% → −18.8%)。全 5 地域負、**sign/Wilcoxon p=0.031**。
これが canonical な UK 数値。[[lessons]] に「unsupervised PCA でも per-fold で fit すべき
(cross-region leak)」を追記候補。
