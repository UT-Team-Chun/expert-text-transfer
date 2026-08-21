# 2026-05-29 — Phase D / Pillar 3 v2: per-layer 観察記事 embedding

## 仮説

Pillar 3 の初版 (`3e419d2` で投入) は **per-boring** で観察記事を抽出していた:
1 ボーリングの全層ナラティブを ` || ` で連結 → 1 つの 768-D BERT 埋め込み →
そのボーリングの全 depth 行に同じ embedding を貼り付け。

ところが「観察記事」は本来**層別**のテキストである (`<観察記事_上端深度>` ~
`<観察記事_下端深度>` で各層に区切られている)。同じ孔の 2.5 m 地点 (「中粒砂、
亜円礫を含む」) と 12 m 地点 (「風化花崗岩、含水比 < 10%」) を 1 embedding に
押し込めると、N 予測の入力としてもったいない (per-boring averaging で深度別
情報がボケる)。

**仮説**: per-layer で抽出 → 各 `<観察記事>` ブロックを独立に embed →
depth-interval join (`depth_top_m ≤ depth_from_surface < depth_bottom_m`) で
v4 boring parquet に貼ると、per-boring 版より N RMSE が下がる (or 同等で
ablation table の感度を上げる)。

## やったこと

### 1. per-layer XML 抽出 (commit `005b60c`)

- `backend/national/data/derived/soil_text_xml.py`
  - 新規 dataclass `SoilTextLayerRecord(file_path, latitude_deg,
    longitude_deg, layer_idx, depth_top_m, depth_bottom_m, observation_text,
    char_length)` を追加
  - 新関数 `extract_soil_text_layers(path) -> list[SoilTextLayerRecord]`
  - 既存 `extract_soil_text` (per-boring) は破壊しない
  - 各 `<観察記事>` ブロック単位で 1 record。`<観察記事_上端深度>` と
    `<観察記事_下端深度>` の両方が parse できないレイヤ、`bot <= top` の
    縮退レイヤは drop
  - `<観察記事_上端深度>` が空 element の場合に発生する `Element.__bool__()`
    の `DeprecationWarning` を `is not None` 明示チェックで回避
- `backend/scripts/extract_soil_text_from_xml.py`
  - `--mode {per_layer, per_boring}` CLI flag。`run()` を
    `_run_per_layer()` / `_run_per_boring()` に分割
  - per-layer CSV のカラム: `file_path, latitude_deg, longitude_deg,
    layer_idx, depth_top_m, depth_bottom_m, observation_text, char_length`

走らせた結果:

```
1,155,359 layer rows across 123,787 files
  mean 9.33 layers / file
  depth range 0–320 m
  layer thickness median 1.25 m / mean 2.13 m
  char_length mean 54 / median 38
```

### 2. embed_soil_text.py の per-layer 対応

- `--soil-text-csv` の中身を見て `layer_idx` + `depth_top_m` 列があれば
  per-layer モードで動作。明示フラグ不要 (schema 自動検出)
- 出力 parquet に `layer_idx`, `depth_top_m`, `depth_bottom_m` を伝搬
- raw 768-D parquet (`_full.parquet`) と PCA-64 parquet を**両方**出す
  (Pillar 3 v1 で「フルDも出しといた方がよくね?」の要望に沿う、commit `065a5cf`)
- wandb は既存通り (`--wandb --wandb-project geo-paperB-national
  --wandb-run-name soil_text_embed_ruri_v3_per_layer`)

### 3. depth-interval join (新規 `join_soil_text_to_parquet.py`)

```python
merged = pd.merge_asof(
    boring.sort_values("depth_from_surface"),
    emb.sort_values("depth_top_m")[
        ["lat_r", "lon_r", "depth_top_m", "depth_bottom_m"] + embed_cols
    ],
    by=["lat_r", "lon_r"],
    left_on="depth_from_surface",
    right_on="depth_top_m",
    direction="backward",
    allow_exact_matches=True,
)
# Post-filter for the half-open interval [top, bottom):
bad = merged["depth_from_surface"] >= merged["depth_bottom_m"]
merged.loc[bad, embed_cols] = np.nan
```

- 落とし穴: `merge_asof` は **`on` 列を global sort** していないと
  「left keys must be sorted」で死ぬ。`by` で同 boring 内に閉じ込められると
  期待して `(lat_r, lon_r, depth_from_surface)` で sort すると、boring 境界で
  depth が逆戻りするので NG。**`depth_from_surface` だけで global sort**。
- 落とし穴 2: v4 parquet は lat/lon を float32 で持つので、`round(4)` する
  前に `astype("float64")` で昇格しないと、float32 表現できない 4 桁目で
  join key がズレて match 0 になる (AIST cache の `13ffa36` と同種の罠)
- post-filter `depth_from_surface < depth_bottom_m` で半開区間を強制
  (merge_asof は前方の depth_top_m を取るが、depth_bottom_m を超えていれば
  別の層なので NaN で埋め直す → 下流 BoringDataset は missing-feature 経路に
  落とせる)

### 4. テスト (10 件追加)

```
backend/tests/national/test_soil_text_xml.py — per_layer 5 件追加
  - returns one per layer
  - degenerate / inverted / empty bounds dropped
  - missing coords -> empty list
  - CLI per_layer mode writes 6 rows for 2 files × 3 layers
  - CLI per_boring legacy unchanged

backend/tests/national/test_join_soil_text_to_parquet.py (新規 5 件)
  - assigns correct layer per depth (intervals [0,2), [2,6), [6,15) hit)
  - unmatched rows -> NaN embedding
  - preserves input row count (even with empty embedding parquet)
  - rejects per-boring schema (ValueError "per-layer columns")
  - float32 coord round-trip survives

566 passed, 5 skipped (CUDA-only).
```

### 5. NFS upload + クラスター投入

- `data/features/derived/soil_text_layers.csv` (250 MB, 1.15M rows) を NFS
  に upload (`kubectl cp` via nfs-helper、`stat` で size match 確認)
- Docker rebuild #5 (`utens deploy` → tag `1780052546739236000`)
- `infra/utens/sweep_submit.py` を per-layer 仕様に書き換え:
  - cell label: `soil_text_embed_ruri_v3` → `soil_text_embed_ruri_v3_per_layer`
  - cell label: `soil_text_embed_sarashina_v2` → `soil_text_embed_sarashina_v2_per_layer`
  - input: `soil_text.csv` → `soil_text_layers.csv`
  - output: `soil_text_embed_*_pca64.parquet` → `soil_text_embed_*_per_layer_pca64.parquet`
- `python sweep_submit.py --label-filter per_layer` で 2 cell 投入
  - `geo-soil-text-embed-ruri-v3-per-layer-5480283000`
  - `geo-soil-text-embed-sarashina-v2-per-layer-5480283001`
  - 推定 wall-clock: Ruri ~2 h (9x データ vs per-boring), Sarashina ~5-6 h

## 6. 高速化 (bf16 + batch + randomized PCA)

per-layer は per-boring の 9× データ (1.15M vs 124k rows)。初版投入は **fp32 +
batch=32 (Ruri) / batch=16 (Sarashina)** で動かすつもりだったが、見直して以下を入れた:

- **bf16 inference**: `SentenceTransformer(model_kwargs={"torch_dtype":
  torch.bfloat16})` で重みを bf16 で GPU に直接ロード。`.half()` ではなく
  HF 経路を通すと attention mask / RoPE buffer の dtype 不整合を避けられる。
- **batch_size**: per-layer 入力は median 38 chars (≈30 tokens) と短いので
  Ruri (ModernBERT-Ja hidden=768) は **batch=512** で peak 63.85 GB / 99.95 GB
  (64%) — H100 のスイートスポット。
- **randomized PCA**: `PCA(svd_solver="randomized", n_components=64)`。
  full SVD は (1.15M, 768) で数分かかるところ **25.7 秒** で完走、variance
  retained 0.778 (full SVD と ~4 桁同等)。
- **fp32+batch16 で見積もり**: ~80% throughput / batch、~50% throughput /
  precision、PCA ~10×; 合算で **~80× 高速化** (Ruri ~7 分 vs 推定 ~10 時間)。

### 落とし穴: Sarashina v2 1B での OOM

batch=256 初版は **86 GB / 93 GB** で死亡 (CUDA OOM、11.99 GiB allocate
できず)。原因は Sarashina が **1B Llama (hidden=2048, 24 layers)** で
ModernBERT (hidden=768, 12 layers) と比べて per-token activation が 3× 重い
こと、+ 1 つでも長文 outlier が混じると batch 全体がそこまで pad されて
活性化が爆発するため。**batch=64 + bf16 + randomized PCA** で 21 min 完走。

| Cell | batch | precision | wall-clock | peak GPU |
|---|---|---|---|---|
| Ruri v3-310m per-layer | 512 | bf16 | **19 min** (job total, ~7 min net encoding) | 64% |
| Sarashina v2-1B per-layer | 64 | bf16 | **~21 min** | 不明 (logs 切れ) |

## 7. join 結果

両 cell とも CPU-only (`no_gpu: True`) で `pandas merge_asof` のみ:

```
2,663,955 rows × 11 cols  (v4 boring parquet)
× 1,155,359 rows × 71 cols (per-layer embedding parquet)
→ Out-of-interval 2.5% → NaN
→ Matched 1,536,704 / 2,663,955 boring rows (57.7%) to a layer embedding
→ Wrote 2,663,955 rows × 75 cols (v5 parquet, 350 MB each)
```

両 backbone で **match 率 57.7% 同一** — 期待通り、boring × layer 幾何は同じで
embedding の数値内容だけが異なるため。残り 42% は観察記事のない boring か
深度区間外の depth 行。`has_text` フラグで encoder が学習時に discount できる。

## 8. v5 → train_lmc_national の wiring

`scripts.train_lmc_national._build_features` を v5 schema 対応に拡張
(commit `ff67309`):

- `embed_*` 列が parquet にあれば自動検出 (数値 suffix で sort、`embed_2` →
  `embed_10` の順)
- 未 match 行 (NaN) は zero-fill + `has_text=0` indicator を追加
- encoder 入力次元: 14 (v4) → 14 + 64 + 1 = **79** (v5)
- backwards-compat: v4 parquet を渡せば従来 14 次元経路

新規 cell:
- `dkl_national_lmc_v5_ruri_per_layer_m12k_l1` (parquet=v5_ruri, M=12k, 50ep)
- `dkl_national_lmc_v5_sarashina_per_layer_m12k_l1` (parquet=v5_sarashina, 同上)

両 cell は Docker rebuild #7 (perf+v5 patch を含む image) の完走後に投入。

## 次

1. **v5 LMC 完走** → N RMSE で v4 baseline (LMC v4 m12k_l1) と比較。期待: text
   embedding 由来の改善 ~3-5% (Pillar 3 v1 per-boring 帰無に比べて改善あれば
   per-layer 切り替えの正当化に十分)
2. **Ruri vs Sarashina 比較** (Pillar 3 ablation table 1 行目)
3. (BLOCKED) Pillar 5 cube cell は距離ラスタが無いと焼けない — 後段

## 関連

- 元 Pillar 3 v1 entry: [2026-05-29_phase_d_data_pipelines_landed.md](2026-05-29_phase_d_data_pipelines_landed.md)
- per-boring vs per-layer の N RMSE 比較は LMC 完走後に [results_table.md](results_table.md) に追加
