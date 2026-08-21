# 2026-06-20 — MINE coordinate-memorization null (--random-init): a REVERSAL

## 仮説（プラン由来）

「coordinates memorize」の中心診断として、学習済みエンコーダの z (24-D bottleneck) と
座標 (lat, lon) の相互情報量 I(z; lat,lon) を MINE で推定し、**random-init（同一アーキ、
重みだけ初期化）の null** と比較する。プランの予想は `random-init ≈ 0.01–0.05 ≪ trained ≈
0.404` — 「学習が座標を z に記憶として焼き込む」。

## 実験設定

| 設定項目 | 値 |
|---|---|
| モデル | `data/runs/dkl_national_full/foundation_model.pt`（全国 DKL+SVGP, n_input=14, n_output=24, Fourier 12-band, rbf+LinearMean, residual_geo=ON）|
| 入力 | 訓練レイアウト一致の 14-D parquet（lat, lon, depth, abs_elev, river_km, coast_km, regime_oh×8）|
| 診断 | `scripts/run_mine_audit.py --strategy leave-region-out --n-rows 8000 --n-iters 1500 --device cpu` |
| null | `--random-init`（commit fd052d7）: load 後にエンコーダ全 submodule を `reset_parameters`（seed=42）|

再現 CLI:
```bash
cd backend
.venv/bin/python -m scripts.run_mine_audit --run-dir ../data/runs/dkl_national_full \
  --parquet <matched_14d.parquet> --out-dir <trained> --strategy leave-region-out \
  --n-rows 8000 --n-iters 1500 --device cpu --no-prefecture --no-plot --force
# null: 同上 + --random-init --out-dir <random>
```

## 結果（予想の逆）

| エンコーダ | global I(z; lat,lon) [nats] | history plateau | 収束 |
|---|---|---|---|
| **trained** | **0.287** | ~0.42 | ✓ (1500 iter 安定) |
| **random-init** | **1.642** | ~1.8 | ✓ (1500 iter 安定) |

**random-init が trained より ~4–6× 高い**。プランの予想（trained ≫ random）の **完全な逆**。

## なぜ逆になったか（重要）

1. **Fourier 周波数は固定 buffer**（`register_buffer("freqs", ...)`, 学習されない、
   `reset_parameters` でリセットされない）。よって random-init でも lat/lon は同一の
   高周波 Fourier 特徴を通り、ランダム線形射影される。高周波 sin/cos は位置のほぼ
   一意な hash なので、ランダム射影でも位置情報はほぼ保存される（≈1.64 nats）。
2. **学習は bottleneck を圧縮する**（information bottleneck）。z は GP が SPT-N 予測に
   使う 24-D に絞られ、タスクに無関係な生の座標情報は捨てられる → trained は 0.29 nats
   に低下。
3. **記憶はそもそも encoder bottleneck に居ない**: SVGP は `add_residual_geo=ON` で
   生 (lat,lon) に Matern kernel を別途持つ。座標記憶が起きるならこの geo-residual
   kernel + kernel に入る Fourier 特徴であって、圧縮された z ではない。MINE-on-z は
   記憶経路を測っていない。
4. **MINE は下界**: trained の真の MI は 0.287 *以上*（bound が緩い可能性）。よって
   「trained < random（真の MI）」は厳密には言えず、言えるのは「下界では trained ≪
   random」。診断としては「trained ≫ random を示せなかった」が結論。

## 結論（正直に）

**MINE-on-encoder-output の trained-vs-random-init 診断は「coordinates memorize」の
positive evidence にならない。** むしろ逆（アーキが座標を注入、学習が圧縮）を示す。
**この診断は論文に memorization の証拠として載せない。** 載せると誤り、かつ査読で刺される。

座標記憶の証拠は別の柱に依拠する（これらは健在）:
- **zero_fourier counter-test**（[lessons](lessons.md) の該当節）: Fourier OFF が contig
  fold で勝つ（12.44 vs Fourier ON 13.98）→ Fourier 座標特徴が spatial-lookup を焼き込み、
  外挿を害する。**これが本来の memorization 証拠**。
- **LRO の失敗**（FM が cross-region で trees に負ける）。
- **cross-national**（座標は定義上 UK に転移できない；転移したのは言葉だけ、−17.3%）。

事前に診断の妥当性を検証したことで、誤った主張を論文に入れずに済んだ。`--random-init`
flag + test 自体は健全（commit fd052d7）で、別アーキ（Fourier 非依存 / residual_geo OFF）
には有効。

## フォローアップ

- 必要なら **encoder ではなく GP geo-residual kernel の寄与** を ablation（residual_geo
  ON/OFF で LRO RMSE 差）で測る方が memorization 経路に忠実。
- もしくは MINE を **kernel 入力（Fourier 特徴そのもの）** に対して取り、trained kernel
  の lengthscale が座標方向に短いか（記憶の符牒）を見る。
- いずれも camera-ready の必須要素ではない（memorization は zero_fourier + LRO +
  cross-national で十分示せている）。優先度は低。
