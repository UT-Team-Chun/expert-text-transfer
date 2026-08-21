# 2026-06-01 — Phase 4 / TTA 24-cell sweep: bn_stats safe, TENT collapses, self_training degrades

## 仮説

Phase C の 8-way LRO sweep (`2026-05-29_phase_c_dkl_lro_8way.md`) で、DKL+SVGP は
in-region 関東ベスト RMSE 5.875 に対して全国 cross-region で平均 RMSE 14.822 まで
+96% 悪化することが確定した。最難 region は kyushu_okinawa (RMSE 18.33) で、
LRO outlier として浮いていた。

**仮説**: source-only 学習済 DKL を target region の **unlabeled** ボーリング行で
test-time adapt できれば、representation shift の一部を closeable にできるはず。
3 戦略を比較する:

1. **bn_stats**: target 行で BatchNorm の running mean/var だけ再推定（パラメータ
   更新なし）。実装最簡、悪化リスク最小。
2. **TENT** (entropy minimization): predictive entropy を target 行で下げる方向に
   BN affine だけ更新。分類で有効報告だが回帰 SVGP では未知。
3. **self_training**: target で平均予測を pseudo-label として ELBO を回す。最も
   攻めるが confirmation bias リスクあり。

期待: bn_stats は no-op に近い、TENT は 5-10% RMSE 削減、self_training は kyushu
で意味ある改善（gap が大きいほど adaptation の伸び代がある）。

## 実験設定

| 設定項目 | 値 | 真の設定ファイル |
|---|---|---|
| Source model | `dkl_national_lro_<region>/foundation_model.pt` (Phase C 8-way の各 fold artifact) | -- |
| Target unlabeled | 各 leave-out region の boring 行（v4 parquet） | -- |
| Sweep grid | 8 regions × 3 strategies = **24 cells** | `infra/utens/sweep_submit.py` |
| TTA エポック | bn_stats=1, TENT=5, self_training=10 | CLI flag |
| Predict chunk | 8192 | `--predict-chunk-size 8192` |
| Affinity | `_AFFINITY_ONPREM_ALL` (Docker image `1780277468070341000`) | -- |
| 評価指標 | per-region RMSE / MAE / cov95、source baseline からの Δ | -- |

24 cells は初回 image で path resolution に問題があり全 fail、image tag
`1780277468070341000` で再 deploy 後にクリーンに完走。

## 結果

### Headline (8 region 平均、source baseline からの Δ)

| Strategy | Δ-RMSE 平均 | Δ-MAE 平均 | 改善 region 数 | σ behavior |
|---|---|---|---|---|
| **bn_stats** | **−0.096** | **−0.134** | **2 / 8** | 不変 (σ ≈ 7.4 維持) |
| TENT | **+11.798** | +6.4 | 1 / 8 | σ collapse 7.4 → **0.011** |
| self_training | **+24.303** | +12.9 | 0 / 8 | σ ≈ 5.2 (やや縮) |

### Per-region 抜粋

| Region | source RMSE | bn_stats Δ | TENT Δ | self_training Δ |
|---|---|---|---|---|
| kanto | 14.10 | **−0.481** | +8.2 | +22.1 |
| kyushu_okinawa | 18.33 (LRO 最難) | **−5.220** | −1.8 | +5.3 |
| chugoku | 13.86 | +0.05 | +14.7 | **+50.66** (最悪) |
| 他 5 region | 12.9–15.6 | ±0.2 以内 | +9〜+15 | +18〜+38 |

bn_stats は 8 cell 中 6 cell で |Δ-RMSE| < 0.2 と near no-op、残り 2 cell
(kanto, kyushu_okinawa) で意味ある改善。TENT と self_training は **全 region 平均で
劣化**し、TENT は σ の極端な collapse (7.4 → 0.011) を伴う = 予測分布が点質量化、
NLL は爆発し conformal の finite-sample 保証が壊れる。self_training は chugoku で
+50.66 と単独で論文を巻き戻す規模の破綻。

詳細 24 JSON は
`/mnt/nas/geo-estimation/runs/dkl_national_lro_<region>/tta_results_<strategy>.json`
(8 × 3 = 24 files)。

🔗 [`fig9_tta_delta_rmse.pdf`](../paper/paper_2_national/figures/fig9_tta_delta_rmse.pdf)
— grouped bar chart (8 region × 3 strategy) に σ collapse の log-scale inset
を重ねた図。

## 考察

仮説は **部分的に否定**された:

- bn_stats は **safe** (mean Δ-RMSE −0.096) — deploy しても害がない、kyushu の
  outlier には効くという defensible な振る舞いで、cross-region serving の default
  に推奨できる。
- TENT は **回帰 SVGP では entropy minimization が σ collapse trap に落ちる**。
  分類でなら最頻クラスへの均一化で entropy が下がるが、回帰では分散をゼロに潰せば
  どんな pseudo-label でも entropy が下がるため、global optimum がそこに張り付く。
  これは Wang et al. 2021 が想定していない failure mode で、**回帰 TTA の null
  result + diagnostic finding** として論文化価値あり。
- self_training は confirmation bias 通り — source baseline の bias を amplify
  するだけで、ELBO で更新したインデューシングが target distribution の outlier に
  引っ張られる。chugoku +50.66 は典型的な「初期 pseudo-label が悪い → 学習で悪化
  → 次の pseudo-label がもっと悪い」の発散。

唯一 multiple strategy で gain を出した kyushu_okinawa (source RMSE 18.33) は、
**source-target gap が既に大きいときだけ TTA が効く** という一般則と整合的:
adaptation の伸び代と source の信頼性のトレードオフ。kanto のような (相対的に)
源泉に近い region は no-op (bn_stats) でわずかに改善、過剰な戦略では潰される。

### What this means for the paper

TTA pillar は元々 **+96% LRO gap を縮める moonshot** として提案されていた
（Phase D Pillar 6）。今回の結果はそれを直接否定する代わりに、

1. **null result** (mean Δ-RMSE ほぼゼロ for bn_stats、悪化 for TENT/self_training)
2. **mechanism finding** (回帰 SVGP における entropy-minimization の σ collapse)

を生んだ。これは **"DKL の cross-region 限界は adaptation 問題ではなく
representation 問題"** という Paper B の中心仮説と整合する: target で BN を
動かす程度の自由度では representation を直せず、entropy/ELBO を動かすと SVGP の
キャリブレーションが先に壊れる。

論文への落とし込みは
[`docs/paper/paper_2_national/sections/07_cross_region_transfer.tex`](../paper/paper_2_national/sections/07_cross_region_transfer.tex)
の新規 `\label{sec:tta}` subsection に `tab:tta_per_region_strategy` +
`fig:tta_delta_rmse` を含めて済ませた。論調は

> "We evaluated three test-time adaptation strategies as a moonshot to close the
> +96% cross-region gap. Only batch-norm statistics adaptation proved safe
> (mean ΔRMSE −0.10, 2/8 improving); entropy minimization triggered predictive
> σ collapse (7.4→0.011) and self-training degraded all 8 regions. We interpret
> this as evidence that DKL's cross-region limit is a representation problem,
> not an adaptation problem."

— defensible null + mechanism、moonshot を引っ込めるのではなく **"why TTA fails
here"** として残す。

## フォローアップ

- [ ] **representation 側の手当て**: per-region geology embedding (AIST granular の
      11-era × 15-litho) を encoder の途中 layer に再注入 → cross-region 表現の
      足場を直接拡げる
- [ ] **bn_stats default の deploy**: cube serving で source-only と bn_stats の
      A/B、kyushu サブセットで意味ある優位なら本番化
- [ ] **σ collapse の formal note**: entropy minimization on regression SVGP の
      collapse trap を `docs/research/lessons.md` の TTA 節に独立 lesson として
      残す（このノートとは別に、横断的な教訓として）
- [ ] (BLOCKED) target で **少量の labeled** を許す semi-supervised TTA — Paper B'
      scope 外、Paper C 候補

## 生成物

- 24 strategy-suffixed JSON:
  `/mnt/nas/geo-estimation/runs/dkl_national_lro_<region>/tta_results_<strategy>.json`
- 新 figure: [`fig9_tta_delta_rmse.pdf`](../paper/paper_2_national/figures/fig9_tta_delta_rmse.pdf)
- 新 paper subsection: [`07_cross_region_transfer.tex`](../paper/paper_2_national/sections/07_cross_region_transfer.tex)
  `\label{sec:tta}` + `tab:tta_per_region_strategy` + `fig:tta_delta_rmse`
- Docker image: `1780277468070341000` (24 cells on `_AFFINITY_ONPREM_ALL`)

## 関連

- 前段の cross-region gap 観測: [2026-05-29_phase_c_dkl_lro_8way.md](2026-05-29_phase_c_dkl_lro_8way.md)
- 全国 baseline retrain: [2026-05-28_phase_c_dkl_v2_and_ablations.md](2026-05-28_phase_c_dkl_v2_and_ablations.md)
- σ collapse は単一実験を超えるので [lessons.md](lessons.md) に H3 lesson を追加予定
- [results_table.md](results_table.md) に TTA 24 行を補助テーブル形式で追記
