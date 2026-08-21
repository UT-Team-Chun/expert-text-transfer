# NC text-transfer 事前登録 — VERDICTS（修正済み機構、2026-08-18）

Pre-registration: [`2026-08-11_nc_text_preregistration.md`](2026-08-11_nc_text_preregistration.md)
（**全計算の前**にコミット済み）
Deviations: [`AMENDMENT 1`](2026-08-14_nc_text_prereg_amendment_1.md)

この文書は P-T1..T10 の各予測について、**事前登録したバーを一字も変えずに引用し**、
達成/未達を記録する。バーに届かなかったものは REFUTED と書く。

> **本版について**: 2026-08-13 の敵対的監査で、borehole ブロック置換 null が
> **置換になっていなかった**（donor へのクリップで 42.7% の行が複製）ことが判明し、
> P-T1/T2/T3/T4/T5 を修正済み機構で全面再実行した。**本文書の数値はすべて
> 修正後のもの**。旧版の数値は 0.6〜1.4 pp 過大だった。経緯は AMENDMENT 1。

## 判定一覧

| ID | バー（事前登録の原文） | 判定 |
|---|---|---|
| **P-T1** | 効果が負 in ≥7/8 regions、grouped-permutation p<0.05、full-population BCa 95% CI が 0 を除外。point estimate は [−20%, −3%] と予測 | **CONFIRMED**（全条件） |
| **P-T2** | 負 in ≥4/5 regions、grouped p<0.05 | **CONFIRMED** |
| **P-T3** | Japan headline content effect の変化 \|Δ\| < 5 percentage points | **CONFIRMED** |
| **P-T4** | 各 family で mean effect 負 かつ held-out unit の ≥70% が負 | **CONFIRMED**（4 family とも 100%。project は全母集団で進行中） |
| **P-T5** | 減衰 ≤5 pt かつ負 in ≥7/8 | **CONFIRMED**（減衰 平均 +0.26 pp、8/8 負） |
| **P-T6** | text few-shot が no-text few-shot を全 budget × 両方向で上回る | **CONFIRMED**（置換 null 不使用のため影響なし） |
| **P-T7** | 新効果が [−27.5%, −17.5%]（±5 pt 以内） | **CONFIRMED**（同上） |
| **P-T8** | marginal coverage gap at α=0.95 ≤ 0.01; width は行分割比 +20% 以内 | **CONFIRMED**（同上） |
| **P-T9** | RMSE ≤ zero-fourier 単独 +2%（どちらに出ても informative として報告） | **CONFIRMED**（4 地域、平均 −0.19%） |
| **P-T10** | 探索的（バー無し）。parser rung は text-由来のみ構成で再計測 | **EXPLORATORY**（手続き要件 達成） |

---

## P-T1 — Japan 8-region primary estimand · CONFIRMED

**バー**: 効果が負 in ≥7/8 regions、grouped-permutation p<0.05、
full-population BCa 95% CI が 0 を除外。point estimate は [−20%, −3%] と予測

### 置換段階（500 files/region、1000 perm、seed 42/43/44）

| held-out region | text RMSE | block null | perm p | content(block) |
|---|---|---|---|---|
| tohoku | 9.061 | 10.942 | 0.000999001 | **-17.19%** |
| kanto | 9.781 | 11.590 | 0.000999001 | **-15.61%** |
| kansai | 9.872 | 11.524 | 0.000999001 | **-14.34%** |
| chugoku | 9.490 | 11.052 | 0.000999001 | **-14.13%** |
| chubu | 9.919 | 11.452 | 0.000999001 | **-13.38%** |
| shikoku | 9.075 | 10.461 | 0.000999001 | **-13.25%** |
| hokkaido | 9.090 | 10.479 | 0.000999001 | **-13.25%** |
| kyushu_okinawa | 10.294 | 11.579 | 0.000999001 | **-11.10%** |

- **8/8 が負**（バー ≥7/8）✓
- **Bonferroni-min p = 0.0080**、Cauchy p = 0.0010（バー <0.05）✓

**p の統合方法について**: 事前登録は Stouffer を指定していたが、Stouffer は fold の
独立を仮定する。leave-one-region-out fold は任意の 2 つが訓練行の 6/8 を共有するため
正に従属で、Stouffer はここでは反保守的である。**任意の従属性の下で妥当な
Bonferroni 補正最小値と Cauchy 統合**を主として報告する。**最も保守的な補正でも
バーを通る**（0.008 < 0.05）ことが主張の強さになる。Stouffer は報告しない。

### 全母集団段階（1,298,728 行 — 点推定と区間はここから、事前登録の二段構えどおり）

| held-out region | text RMSE | block null | content(block) |
|---|---|---|---|
| tohoku | 9.509 | 11.340 | **-16.15%** |
| hokkaido | 10.373 | 12.287 | **-15.58%** |
| kanto | 9.569 | 11.262 | **-15.03%** |
| shikoku | 9.507 | 10.883 | **-12.64%** |
| chugoku | 9.947 | 11.297 | **-11.95%** |
| kansai | 9.368 | 10.623 | **-11.82%** |
| chubu | 9.869 | 11.003 | **-10.31%** |
| kyushu_okinawa | 10.453 | 11.356 | **-7.95%** |

| | 値 |
|---|---|
| 地域平均 content(block) | **-12.68%** |
| 負の地域 | **8/8** |
| pooled θ̂ | **-11.35%** |
| **full-population BCa 95% CI** | **[-11.49, -11.21]**（0 を除外）✓ |
| n_boreholes | 97,903 |

点推定 -12.68% は予測レンジ [−20%, −3%] の内側 ✓

**バーの 4 条件すべてを満たした。**

### 2 つの推定量を併記する理由

-12.68% は「地域ごとの content% の平均」（地域を等重み）、
-11.35% は「全ボーリングの paired loss をプールした比」
（ボーリングを等重み、区間はこちらに対して構成）。どちらも負で CI は 0 を除外するが
値が違うので、**片方だけ出すのは選択的報告**になる。論文では両方を明示する。

### 部分標本の取り方への頑健性

| campaign | 地域平均 | pooled θ̂ | BCa 95% | 負の地域 | 地域レンジ |
|---|---|---|---|---|---|
| subsample s42 | -14.03% | -11.58% | [-12.29, -10.87] | 8/8 | -11.10% … -17.19% |
| subsample s43 | -13.08% | -10.68% | [-11.45, -9.92] | 8/8 | -8.33% … -17.70% |
| subsample s44 | -12.51% | -11.27% | [-12.04, -10.49] | 8/8 | -6.92% … -17.21% |
| **全母集団** | -12.68% | -11.35% | [-11.49, -11.21] | 8/8 | -7.95% … -16.15% |

3 つの独立な subsample と全母集団が −12.5% 〜 −14.0%（地域平均）、
−10.7% 〜 −11.6%（pooled）の帯に収まる。**どの 500 files/region を引いたかに
headline は依存しない。**

### null が置換であったことの証拠

全シャードが `null_permutation` ブロックを持ち、`all_draws_bijective = true`、
`strata_columns = ["region", "aist_litho_macro_code"]`（事前登録どおり）、
`n_draws_checked = 1000`。**公開 JSON だけで検証でき**、欠陥版なら全 draw で
false になるため新旧の判別もつく。

**Artefacts**: `2026-08-18_grouped_null_japan_s42.json`,
`..._s43.json`, `..._s44.json`, `..._fullpop.json`

---

## P-T2 — UK 5-region 複製 · CONFIRMED

**バー**: 負 in ≥4/5 regions、grouped p<0.05

| held-out region | text RMSE | block null | perm p | content(block) |
|---|---|---|---|---|
| wales_west | 15.826 | 18.677 | 0.000999001 | **-15.26%** |
| north_england | 13.524 | 14.779 | 0.000999001 | **-8.49%** |
| south_england | 15.480 | 16.199 | 0.000999001 | **-4.44%** |
| scotland | 16.185 | 16.885 | 0.000999001 | **-4.14%** |
| midlands | 18.621 | 18.658 | 0.372627 | **-0.20%** |

- **5/5 が負**（バー ≥4/5）✓
- **Bonferroni-min p = 0.0050**（バー <0.05）✓
- 平均 content(block) = **-6.51%**
- borehole-block **BCa 95% CI = [-5.04, -2.93]**、
  θ̂ = **-4.04%**、n_boreholes = 2,597（0 を除外）

**正直に書くべき異質性**: 地域差が大きい。midlands は実質ゼロ
（-0.20%、p = 0.373）、
wales_west は -15.26%。
**「どの地域でも効く」は誤り**。正確には「5 地域すべてで符号は負だが、大きさは
地域依存で、1 地域では検出できない」。

**baseline の非対称性（記録）**: UK の非テキスト baseline は depth と ground_level の
2 列のみで、Japan の 35 列（AIST regime/litho/era one-hot + river/coast + KNN prior）に
対応する変数が UK アーカイブに存在しない。**「同一 spec」ではない**。UK の効果が
Japan より小さい理由の候補でもあり、そう書く。

**層別の非対称性（記録）**: UK には AIST 岩相コードが無いため null は region 単独で
層別される。成果物の `strata_columns` がそれを記録している。

---

## P-T3 — 行 null → borehole ブロック null の感度 · CONFIRMED

**バー**: |Δ| < 5 percentage points

| | 行 null | block null | Δ (block − row) |
|---|---|---|---|
| Japan（8 地域平均） | -13.44% | **-14.03%** | **-0.59 pp** |
| UK（5 地域平均） | -6.04% | **-6.51%** | **-0.47 pp** |

いずれもバー 5 pp の内側 ✓。stop clause 2 は**発火しない**。

**この判定が答えていること／いないこと**: これは「同一データ・同一 baseline で
null の単位だけを変えた」対比であり、兄弟層が null 側に残ることで効果が水増しされる
という懸念に答える。**論文の既報値（−11.3%/−9.5%）との差に対する判定ではない** —
既報値は within-lithology-class 行 null・別 baseline・ほぼ別標本の**別推定量**であり、
直接比較できない（AMENDMENT 1 §4）。
---

## P-T5 — 固有名詞 strip + テンプレート正規化 · CONFIRMED

**バー**: 減衰 ≤5 pt かつ負 in ≥7/8

**何を測るか**: 「埋め込みが読んでいるのは*誰が記載したか・どこか*であって
地質ではない」という批判への直接の反証。層記載から (a) そのボーリング自身の
ヘッダ（調査名・発注機関・調査会社）に現れる 2〜16 文字の部分文字列、(b) 行政・
地理接尾辞の直前 2〜8 文字の漢字/カタカナ列、(c) 同一プロジェクトの 90% 以上の層に
現れる最大部分文字列（テンプレート）を除去したうえで、P-T1 と同一プロトコルを回す。

**結果**（8 地域、1000 perm、seed 42/43/44、全 draw 全単射）:

| held-out region | frozen strip | 固有名詞 strip 後 | 減衰 |
|---|---|---|---|
| tohoku | -17.19% | **-17.21%** | -0.02 pp |
| kanto | -15.61% | **-14.86%** | +0.75 pp |
| kansai | -14.34% | **-13.94%** | +0.40 pp |
| chugoku | -14.13% | **-13.61%** | +0.52 pp |
| chubu | -13.38% | **-13.50%** | -0.12 pp |
| hokkaido | -13.25% | **-13.40%** | -0.15 pp |
| shikoku | -13.25% | **-12.63%** | +0.62 pp |
| kyushu_okinawa | -11.10% | **-11.04%** | +0.05 pp |

| | 値 |
|---|---|
| 地域平均 | **-13.775%**（frozen は -14.031%） |
| 負の地域 | **8/8**（バー ≥7/8）✓ |
| **減衰** | **平均 +0.26 pp、最悪 +0.75 pp**（バー ≤5 pp）✓ |
| Bonferroni-min p | 0.0080 |
| pooled θ̂ / BCa 95% | -11.54% / [-12.27, -10.80] |

**strip の副作用**: この腕が実際に走った母集団（500 files/region、52,806 層）で
成果物が記録する値は、文字削除 **5.9%**、空になった層 **7 件 = 0.013%**
（`config.strip_stats`）。内訳は正規化 5.3% / ヘッダ由来 540 文字・315 層 /
地名 68 文字・17 層 / テンプレート 0.6%・2,006 層、テンプレートを持つプロジェクト 90 件。

**地質語彙の保護について、当初の主張を訂正する。** 私は「火山灰 / 安山 / 段丘 /
条線 / 区分 の削除はゼロ」と絶対的に書いていたが、これは**母集団依存**であり
一般には成立しない。正確には:

- **固有名詞 strip の 2 操作（ヘッダ由来・地名規則）は構成上、地質語彙を触れない** —
  保護語彙を private-use のプレースホルダに退避してから実行し、後で戻すため
- **テンプレート正規化はこの保護を受けない**。実測で、1,155,359 行の層テーブル上では
  **区分 の出現の 1.8% を削除する**（プロジェクト内で 90% 以上に現れる文字列を消す
  規則なので、定型的に使われる地質語は原理的に巻き込まれ得る）
- この腕が走った 500 files/region の抽出では、当該 5 語の削除は実測ゼロ、
  礫 −0.391%、砂 −0.167%

論文には「構成による保護」＋成果物が記録する suffix 監査
（棄却した語尾 線 68% / 区 90% / 丘 92% が地質語である実測）を書き、
検証できないコーパス全体の「ゼロ」という数え上げは書かない。

**過剰削除の向きについて**: テンプレート正規化が地質語を巻き込む場合、それは
text 腕から情報を削る方向に働く。すなわち**効果を薄める側の誤り**であり、
水増しはしない。減衰が +0.26 pp に留まったこととも整合する。

**集計上の注意（記録）**: NAS 上に本番 `pt5/`（8 件）と事前検証 `pt5_verify/`
（shikoku 1 件、3 draw）が併存しており、最初の集計で shikoku が検証値
（−11.569%、減衰 +1.68 pp）を拾っていた。`perm_p_block = 0.25`（= 1/4、3 draw の
下限）が本番の 0.000999 と違うことで検出。本番値は −12.633% / 減衰 +0.62 pp。
**成果物に draw 数を記録していたから気づけた**ケース。

**Artefact**: `2026-08-18_grouped_null_japan_pt5_deperson.json`

---

## P-T6 / P-T7 / P-T8 / P-T10（置換 null 不使用のため無変更）

いずれも borehole ブロック置換 null を使わないため、機構修正の影響を受けない。

- **P-T6** — CONFIRMED、12/12。詳細は [ノート](2026-08-12_pt6_fewshot_borehole_curve.md)
- **P-T7** — CONFIRMED。identity join で RMSE **8.505 ± 0.162** vs baseline 11.333 →
  **−24.95%**（バー [−27.5%, −17.5%] の内側）、MAE **5.319 ± 0.121** → **−33.69%**。
  詳細は [artefact](2026-08-14_pt7_identity_join_3fold.json)
- **P-T8** — CONFIRMED、informative negative。詳細は [ノート](2026-08-12_pt8_conformal_grouped_split.md)
- **P-T10** — EXPLORATORY、手続き要件達成。詳細は [ノート](2026-08-12_pt10_descriptor_families.md)。
  ただし監査は本項が凍結共通プロトコルから逸脱（thin baseline + raw lat/lon + 行 null +
  未 strip テキスト）していると指摘しており、**探索的である旨を論文でより明確に書く**
---

## P-T4 — Provenance folds · CONFIRMED（修正済み機構で再実行）

**バー**: 各 family で mean effect 負 かつ held-out unit の ≥70% が負

**結果**（`nc_provenance_folds`、500 files/region、borehole-block null 500 perm、
seed 42/43/44、(family, fold) 単位で 26 シャード。**全シャードが
`all_draws_bijective: true`**）:

| family | 保持列 | mean content(block) | 負の unit |
|---|---|---|---|
| client | `orderer_name` | **-14.33%** | **7/7** |
| year | `year_bin` | **-12.80%** | **6/6** |
| contractor | `surveyor_name` | **-10.52%** | **8/8** |
| dtd | `dtd_version` | **-10.42%** | **5/5** |

4 family すべてで mean 負、負の unit 比率は**全 family 100%**（バー 70%）✓

**これが答える批判**: 「効果は記載組織の癖・年代の癖・スキーマ世代の癖を
モデルが覚えているだけではないか」。発注機関・調査会社・年代・スキーマ世代の
いずれを丸ごと held-out にしても効果は −10% 〜 −14% で残る。

**5 番目の family（leave-project-out）**: subsample では 300 行以上のプロジェクトが
1 件しかなく測定不能（全母集団では 920 件）。全母集団で実行中で、
最初の fold は `地質調査業務`（n_te = 3,768）で **-12.40%**。

**層別の非対称性（記録すべき差異）**: P-T4 の null は `geo_region` 単独で層別される
（`nc_provenance_folds` が独自にそう渡している）。P-T1/P-T2 の
`region × lithology-macro` とは異なり、**同一 campaign 内で層別が揃っていない**。
事前登録の層別指定は primary estimand（P-T1）に対するもので P-T4 には及ばないが、
差異は差異として記録する。P-T1 での実測では層別の違いは ≤0.4 pp・符号不揃いだった
ので、結論には影響しない見込み。

**Artefact**: `2026-08-18_provenance_folds_japan.json`

---

## P-T9 — 真の座標フリー腕 · CONFIRMED

**バー**: RMSE ≤ zero-fourier 単独 +2%（どちらに出ても informative として報告）

**なぜ必要か**: `foundation.py` は encoder の Fourier 経路とは別に、GP 側へ raw
(lat, lon) の Matérn-3/2 residual カーネルを**常時加算**する（`add_residual_geo`
既定 ON）。したがって既存の「Fourier OFF」腕は**座標フリーではなかった**。

**結果**（leave-region-out、4 地域 × 2 腕。held-out RMSE は各 run の
`predictions.npz` から算出）:

| 地域 | A: `--zero-fourier` | B: `+ --no-residual-geo` | B − A | n_eval |
|---|---|---|---|---|
| chubu | 12.4752 | 12.3330 | **−1.14%** | 457,691 |
| kansai | 11.4477 | 11.5942 | +1.28% | 493,613 |
| kyushu_okinawa | 11.6201 | 11.6043 | −0.14% | 431,229 |
| tohoku | 11.7715 | 11.6815 | −0.76% | 287,496 |

- **平均 B − A = −0.19%**、最大 |B − A| = 1.28% → バー ≤ +2% の内側 ✓
- 対ごとに `zero_fourier` / `add_residual_geo` フラグ、epoch 数、誘導点数、
  特徴数、評価行数が一致することを確認済み（腕の差は `--no-residual-geo` のみ）

**意味**: raw 座標の residual カーネルを外しても外挿は悪化せず、むしろ僅かに改善する。
**「座標は転移しない」という主張が、初めて本当に座標フリーな腕の上に立つ。**

**スコープ（絞ったことを絞ったものとして記録）**: 8 地域ではなく **4 地域**。
GPU 容量の制約による。地域は LRO の難易度が異なるものを選んでいる。
各地域の 2 腕は同一ハードウェアに配置した — pt7 の実測で、同一モデル・同一設定・
同一分割でも H100 と RTX 6000 Ada で **0.55%** 違ったため、バーが 2% 差である以上
ハードウェア混在は許容できない。

---

## 最終集計

| | 件数 |
|---|---|
| **CONFIRMED** | **9**（P-T1 / T2 / T3 / T4 / T5 / T6 / T7 / T8 / T9） |
| **EXPLORATORY**（バー無しで登録） | **1**（P-T10、手続き要件は達成） |
| **REFUTED** | **0** |
| **PENDING** | **0** |

**Stop clause は 3 つとも発火しなかった。**

1. 「P-T1 が負に出ない、または p≥0.05」→ 8/8 負、Bonferroni-min p = 0.008 で不発
2. 「P-T3 で \|Δ\| ≥ 5 pt」→ Japan −0.59 pp / UK −0.47 pp で不発
3. 「P-T7 で範囲外」→ −24.95%、バー [−27.5%, −17.5%] の内側で不発

**バーに届かなかった予測は無い。** ただし予測が外れた箇所は 1 つあり、P-T6 の副予測
「row→borehole の再設計で ρ は減衰する」は、はっきりした減衰としては観測されなかった。
主予測ではないが、外れた側として記録する。

**事前登録に対する逸脱**は [AMENDMENT 1](2026-08-14_nc_text_prereg_amendment_1.md) に
すべて開示済み（null の層別、null 実装の欠陥、推論 4 点、P-T4 の family 構成）。
