# P-T10: Descriptor-family ablation — parser rung の純化と、効果の「分散」局在

- **日付**: 2026-08-12
- **フェーズ**: NC pre-review 対応 R1（事前登録キャンペーン）
- **担当**: R. Okauchi
- **モデル訓練**: なし（HGB(400) の LRO 評価のみ。DKL/SVGP 訓練を伴わないので
  `diagnostics.png` は生成されない — 一次証拠は下記 JSON）

## 事前登録と判定

- **P-ID**: **P-T10** — [NC text-transfer PRE-REGISTRATION (R1)](2026-08-11_nc_text_preregistration.md)
- **事前登録バー（逐語）**:

  > **Descriptor-family ablation**（parser を text 由来のみに分離した上で、grain-size / weathering / water-state / colour / angularity / 組成% を 1 family ずつ除去） | 探索的（バー無し）。ただし parser rung の再計測は text-由来のみ構成で行い、archive コード混入版と分離して報告する

- **判定**: **EXPLORATORY**（数値バー無しで事前登録された唯一の P-T 項目）。
  唯一の**手続き上の要件**「parser rung を text 由来のみ構成で再計測し、
  archive コード混入版と分離して報告する」は **満たした**（両者を下記に併記）。
  family 別の減衰量は point estimate による mechanism localisation であり、
  仮説検定ではない。

## 動機

2 つの独立な理由でこの ablation が要る。

1. **査読側の汚染懸念**: 既報の parser rung（structured lithology parser を
   非テキスト baseline に積む段）は、Japan 側で **AIST archive コード
   （regime / litho-macro の one-hot）を混ぜていた**。archive コードは地図メタデータで
   あってテキストではないので、「language-model-free なテキストの読み取り」として
   parser を引用する限り、混入は主張を弱める（あるいは無効化する）。
2. **メカニズムの局在**: 「テキストが効く」から
   「**これらの地質観察**が転移信号を担う」へ論文を進めたい。parser は
   LM を通さないので埋め込みの交絡が無く、family 単位の leave-one-out ができる。

## 設定

| 項目 | 値 | 真の設定ファイル |
|---|---|---|
| 母集団 | japan、ID-join 済み text-bearing positive-N 行の balanced subsample **52,806 行**（500 files/region、`sample_seed=42`） | [`backend/scripts/nc_descriptor_families.py`](../../backend/scripts/nc_descriptor_families.py) |
| 転移チャネル | structured lithology parser（**LM なし**、one-hot + 組成% bins） | `backend/scripts/text_leakage_controls.py:structured_families` |
| モデル | HGB(400)、leak-proof per-fold PCA、8 地方 leave-region-out | `scripts/uk_transfer_test.py:_evaluate_lro` |
| null | 行 shuffle null（content effect = raw − null の分解） | 同上 |
| 乱数シード | 42 / 43 / 44 | -- |
| family | dictionary / grain_size / sorting / lith_class / weathering / water_state / angularity / colour / composition_pct | `structured_families` |

再現:

```bash
cd backend
.venv/bin/python -m scripts.nc_descriptor_families --domain japan \
    --out ../docs/research/2026-08-12_descriptor_families_japan.json
```

## 結果

一次証拠: [2026-08-12_descriptor_families_japan.json](2026-08-12_descriptor_families_japan.json)

### 1. Parser rung の純化 — archive コードを抜くと **強くなる**

| arm | 特徴量数 | content effect | 負の地方 |
|---|---|---|---|
| `parser_text_only`（text 由来のみ） | 66 | **−17.531%** | **8/8** |
| `parser_with_codes`（legacy、AIST archive コード混入） | 88 | −16.216% | 8/8 |

地方別:

| 地方 | text-only | with codes |
|---|---|---|
| hokkaido | −21.194 | −18.631 |
| tohoku | −19.596 | −19.619 |
| kanto | −23.517 | −21.564 |
| chubu | −15.266 | −13.957 |
| kansai | −11.491 | −12.004 |
| chugoku | −12.867 | −12.724 |
| shikoku | −17.154 | −17.498 |
| kyushu_okinawa | −18.075 | −13.010 |

**公表済みの −16.2% は混入版（`parser_with_codes`）に対応する。**
archive コードを外すと効果は **−16.216% → −17.531%（+1.3 pt 強化）** であり、
査読側が恐れた汚染の**逆向き**である。archive コードはテキストと重複する情報を
持ち込み、しかもノイズ（AIST 参照の粒度・UNKNOWN 残り）を伴うので、
純テキスト構成のほうがクリーンに効く、という読み方になる。

### 2. Mechanism localisation — leave-one-family-out

減衰量 = `parser_text_only` の −17.531% から何 pt 戻ったか（大きいほどその family が重要）。

| 除去した family | 特徴量数 | content effect | 減衰 (pt) | 負の地方 |
|---|---|---|---|---|
| lith_class | 49 | −15.711% | **+1.820** | 8/8 |
| grain_size | 56 | −16.480% | **+1.051** | 8/8 |
| water_state | 62 | −16.909% | +0.622 | 8/8 |
| dictionary | 56 | −17.022% | +0.509 | 8/8 |
| sorting | 60 | −17.107% | +0.424 | 8/8 |
| angularity | 62 | −17.260% | +0.271 | 8/8 |
| composition_pct | 62 | −17.289% | +0.242 | 8/8 |
| colour | 58 | −17.360% | +0.171 | 8/8 |
| weathering | 63 | −17.456% | +0.075 | 8/8 |

**どの単一 family を落としても content effect は −15.7% 以下（＝より負）に留まり、
かつ 8/8 地方で負のまま。** 最大の寄与者 lith_class ですら 1.82 pt しか動かさない
（全体 17.5 pt の約 10%）。9 family の減衰量の合計も 5.19 pt で、
全体の 30% にしかならない（＝family 間で情報が冗長に重複している）。

## 解釈（正直に）

1. **parser rung の純化は主張を強める**。公表値 −16.2% は混合版であり、
   text-derived only では −17.5%。論文では **text-only を正史**として引用し、
   混合版は「archive コードを足しても効果は −16.2% と同水準」という
   補足（robustness）に降格するのが正しい。数値の出所を取り違えないよう
   本ノートで両方を明記した。
2. **単一の descriptor family が効果を担っていない**。最重要の lith_class を
   落としても 90% が残る。これは「特定キーワード（"礫" や "N 値換算語"）を
   拾っているだけ」という読み方を**この解像度で否定**する。
   信号は fine-grained な記載全体に**分散して**存在する。
3. **だからこれは keyword effect ではなく description effect**である、という
   論文タイトルの主張と整合する。粗い辞書（`dictionary` family）を落としても
   0.51 pt しか動かないのに、その辞書だけを使った既報の粗辞書 baseline が
   ≈0% だった事実（[2026-06-23 structured-litho baseline](2026-06-23_structured_litho_baseline.md)）と
   合わせると、「語彙の粒度」ではなく「記載の細部の組み合わせ」が効いている。
4. **限界を正直に**: (a) これは point estimate による局在化であり、
   family 減衰の順位に検定は付いていない（事前登録どおり EXPLORATORY）。
   (b) 母集団は balanced subsample 52,806 行であり、full-population の primary
   estimand（P-T1）とは母集団が違う。(c) family は互いに直交していない
   （"細砂" は grain_size と lith_class の双方に触れる）ので、
   leave-one-out の減衰は「その family 固有の寄与」ではなく
   「他 family で代替できない残余」の下界である。減衰の合計が全体に
   届かないのはまさにこの冗長性の表れ。

## 次のアクション

- [ ] 新 Fig. 4（mechanism 図）を family 減衰の棒グラフとして作図
- [ ] UK 側（`--domain uk`）でも同一 ablation を回し、family 順位が
      アーカイブ間で保存されるか確認（BS5930 語彙は family 構成が違う）
- [ ] 減衰の冗長性を測るため、family を 1 つだけ**残す** leave-all-but-one も
      走らせて上界・下界で挟む
- [ ] 論文本文・SI の「parser rung −16.2%」を text-only −17.5% に差し替え、
      混合版は robustness 行に移す

## 生成物

- 結果 JSON: [2026-08-12_descriptor_families_japan.json](2026-08-12_descriptor_families_japan.json)
- runner: [`backend/scripts/nc_descriptor_families.py`](../../backend/scripts/nc_descriptor_families.py)
- 事前登録: [2026-08-11_nc_text_preregistration.md](2026-08-11_nc_text_preregistration.md) P-T10
- 同 R1 キャンペーンの姉妹ノート:
  [P-T8 conformal grouped split](2026-08-12_pt8_conformal_grouped_split.md) /
  [P-T6 borehole 単位 few-shot 曲線](2026-08-12_pt6_fewshot_borehole_curve.md)
- 前身: [2026-06-23 structured-lithology parser baseline](2026-06-23_structured_litho_baseline.md) /
  [2026-07-04 rich baseline ladder](2026-07-04_rich_baseline_ladder.md)（parser rung の公表値の出所）
- **diagnostics.png は無し** — 訓練を伴わない評価のみの再解析のため（上記 JSON が一次証拠）
