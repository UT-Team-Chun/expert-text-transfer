# P-T6: Few-shot 曲線を borehole 単位に再設計 — 12/12 で text が勝ち、shuffled null は方向で非対称

- **日付**: 2026-08-12
- **フェーズ**: NC pre-review 対応 R1（事前登録キャンペーン）
- **担当**: R. Okauchi
- **モデル訓練**: なし（HGB(400) の適応学習のみ。DKL/SVGP 訓練を伴わないので
  `diagnostics.png` は生成されない — 一次証拠は下記 JSON）

## 事前登録と判定

- **P-ID**: **P-T6** — [NC text-transfer PRE-REGISTRATION (R1)](2026-08-11_nc_text_preregistration.md)
- **事前登録バー（逐語）**:

  > **Few-shot を borehole 単位に再設計**（budget {0,10,25,50,100,300} 本、全 budget 共有 holdout、3 seed）: row 単位 ρ=0.506/0.461 は兄弟層リークでインフレしている | text few-shot が no-text few-shot を全 budget × 両方向で上回る。row→borehole で ρ は減衰すると予測（減衰幅は予測しない）

- **判定**: **CONFIRMED** — 6 budget × 2 方向 = **12/12 cell** で
  `depth_text` が `depth_only` を Spearman ρ で上回った（z-RMSE でも 12/12 で下回った）。
  副次予測（row→borehole で ρ 減衰）については後述のとおり
  **「行数を揃えた近似比較でのみ見える」**ので、正直に条件付きで報告する。

## 動機

旧 few-shot 曲線は **行**で budget を切っていたため、1 本のボーリングの兄弟層が
適応側と評価側の両方に現れる。ボーリングは層序・記録者・位置を共有するので、
これは適応値を押し上げる。旧公表値 ρ=0.506/0.461（n=1,000 行）は、まさにこの
リークの疑いがかかっていた。

今回の再設計:

- 評価 **holdout は target ボーリングの固定 50%（seed 0）**で、
  **全 budget・全 arm・全 seed で共有**。旧設計は budget ごとに holdout が違ったので
  budget 間の比較が原理的に成立していなかった（この修正のほうが重要かもしれない）。
- 適応 budget は **ボーリング本数** {0, 10, 25, 50, 100, 300}、
  holdout 外プールから seed ごとに抽出。
- 各 budget で `depth_only` と `depth_text` の両 arm を回す
  → 「テキストチャネルは target アーカイブのボーリング何本ぶんの価値か」に答える。

## 設定

| 項目 | 値 | 真の設定ファイル |
|---|---|---|
| 表現 | lithology_only（strength-stripped v2）, multilingual-e5, **source-fit** PCA-64 | [`backend/scripts/nc_fewshot_curve.py`](../../backend/scripts/nc_fewshot_curve.py) |
| モデル | HGB(400) | 同上 |
| budget | 0 / 10 / 25 / 50 / 100 / 300 **ボーリング** | -- |
| holdout | target ボーリングの 50%（seed 0 固定）、全 budget/arm/seed 共有 | -- |
| 乱数シード | 42 / 43 / 44 | -- |
| japan→uk | source 41,146 行 / target 18,157 行・2,597 本、holdout 8,730 行・1,298 本 | -- |
| uk→japan | source 18,157 行 / target 41,146 行・3,116 本、holdout 20,309 行・1,558 本 | -- |

再現:

```bash
cd backend
.venv/bin/python -m scripts.nc_fewshot_curve \
    --out ../docs/research/2026-08-12_fewshot_borehole_curve.json
```

## 結果

一次証拠: [2026-08-12_fewshot_borehole_curve.json](2026-08-12_fewshot_borehole_curve.json)

### Japan → UK（target-trained depth-only リファレンス: ρ 0.4084 / z-RMSE 1.1003）

| budget (本) | depth_only ρ (mean ± sd) | depth_text ρ (mean ± sd) | depth_only z-RMSE | depth_text z-RMSE | text 勝ち |
|---|---|---|---|---|---|
| 0 | 0.1157 ± 0.0118 | **0.3503 ± 0.0160** | 1.3006 | 1.1706 | ✓ |
| 10 | 0.1975 ± 0.0446 | **0.2687 ± 0.0920** | 1.2759 | 1.2446 | ✓ |
| 25 | 0.1405 ± 0.0599 | **0.3118 ± 0.0208** | 1.3013 | 1.2141 | ✓ |
| 50 | 0.1840 ± 0.0528 | **0.3376 ± 0.0370** | 1.2834 | 1.1877 | ✓ |
| 100 | 0.1758 ± 0.0111 | **0.4280 ± 0.0227** | 1.2908 | 1.1021 | ✓ |
| 300 | 0.1400 ± 0.0429 | **0.5062 ± 0.0046** | 1.3094 | 1.0206 | ✓ |

### UK → Japan（target-trained depth-only リファレンス: ρ 0.2838 / z-RMSE 1.2003）

| budget (本) | depth_only ρ (mean ± sd) | depth_text ρ (mean ± sd) | depth_only z-RMSE | depth_text z-RMSE | text 勝ち |
|---|---|---|---|---|---|
| 0 | 0.2705 ± 0.0029 | **0.3066 ± 0.0319** | 1.2229 | 1.1982 | ✓ |
| 10 | 0.2671 ± 0.0087 | **0.3413 ± 0.0222** | 1.2288 | 1.1709 | ✓ |
| 25 | 0.2610 ± 0.0139 | **0.3454 ± 0.0081** | 1.2319 | 1.1667 | ✓ |
| 50 | 0.2565 ± 0.0096 | **0.3853 ± 0.0038** | 1.2376 | 1.1383 | ✓ |
| 100 | 0.2578 ± 0.0014 | **0.4306 ± 0.0074** | 1.2374 | 1.1006 | ✓ |
| 300 | 0.2587 ± 0.0037 | **0.5076 ± 0.0057** | 1.2515 | 1.0322 | ✓ |

### Zero-shot 分解（同一 holdout 上）

| 方向 | depth_only | depth_text | depth_shuffled | target-trained depth-only 参照 |
|---|---|---|---|---|
| japan→uk | 0.1157 | **0.3503** | 0.2854 | 0.4084 |
| uk→japan | 0.2705 | **0.3066** | 0.1666 | 0.2838 |

（full target 全体での zero-shot は japan→uk 0.1181 / **0.3333** / 0.2731、
uk→japan 0.2575 / **0.2969** / 0.1568。）

## 解釈（正直に）

### 1. コスト換算の結果（Japan → UK）— 論文が必要としていた数字

UK 側の `depth_only` arm は **budget 300 本まで一度も ρ=0.198 を超えない**
（最大は budget 10 の 0.1975）。一方、**現地ボーリング 0 本**のテキスト転移は
**ρ = 0.350**。UK 自身のボーリングで学習した depth-only リファレンスは 0.408。
すなわち:

- テキストチャネルは **現地 0 本で、現地学習 depth-only モデルの 86%**
  （0.3503 / 0.4084 = 0.858）に到達する。
- そして **現地で 300 本掘って記載しても、テキスト無しでは届かない水準**
  （budget 300 の depth_only = 0.140 ≪ 0.350）を超えている。

**caveat（必ず併記する）**: この比較の相手である depth-only arm は
**budget を通じてフラットかつノイジー**（0.116 → 0.198 → 0.141 → 0.184 → 0.176 → 0.140、
sd は最大 0.060）で、単調な学習曲線になっていない。つまり
「300 本ぶんを超える」という言い方は、**弱くて飽和したローカル baseline に対する
超え方**であり、「300 本の掘削費用に相当する」という強い経済的主張には使えない。
UK 側の非テキスト共変量が depth と ground level しかない（W2 で既知の
covariate-poor アーカイブ）ことが背景にある。

### 2. UK → Japan

zero-shot text **0.3066** vs depth-only 0.2705。target-trained depth-only
リファレンスは **0.2838** で、テキスト arm は現地 0 本でこれを超える。
few-shot では **10 本（0.3413）でリファレンスを明確に上回る**。
こちらは depth-only が 0.257–0.271 の狭い帯で安定しており（sd ≤ 0.014）、
Japan→UK のようなノイズは無い。

### 3. Shuffled-embedding null は **方向で非対称** — 両方向を対称に報告する

| 方向 | depth_only | shuffled | text | shuffled が埋めた gap | 純内容の増分 |
|---|---|---|---|---|---|
| japan→uk | 0.1157 | **0.2854** | 0.3503 | (0.2854−0.1157)/(0.3503−0.1157) = **72%** | 0.350 − 0.285 = **+0.065** |
| uk→japan | 0.2705 | **0.1666** | 0.3066 | depth-only を**下回る**（負） | 0.307 − 0.167 = **+0.140** |

- **japan→uk**: shuffled 埋め込みだけで text−depth gap の約 **72%** が復元される。
  つまりこの方向の見かけの転移の大半は、**地質内容ではなく埋め込み空間の
  一般的な構造**（層の並び・分布の形）で説明できてしまう。
  真に内容由来と言えるのは **+0.065** に過ぎない。
- **uk→japan**: shuffled は **0.1666** で depth-only の 0.2705 を**下回る**。
  この方向では shuffled 埋め込みは積極的に**害**であり、
  純内容の増分は **+0.140** と大きい。
- **この 2 つを平均してはいけないし、都合のよい方向だけ載せてもいけない。**
  非対称そのものが所見である（source-fit PCA を UK の薄い語彙で張るか
  日本の厚い語彙で張るかで、shuffled null の性格が変わる）。
  論文では両方向の表をそのまま出し、
  「japan→uk の headline は shuffled 統制後に +0.065 まで縮む」と明記する。

### 4. Zero-shot は再設計の影響を受けていない（continuity）

full target 上の zero-shot ρ は **japan→uk 0.3333 / uk→japan 0.2969** で、
行 budget 時代の [2026-07-04 cross-archive transfer](2026-07-04_cross_archive_transfer.md)
の値と**完全に一致**する（depth_only 0.1181 / 0.2575、shuffled 0.2731 / 0.1568 も一致）。
zero-shot は budget の切り方に依存しないので当然であり、
**再設計が変えたのは few-shot arm だけ**であることの確認になる。

### 5. 副次予測「row→borehole で ρ は減衰する」について

- **budget の端点だけ見ると減衰していない**: 300 本で ρ = 0.5062 / 0.5076 に対し、
  行 budget 時代の n=1,000 行は 0.5059 / 0.4613。uk→japan はむしろ**上がっている**。
- ただし単位が違うので端点比較は不当。JSON の行数/本数から
  1 本あたり **6.99 行（japan→uk）/ 13.20 行（uk→japan）**なので、
  300 本 ≈ 2,098 行 / 3,961 行であり、旧 n=1,000 行より**多い**。
  行数を揃えた近似点（japan→uk ≈143 本、uk→japan ≈76 本）で曲線を読むと
  ρ ≈ 0.43–0.45 / 0.39–0.43 となり、旧 0.506 / 0.461 より**低い**＝
  予測どおりの減衰が見える。
- **正直な結論**: 減衰は「行数を揃えた近似比較では見える」が、
  評価集合も適応プールも旧設計と違う（旧は full target 評価、新は 50% ボーリング
  holdout 評価）ので、**厳密な matched comparison ではない**。
  事前登録は減衰幅を予測していないので、この所見は記述として残し、
  「兄弟層リークで 0.506 が何 pt 膨らんでいたか」を単一の数字で言うことはしない。

### 6. 小さいが実在する異常

japan→uk の budget 10 で、text arm は **zero-shot（0.3503）より下がる（0.2687 ± 0.0920）**。
現地 10 本の適応データは、source で張った表現に対して追加ノイズにしかなっていない。
sd も 0.092 と最大で、seed 依存が強い。曲線を「単調に上がる」と描写しないこと。

## 次のアクション

- [ ] Fig.（few-shot 曲線）に shuffled null を水平線として重ね、
      japan→uk の +0.065 / uk→japan の +0.140 を図中に明示
- [ ] budget 10 の落ち込みが「適応セットの regime 偏り」で説明できるか、
      抽出されたボーリングの regime 分布を確認
- [ ] 行 budget と本 budget を**同一 holdout 上**で並走させ、
      兄弟層リークの寄与を matched に測る（今回は近似比較に留まった）
- [ ] source-fit PCA を target-fit / joint-fit に変えたとき shuffled null の
      非対称が反転するか（表現空間側の説明の検証）

## 生成物

- 結果 JSON: [2026-08-12_fewshot_borehole_curve.json](2026-08-12_fewshot_borehole_curve.json)
- runner: [`backend/scripts/nc_fewshot_curve.py`](../../backend/scripts/nc_fewshot_curve.py)
- 事前登録: [2026-08-11_nc_text_preregistration.md](2026-08-11_nc_text_preregistration.md) P-T6
- 同 R1 キャンペーンの姉妹ノート:
  [P-T8 conformal grouped split](2026-08-12_pt8_conformal_grouped_split.md) /
  [P-T10 descriptor-family ablation](2026-08-12_pt10_descriptor_families.md)
- 前身（supersede 対象）: [2026-07-04 cross-archive zero-shot transfer](2026-07-04_cross_archive_transfer.md)
  の行 budget few-shot arm（zero-shot 部分は不変で有効）
- **diagnostics.png は無し** — 訓練を伴わない評価のみの再解析のため（上記 JSON が一次証拠）
