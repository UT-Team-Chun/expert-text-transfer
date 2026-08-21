# P-T8: Conformal を borehole / site grouped split で再実行 — 兄弟層依存は marginal coverage を動かさない（INFORMATIVE NEGATIVE）

- **日付**: 2026-08-12
- **フェーズ**: NC pre-review 対応 R1（事前登録キャンペーン）
- **担当**: R. Okauchi
- **モデル訓練**: なし（既存 `predictions.npz` の再評価のみ。quantile fit だけなので
  `diagnostics.png` は生成されない — 一次証拠は下記 JSON）

## 事前登録と判定

- **P-ID**: **P-T8** — [NC text-transfer PRE-REGISTRATION (R1)](2026-08-11_nc_text_preregistration.md)
- **事前登録バー（逐語）**:

  > **Conformal を borehole ブロック分割で再実行**（cal/eval を mesh/borehole 単位分割、interval width + Winkler 併記） | marginal coverage gap at α=0.95 ≤ 0.01（行分割の ~0.000 からの悪化を許容しつつ実用域）; width は行分割比 +20% 以内

- **判定**: **CONFIRMED** — ただし科学的な中身は **informative negative**。
  - α=0.95 の marginal gap：borehole split で最悪 **+0.00341**（kyushu_okinawa）、
    site split で最悪 **−0.00527**（chugoku）、全国フルコーパスで **−0.00017 / −0.00024**。
    全 cell がバー（≤ 0.01）の内側。
  - width 比：LRO 全 cell・全 α で行分割比 **0.989–1.010 倍**（バーは +20% 以内）。
  - すなわちバーは達成したが、達成の仕方は「grouped にしても何も起きなかった」であり、
    「grouped でも耐えた」ではない。区別して報告する。

## 動機

公表済みの target-region conformal 図（marginal gap ≈ 0.000）は、cal/eval を
**ランダムな行**で分割している。1 ボーリングあたり平均 9.3 層あるため、同一
ボーリングの兄弟層が分割の両側に落ち、ボーリング固有の誤差が cal 側と eval 側で
共有される。査読側の指摘は「それは coverage を簡単にしているのではないか」。
指摘は原理として正しいので、**split unit だけを変えた 3 条件**
（row / borehole / ~500 m site）で同一の予測 artefact を再評価した。
coverage は width で買えるので、**mean interval width と Winkler interval score を
必ず併記**する（査読 P0-8）。

## 設定

| 項目 | 値 | 真の設定ファイル |
|---|---|---|
| 予測 artefact | `data/runs/dkl_national_full_v2/predictions.npz`（2,663,955 行）+ `data/runs/dkl_national_lro_<region>/predictions.npz` × 8 | `run_leave_region_out` |
| ボーリング identity | `data/features/borings_japan_v4id.parquet` の `boring_file` | [`backend/scripts/attach_identity_to_parquet.py`](../../backend/scripts/attach_identity_to_parquet.py) |
| split unit | `row`（legacy）/ `borehole`（`boring_file` 単位）/ `site`（lat・lon を小数第 2 位で丸めたセル、丸め半幅 ≈ 500 m） | [`backend/scripts/run_mondrian_recal_grouped.py`](../../backend/scripts/run_mondrian_recal_grouped.py) |
| conformal | `ConformalCalibrator.fit_mondrian`（regime 別 quantile、`min_group_n=30`、fallback は marginal） | `backend/national/evaluation/calibration.py` |
| α | 0.5 / 0.8 / 0.95 | -- |
| 乱数シード | 42 / 43 / 44（`cal_fraction=0.5`） | -- |
| 指標 | marginal coverage / per-regime coverage / mean interval width / **Winkler interval score** | -- |

再現:

```bash
cd backend
# 全国フルコーパス
.venv/bin/python -m scripts.run_mondrian_recal_grouped \
    --run dkl_national_full_v2 \
    --out ../docs/research/2026-08-12_conformal_grouped_split.json
# leave-region-out 8 地方
for r in hokkaido tohoku kanto chubu kansai chugoku shikoku kyushu_okinawa; do
  .venv/bin/python -m scripts.run_mondrian_recal_grouped \
      --run "dkl_national_lro_${r}" --out "/tmp/pt8/${r}.json"
done
```

## 結果

一次証拠: [2026-08-12_conformal_grouped_split.json](2026-08-12_conformal_grouped_split.json)（全国フルコーパス）と
[2026-08-12_conformal_grouped_split_lro.json](2026-08-12_conformal_grouped_split_lro.json)（8 地方 LRO + フルコーパス集約）。
**provenance 注記**: 本ノート執筆開始時点で LRO 集約 JSON は未着地であり、数値は
`/tmp/pt8/{hokkaido,tohoku,kanto,chubu,kansai,chugoku,shikoku,kyushu_okinawa}.json`
の 8 個の地方別ファイル（および同ディレクトリの `all_lro.json`）から読んだ。
執筆中に集約ファイルが `docs/research/` に着地したので突き合わせ、
**8 cell 分の内容が per-region ファイルと完全一致**（`==` で検査）、
フルコーパス側も単独 JSON と完全一致であることを確認済み。

### 全国フルコーパス（`dkl_national_full_v2`、全 2,663,955 行、n_cal ≈ 1.33M 行 — borehole split では 81,229 本、3 seed 平均）

| α | split | coverage | gap | mean width | Winkler |
|---|---|---|---|---|---|
| 0.50 | row | 0.50009 | +0.00009 | 4.6071 | 17.2083 |
| 0.50 | borehole | 0.49967 | −0.00033 | 4.5952 (−0.26%) | 17.2079 (−0.00%) |
| 0.50 | site | 0.50025 | +0.00025 | 4.6141 (+0.15%) | 17.2279 (+0.11%) |
| 0.80 | row | 0.80005 | +0.00005 | 16.5692 | 28.6631 |
| 0.80 | borehole | 0.79997 | −0.00003 | 16.5563 (−0.08%) | 28.6708 (+0.03%) |
| 0.80 | site | 0.79944 | −0.00056 | 16.5567 (−0.08%) | 28.7136 (+0.18%) |
| 0.95 | row | 0.95021 | +0.00021 | 33.8673 | 44.3578 |
| 0.95 | borehole | 0.94983 | −0.00017 | 33.8078 (−0.18%) | 44.3803 (+0.05%) |
| 0.95 | site | 0.94976 | −0.00024 | 33.8250 (−0.12%) | 44.4305 (+0.16%) |

### Leave-region-out 8 地方（各 3 seed 平均、α=0.95 の coverage と、3 α を通じた row 分割との最大乖離）

| 地方 | held-out 行数 | ボーリング数 | cov@.95 row | cov@.95 borehole | cov@.95 site | max \|Δcov\| (bore / site) | max Δwidth (bore / site) |
|---|---|---|---|---|---|---|---|
| hokkaido | 216,692 | 15,690 | 0.94939 | 0.94902 | 0.94858 | 0.0034 / 0.0027 | 0.33% / 0.43% |
| tohoku | 287,496 | 17,751 | 0.95028 | 0.95256 | 0.95109 | 0.0045 / 0.0062 | 0.55% / 1.11% |
| kanto | 495,725 | 22,352 | 0.95014 | 0.94883 | 0.95072 | 0.0027 / 0.0030 | 0.47% / 0.61% |
| chubu | 457,691 | 26,894 | 0.94996 | 0.95003 | 0.94924 | 0.0017 / 0.0074 | 0.33% / 1.56% |
| kansai | 493,613 | 27,758 | 0.94957 | 0.94903 | 0.95173 | 0.0035 / **0.0176** | 0.85% / **4.97%** |
| chugoku | 417,227 | 30,514 | 0.94948 | 0.94982 | 0.94473 | 0.0014 / 0.0102 | 0.39% / 1.60% |
| shikoku | 242,488 | 17,892 | 0.95040 | 0.95084 | 0.95007 | 0.0030 / 0.0014 | 0.48% / 0.15% |
| kyushu_okinawa | 431,229 | 27,264 | 0.94939 | 0.95341 | 0.94831 | **0.0057** / 0.0015 | 0.93% / 0.64% |

**乖離幅の正直な要約**（集約 JSON の `interpretation` 文は 8 地方平均で書かれているので、
per-cell 最悪値と分けて記す）:

| 比較 | 8 地方平均（集約 JSON の値） | per-cell 最悪 |
|---|---|---|
| borehole vs row, coverage | 最悪地方 gap +0.00565 (α=0.8) | **0.0057**（kyushu_okinawa, α=0.8） |
| site vs row, coverage | 最悪地方 gap +0.01652 (α=0.5) | **0.0176**（kansai, α=0.5） |
| borehole vs row, width | −0.11%…+0.12% | **0.93%**（kyushu_okinawa, α=0.95） |
| site vs row, width | −0.12%…+0.69% | **4.97%**（kansai, α=0.5） |
| borehole vs row, Winkler | −0.11%…−0.08% | **1.36%**（kyushu_okinawa, α=0.95） |
| site vs row, Winkler | −0.20%…+0.28% | **1.88%**（kansai, α=0.5） |

つまり「row 分割を **0.006 coverage / 約 1% width / 約 0.3% Winkler** の範囲で再現する」
という要約は **borehole split（と、地方平均で見た site split）については正しい**。
site split の per-cell では α=0.5 の kansai が coverage 0.0176・width 4.97% とはみ出す
（α が小さいほど区間が狭く、site グループ化による有効標本減が効きやすい）。
**論文で引用するのは borehole split の数値に限定する**のが安全。

### 条件付き（per-regime）coverage — 生き残る caveat

borehole split・α=0.95・seed 42 の LRO 全 58 regime cell の中央値は 0.9515、
最小 0.9074、最大 1.0000。バーの外側に落ちる worst cell:

| 地方 | regime | coverage |
|---|---|---|
| shikoku | 6 (LIMESTONE) | **0.9074** |
| hokkaido | 2 (VOLCANIC_ASH) | 0.9389 |
| kyushu_okinawa | 6 (LIMESTONE) | 0.9404 |
| chubu | 7 (UNKNOWN) | 0.9423 |

α=0.5 では条件付きの荒れはさらに大きい（同じ split・seed で kanto regime 2 = 0.1548、
chubu regime 6 = 0.3567、shikoku regime 6 = 0.4259 ↔ chubu regime 2 / kanto regime 6 は 1.0000）。
フルコーパスでも site split・α=0.5・seed 42 の regime 6 が 0.1219 と落ちる
（seed 43/44 では 0.6917 / 0.7392 と振れる）＝ **希少 regime × 狭い区間**で
Mondrian bin が不安定になる。

## 解釈（正直に）

1. **査読者の懸念は原理としては正しいが、このデータでは immaterial**。
   grouped split は「同一ボーリングの誤差共有」を確かに断つが、marginal quantile は
   どちらの分割でも **10 万本以上の残差**から推定される（LRO で最小の hokkaido でも
   n_cal = 108,107 行 / 7,845 ボーリング、フルコーパスは 1.33M 行 / 81,229 ボーリング）。
   グループ化が変えるのは conformity score の**有効標本数**であって、推定される
   **quantile の値そのものではない**。だから coverage も width も Winkler も動かない。
   これは「頑健性が示された」というより「この検査は情報を持たなかった」という
   negative result であり、そう書く。
2. **width と Winkler を併記した意味**: coverage は区間を広げれば買える。
   ここでは width が行分割比 ±1%（borehole）で、Winkler も同水準。
   つまり「coverage を保つために区間を広げた」わけではない、と言い切れる。
   coverage 単独の一致だけを見せるより主張が強い。
3. **生き残る caveat は marginal ではなく conditional**。
   split unit をどう変えても marginal は α に張り付くが、**geological regime 別では
   ばらつく**（α=0.95 で 0.907–1.000、α=0.50 では 0.15–1.00）。
   LIMESTONE / VOLCANIC_ASH / UNKNOWN といった希少 regime で under-cover するのが
   典型。Mondrian binning はこれを緩和するために入れているが、消してはいない。
   論文の coverage 主張は **marginal であることを明示**し、per-regime 表を SI に置く。
4. **artefact bug（今後の全消費者への警告）**: LRO runner は `predictions.npz` に
   **訓練 fold 側の `regime` 配列**を保存していた。kanto では
   **2,168,230 行に対して予測は 495,725 行**（hokkaido 2,447,263 / tohoku 2,376,459 /
   chubu 2,206,264 / kansai 2,170,342 / chugoku 2,246,728 / shikoku 2,421,467 /
   kyushu_okinawa 2,232,726 も同様に train-sized）。
   既に [`backend/scripts/run_tta_lro.py:143`](../../backend/scripts/run_tta_lro.py) に
   quirk として記述がある（TTA では regime を消費しないのでサイズ不一致フィールドを
   警告付きで捨てている）。
   [`backend/scripts/run_mondrian_recal_grouped.py`](../../backend/scripts/run_mondrian_recal_grouped.py)
   は LRO run に対しては **v4id parquet の `regime_code` を正とする**。
   その際、held-out region の bounding box 選択に対して `y_true` の
   **byte 等価性を検証**してから identity を付ける（不一致なら `SystemExit`）。
   → **LRO の npz の `regime` フィールドは、いかなる将来の consumer も信用してはならない。**
   Mondrian bin をこの配列から作った解析があれば、bin が全部でたらめになる。

## 次のアクション

- [ ] LRO runner 側で `predictions.npz` の `regime` を held-out fold の配列に修正
      （または当該フィールドを削除して parquet 参照を強制）。今は下流での回避に頼っている
- [ ] per-regime coverage 表（α=0.95、borehole split）を SI 表として整形
- [ ] 希少 regime（LIMESTONE / VOLCANIC_ASH）の under-cover を Mondrian の
      `min_group_n` 引き上げ or 階層 pooling で改善できるか（別実験）
- [ ] site split の per-cell 外れ（kansai α=0.5）が丸め幅由来か標本数由来かの切り分け

## 生成物

- 全国フルコーパス結果: [2026-08-12_conformal_grouped_split.json](2026-08-12_conformal_grouped_split.json)
- LRO 8 地方 + 集約: [2026-08-12_conformal_grouped_split_lro.json](2026-08-12_conformal_grouped_split_lro.json)
  （執筆時の一次読み取りは `/tmp/pt8/*.json`、内容一致を確認済み）
- runner: [`backend/scripts/run_mondrian_recal_grouped.py`](../../backend/scripts/run_mondrian_recal_grouped.py)
- 事前登録: [2026-08-11_nc_text_preregistration.md](2026-08-11_nc_text_preregistration.md) P-T8
- 同 R1 キャンペーンの姉妹ノート:
  [P-T10 descriptor-family ablation](2026-08-12_pt10_descriptor_families.md) /
  [P-T6 borehole 単位 few-shot 曲線](2026-08-12_pt6_fewshot_borehole_curve.md)
- 前身: [2026-05-29 Phase C Mondrian per-regime conformal recalibration](2026-05-29_phase_c_mondrian_recal.md) /
  [2026-08-11 join 監査](2026-08-11_join_audit.md)（identity spine の出所）
- **diagnostics.png は無し** — 訓練を伴わない評価のみの再解析のため（上記 JSON が一次証拠）
