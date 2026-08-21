# PRE-REGISTRATION — NC text-transfer 統計強化キャンペーン (R1)

**Committed BEFORE any R1 computation.** 本文書のコミット時点で、以下の
P-T 系列のいかなる結果も未知である。ハーネス改修コード（R1-1)は本文書の
後にコミットされるが、いかなる実験結果も本文書に先行しない。
NMI campaign の prereg 慣行（P-ID + 数値バー + stop clause + verdicts）に従う。
これは text-transfer 系列（paper_2_national）で初の事前登録である。

## 検証する原理

> 専門家の層記載テキストの細粒度内容は、(a) 最良の非テキスト構造化情報を
> 与えた後にも SPT-N の予測情報を持ち、(b) その効果は行単位ではなく
> ボーリング単位の統計処理の下でも消えず、(c) 記載組織・年代・DTD 世代を
> 越えて残存する。

反証可能性: (a)(b)(c) のいずれも、下記の数値バーを満たさなければ
対応する主張を論文から降格・削除する。

## 共通プロトコル（全 P-T で凍結）

- **母集団**: ID-join 済み text-bearing positive-N (0<N≤100) 行
  （`borings_japan_v4id.parquet` 系譜、join は `--join-key file`）
- **モデル**: HGB (max_iter=400, lr=0.05)、per-fold PCA-64（train 側のみ、
  rs=0）— 既存 `_evaluate_lro` と同一
- **テキスト腕**: strength-stripped v2 (lithology-only) 埋め込み
- **null**: **borehole ブロック置換**（ボーリング単位で embedding ブロックを
  入れ替え; 層内順序は保持）、region×lithology-macro 層内で置換
- **置換数**: ≥1000 / fold、real と同じ HGB seed 系列で seed-paired、
  p は (1+r)/(1+n) 補正、fold 統合は Stouffer（p の平均は使わない）
- **CI**: borehole ブロック BCa（paired per-borehole loss、refit なし、
  full-population 対応）+ region-level bootstrap (10^4) 併記
- **metric**: borehole 加重 RMSE（primary）、MAE / Spearman ρ /
  N≥10・N≥30 threshold accuracy（secondary）
- **計算の二段構え**: permutation p は balanced subsample スケール
  （500 files/region、独立 3 subsample × 3 HGB seed）で確定し、
  point estimate と BCa CI は full text-bearing population で計算する
- 全実験は docs/research ノート + results_table 行 + provenance JSON 付き

## Predictions

| ID | 内容 | バー（これを満たせば CONFIRMED） |
|---|---|---|
| **P-T1** | **Primary estimand**: 最良非テキスト baseline（depth + elev + river/coast + regime(8) + litho-macro(14) + era(9) + train-side KNN spatial prior）に対する strength-stripped text の増分、Japan 8-region LRO | 効果が負 in ≥7/8 regions、grouped-permutation Stouffer p<0.05、full-population BCa 95% CI が 0 を除外。point estimate は [−20%, −3%] と予測 |
| **P-T2** | UK 5-region で同一 spec の複製（loca_id グループ） | 負 in ≥4/5 regions、grouped p<0.05 |
| **P-T3** | **行 null → borehole ブロック null の感度**: 同一データ・同一 baseline で null の単位だけ変える | Japan headline content effect の変化 |Δ| < 5 percentage points（= 既報が行シャッフルのアーティファクトでない） |
| **P-T4** | **Provenance folds**: leave-project-out（上位プロジェクト）/ leave-contractor-out（上位 8 社）/ leave-year-out（年代 bin）/ leave-DTD-out | 各 family で mean effect 負 かつ held-out unit の ≥70% が負 |
| **P-T5** | 固有名詞 strip（調査名・地名 token 除去）+ テンプレート正規化後も効果残存 | 減衰 ≤5 pt かつ負 in ≥7/8 |
| **P-T6** | **Few-shot を borehole 単位に再設計**（budget {0,10,25,50,100,300} 本、全 budget 共有 holdout、3 seed）: row 単位 ρ=0.506/0.461 は兄弟層リークでインフレしている | text few-shot が no-text few-shot を全 budget × 両方向で上回る。row→borehole で ρ は減衰すると予測（減衰幅は予測しない） |
| **P-T7** | **ID-join 再計算**: in-distribution 3-fold LMC の text 効果（旧 −22.5%）を identity-join parquet で再計算 | 新効果が [−27.5%, −17.5%]（±5 pt 以内）。8.9% の誤テキストはノイズ源だったので、効果は不変〜微改善と予測 |
| **P-T8** | **Conformal を borehole ブロック分割で再実行**（cal/eval を mesh/borehole 単位分割、interval width + Winkler 併記） | marginal coverage gap at α=0.95 ≤ 0.01（行分割の ~0.000 からの悪化を許容しつつ実用域）; width は行分割比 +20% 以内 |
| **P-T9** | **真の座標フリー腕**: `--zero-fourier --no-residual-geo` の LRO | 予測: raw-coordinate residual の除去は extrapolation を悪化させない（RMSE ≤ zero-fourier 単独 +2%）。どちらに出ても informative として報告 |
| **P-T10** | **Descriptor-family ablation**（parser を text 由来のみに分離した上で、grain-size / weathering / water-state / colour / angularity / 組成% を 1 family ずつ除去） | 探索的（バー無し）。ただし parser rung の再計測は text-由来のみ構成で行い、archive コード混入版と分離して報告する |

## Stop clauses

1. **P-T1 が負に出ない、または p≥0.05** → 「transferable content」を主張から
   降格し、論文は in-distribution 増分 + 正直な null 報告に再構成する。
   別のベースライン・別の null を探して「効かせる」ことはしない。
2. **P-T3 で |Δ| ≥ 5 pt** → 既報の行 null 由来の全数値（−21.2/−19.8/−11.3%
   等）を隔離し、grouped 数値のみを正史とする。
3. **P-T7 で範囲外** → 旧 join 由来の全 in-distribution 数値を隔離、
   新数値のみ報告。
4. バーに達しなかった予測は verdicts 文書で REFUTED と記録し、
   silent reframe をしない。

## Amendment 規律

追加・変更は本文書への追記コミットとして行い、対象実験の実行前で
あることをコミットメッセージに明記する。
