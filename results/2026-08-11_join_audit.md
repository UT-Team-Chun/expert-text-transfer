# Join 監査: 丸め座標 text-join の欠陥定量と ID-exact join への切替根拠（R0-2）

- **日付**: 2026-08-11
- **フェーズ**: NC pre-review 対応 R0（信頼の土台）
- **担当**: R. Okauchi
- **モデル訓練**: なし（データ監査のみ）

## 仮説 / 動機

NC 投稿前レビュー P0-6:「10m 丸め座標 + depth interval の text join は、密集地で
別ボーリングの層記載を SPT 行に付与し得る。監査せよ」。v4 parquet にボーリング
ID が無かったため丸め座標が事実上の識別子になっていた。R0-1 で identity spine
（`borings_japan_v4id.parquet` + `kunijiban_metadata.parquet`）を再構築し、本監査で
legacy join と ID-exact join を全数比較した。

## 設定

| 項目 | 値 | 真の設定ファイル |
|---|---|---|
| SPT 側 | `data/features/borings_japan_v4id.parquet`（2,663,955 行、162,459 ボーリング） | `scripts/attach_identity_to_parquet.py` |
| text 側 | `data/features/derived/soil_text_layers.csv`（1,155,359 層） | `scripts/extract_soil_text_from_xml.py` |
| メタデータ | `data/features/derived/kunijiban_metadata.parquet`（191,572 件） | `scripts/extract_kunijiban_metadata.py` |
| join 再現 | `join_soil_text_to_parquet.py` と同一の merge_asof 論理（embedding 列の代わりに層ラベル付与） | `scripts/audit_text_join.py` |

再現:

```bash
cd backend
.venv/bin/python -m scripts.audit_text_join \
    --out ../docs/research/2026-08-11_join_audit.json
```

## 結果（[2026-08-11_join_audit.json](2026-08-11_join_audit.json)）

### 衝突の規模（corpus_keys）
- 丸めキー 148,433 のうち **7.42% (11,017)** が複数ボーリングを保持
- ボーリングの **15.41%** が multi-key 下、**13.03% (21,168 本)** は
  **byte-identical な float32 座標**を共有（丸め幅調整では原理的に分離不能）
- 1 キー最大 **28 ボーリング**

### 衝突の悪性度（collision_mix、全 11,017 キー全数 — 従来は 1,200 キー標本）
- 別プロジェクト混在: **11.7%**、別調査会社: 9.3%、別調査年: 9.9%、
  **別 DTD 世代: 24.2%** → 座標 join のままでは provenance fold も汚染される

### 両 join の差分（join_delta — 決定的数値）
- legacy coord join: **1,536,599 行マッチ (57.68%)** — 論文記載の
  1,536,704 / 57.7% をほぼ厳密に再現（差 105 行）。監査は原稿の数値系と直結
- **coord マッチの 8.87% (136,294 行) は別ボーリング由来の記載**（実現ミス付与率。
  事前推計の ~16% は「曖昧候補が存在する行」の上界だった）
- **ID join は 1,619,172 行 (60.78%) にマッチ** — 正当なマッチを **+82,573 行純増**
  （coord 丸めがテキスト側と SPT 側で別キーに割れて取り逃していた分 87,701 行 −
  coord のみ 5,128 行）。ID join は「より正しい」だけでなく「より多い」
- 共通マッチのうち層ラベルが変わる行: 8.56%

### 境界感度（boundary_sensitivity — 査読者要求の ±ε 頑健性の土台）
- ID join マッチの 5.89% / 18.46% / 37.64% が層境界から 0.1 / 0.25 / 0.5 m 以内

## 解釈（正直に）

1. **欠陥は現実だが、破滅的ではない**: 実現ミス付与率 8.9%。しかも誤付与の一部は
   「同一ボーリングの重複提出ファイル」（identity 曖昧群 7.13%）に由来するため、
   物理的に誤った記載の率はさらに下がる。ただし transfer 系 headline
   （−21.2/−19.8/−11.3%）はもともとファイル内 join で**無影響**、影響を受けるのは
   parquet join 経由の −22.5%（in-dist LMC）と −7.3%（conservative arm）のみ。
2. **修正は改善のみ**: ID join はミス付与ゼロ化に加えマッチ +5.4%。再計算
   （R1、クラスタ）で −22.5% がどう動くかが残る確認事項。
3. 8.87% の誤テキストは典型的に「同一 10m キー内の隣接ボーリング」であり層序が
   類似 → 効果を大きく歪めていない可能性が高いが、これは再計算で証明する
   （憶測で済ませない）。
4. 論文 §2 の join 記述（丸めを float 精度対策としてのみ説明）は R2 で全面改訂し、
   本監査を SI 新節として掲載する。

## 次のアクション
- R1-0 prereg に「ID join 済み母集団」を primary population として明記
- R1 で −22.5% / −7.3% を ID join parquet 上で再計算（utens k8s）
- SI 節「Join audit」の執筆（R2）
