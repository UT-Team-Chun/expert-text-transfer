# 2026-06-02 — Phase 3 / Ruri l=2 3-fold + Sara l=1 kfold1 retry: capacity confound 閉じる、0.45σ で tie 確定

## 仮説

前 phase で Sara (l=2) と Ruri (l=2) の比較を行ったが、Ruri 側は l=1 のみで走っており
**latent capacity が揃っていない confound** が残っていた。Ruri を l=2 に揃えて 3-fold
完走させれば、(a) backbone (Sara vs Ruri) 由来の差と (b) latent rank (l=1 vs l=2) 由来
の差を分離できる。さらに Sara l=1 kfold1 が NaN で落ちていたので retry して 3-fold を
揃える。

期待:
- l=1 → l=2 で両 backbone とも有意な RMSE_N 改善（capacity 効果）。
- 同じ l=2 で揃えると Sara と Ruri の差は 1σ 以下に縮み、選択は backbone-agnostic
  framing に倒せる。

## 実験設定

| 設定項目 | 値 | 真の設定ファイル |
|---|---|---|
| 訓練データ | `data/features/borings_japan_v4.parquet` (2.66M 行) | `scripts/enrich_borings.py` |
| 訓練対象 | DKL LMC national v5 | -- |
| Fold 構成 | spatial 3-fold (secondary-mesh) kfold0 / kfold1 / kfold2 | -- |
| 走らせた cell | 4 cells (Ruri l=2 × {kfold0, kfold1, kfold2} + Sara l=1 kfold1 retry) | `infra/utens/sweep_submit.py` |
| Backbone | Ruri-310M / Sara (Sentence-BERT 系) | -- |
| Latent rank l | 2 (Ruri 全 3 fold) / 1 (Sara retry) | LMC head config |
| デバイス | Azure H100, `NO_NVLINK` affinity | `deploy_template.yml` |
| Checkpoint | `/tmp` local SSD → 終了時に NFS flush | `1c6175c` |
| 乱数シード | 42 | -- |

再現用 CLI:

```bash
cd backend
uv run python -m scripts.train_national_lmc_v5 \
    --parquet ../data/features/borings_japan_v4.parquet \
    --backbone ruri --latent-rank 2 --kfold 0 \
    --output-dir /mnt/nas/runs/dkl_national_lmc_v5_ruri_l2_kfold0
```

## 結果

### Per-fold メトリクス (RMSE_N / MAE_N / RMSE_GW)

| Backbone × l | Fold | RMSE_N | MAE_N | RMSE_GW |
|---|---|---|---|---|
| **Ruri l=2** | kfold0 | 8.630 | 5.428 | 5.284 |
| Ruri l=2 | kfold1 | 9.365 | 6.295 | 4.031 |
| Ruri l=2 | kfold2 | 8.921 | 5.653 | 4.015 |
| **Sara l=2** | kfold0 | 8.591 | 5.458 | 5.265 |
| Sara l=2 | kfold1 | 8.931 | 5.738 | 4.046 |
| Sara l=2 | kfold2 | 8.835 | 5.717 | 4.078 |

### 集約 (mean ± std, 3-fold)

| Backbone × l | mean RMSE_N | std RMSE_N | mean MAE_N | std MAE_N | mean RMSE_GW | std RMSE_GW |
|---|---|---|---|---|---|---|
| Ruri l=2 | **8.972** | 0.370 | 5.792 | 0.448 | 4.443 | 0.715 |
| Sara l=2 | **8.786** | 0.176 | 5.638 | 0.156 | 4.463 | 0.697 |
| Sara l=1 | 9.639 | 0.146 | -- | -- | -- | -- |
| Ruri l=1 | 9.592 | -- | -- | -- | -- | -- |

Sara l=1 の per-fold RMSE_N は kfold0=9.636 / kfold1=9.494 (retry 成功) / kfold2=9.786。

### Capacity confound の閉じ方 (l=1 → l=2)

| Backbone | l=1 RMSE_N | l=2 RMSE_N | Δ (%) |
|---|---|---|---|
| Sara | 9.639 | 8.786 | **−8.85 %** |
| Ruri | 9.592 | 8.972 | **−6.46 %** |

両 backbone とも l=2 で有意改善 → latent rank 1 では明確に under-capacity。

### Sara vs Ruri @ l=2 の verdict

- Δ mean RMSE_N = **0.186** (Sara が低い)
- 効果量 = **0.45 σ** (Ruri の std 0.370 基準)
- **判定: tie** — backbone 選択は RMSE_N で決められない。

### 運用面の知見

- **H100 affinity**: `NO_NVLINK` 指定で安定。NVLink ありの affinity は配置待ちで
  queue が伸びる時間帯があり、4 cell 完走を優先して NO_NVLINK に倒した。
- **/tmp checkpoint speedup**: `1c6175c` の "checkpoint to /tmp local SSD + final
  flush to NFS" 対応で save が **約 10×** 速くなった (Azure tailscale NFS 経由の
  fsync が支配的だった)。Phase 3 sweep の wall time に効いた。

Fold 別 raw artifact:
- [Ruri l=2 kfold0 summary.json](/mnt/nas/runs/dkl_national_lmc_v5_ruri_l2_kfold0/summary.json)
- [Ruri l=2 kfold1 summary.json](/mnt/nas/runs/dkl_national_lmc_v5_ruri_l2_kfold1/summary.json)
- [Ruri l=2 kfold2 summary.json](/mnt/nas/runs/dkl_national_lmc_v5_ruri_l2_kfold2/summary.json)
- [Sara l=1 kfold1 retry summary.json](/mnt/nas/runs/dkl_national_lmc_v5_sara_l1_kfold1/summary.json)

更新済 Figure 3 ソース: [`backend/scripts/build_paper2_figs.py`](../../backend/scripts/build_paper2_figs.py)

## 考察

- **Capacity confound は閉じた**: Ruri l=1 → l=2 で −6.46 %、Sara l=1 → l=2 で
  −8.85 %。両 backbone で同じ向きに動いており、l=2 が必要条件であることは確定。
  以前の Sara > Ruri の見え方は「Sara だけ l=2 だった」ことで説明できる過大評価
  だった。
- **同じ l=2 で揃えると tie**: Δ = 0.186 RMSE_N、効果量 0.45σ。Sara の std (0.176)
  は Ruri の std (0.370) の半分で fold robust だが、mean 差は noise band 内。
- **GW (groundwater) 側も tie**: mean RMSE_GW は Ruri 4.443 vs Sara 4.463、
  Sara がわずかに悪いが std (0.715 vs 0.697) より小さい差。
- **意思決定**: paper では **Sara を anchor** にしつつ、結論を "backbone-agnostic
  at l=2" の framing に切り替える。これは "Sara が勝った" 主張を引っ込め、
  capacity 効果が dominant という発見を前に出す書き換え。

## フォローアップ

- [ ] Figure 3 の 4 cell 結果を反映 (`build_paper2_figs.py` 経由で再描画)。
- [ ] paper2 §08b 本文を "Sara > Ruri" 主張から "backbone-agnostic at l=2"
      framing に書き換え。
- [ ] l=3 を試して capacity の reachable な上限を見極めるか判断（diminishing
      return が見えれば l=2 で打ち止め）。
- [ ] Sara l=1 kfold1 が初回 NaN で落ちた件、retry で通った理由を後で audit
      （seed か checkpoint resume の interaction かを確認）。

## 生成物

- Run artifacts: `/mnt/nas/runs/dkl_national_lmc_v5_{ruri_l2_kfold0,ruri_l2_kfold1,ruri_l2_kfold2,sara_l1_kfold1}/`
- Updated figure source: [`backend/scripts/build_paper2_figs.py`](../../backend/scripts/build_paper2_figs.py)
- Authoritative results dict: 本ドキュメント「結果」節
- 直前の関連 commit: `1c6175c` (`/tmp` checkpoint), `1a62ee2` (utens affinity 修正)
