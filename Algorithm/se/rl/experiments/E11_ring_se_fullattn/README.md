# E11：Ring_SE 選候選範圍，100 顆完整輸入＋範圍旗標＋Transformer（FullNet）

- 狀態（10/26）：b1000 FullNet（plain_delta、soft τ=1、全 0 也訓練）訓練＋lifetime 評估中（run.sh）。
- 動機（使用者 10/26）：E10 範圍外平均‖最大沒幫助；改成 100 顆完整輸入、不池化，讓網路自己看範圍內外的關係。
  範圍不給起點／終點數字讓網路自己推，而是換成每顆 1 個旗標（資訊相同）。
- 和 E07 共用：資料（E07 results/data）、env、RoundStore／norm_stats、train_sup 的 load_labels／PairData／loss_fn／metrics。

## 網路（每個候選，4 個候選共用權重）

每顆 [token 25（z-score）‖ 在範圍內 1] → 線性 26 → 64，前面加 CLS → 101 × 64
→ Transformer 2 層（d 64、4 頭、FFN 128、pre-norm、無 dropout）→ LayerNorm → CLS 64
→ [CLS 64 ‖ 候選 10 ‖ 全域 10] ＝ 84 → 128 → 64 → 1；4 個 logits 做 softmax。

- 88,065 參數；不加 sensor 編號位置嵌入（避免背地圖，`--full 8` 的教訓）。
- 冒煙測試：候選順序換了分數跟著換（等變）。
- 推論一回合（64 組 × 4 候選）：GPU 3.7 ms、單執行緒 CPU 128 ms；同一回合 128 個子代 fitness 約 23.6 ms（184 µs／次，E09 量的）。

## 檔案

- `fullnet.py`：FullNet、`load_fullnet`。
- `train_full.py`：訓練＋test → `results/full_b{B}_m{N}_{label}{loss}/seed0/`（評估分塊 1024 組）。
- `evaluate_full.py`：lifetime 評估（`--algorithm e11_full`，checkpoint 由 `E11_CKPT` 給）。
- `run.sh`：訓練 → MAP01–06 × seed 7–11 的 1000／3000／10000 → `experiment_results/s711_e11_full_b1000_m140_plain_delta_soft1_eval{B}`。
- `results` → `runs/2027WSN/rl/E11_ring_se_fullattn`。

## 結果

b1000、plain_delta、soft τ=1、全 0 也訓練、140 張、val 早停（20 epoch，best 19；一個 epoch 154 秒）；test 30 張，同分算對：

| 模型 | 全部交易 | 正值交易選到最大 | 標籤比 |
| --- | --- | --- | --- |
| E07 SupNet | 59.29% | 50.04% | 57.31% |
| E08 RelNet | 62.43% | 53.90% | 61.20% |
| E10 OutNet | 59.94% | 50.84% | 58.14% |
| **E11 FullNet** | 60.40% | 51.40% | 58.65% |

lifetime（MAP01–06 × seed 7–11，30 場，成對 Wilcoxon）：

| 預算 | E11 | vs ring_se | vs E07 | vs E08 | vs E10 |
| --- | --- | --- | --- | --- | --- |
| 1000 | 948.3 | +5.37%（24/6，p<.001） | +0.33%（p=.75） | −0.16%（p=.81） | +1.13%（p=.32） |
| 3000 | 1023.6 | +2.66%（25/5，p=.001） | +0.07%（p=.94） | +0.53%（p=.96） | +0.18%（p=.93） |
| 10000 | 1059.6 | −0.42%（p=.57） | +0.02%（p=.83） | −0.06%（p=.68） | −0.89%（p=.52） |

→ 100 顆完整輸入＋Transformer 準確度比 E07 +1.4 點（仍低於 E08），lifetime 對 E07／E08／E10 全 ns。
