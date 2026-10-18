# E04_ring_se_seg8

- 日期：2026-10-17
- 狀態：**seed 0 訓練與 lifetime 完成**（2026-10-18）
- 基底演算法：`ring_se`（Ring_SE，SensorEncoding，n8 w8 μ0.4，EN 更新），**h 改 8、範圍改成整段**
- 設計圖：`design_A.svg`（設計 A）、`design_B.svg`（設計 B）；`design.svg` 是兩版合在一起的最早提案

## 1. 目的

E03 的「4 個隨機候選範圍選 1」作弊版天花板太低（1000 +7.5%，10000 只 +1.5%）。
E04 改成環固定切 8 段、整段交換，網路直接選段。8 段整段的作弊版在 1000 時對 8 段隨機 +7.8%（28/2）
（`tmp/whole_segment8`），所以只在預算 1000 訓練和評估。

1. 用 RL（不先做監督式）能不能學到選段，贏 8 段隨機？
2. 段之間做 attention（設計 B）有沒有比經典 Dueling 串接（設計 A）好？

## 2. 問題設定

- **決策單位**：每回合每組 searcher-good 交易（8 × 8 = 64 次／回合；預算 1000 約 8 回合）。
- **動作**：8 段選 1。sensor 順序是「由內到外的距離環、環內依角度」，段 0 最內圈（靠近 BS）、段 7 最外圈，
  段 k 和 k±1 相鄰。選到第 k 段就交換第 k 段全部 sensor，突變也只在第 k 段內；EN 更新同 ring_se。
- **環境**：fitness v3、SensorEncoding；沿用 E01 的訓練快照（190 張訓練圖、10 張驗證圖）；測試 MAP01–06。

## 3. 輸入（兩版共用，`env.py`）

| 部分 | 形狀 | 內容 |
| --- | --- | --- |
| sensor token | 8 段 × 13 × 25 | S 解碼 10 ‖ G 解碼 10 ‖ E/E0、離 BS 距離 ‖ 差異旗標 3（同 E03）；12 顆的段補 1 個空位並遮罩 |
| 段特徵 | 8 × 7 | 瓶頸分數 b_k（有效差異 sensor 在 S、G 的緊迫度總和）、有效差異數、無效差異數、範圍外影響 4（同 E03） |
| 全域 | 10 | 同 E03 |

所有特徵做 z-score 標準化，平均／標準差由 `norm_stats.py` 用隨機選段跑 40 段最佳化算一次（`data/norm.npz`）。

## 4. 網路

| 子資料夾 | 網路 | 參數 |
| --- | --- | --- |
| [`dueling_concat/`](dueling_concat/README.md) | 設計 A：8 段向量依段號串接＋全域 → 經典 Dueling | 140,233 |
| [`attention/`](attention/README.md) | 設計 B：全域 token＋8 段 token 過 2 層 Transformer（段距離偏置），8 段共用 A 頭 | 116,490 |

段編碼器兩版共用（`encoder.py`）：φ 25→64→64 → 遮罩後平均‖最大‖總和÷13 → ‖段特徵 → 199→128→64。

## 5. 訓練（兩版相同，`train.py`、`config.json`）

- **D3QN**：Double DQN（1-step＋3-step，Huber）＋Dueling＋target network（每 1000 次更新）＋比例式 PER（α 0.4、β 0.6→1）。
  不用示範資料、不用監督式暖啟動。
- **獎勵**（每組交易）：Δ = max(0, f(c1) − f(searcher)) + max(0, f(c2) − f(good))；Δ ≤ 1e-12 → 0，
  否則 min(Δ ÷ 本回合正 Δ 的中位數, 5)。
- **γ = 0.5**；轉移＝同一個 searcher、同一個 good 位置的下一回合（同 E01），最後一回合為終止。
- **episode**：一段最佳化，預算 1000，從訓練快照均勻抽。800 個 episode；ε 1.0 → 0.05（前 50%）；
  每個 episode 更新 4 × 回合數次，batch 128；學習率 A 3e-4、B 1e-4；梯度裁剪 10。
- **驗證**：每 50 個 episode，10 張驗證圖 × 快照 {1, 4}（20 份），同 seed 和 8 段隨機比最終 fitness；
  瓶頸規則同條件算一次當參考。`best.pt` 依對隨機的平均差選。
- replay 每個 state 只存一份（float16），轉移用索引指到下一個 state（`StateReplay`）。

## 6. 評估

- `run_eval.sh`（等 TRAIN_DONE 後自動跑）：MAP01–06 × seed 7–36（180 場），預算 1000，`evaluate.py` 註冊
  e04_random（8 段整段隨機）、e04_bottleneck（瓶頸規則）、e04_concat／e04_attention（best.pt，greedy）。
- 結果：`experiment_results/s736_e04_{random,bottleneck,concat,attention}_eval1000`；進度 `results/eval_progress.txt`。
- 成對 Wilcoxon 對 e04_random；作弊版參考 `wholeseg8_6maps_eval1000`（只有 seed 7–11）。

## 7. 結果（seed 0，`summary.py`）

訓練中驗證（20 份快照、單段最終 fitness）：兩版都沒贏 8 段隨機；best.pt 是 A 第 650 個 episode、B 第 500 個 episode。

lifetime（MAP01–06 × seed 7–36，180 場，預算 1000）：

| 方法 | lifetime | 對 8 段隨機 | 勝/負 | p |
| --- | --- | --- | --- | --- |
| 8 段隨機 | 903.2 | — | — | — |
| 8 段瓶頸規則 | 894.0 | −1.03% | 75/104 | .028 |
| 設計 A 串接 Dueling | 894.4 | −0.98% | 82/96 | .065 |
| **設計 B Attention** | **911.0** | **+0.86%** | 106/74 | .138 |
| ring_se（4 段隨機範圍） | 906.6 | +0.37% | 98/81 | .54 |
| 8 段作弊版（seed 7–11，30 場） | 971.0 | +7.28% | 26/4 | <.001 |

成對：B 對 A +1.86%（114/65，p=.0001）；B 對瓶頸規則 +1.91%（p=.0001）；A 對瓶頸規則 +0.05%（ns）。

- 兩版都沒有顯著贏 8 段隨機；B 拿到天花板約 12%。
- B 顯著比 A 好 → 段之間的 attention（或共用 A 頭）有幫助。
- 瓶頸規則在 8 段整段反而輸隨機（E03 的 4 候選時是 +4.2%）：它偏好內圈段，整段交換時選段太集中。
