# E08：Ring_SE 選候選範圍，加入範圍外關係與交換後覆蓋變化

- 狀態（10/25）：b1000 RelNet（plain_delta、soft τ=1、全 0 也訓練）訓練中；還沒跑 lifetime。
- 動機：E07 不論換哪種標籤／loss，test 正值交易選到最大都卡在 ~50%；背答案測試（tmp/e07_memorize）
  大網路 30 張圖 train 98.5%、val 從 47.9% 掉到 39.5% → 容量夠，是輸入裡能推廣的資訊不足。
- 和 E07 共用：資料（E07 results/data，標籤、原輸入）、env、RoundStore／norm_stats、train_sup 的
  load_labels／PairData／loss_fn／metrics。E08 只寫新的部分。

## 輸入（每個候選，4 個候選共用權重）

| | 內容 | 維度 |
| --- | --- | --- |
| ① 範圍內 | φ(token) 範圍內 平均‖最大（同 E07） | 128 |
| ② 範圍外路由相關 | 範圍外、S 或 G 的解裡往 BS 的路徑經過範圍內的 sensor，φ 平均‖最大 | 128 |
| ③ 範圍外覆蓋相關 | 範圍外、和範圍內開啟的 sensor 覆蓋同一個 target 的 sensor，φ 平均‖最大 | 128 |
| ④ 候選特徵 | 同 E07 cand | 10 |
| ⑤ 交換後覆蓋變化 | child1（S 範圍內換 G 的等級）、child2 各 4：失去覆蓋 ÷T、新增覆蓋 ÷T、開啟數變化 ÷N、等級總和變化 ÷5N | 8 |
| ⑥ 全域 | 同 E07 glob | 10 |

412 → 128 → 64 → 1；4 個 logits 做 softmax。②③⑤ 只用父代解碼後的等級、父節點和地圖覆蓋表（features.py），
不解碼子代、不做路由；等級用解碼後的（解碼器 v1/v2 會調整等級，基因上的等級不準），子代實際解碼可能再調整，⑤ 是近似。

驗證：② 只取 S 時的顆數 ÷N 和 E07 cand[4] 24576/24576 相同；覆蓋表重建的每顆覆蓋數和 E07 特徵第 8 欄相同。
成本：②③⑤ 一回合（64 組 × 4 候選）4.6 ms ＝ 每個候選 ~18 µs；一次 fitness 評估（decode＋evaluate_state）184 µs。

## 檔案

- `features.py`：②③⑤ 的計算（`round_extras`），離線、線上共用。
- `build_features.py --budget B`：從 E07 資料算 ②③⑤ → `results/data/b{B}/{train,val,test}/RLTRAIN??.npz`（b1000 48 MB）。
- `relnet.py`：RelNet、`load_relnet`。
- `train_rel.py`：訓練＋test（另算同分算對的準確度 `test_acc_all`、`test_acc_pos`）→ `results/rel_b{B}_m{N}_{label}{loss}/seed0/`。
- `results` → `runs/2027WSN/rl/E08_ring_se_relations`。

## 結果

b1000、plain_delta、soft τ=1、全 0 也訓練、140 張圖、val 早停（第 20 epoch 最好，還在進步）；test 30 張，同分算對：

| 模型 | 全部交易 | 正值交易選到最大 | 標籤比 |
| --- | --- | --- | --- |
| 隨機 | 39.1% | 25.2% | 31.4% |
| E07 SupNet（`sup_b1000_m140_plain_delta_soft1`） | 59.29% | 50.04% | 57.31% |
| **E08 RelNet**（`rel_b1000_m140_plain_delta_soft1`） | **62.43%** | **53.90%** | **61.20%** |

train／val 選到最大 54.8%／53.4%（E07 約 50%／49.5%）。

lifetime（`evaluate_rel.py --algorithm e08_rel`，MAP01–06 × seed 7–11，`experiment_results/s711_e08_rel_b1000_m140_plain_delta_soft1_eval{B}`）：

| 預算 | ring_se | E08 | E07 soft1 | E08 vs E07 | plain 作弊版 |
| --- | --- | --- | --- | --- | --- |
| 1000 | 899.9 | 949.8（+5.54%，p<.001） | 945.1（+5.02%） | +0.49%（p=.44） | 979.8（+8.87%） |
| 3000 | 997.0 | 1018.1（+2.12%，p=.039） | 1022.8（+2.59%） | −0.46%（p=.38） | 1049.6（+5.28%） |
| 10000 | 1064.1 | 1060.3（−0.36%） | 1059.4（−0.44%） | +0.08%（p=.67） | 1087.5（+2.20%） |

→ 準確度 +4 點換不到 lifetime；和 E07 都沒有顯著差別。

缺口：網路沒有原始等級基因（只有解碼後的等級），也沒有解碼器排序用的 energy_score、proximity_score，
所以輸入不能完全決定子代的解碼結果。
