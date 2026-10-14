# Rainbow（Hessel et al., 2018）

把 DQN 的六個改良合在一起。實驗 README 要寫清楚**每個元件開或關**和參數，方便做消融。

| 元件 | 做什麼 | 主要參數 |
| --- | --- | --- |
| Double Q-learning | online 網路挑動作、target 網路估值，降低高估（同 D3QN） | — |
| Dueling | Q = V + A − mean(A)（同 D3QN） | — |
| Prioritized replay（PER） | 依 TD error（Rainbow 用分布的 KL 損失）抽樣，重要樣本多學；用 importance-sampling 權重修正偏差 | α（優先度次方）、β（IS 修正，通常從 0.4 線性升到 1） |
| Multi-step（n-step） | 目標用 n 步累積回報 `Σ γ^k r_{t+k} + γ^n Q(s_{t+n})`，獎勵傳得較快 | n（常用 3） |
| Distributional（C51） | 網路輸出每個動作的回報**分布**（N 個 atom 的機率），不是單一 Q 值；損失是投影後分布的交叉熵 | atom 數（51）、支撐範圍 [V_min, V_max] |
| Noisy Nets | 線性層的權重加上可學習的雜訊，取代 ε-greedy 做探索 | 初始雜訊 σ₀（0.5） |

## 和 D3QN 的差異（實驗 README 要特別寫）

- **輸出層不同**：Distributional 時每個動作輸出 N 個 atom 的 softmax，Q = Σ z_i p_i；
  輸出維度 = 動作數 × N。
- **探索方式不同**：Noisy Nets 開啟時通常 ε = 0。
- **獎勵範圍要先定**：C51 的 [V_min, V_max] 要涵蓋 n-step 回報的實際範圍，否則會被截斷。
- **回放不同**：PER 要記錄每筆樣本的優先度。

## 實驗 README 要寫的參數

D3QN 的全部參數，加上：PER 的 α、β 排程；n；atom 數與 [V_min, V_max]；Noisy 的 σ₀。
