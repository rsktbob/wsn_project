# RL 實驗

SE 演算法上的強化學習實驗。

## 規則

1. **一個實驗一個資料夾**：`experiments/E{編號}_{基底演算法}_{重點}/`（例如 `E01_ring_sets_segment`）。
   不同實驗一律分開，即使輸入狀態相同，README 也要自己寫完整，不寫「同 E01」。
   **每個實驗只針對一個基底 SE 演算法**（sa_sets、si_sets、ring_se 或其中一個變體，如 ring_sepri），
   不在同一個實驗裡混多個基底演算法；要換基底演算法就開新實驗。
2. **同一個實驗裡比較不同演算法**：每個演算法一個子資料夾（`d3qn/`、`rainbow/`…），
   各自有 README，寫**它自己的**網路結構（附 `network.svg`）、輸入狀態、輸出、獎勵訊號、訓練設定與結果
   ——不同演算法的網路與輸出（例如 Q 值 vs. 分布）會有差異，不能共用一份。
3. **實驗層 README**：目的、問題設定（決策單位、動作空間、環境與預算）、評估方式、各演算法對照表、結論。
4. **先設計後寫程式**：README 的設計段落填好、和使用者確認後才寫程式。
5. **checkpoint、log 不進 git**：放在 `runs/2027WSN/rl/E{編號}_…/{演算法}/`，實驗資料夾內以 `results` symlink 指過去。
6. **框架用 PyTorch**（wsnenv 已安裝）。
7. **超參數只放 `config.json`**，程式不寫死；`network.svg` 由 `common/draw_network.py` 依 config 產生（之後建立）。
8. **贏了才正式化**：實驗結果確定後，才整理成 `Algorithm/se/ring_se/`（或對應家族）裡的正式演算法並註冊。

模板：`TEMPLATE_experiment.md`（實驗層）、`TEMPLATE_algorithm.md`（演算法層）。
演算法原理：`algorithms/`（各演算法寫一次；演算法層 README 只寫用了哪些元件與參數）。
共用程式：`common/`（狀態擷取、獎勵、replay、評估等，避免各實驗複製後走樣）。

## 實驗一覽

| 編號 | 日期 | 基底演算法 | 目的 | 決策單位／動作 | RL 演算法 | 狀態 | 結論 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [E01](experiments/E01_ring_se_segment/README.md) | 2026-10-13 | ring_se | 學到的策略選 ring 區段能否贏隨機、拿到作弊版天花板（1000：+9.1%）的多少 | 每組交易選 4 段之一 | D3QN、Rainbow（經典 Dueling＋DQfD 式模仿） | 訓練中（D3QN DQfD seed 0） | |
| [E03](experiments/E03_ring_se_decoded/README.md) | 2026-10-14 | ring_se | 用解碼後的 sensor token 選候選範圍 | 每組交易 4 候選選 1 | 監督式 DeepSets、瓶頸規則 | 監督式完成；RL 待設計 | 瓶頸規則 +4.2% 最好，DeepSets +2.9% |
