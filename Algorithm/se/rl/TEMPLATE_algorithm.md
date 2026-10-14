# E{編號} / {演算法}

- 演算法：（D3QN、Rainbow…；原理見 `../../algorithms/`）
- 啟用的元件：（例如 Rainbow：Double／Dueling／PER／n-step／Noisy／Distributional 各開或關）
- 框架：PyTorch（版本：）

## 1. 輸入狀態

網路實際看到的向量。

| 特徵 | 維度 | 怎麼算 | 正規化 |
| --- | --- | --- | --- |

總維度：

## 2. 輸出

- 輸出是什麼：（每個動作一個 Q 值／每個動作一個分布（atoms 數、支撐範圍）／…）
- 怎麼選動作：（argmax／ε-greedy／Noisy 探索）

## 3. 獎勵訊號

- 公式：
- 什麼時候給：（每步／每回合／episode 結束）
- γ、n-step：

## 4. 網路結構

![network](network.svg)

| 層 | 輸入 → 輸出 | 啟動函數 | 參數量 |
| --- | --- | --- | --- |

總參數量：

## 5. 訓練設定

- 訓練地圖／episode 數／每 episode 預算
- 學習率、batch、replay 大小與 warmup、target 更新頻率、探索排程
- 驗證：地圖、頻率、挑 checkpoint 的規則
- 超參數全部在 `config.json`

## 6. 結果

## 7. 可重現資訊

- git commit：
- 環境雜湊（fitness、解碼、Problem 實作）：
- 訓練指令：
- 評估指令：
- checkpoint：`results/…`
