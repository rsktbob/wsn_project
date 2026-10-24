# E06 / d3qn：D3QN（γ = 0 主設定、γ = 0.5 消融）

- 演算法：D3QN（原理見 `../../../algorithms/d3qn.md`）
- 啟用的元件：Double DQN、Dueling、target network、比例式 PER；γ > 0 時另加 3-step 目標。γ = 0 時目標就是 r（contextual bandit），Double／target 不起作用。
- 框架：PyTorch 2.x
- 程式：`../train_rl.py`；結果：`../results/d3qn/{g0_scratch, g0_warm, g05_warm}_seed{seed}/`

## 1. 輸入狀態

同實驗層 README 第 3 節（每組交易：sensor token 100 × 25、父節點 2 × 100、候選遮罩 4 × 100、候選特徵 4 × 10、全域 10），z-score 標準化。

## 2. 輸出

4 個 Q 值（每個候選一個，由候選自己的內容算，不依候選編號）；訓練時 ε-greedy，評估時取最大。

## 3. 獎勵訊號

- 每組交易的差異獎勵（實際子代，含突變）÷ 上一回合實際一般 Δ 的正值中位數，截在 5；定義見實驗層 README 5.1。
- 每回合給（每組交易一個）。
- γ = 0（主設定）、γ = 0.5＋3-step（消融）；轉移＝同一個 searcher、同一個 good 位置的下一回合，最後一回合終止。

## 4. 網路結構

同 `../supervised/README.md` 第 4 節（154,170 參數）。暖啟動版載入 `../results/supervised/seed{seed}/best.pt`。

## 5. 訓練設定

- 800 個 episode，每個 episode 是一段最佳化（預算 1000），從 190 張訓練圖的快照均勻抽。
- 每回合更新 16 次（約 10 萬次），batch 128，Adam lr 1e-4、L2 1e-5，梯度裁剪 10，target 每 1000 次更新，warmup 5000 組。
- PER α 0.4、β 0.6 → 1、ε_p 1e-3。
- ε：從零 1.0 → 0.05、暖啟動 0.2 → 0.05（前 50% episode 線性下降）。
- 驗證：每 50 個 episode，20 份驗證快照和隨機比最終 fitness，`best.pt` 依平均差選（暖啟動版第 0 個 episode 也驗證一次）。
- 超參數在 `../config.json` 的 `rl`。

## 6. 結果

| 版本 | lifetime | 對 ring_se | 對監督式 |
| --- | --- | --- | --- |
| γ=0 從零 | 933.9 | +3.02% | −1.13% |
| γ=0 暖啟動 | 943.3 | +4.05% | −0.14%（p=.74） |
| γ=0.5 暖啟動 | 939.3 | +3.60% | — （對 γ=0 暖啟動 −0.43%，p=.15） |

暖啟動接 RL 後驗證成績先掉、沒再超過起點；γ=0.5 沒有比 γ=0 好。詳見實驗層 README 第 8、9 節。
