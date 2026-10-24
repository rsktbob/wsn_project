# E04 / 設計 B：段之間做 Attention

- 圖：`../design_B.svg`；程式：`model.py`（AttentionDueling），訓練用 `../train.py --net attention`
- 演算法：D3QN（Double DQN＋Dueling＋target＋PER），不用示範、不用監督式暖啟動
- 框架：PyTorch 2.14.1
- 參數：116,490

## 1. 網路

```
段編碼器（8 段共用，與設計 A 相同，../encoder.py）→ h_k（64）
token：全域 10→64；段 h_k ＋ 可學段嵌入 e_k（8×64）→ 共 9 個
Transformer 2 層：pre-LN、d=64、4 頭、FFN 128、不用 dropout
  段距離偏置：score(i, j) += r[頭, d(i, j)]，d = |i−j| ∈ 0…7（段 0 最內圈、段 7 最外圈），
  有一方是全域 token 時 d = 8；每層每頭 9 個可學參數
最後 LayerNorm
V：全域 token 輸出 64→64→1
A：每個段 token 輸出 64→64→1（8 段共用權重）
Q_k = V + A_k − mean(A)
```

輸入先做 z-score 標準化（`../data/norm.npz`，存在網路的 buffer 裡）。

## 2. 獎勵與轉移

同設計 A：r = min(Δ ÷ 本回合正 Δ 的中位數, 5)；γ = 0.5、n = 3。

## 3. 超參數

學習率 1e-4（比設計 A 小，Transformer 在 RL 較不穩；其餘見 `../config.json`）。

## 4. 結果

`results/seed0/`（log.jsonl、best.pt、last.pt）。尚未完成。
