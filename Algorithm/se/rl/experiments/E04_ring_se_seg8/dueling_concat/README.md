# E04 / 設計 A：串接＋經典 Dueling

- 圖：`../design_A.svg`；程式：`model.py`（ConcatDueling），訓練用 `../train.py --net concat`
- 演算法：D3QN（Double DQN＋Dueling＋target＋PER），不用示範、不用監督式暖啟動
- 框架：PyTorch 2.14.1
- 參數：140,233

## 1. 網路

```
段編碼器（8 段共用，../encoder.py）：
  13 token（25 維）→ φ 25→64→64 → 遮罩後 平均 ‖ 最大 ‖ 總和÷13（192）
  ‖ 段特徵 7 → 199→128→64 = h_k
串接：h₀ … h₇ ‖ 全域 10 = 522（依段號固定位置，所以不需要段編號）
共用層 522→128→128（ReLU）
V：128→64→1；A：128→64→8
Q_k = V + A_k − mean(A)，第 k 個輸出固定＝第 k 段
```

輸入先做 z-score 標準化（`../data/norm.npz`，存在網路的 buffer 裡）。

## 2. 獎勵與轉移

- r = min(Δ ÷ 本回合正 Δ 的中位數, 5)，Δ = 兩個子代的改善量和；沒改善為 0。
- γ = 0.5、n = 3；轉移＝同一個 searcher、同一個 good 位置的下一回合。

## 3. 超參數

學習率 3e-4（其餘見 `../config.json`）。

## 4. 結果

`results/seed0/`（log.jsonl、best.pt、last.pt）。尚未完成。
