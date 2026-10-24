# E06 / supervised：監督式回歸 Q

- 演算法：監督式回歸（作弊版標籤），輸出 Q，之後可直接當 RL 的初始權重
- 框架：PyTorch 2.x
- 程式：`../train_sup.py`；結果：`../results/supervised/seed{seed}/`、消融 `../results/supervised_l0/seed{seed}/`

## 1. 輸入狀態

同實驗層 README 第 3 節（每組交易：sensor token 100 × 25、父節點 2 × 100、候選遮罩 4 × 100、候選特徵 4 × 10、全域 10），z-score 標準化。

## 2. 輸出

4 個 Q 值（每個候選一個）；選 Q 最大。

## 3. 標籤

每個候選「只交換、不突變」的差異獎勵（其他交易固定為作弊版的選擇），除以上一回合實際一般 Δ 的正值中位數，截在 5；
定義見實驗層 README 5.1。等於 γ = 0 的 Q。

## 4. 網路結構

同實驗層 README 第 4 節（`../model.py` 的 E06Net；圖見 `../design.svg`）。

| 部分 | 輸入 → 輸出 | 參數量 |
| --- | --- | --- |
| φ | 25 → 64 → 64 | 5,824 |
| sensor Transformer × 2（含環距離、路由父子偏置、最後 LayerNorm） | 100 × 64 → 100 × 64 | 67,192 |
| 候選 MLP | 202 → 128 → 64 | 34,240 |
| 全域 MLP | 10 → 64 → 64 | 4,864 |
| 候選 Transformer × 1（含最後 LayerNorm） | 5 × 64 → 5 × 64 | 33,600 |
| V、A 頭 | 64 → 64 → 1（各） | 8,450 |

總參數量 154,170（消融 sensor Transformer 0 層：86,978）。

## 5. 訓練設定

- 資料：`../collect.py --policy oracle`，800 episode、409,600 組交易；依地圖切 10%（19 張圖）當驗證。
- Huber 損失（4 個候選都算），Adam lr 3e-4、L2 1e-5、batch 256、梯度裁剪 10，最多 40 epoch，驗證損失 6 次沒進步就停。
- 驗證指標（只看至少一個標籤 > 0 的交易）：命中率（選到標籤最大的）、標籤比例（選到的 ÷ 最大），以及用只交換的一般 Δ 算的同兩個量。
- 訓練完用 best.pt 在 20 份驗證快照自己推進，和隨機、瓶頸規則比最終 fitness。

## 6. 結果

監督式（2 層）lifetime 944.6（+4.20% 對 ring_se），0 層消融 945.7（+4.31%），兩者 −0.11%（p=.96）；驗證見實驗層 README 8.2。
best epoch 14；驗證標籤比例 0.461（隨機 0.27）。checkpoint：`../results/supervised/seed0/best.pt`、`../results/supervised_l0/seed0/best.pt`。
