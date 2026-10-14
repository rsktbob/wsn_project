# E01 / oracle（步驟 1：天花板測試，不訓練）

- 目的：如果每組交易都「選對段」，ring_se 的 lifetime 最多能贏隨機選段多少？這是 E01 任何學習方法的上限。
- 程式：`variants.py`（`Ring_SE_Oracle`）、`run.py`（執行時註冊 `ring_se_oracle`，不改 registry）、`run.sh`、`summary.py`。
- 狀態：**執行中**（2026-10-13）

## 作法（一組交易）

1. 在 4 段各抽一個 80/20 候選範圍（同 RL 版的 PairSegmentMarket）。
2. **偷看**：4 段都**只做交換、不突變**，評估 4 組交換後的子代（每組 child1、child2）。
   偷看只走評估子程序池，**不計入預算、不更新全程最佳解**。
3. **選段**：先比 Δ = max(0, f(child1) − f(searcher)) + max(0, f(child2) − f(good))（≤ 1e-12 視為 0）；
   Δ 相同（大多是都沒改善）時比 f(child2)，因為 EN 更新下 good 會被造訪者中最好的 child2 取代。
4. **實際使用**：拿選中那段的交換結果，照 ring_se 規則讓每個子代以 μ = 0.4 機率在該段突變，
   再照常評估（計入預算）、照常更新（searcher 菁英、good EN）。

所以天花板只來自「依交換結果選對段」，突變的運氣和一般 ring_se 相同。
評估次數：偷看 4 段 × 2 子代 = 8 次（不計預算）＋實際 2 次（計預算，和 ring_se 相同）。

（2026-10-13 更早的版本偷看時含突變，另有「直接沿用偷看子代」的含運氣版本；依使用者要求刪除，結果未保留。）

小測（MAP01、seed 7、第一段、預算 1000）：評估數 1040（和 ring_se 相同）；
4 段交換裡至少一段有改善的交易 58.8%，選中段突變後實際有改善 56.1%；每段最佳化約 1 秒（ring_se 0.4 秒）。

## 評估

MAP01–06 × seed 7–11，預算 1000、3000，fitness v3。
比較對象：ring_se 正式版（每組隨機選段，`experiment_results/ringpair_prod_6maps_eval{B}`）、
舊版 ring_se（每 searcher 一段，`se3_6maps_eval{B}` 的 `ring_sets`）。
結果：`experiment_results/e01_oracle_6maps_eval{B}`；log 在 `tmp/e01_oracle/`。

## 結果

（執行中）
