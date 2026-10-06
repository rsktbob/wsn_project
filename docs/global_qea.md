# 全局 QEA 與對照實驗

`qea` 集中搜尋整個網路，每顆感測器包含感測選項與路由優先序兩個基因。
共用 `SensorEncoding.decode()`（目前 v3）及實驗指定的 fitness service；
全局表示集中決策，不保證找到全局最優解。這是 CPU 上的量子啟發式搜尋，
沒有量子硬體、糾纏或量子加速。

## 編碼與更新

`State/QuantumSensorEncoding.py` 保存 QEA 的條件機率位元配置，提供
`observe()` 取樣和 `elite_path()` 更新路徑。它與 `SensorEncoding` 職責不同：
前者產生後者，不能直接互換。QEA 的角度族群與旋轉更新由 QEA 自己管理。

`observe()` 回傳標準 `SensorEncoding`，再經共用 `Algorithm.evaluate()`
呼叫 `SensorEncoding.decode()`。`QuantumSensorEncoding` 沒有自己的 decode，
也不取代既有的感測排程與路由解碼器。

K 個選項使用 K−1 個條件 Bernoulli 閘；從第一個閘開始連續遇到 1 就繼續，
遇到第一個 0 停止，連續 1 的數量即選項索引。三態是 `0* → 0`、`10 → 1`、
`11 → 2`。尾端未使用的位元無須修復，不使用 modulo，所以不會把多餘二進位
碼偏置映射至低等級。支援 discrete、bucketed、exact 的不同感測器選項數。

第 j 個閘（從 0 開始）的初始 p(1)=(K−j−1)/(K−j)，使每個選項初始機率
恰為 1/K。K=1 不需閘。感測選项與路由 rank 都採相同規則；目前 rank 有 10 態。
這種表示需要 O(sum(K−1)) 記憶體，並不是最省位元的二進位表示。

振幅為 α=cos(θ)、β=sin(θ)，p(1)=sin²(θ)。每代評估 n 個觀測解後，朝歷史
最佳染色體旋轉；只更新最佳解有效路徑上與本次觀測位元不同的位置。目標位元
是 1 時 θ 增加，0 時減少，步長預設 0.01π。角度限制在第一象限，機率保留
探索下限 min(0.01, 1/K)，避免完全塌縮，也保留大選項數時的均勻初始分布。

這是依 Han & Kim 的 Q-bit 與 rotation gate 概念設計的多值問題變體，並非
逐項重現原文二進位旋轉查表或 migration 策略：
[Quantum-Inspired Evolutionary Algorithm for a Class of Combinatorial Optimization](https://doi.org/10.1109/TEVC.2002.804320)。

## 對照組與解讀

- `qea`：旋轉角度更新。
- `qea_classical`：相同族群、初始分布、最佳解路徑和更新遮罩，改用
  p ← p + 0.05(target − p) 的傳統機率更新。
- `qea_random`：相同初始分布但不更新，等同合法選項的均勻隨機搜尋。
- `ga`：保留既有 GA，使用原本 coverage-seeded 初始化。因此它是實務基準，
  不是只改一個運算子的消融實驗。

擊敗 random 只代表學習有用；擊敗 classical 才支持這組參數下旋轉更新較好。
這些結果都不能證明量子硬體優勢；rotation 與 classical 的學習率也尚未調參。
首批觀測在三個 QEA 版本的同 seed 下完全相同，所有評估（含初始化）都計數。
QEA 保留歷史最佳解，每次 fitness 評估都有一筆 best-so-far history。

## 執行

在 `codes/2027WSN` 下，使用已安裝 NumPy 的專案 Python：

```bash
python experiment_algorithms.py --algorithm qea qea_classical qea_random ga --mode single --runs 5 --evaluate 3006
python experiment_algorithms.py --algorithm qea --mode lifetime --runs 5 --evaluate 10000
python experiments/benchmark_qea.py --map-count 3 --runs 5 --budget 3006 --output /tmp/qea_pilot
python tests/smoke_qea.py
```

`benchmark_qea.py` 預設選 manifest 前三張固定地圖，五個種子 7–11，每次 3006 次
評估，fitness v2，離散感測。3006=30+31×96 配合現有 GA 的完整世代計數，
因此四種方法實際評估次數都相同；一般 CLI 下 GA 仍可能按完整世代超過要求預算。
benchmark 先暖機解碼核心，並輪替方法執行順序，記錄每次評估的最佳 fitness。

輸出 `results.csv`（含可行性、耗能與時間）、`summary.json`（含參數、地圖與
成對勝負）、`histories.npz`、`convergence.png`。只比較單次最佳化，不把 fitness
提升當成實際網路壽命提升。要驗證壽命需另跑 lifetime 實驗。

一般 `run()` 預設最多 800 代；若預算大於 n×800，會回報 early_stopped。
需要更多代可傳入 `max_iteration`。每次搜尋重新初始化分布，不沿用上一輪
lifetime 的機率模型或傳入 state，避免拓樸與能源改變後使用過時先驗。
