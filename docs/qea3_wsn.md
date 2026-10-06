# QEA3-WSN

新增 `--algorithm qea3`，既有 `qea` 保留。未修改 SensorEncoding 或其 decode。

這是採用 Han & Kim (2002) QEA3 分享機制的 WSN 變體，**不是原論文完整重現**。
原文：https://khhan.com/assets/docs/TEVC2002_V6N6.pdf （流程 pp.582–583；設定 p.587）。

## 設定與差異

- 族群 30，與目前 QEA 比較；原文 QEA3 實驗族群為 10。
- 沿用 QuantumSensorEncoding 的條件取樣與 SensorEncoding.decode v3。
- 沿用 WSN QEA 的有效路徑遮罩、第一象限旋轉、步長 0.01π、機率界限；
  沒有重現原文完整 fitness/振幅符號旋轉查表。
- 明確採用固定不重疊配對 (0,1)、(2,3)…；奇數族群最後一個單獨保留。
  原文說明鄰近解之間分享，並未完整列出配對索引及實作順序。
- 每代局部分享，每 100 代全局分享；全局分享代取代局部分享。
- 分享只複製具體參考解與其分數，不直接複製或重設機率模型。

## 執行順序

初始化：觀測並評估 n 個解，建立每個個體的 reference。初始化是第 0 代，
不旋轉、不 migration；初始化的評估也扣預算。

後續每代：

1. 取樣、decode、評估各候選解。
2. 依 **上一代分享後的 references** 旋轉模型。
3. 本代候選解較好時，替換該個體 reference；同分保留原解。
4. 局部分享（兩份記錄更新成組內較好者，同分選左側）或定期全局分享。

共用 `score()` 會在評估時即記錄全局 best；步驟 2 完全不讀它，因此本代新解
不會提早成為旋轉目標。全局分享是在步驟 3 之後使用更新後的 global best。
局部分享後同組 references 內容相同，不能稱為永久獨立的 personal best。

每次 lifetime 重排程重新初始化所有模型、references 與 migration 計數。
預算不足一批時，只評估剩餘數量，最後這個不完整批次旋轉與更新 references，
但不執行 migration。n=30、budget=10000：30 次初始化 + 332 個完整更新代
+ 10 次最後部分批次，全局分享在更新代 100、200、300。
預設 max_iteration=800 限制初始化後的更新批次；提早停止會回報 early_stopped。

## 執行

在 codes/2027WSN 下：

```bash
python experiment_algorithms.py --algorithm qea3 --maps maps/maps_100100100.json --mode lifetime --runs 5 --seed 7 --evaluate 10000 --moving false
python tests/smoke_qea3.py
```

族群與 global_period 可在 experiments/presets.py 的 qea3 項目設定，
或直接透過 QEA3(problem, n=10, global_period=100, seed=7) 建立。

`smoke_qea3.py` 驗證先旋轉後更新 reference、局部/全局分享、複製隔離、不修改
角度、奇數族群、小額及非整批預算、三種 sensing mode、seed 重現與重複 run 重設。
