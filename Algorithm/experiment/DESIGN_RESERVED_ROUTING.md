# 無需訓練的分散式路由設計

題目：**考量剩餘能量與轉送負載之無線感測器網路分散式路由與生命週期研究**。

這版用局部規則、容量查詢與流量預留，不用強化學習、神經網路、模型訓練或量子方法。研究假說是：用「接收新增流量後的可服務時槽數」選下一跳，配合僅在排程變更時重建路由，可減少重要中繼的過早耗盡。改善幅度需要驗證。

## 範圍與比較條件

目前是**集中式 SA-SETS 感測排程 + 分散式路由**的隔離實驗，不是整個網路完全去中心化。每次仍由 SA-SETS 以預算 10000 產生感測排程；本版只讀 levels，不讀其 next_hops、染色體優先序或全域 fitness 來選下一跳。SA-SETS 本身仍以原中央路由評估候選排程，因此尚未共同最佳化新路由；這是已知限制。

沿用原資料地圖、離散等級 0～5、原合法資料鏈路、EnergyService、fitness v2、Problem.LifeCheck 與既有 _advance_lifetime。允許冗餘感測器關閉；不要求所有 100 個 sensor 一直存活。移除前版额外 30 距離通訊上限，且不固定最高感測等級。節點位置固定。

三組使用相同 SA-SETS 設定與 seed：原 SA-SETS、分散式路由不計控制成本、分散式路由計入控制成本。第一段感測排程相同，後續因各組剩餘能量不同，排程也會不同；比較的是同一排程演算法下的整體生命週期，不是假設所有時間的 levels 永遠相同。

## 狀態與選路

每個已加入的節點只維護自己的 parent、剩餘封包容量、當前累積轉送負載、能量與自己的感測/鏈路成本。新節點只能向合法且已連上 BS 的候選節點詢問，已連線節點不反過來依賴未加入的節點，因此不會形成環。

加入競爭採理想、無碰撞的時序：自身剩餘能量越高越早，同能量以到 BS 距離、id 決定。模擬器以排序實現事件時序，但不替節點選 parent。真實無線部署仍需設計對應的計時/競爭協定；此版不宣稱已解決 MAC。

候選鄰居將查詢沿既有 parent 逐跳傳遞。每個 relay 用自己的負載和能量計算加入來源流量 g 後的可服務時間：

`slots = energy / [sensing_cost + (tx_load + g) × (amp_cost + 2 × circuit_cost) − own_generated_load × circuit_cost]`

每一跳只回傳自己與上游回覆的最小 slots，以及剩餘封包容量的最小值；不回傳整條路徑或全網能量。來源節點計算自己使用該鄰居的可服務時間，選擇使 `min(自身 slots, 候選路徑 slots)` 最大的合法候選。同分以剩餘封包餘裕、parent id 決定。

選定後沿路逐跳預留 g，更新每個 relay 的累積流量/剩餘容量，ACK 返回後來源加入。採序列化原子預留，後面的來源會看到已更新的負載，避免多個來源同時搶用相同剩餘容量。

該公式是當前部分路由下的預估，未預知尚未加入的其他來源，不能保證全局最優。容量預留只保障目前選定的資料量；計入控制成本後仍須檢查是否可服務。

## 按需重建與能耗

路由建立後沿用整段服務期間，直到 LifeCheck 無法繼續才重新最佳化感測排程及建立路由。沒有每時槽全網 HELLO、重選路由或迴圈廣播。

每次來源與候選的 query/reply，以及沿上游的查詢、reservation/ACK，皆逐跳計數，預設每訊息 128 bits。控制耗能使用同一電路與幾何放大器公式，傳送及接收分開扣除；BS 不受能量限制。控制訊息只在安裝新排程時一次收費，資料與感測則每 slot 扣除。控制通道假設可靠且可逆，即使資料候選遮罩是有向。

不計共同的中央排程分發、鄰居發現、無線碰撞/重傳、等待與計算耗能；原 SA-SETS 基準也未計這些項目。所以這是**額外路由維護成本**的比較，不是完整部署成本評估。控制 bits=0 組保留相同邏輯和訊息數，但不收控制能量。

若新排程加安裝成本連一個 slot 都無法服務，記 infeasible_solution；失敗建立嘗試成本記在 trace，但不計入已付控制成本。終止是此策略未能找到下一個可服務配置，不代表數學上已不存在其他可行解。

## 檔案與執行

- `reserved_routing.py`：Offer 局部選擇、逐跳容量與時間查詢、流量預留。
- `run_reserved_lifetime.py`：獨立比較 runner，讀取既有演算法及服務，不修改外部檔案。
- `test_reserved_routing.py`：共享上游容量、關閉節點、路徑無環及控制成本消融測試。
- `results/`：本目錄內的參數、逐段 levels/next_hops、訊息能耗與 lifetime。

單圖驗證（從 codes/2027WSN 執行）：

```bash
PYTHONDONTWRITEBYTECODE=1 python Algorithm/experiment/test_reserved_routing.py
PYTHONDONTWRITEBYTECODE=1 python Algorithm/experiment/run_reserved_lifetime.py \
  --map MAP01 --budget 10000 --workers 3 --output results/reserved_v3_map01
```

不需要訓練；時間主要花在沿用的 SA-SETS 排程。拿掉 --map 可跑六圖，--runs 5 可跑五 seeds，但本輪依使用者要求只做單圖設計驗證，不啟動完整大規模比較。單圖結果不可当作六圖平均，也不能宣稱普遍超過 SA-SETS 的一半。

`external_source_audit.json` 在 runner 前後比對 Algorithm/experiment 以外的 .py/.md/.json 檔案。Python bytecode 禁止寫入，Numba cache 也放在本目錄內。
