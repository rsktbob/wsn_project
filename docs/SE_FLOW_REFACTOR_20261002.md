# SE 流程重構與 A2 回歸驗證（2026-10-02）

> 後續更新：BaseSA／BaseSI 與 market.search 已於 2026-10-03 移除，現行架構請看 [共用流程](SE_SHARED_FLOW_20261003.md)。本文件保留為第一階段歷史紀錄。

## 範圍

依本次要求整理兩套基礎流程：全區投資 `BaseSA`、指定範圍投資 `BaseSI`。
完整數值驗證限定 A2 地圖的 `sa_sets`、`si_setsv2`，沒有跑全部 SE 方法。
RL 類別不改其學習流程；繼承到共用元件的其他方法未做完整數值等價宣稱。

## 架構

```text
Algorithm
└─ BaseSE                         搜尋生命週期、預算迴圈、回呼、結果
   ├─ BaseSA                      全區投資流程
   │  ├─ SETS
   │  │  └─ SA_SETS               選取身分感測器與 CCS 初始化
   │  │     ├─ SA_SETS_Target
   │  │     └─ RegionSelectionSA_SETS → 各選區策略
   │  ├─ CodingSE
   │  ├─ CodingSEv2
   │  └─ SES                      排程版本的相容繼承調整
   ├─ BaseSI                      指定範圍投資流程
   │  ├─ SI_SETS                  與 SA_SETS 無繼承關係
   │  │  ├─ SI_SETSv2             角色分離雙子代
   │  │  └─ RL_SETS → 既有 RL 變體（未納入驗證）
   │  └─ Ring_SETS                不繼承具體 SI_SETS
   │     └─ Ring_SETSv2
   └─ SETSv2                      保留原本獨立的特殊流程
      └─ SA_SETSv2
```

共用操作放在 `sensor_operators.py`，含 `SensorOperators` 與
`AdaptiveSensorInitialization`，不帶 SA/SI 市場迴圈。相同的族群接受規則
放在 `population_updates.py`；商品、投資者的替換都由主程序決定。

## 每輪順序

### BaseSA

1. `market.search()`：主程序以各區獨立 RNG 產生全部區域投資；worker 只評估。
2. 主程序按原區域順序合併評估次數及各區最佳解。
3. 使用更新前的商品快照與本輪投資分數計算 `region_probabilities()`。
4. `select_regions()`。
5. `update_searchers()`：讀舊商品，只有改善才取代投資者。
6. `update_goods()`：訪客的最佳投資解取代商品，維持原本允許退步的規則。
7. 保存選區、執行既有策略回饋；返回 `BaseSE` 更新記憶與歷史。

原本商品更新在 worker 回傳前。現在移到投資者更新之後，兩個更新函式
與 BaseSI 同名、同參數責任；全區與指定區的投資結果維度仍由各自流程管理。
舊商品先供投資者讀取，再更新商品，保留原本資料依賴。

### BaseSI

1. 讀取本輪 `selected_regions` 與更新前商品分數。
2. `create_investments()`：透過 `goods_for_region()`、`make_offspring()` 建立投資列。
3. 攤平後呼叫 `evaluate_many(..., batch_size=每列寬度)`，worker 結果依原序還原。
4. `summarize_round()` 更新本輪造訪品質，其餘維持既有紀錄。
5. `region_probabilities()`、`select_regions()` 選下一輪區域。
6. `update_searchers()`。
7. `update_goods()`。
8. 保存選區，返回 `BaseSE` 更新記憶與歷史。

`SI_SETSv2` 和 `Ring_SETS` 不再複製完整迴圈，只覆寫操作、品質及更新差異。
Ring 使用共用商品，`goods_for_region()` 不採用 SI 的身分分區商品池。

## 數值相容性要點

- SA 舊 worker 的區域 RNG 與局部最佳紀錄改由主程序內的獨立上下文保存。
  同樣消耗 h 個區域 seed；純評估 worker 不額外消耗演算法 RNG。
- SA 的初始商品評估仍記在各區上下文，於首輪合併，維持原 FE／歷史邊界。
- 同分優先順序、區域合併順序、child1/child2 排列都保持。
- SA 商品可退步；SI v2 商品需改善超過 1e-12，投資者嚴格改善即可。
- SI v2 仍只以自己的 child1 更新投資者，以各商品對應訪客的 child2 更新商品。
- `evaluate_investments()` 留作既有呼叫者的相容轉接，新的 BaseSI 主流程使用
  `evaluate_many()`；尚未啟動 pool 的初始化走循序評估。
- 原有 SETSv2 特殊的父代重新評估、歷史區域最佳與對齊順序未重寫。

## A2 驗證

設定：100 顆感測器、100 個目標、初始能量 10、靜態、discrete、
SensorEncoding v3、fitness v2、routing v1、Numba；n=8、h=4、w=2、mu=0.4。
每次最佳化名義預算 10,000 FE。

### 單次搜尋逐輪比對

seeds 7、11，各方法各 2 次，總共 4 組。

- SA：每組 156 輪，實際 10,000 FE。
- SI v2：每組 312 輪，實際 10,000 FE。
- 四組逐輪投資者基因／fitness、所選區域、ta/tb、最佳基因、解碼後 levels／
  next_hops／tx_load、最終 fitness 及完整歷史雜湊均完全相同。
- SI v2 另比對每輪商品基因、商品 fitness、投資品質紀錄，均完全相同。
- SA 舊 worker 未直接提供商品逐輪快照，因此沒有宣稱直接比對每輪 SA 商品；
  其後續搜尋軌跡與完整壽命紀錄另有比對。

### 完整壽命

A2、seed=7，無移動、無 lifetime cap；與改前比對 summary 與完整 trace，
只排除 `elapsed_seconds`、`optimize_elapsed_seconds`。

| 方法 | 改前壽命 | 改後壽命 | 總 FE | trace 列数 |
|---|---:|---:|---:|---:|
| SA_SETS | 949 | 949 | 180,000 | 18 |
| SI_SETSv2 | 1026 | 1026 | 240,000 | 24 |

trace 包含末次不可行候選的停止紀錄。以上兩個案例均以 `infeasible_solution` 結束。
非耗時欄位完全相同；這是限定案例的回歸結果，不是對所有地圖與 SE 變體的數學證明。

## 可重跑方式

在專案根目錄 `/home/g114056067/wsn_project`：

```bash
/home/g114056067/miniconda3/envs/wsnenv/bin/python \
  codes/2027WSN/tests/check_se_flow_a2.py \
  artifacts/se_flow_refactor_20261002/recheck.json \
  artifacts/se_flow_refactor_20261002/before.json
```

原始資料位於專案根目錄 `artifacts/se_flow_refactor_20261002/`：
`before.json`、`after.json`、`lifetime_before/`、`lifetime_after_final/`，
以及 `verification.json`。耗時受主程序生成子代與程序間資料傳输影響，允許與改前不同。
