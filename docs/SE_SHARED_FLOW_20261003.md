# SE 共用每輪流程（2026-10-03）

本次取代 10 月 2 日的 BaseSA／BaseSI 中間層。兩個檔案已移除，
`ParallelSEMarket.search()` 也已移除。這是介面與流程重構，不變更接受規則。

## 繼承與責任

```text
Algorithm
└─ BaseSE
   ├─ SETS → SA_SETS → Target／選區策略變體
   ├─ SI_SETS → SI_SETSv2
   ├─ Ring_SETS → Ring_SETSv2
   ├─ CodingSE、CodingSEv2、SES
   └─ SETSv2 → SA_SETSv2（保留專屬每輪順序）
```

省略圖中的操作、評估與記憶 mixin。SI 不繼承 SA；Ring 不繼承具體 SI。
`SA_SETS`、`SI_SETS`、`SI_SETSv2`、`Ring_SETS` 使用同一個
`BaseSE.vision_search()`。SA 選區策略的既有前置回饋包裝及 RL 專屬迴圈保留。

## 唯一的共用每輪流程

```text
保存 active_regions、舊商品快照
  ↓
create_investments(problem, goods, active_regions)
  ↓
evaluate_many(problem, flat_candidates, batch_size=...)
  ↓
summarize_round(active_regions, scores)
  ↓
region_probabilities(goods_before, quality)
  ↓
select_regions(probabilities)
  ↓
update_searchers(children, scores, selected, goods_before)
  ↓
update_goods(children, scores, active_regions)
  ↓
保存 selected_regions，執行既有選區策略回饋
```

外層 `BaseSE.search()` 再更新搜尋記憶、歷史及 iteration callback。
商品參數明確傳入 `create_investments()`；此函式只生成子代，不評估、不更新族群。

- 全區投資使用 `BaseSE.create_investments()`：區域 × 投資者 × 商品。
- SI 覆寫同名函式，生成投資者 × 指定區域商品的子代列。
- Ring 覆寫同名函式，生成共用商品／指定操作環段的子代列。
- SI、Ring 使用共用 `create_selected_investments()` 工具，沒有第二份流程迴圈。
- 產生結果攤平成一維交給 `evaluate_many()`，評估後恢復原形狀與 worker 回傳候選解。
- SI v2／Ring 保留 child1、child2 角色與各自接受條件。

## 平行評估與相容性

程序池仍使用同一個 `PersistentWorkerPool`（multiprocessing Process／Pipe）。
`ParallelSEMarket` 現在只保存全區資源及提供 `evaluate_many()`，不產生子代、不選區、
不更新商品。`PooledEvaluation` 提供 SI／Ring 的同名評估介面。

全區評估保留各區 RNG、局部最佳檔案與原有合併順序；初始商品 FE 仍於第一次合併計入。
因此不直接以普通的全域逐子代更新替代區域檔案歸併。這是保留數值結果所需的評估差異。

SETSv2 的特殊父代重新評估、區域最佳及對齊順序不變，仍覆寫 `vision_search()`。
本次未擴大執行其他 SE／RL 方法的完整實驗。

## A2 回歸

沿用 2026-10-02 重構前基準：A2、100 sensor／100 target、F=10、靜態 discrete、
SensorEncoding v3、fitness v2、routing v1、Numba，n=8、h=4、w=2、mu=0.4。

- 單次最佳化：seeds 7、11，各 10,000 FE，SA 每組 156 輪、SI v2 每組 312 輪。
- 比對每輪投資者、選區、記憶、最佳解、完整 fitness 歷史及 FE；SI v2 另比對商品及品質紀錄。
- 完整壽命：seed 7，逐欄比較 summary 和 trace，僅排除耗時欄位。
- 期望與實測壽命：SA 949，SI v2 1026。詳細結果見專案根目錄
  `artifacts/se_common_flow_20261003/verification.json`。

可重跑（在專案根目錄）：

```bash
/home/g114056067/miniconda3/envs/wsnenv/bin/python \
  codes/2027WSN/tests/check_se_flow_a2.py \
  artifacts/se_common_flow_20261003/recheck.json \
  artifacts/se_flow_refactor_20261002/before.json
```

此結果限定上述兩個方法與 A2 案例，不宣稱所有變體／地圖已驗證。
