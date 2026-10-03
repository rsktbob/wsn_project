# 2023 WSN Refactoring Notes

## 目標

本次重構的重點是降低 2023 論文實作程式碼的耦合度，讓問題建模、狀態表示、演算法模板與視覺化輸出各自負責清楚的職責，同時保留舊 API 的相容性，避免既有實驗腳本一次全部失效。

## Problem 架構

`Problem` 是統一的問題入口，負責保存 WSN 問題資料與提供公開 API。

內部依功能拆成三個 service：

- `CoverageService`：覆蓋度、target 覆蓋關係、未覆蓋 target 檢查。
- `RoutingService`：routing priority、forwarding path、route capacity、斷線 sensor 檢查。
- `EnergyService`：scheduling cost、routing cost、total cost、remaining energy、target remaining energy。


## Algorithm

只有真正被多個演算法共用的模板流程才保留 `Base*`：

- `BaseGA`
- `BasePSO`
- `BaseGWO`
- `BaseEDA`
- `BaseSE`
- `BaseNSGAII`

`BaseSE` 保存 SE 家族共用參數、必要操作介面及標準市場搜尋流程，不再
另外保留只有轉接用途的 `SE`。早期過渡用的 `SETSv1`、`SETSv2` 已移除；
原 `SETSv3`、`SETSv4` 重新編號為目前的 `SETSv1`、`SETSv2`。

同樣地，只有單一實作者的 `BaseCS`、`BaseRIME` 與 `BaseSETSM` 不再作為
獨立基底類別。

同時新增 snake_case 方法 alias，例如：

- `run_budget`
- `_evaluate_state`
- `_create_state`
- `_create_investments`
- `_invest`
- `_score_regions`
- `_align_to_region`

## Draw 委託

演算法不再直接呼叫 `Draw` 或 `Draw.ShowImage`。

目前改為：

```python
from Algorithm.visualization import show_state
```

演算法只呼叫 `show_state(...)`，實際繪圖由 `Algorithm.visualization.DrawVisualizer` 委託給 `Draw.Draw.ShowImage`。

如果繪圖依賴不存在，`NullVisualizer` 會以 no-op 方式略過繪圖，避免演算法因為視覺化套件缺失而無法 import 或執行。

## 相容性

目前保留兩層相容：

- 舊 class 名稱仍可使用，例如 `GA`、`PSO`、`SETSM`。
- 舊 CamelCase 方法仍可使用，新 snake_case 方法作為漸進式改善入口。

## 驗證

已執行：

```bash
python tests/smoke_problem.py
```

結果：

```text
smoke_ok
```

同時確認：

- `Algorithm` 全部 Python 檔可編譯。
- 除 `Algorithm.visualization` 委託層外，演算法檔案不再直接引用 `Draw`。
- `Base*` alias 與主要 Combine 類別的 snake_case alias 可正常 import。

## 舊 State 類別已移除（2026-09-27）

`CodingState` 與 `TargetCodingState` 已刪除，所有演算法改走
`SensorEncoding`／`TargetEncoding` 解碼出 `State`。原本唯一還在用
`TargetCodingState` 的 `SRIME` 已改為 `TargetEncoding` + `State`。

刪除前兩組實作已不等價，這會影響 `SRIME` 的結果：

**sensor 側**（`CodingState` vs `SensorEncoding`）—— priority 公式不同，
`SensorEncoding` 多了 `0.5 +` 下限（避免 gene=0 的 sensor 完全失去排序權重）。

**target 側**（`TargetCodingState` vs `TargetEncoding`）—— priority 公式與
`target_assignment` 相同，分歧在冗餘 sensor 移除的方向：舊版從 `open_order`
前端開始嘗試刪除，`TargetEncoding` 從後端（低優先）開始刪除。因此 `SRIME`
改用 `TargetEncoding` 後，結果與改版前的舊實驗不可直接比較。


## 2026-10-02 SE 全區／指定範圍流程分離

新增 `BaseSA`、`BaseSI`，SI 脫離 SA，SA 商品更新移至投資者更新後；worker 專注評估。
詳細流程與限定 A2 回歸結果見 [SE 流程重構說明](docs/SE_FLOW_REFACTOR_20261002.md)。


## 2026-10-03 SE 共用每輪流程

移除 `BaseSA`、`BaseSI` 與 `market.search()`；SA／SI／SI v2／Ring 共用 `BaseSE.vision_search()`，各步驟同名覆寫。SI 保持獨立於 SA。詳見 [現行架構及 A2 驗證](docs/SE_SHARED_FLOW_20261003.md)。
