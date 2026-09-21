# 2023 WSN Refactoring Notes

## 目標

本次重構的重點是降低 2023 論文實作程式碼的耦合度，讓問題建模、狀態表示、演算法模板與視覺化輸出各自負責清楚的職責，同時保留舊 API 的相容性，避免既有實驗腳本一次全部失效。

## Problem 架構

`Problem` 是統一的問題入口，負責保存 WSN 問題資料與提供公開 API。

內部依功能拆成三個 service：

- `CoverageService`：覆蓋度、target 覆蓋關係、未覆蓋 target 檢查。
- `RoutingService`：routing priority、forwarding path、route capacity、斷線 sensor 檢查。
- `EnergyService`：scheduling cost、routing cost、total cost、remaining energy、target remaining energy。

`Problem.CodingProblem.CodingProblem` 保留為相容 wrapper，實際繼承並使用 `Problem`。

## CodingState

`CodingState` 保留原本名稱，避免語意與既有使用習慣被破壞。

新增較清楚的 snake_case API，例如：

- `create_zero_code`
- `create_random_code`
- `random_gene_value`
- `evaluate`
- `decode`
- `decode_routing_priority`
- `decode_schedule`
- `decode_routes`
- `copy`

舊方法仍保留，新的方法以 wrapper 方式委託舊方法，方便逐步遷移。

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

## 舊 State 與新 Encoding 已不再等價（2026-09-21 查證）

重構當時的 `tests/regression_coding_split.py` 用固定染色體比對「舊 State」
與「新 Encoding」的解碼結果必須完全相同。該遷移已完成，兩邊之後各自演進，
等價關係被刻意打破，因此該測試已移除。實測的分歧點：

**sensor 側**（`CodingState` vs `SensorEncoding`）—— priority 公式不同：

```python
SensorEncoding.py   (energy_score + proximity_score) * (0.5 + gene/(RANK_PRECISION-1))
CodingState.py      (proximity_score + energy_score) * ((gene % rank_prec)/(rank_prec-1))
```

那個 `0.5` 下限是後來加的（避免 gene=0 的 sensor 完全失去排序權重），
`CodingState` 沒有跟上。已驗證：把 `SensorEncoding` 換回舊公式後，
discrete／bucketed／exact 三種模式的兩邊結果完全相同。

**target 側**（`TargetCodingState` vs `TargetEncoding`）—— `_target_priority`
與 `_target_priorities` 的公式相同，`target_assignment` 也完全一致，但之後的
`levels`、`next_hops`、`paths` 與 fitness 都不同，分歧在覆蓋解碼之後的冗餘移除
或路由階段，尚未定位到確切位置。

目前沒有演算法依賴這兩組實作彼此一致：`CodingState` 已無演算法使用
（只剩 `tests/smoke_problem.py` 與 `plotting/rescan.py` 引用），
`TargetCodingState` 僅由 `SRIME` 使用，其餘走 `SensorEncoding`／`TargetEncoding`。
