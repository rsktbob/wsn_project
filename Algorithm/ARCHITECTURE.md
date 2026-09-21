# Algorithm 架構

## 設計原則

- 一個可執行演算法對應一個 class。
- `Algorithm` 是薄介面，只提供 `run(problem, budget=...)`、`search()` 契約、
  `rng`／`random`、迭代事件與 `AlgorithmResult`。
- 家族 Base 使用短名稱，例如 `BaseSE`、`BaseGA`、`BaseGIGOMEA`。
- 只有流程確實相同的演算法才共用 family Base。
- encoding、decode、fitness 與 operator 留在 family/concrete algorithm。
- `search(problem, budget)` 一律回傳已解碼的 `State`；encoding 不離開演算法。
- 多個家族重複使用的程序評估由 `BatchEvaluator` 組合元件負責。
- 所有演算法統一使用 `run(problem, budget=...)`。

## 目錄與責任

```text
Algorithm/
├─ Algorithm.py          # 薄 Algorithm 契約與迭代事件
├─ core/
│  ├─ result.py          # AlgorithmResult
│  └─ evaluation.py      # serial/process BatchEvaluator
│
├─ BaseGA.py             # GA 族群演化流程
├─ BasePSO.py            # PSO 粒子更新流程
├─ BaseEDA.py            # EDA 分布取樣與更新流程
├─ BaseGWO.py            # GWO leader 與位置更新流程
├─ BaseNSGAII.py         # NSGA-II Pareto 排序與族群演化
├─ BaseSE.py             # SE 市場、投資與信念更新流程
├─ BaseGIGOMEA.py        # GI-GOMEA linkage learning 與 mixing
│
├─ Combine/              # 同時求解排程與路由
├─ Scheduling/           # 只求解排程
└─ Routing/              # 只求解路由
```

`Algorithm2/` 僅作行為與設計比較，不是本目錄的父層，也沒有直接複製其
「單一 class 配合 mode/factory 切換多個演算法」的作法。

## 主要繼承關係

```text
Algorithm
├─ BaseGA
│  ├─ Combine.GA
│  ├─ Scheduling.SGA
│  └─ Routing.RGA
├─ BasePSO
│  ├─ Combine.PSO
│  └─ Routing.KPSOO
├─ BaseEDA
│  ├─ Combine.EDA
│  └─ Scheduling.NBEDA
├─ BaseGWO
│  ├─ Combine.GWO
│  └─ Routing.CBGWO
├─ BaseNSGAII
│  ├─ Combine.NSGAII
│  └─ Scheduling.SNSGAII
├─ BaseSE
│  ├─ Combine.CodingSE
│  ├─ Combine.CodingSEv2
│  ├─ Combine.SA_SETS
│  │  ├─ Combine.SETSv1
│  │  └─ Combine.SA_SETS_Target
│  ├─ Combine.SETSv2
│  │  └─ Combine.SA_SETSv2
│  └─ Scheduling.SES
├─ BaseGIGOMEA
│  ├─ Combine.GI_GOMEA
│  └─ Combine.GI_GOMEA_Target
├─ Combine.RLHH_SETS
├─ Combine.ALNS
├─ Combine.NSOA
├─ Combine.CS
└─ Scheduling.SRIME
```

舊 `SETSv1`、`SETSv2` 過渡實作已移除；原 `SETSv3`、`SETSv4` 分別
重新編號為目前的 `SETSv1`、`SETSv2`。

`RLHH_SETS` 也不繼承 `SA_SETS`：它是 serial RL hyper-heuristic runner，
只組合 SA-SETS 的 state/operator 能力，不繼承多程序市場流程。

## Base、組合與個別演算法的界線

放在 `Algorithm`：

- `run(problem, budget, state=None) -> AlgorithmResult`
- `search(problem, budget, state=None) -> State`
- `rng`（NumPy 陣列／code 產生）與 `random`（單次選擇／機率判斷）
- iteration callback/listener
- 統一結果欄位

放在 family Base：

- GA selection/crossover/mutation 的固定迴圈
- PSO velocity、personal/global best 的固定迴圈
- EDA sampling/selection/distribution update
- NSGA-II non-dominated sorting 與 crowding distance
- SE `search()`、encoding 評估、region 對齊契約、investment 建立、region 選擇、
  `update_search_memory()`
- GI-GOMEA linkage learning 與 gene-pool optimal mixing

使用組合：

- `BatchEvaluator`：PSO、EDA、GWO 共用的 process workers
- RLHH 使用的 SA-SETS operators

留在 individual algorithm：

- state 類型與 decode
- fitness 定義
- mutation、repair、destroy、crossover 等變體 operator
- target、critical、coding 等 gene domain 與 linkage 限制
- 演算法獨有的停止或接受規則

## 已移除的舊名稱對照

| 舊名稱 | 新內部名稱 | 語意 |
| --- | --- | --- |
| `CreateState` | `create_candidate` | 建立一個候選編碼 |
| `Evaluate` | `Algorithm.evaluate` | 解碼並評估一個候選解 |
| `FitnessFunction` | `evaluate_many` / `evaluate_population` | 評估族群 |
| `StateCrossOver` | concrete `_crossover` | 交配兩個候選解 |
| `Transition` | concrete `transition` | 執行演算法專屬變異操作 |
| `GoToRegion` | `align_region` | 對齊 SE region |
| `MasterTransition` | `create_investments` | 建立投資品陣列 |
| `ExpectedValue` | `region_probabilities` | 計算 region 投資機率 |
| `Determination` | `select_regions` | 選擇 region |
| `MarketingResearch` + history | `update_search_memory` | 更新信念、最佳解與歷史 |
| `CreateBird` | `create_swarm` | 建立 PSO swarm |
| `LocalBestUpdate` | `update_personal_best` | 更新粒子最佳解 |
| `NewVelocity` | `update_velocity` | 更新粒子速度 |
| `CreateVector` | `create_probability_vector` | 建立 EDA 分布 |
| `UpdateVector` | `update_probability_vector` | 依 elites 更新分布 |

演算法實作統一使用 snake_case；不再保留 PascalCase 相容入口。
