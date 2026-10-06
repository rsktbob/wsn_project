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
│  └─ Combine.PSO
├─ BaseEDA
│  ├─ Combine.EDA
│  └─ Scheduling.NBEDA
├─ BaseGWO
│  └─ Combine.GWO
├─ BaseNSGAII
│  ├─ Combine.NSGAII
│  └─ Scheduling.SNSGAII
├─ BaseSE
│  ├─ Combine.CodingSE
│  ├─ Combine.CodingSEv2
│  ├─ se.SETS                   （家族一：identity sensor 分區，跟所有商品投資）
│  │  └─ se.SA_SETS
│  │     ├─ se.RegionSelectionSA_SETS → EXP3 / LinUCB / Thompson / CTIG
│  │     └─ se.SA_SETS_Target
│  ├─ se.SI_SETS                （家族二：只跟選到的區域投資；流程不同，不繼承 SA_SETS）
│  ├─ se.Ring_SETS              （家族三：區域只決定交配範圍，共用商品）
│  │  ├─ se.Ring_SETSpriority
│  │  ├─ se.LinUCB_Ring_SETS / LinUCB_Ring_SETSpriority（每組交易用 LinUCB 選區段）
│  │  └─ se.RL_Ring_SETS / RL_Ring_SETSpriority（每組交易用 D3QN 選區段）
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
重新編號為 `SETSv1`、`SETSv2`。2026-09-27 起 `SETSv1` 改名為 `SETS`，成為
SA_SETS 的父類別；原 `SA_SETSv3`、`SA_SETSv4` 改名為 `Ring_SETS`、`Ring_SETSv2`，2026-10-04 起 `Ring_SETSv2`（及其 LinUCB／RL 版）再改名為 `Ring_SETSpriority`（C4 PriorityEncoding 版）。
三個家族共用的 Beta 記憶、平行評估與角色分離更新放在 `se/market_components.py`；
sensor 分區、初始解與交配突變放在 `se/sensor_operators.py`（以 mixin 給 SETS、SA_SETS、
SI_SETS 共用）。2026-10-03 起 SI_SETS 直接繼承 BaseSE；論文版 `SETSv2`、`SA_SETSv2` 已刪除；原 `SI_SETS`（單子代、無條件換 goods）刪除，`SI_SETSv2` 改名為 `SI_SETS`。

逐組選區段的 Ring 變體（2026-10-04）：共用部分在 `se/ring_segment_policy.py`
（每組交易的候選範圍與特徵、LinUCB、區段共用權重的 D3QN），訓練用
`train_rl_ring_sets.py`。

RL_SETS v1–v5、`rl_sets_components.py`、`train_rl_sets.py` 及其測試與文件，
2026-10-04 起移到專案根目錄的 `legacy/`，不再註冊、不在測試套件內；
`environment_contract` 已搬到 `se/ring_segment_policy.py`（內容不變）。
以下 mixin 表只描述 legacy 裡的舊版本。

| mixin | 內容 | 使用版本 |
| --- | --- | --- |
| `EnergyConstraintMetrics` | 能量耗損與限制違反摘要 | v2、v3、v4、v5 |
| `LifetimeMetrics` | 加上預估壽命摘要 | v4、v5 |
| `EnvironmentObservation` | 全域能量觀察特徵 | v3、v4、v5（v5 覆寫特徵內容） |
| `ScaledPerturbation` | 依比例的 sensor 數、相鄰值突變 | v4、v5 |
| `RandomPolicyOption` | `random_policy` 均勻隨機基準 | v4、v5 |

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
