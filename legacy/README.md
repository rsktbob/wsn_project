# legacy

不再使用、但保留原始碼與紀錄的實作。這裡的檔案**不在 registry、不在測試套件內**，
也不保證能直接執行（import 路徑仍指向原本的位置）。

## RL-SETS v1–v5（2026-10-04 移入，檔案在 `rl_sets/`）

SI_SETS 上的 D3QN 超啟發式：每個 searcher 每回合選一個運算子，事先在訓練地圖上訓練、
正式實驗凍結使用。被 Ring 家族的逐組區段選擇（`LinUCB_Ring_SETS`、`RL_Ring_SETS`）取代。

| 檔案 | 原位置 | 內容 |
| --- | --- | --- |
| `RL_SETS.py` | `Algorithm/se/` | 基底：每個 searcher 從 4 個運算子（區段交換、level＋priority 突變、只改 level、只改 priority）選一個 |
| `RL_SETSv2.py` | `Algorithm/se/` | 角色分離子代（child1／child2）＋能量限制回饋 |
| `RL_SETSv3.py` | `Algorithm/se/` | 一般突變 vs Lévy 重尾突變 |
| `RL_SETSv4.py` | `Algorithm/se/` | 小／中／大三種擾動尺度，回饋含預估壽命 |
| `RL_SETSv5.py` | `Algorithm/se/` | 交配強度 × 突變強度分開選（以 SA 為中心），回饋含 fitness 與壽命 |
| `rl_sets_components.py` | `Algorithm/se/` | v2–v5 共用的 mixin（能量／壽命摘要、環境觀察、隨機策略基準等） |
| `train_rl_sets.py` | 專案根目錄 | v1–v5 的訓練腳本 |
| `smoke_rl_sets.py`、`smoke_rl_setsv5.py`、`smoke_train_rl_sets.py` | `tests/` | 對應的 smoke 測試 |
| `RL_SETS_reward_20260909.md`、`RL_SETSv2_training.md`、`RL_SETSv61.md` | `docs/` | 回饋設計、訓練紀錄與 v61（現 v5）設計文件 |

**結果摘要：** 完整 lifetime 比較（單一訓練 seed）最好約 1031，仍低於同設定 SA-SETS 的 1046；
沒有在多個訓練 seed 下證明勝過 SA-SETS（見 `RL_SETS_reward_20260909.md`）。

**仍留在原處的：**
- 訓練好的模型 `Models/rl_setsv6_60maps_1pass.npz`（現 v4）、`Models/rl_setsv61_*.npz`（現 v5）及其 `.training.json`。
- 歷史實驗結果在 `experiment_results/`。
- `Algorithm/se/d3qn.py`（NumPy D3QN）仍由 `RL_Ring_SETS` 使用。
- `environment_contract()` 已搬到 `Algorithm/se/ring_segment_policy.py`，內容不變。

**若要恢復：** 把檔案移回原位置，並在 `experiments/registry.py`（`ALGORITHM_IMPORTS`、
`RL_SETS_ALGORITHMS`）、`experiments/presets.py`、`experiments/parameter_basis.json`、
`experiments/builder.py`（`rl_sets` 的特殊建構、v4/v5 的隨機策略基準）、
`experiments/config.py`（v4/v5 預設 5000 次評估）、`tests/baseline.json`、
`tests/smoke_selected_algorithms.py`、`tests/smoke_experiment_algorithms.py` 加回對應項目；
`RL_SETS.py` 需要改回自己定義或 import `environment_contract`。

## Retired algorithm families (2026-10-08)

Moved `Algorithm/misc`, `Algorithm/nsga`, `Algorithm/Routing`, and
`Algorithm/Scheduling` into `legacy/algorithms/`. Their former registry entries,
presets and parameter notes were removed, so they no longer appear in
`--algorithm all`. Tests that directly imported these families are retained
in `legacy/tests/` and are not part of the active test suite. They still use
the old import paths and are kept as historical references.

`Algorithm/map_elites` was already empty when this cleanup began: its two
tracked files were deleted in the working tree. The last committed versions
were copied from Git into `legacy/algorithms/map_elites/` without changing
those existing deletions. Its stale registry and preset entries were removed
and its tests archived.

## GOMEA family (2026-10-08)

`Algorithm/gomea` was moved to `legacy/algorithms/gomea/`. The GI-GOMEA,
GPU-GOMEA, and target-coded GI-GOMEA registry entries, presets and parameter
notes were removed. Their dedicated tests and parameter benchmark are in
`legacy/tests/`; the active sensing-mode test no longer imports them.

## SA-SETS 選區變體與 QEA3／QEA 對照組（2026-10-08）

| 檔案 | 原位置 | 內容 |
| --- | --- | --- |
| `algorithms/se/RegionSelectionSA_SETS.py` | `Algorithm/se/` | SA-SETS 加上可替換選區策略的基底 |
| `algorithms/se/EXP3_SA_SETS.py`、`LinUCB_SA_SETS.py`、`Thompson_SA_SETS.py`、`ContextualThompsonIG_SA_SETS.py` | `Algorithm/se/` | 以 bandit 選學長分區的 SA-SETS（`exp3_sa_sets` 等） |
| `algorithms/quantum/QEA3.py`、`qea3_wsn.md` | `Algorithm/quantum/`、`docs/` | QEA3 遷移變體與說明 |
| `tests/smoke_qea3.py` | `tests/` | QEA3 smoke 測試 |
| `plotting/plot_region_policy_first_segments.py` | `plotting/` | 選區策略第一段的繪圖 |

registry、presets、parameter_basis 已移除 `exp3_sa_sets`、`linucb_sa_sets`、`thompson_sa_sets`、
`contextual_thompson_ig_sa_sets`、`qea3`、`qea_random`、`qea_classical`。
`qea_random`／`qea_classical` 只是 `QEA` 的 `update_rule='none'`／`'classical'` 設定，
`Algorithm/quantum/QEA.py` 仍保留這兩種規則；`tests/smoke_qea.py` 與 `experiments/benchmark_qea.py`
已改成只用 `qea`。選區變體被淘汰的原因：學長分區已證明無效，主線改用 Ring。

## Retired core helpers (2026-10-08)

`Algorithm/core/evaluation.py` and `Algorithm/core/initialization.py` were
moved to `legacy/core/`. They are referenced only by archived algorithm families.
The active `Algorithm.core` package no longer exports `BatchEvaluator`.
Restore these files to their former paths along with the retired algorithms
if those implementations are brought back.

## Evaluation budget helper (2026-10-08)

`Algorithm/core/budget.py` was moved to `legacy/core/budget.py`. The active GA
now checks `self.evatime < evaluate` directly before each generation. This
preserves its whole-generation behavior, including a slight budget overshoot.
The removed `iterations_to_reach_budget` helper had no active callers.

## GA 基底封存（2026-10-09）

現行 GA 已合併為 `Algorithm/ga/GA.py`，直接繼承 `Algorithm`。
原 `BaseGA` 保留於 `algorithms/ga/BaseGA.py`，SGA、RGA 的引用已指向此處。
其餘 legacy 執行限制仍如上述，不屬於現行測試套件。

## SA-SETS 2023 原始程式重現（2026-10-13，檔案在 `sa_sets2023/`）

照學長 `codes/2023以恩MWSN/Algorithm/Combine/SA_SETS.py`、`SETSM.py`、`State/CodingState.py` 的**原始程式**
（包含已知錯誤）重寫，只用來重現原版、說明修正前後差多少，不進正式比較。正式比較用 `Algorithm/se/SA_SETS.py`。
這個資料夾**可以直接執行**（`run.py` 在執行時註冊，不改 registry）：

```
python legacy/sa_sets2023/run.py --algorithm sa_sets2023 --fitness-service v1 --maps maps/maps_100100100_a2.json ...
```

| 檔案 | 內容 |
| --- | --- |
| `SA_SETS2023.py` | 繼承正式 SA_SETS，改：軟分區（只強制開、只看原值 == 0）、基因 0–9（FMAX=10，含感測等級）、回傳最好的 searcher |
| `SensorEncoding2023.py` | 原版解碼：v1（無冗餘移除）、原版選父節點打分 `min(⌊連線⌋+路徑/最大路徑, ⌊路徑⌋+連線/最大連線) − 封包量`（BS 路徑容量 0x3F3F3F3F）、容量不足寫回染色體 |
| `verify_decode.py` | 和學長 `CodingState.Decode` 逐顆比對（先跑 `tmp/sa_sets_2023/orig_side.py` 產生 npz） |
| `run.py` | 註冊 `sa_sets2023`（參數同 sa_sets：n8 h4 w2 mu0.4）後呼叫 experiment_algorithms |

**和正式 SA_SETS 的差異：** 分區（軟／強制）、解碼（v1／v3）、選父節點打分、容量不足寫回、基因值域（0–9 mod 6／均勻）、
回傳（最好 searcher／全程最佳）、fitness（v1／v3，用 CLI 指定）。路由優先值兩邊都沒有 0.5（10/13 正式版已拿掉）。
市場流程、identity sensor、CCS、交配、突變組數、goods 更新、選區、期望值、Beta、步長兩邊相同。

**驗證（10/13）：** A2 滿電 200 組學長 CreateCode 染色體，`RING=0`（距離排序，同學長）時開關／等級／下一跳／耗能 200/200 逐顆相同。

**未照抄：** 亂數產生方式（原版子程序的 `random` 沒設種子、無法重現）；原版不算 goods 初始評估、每回合多算 1 次 best_w、迴圈條件 `<=`。

**論文 vs 原始程式：** 論文（第 35–43 頁）寫強制分區、式 44 電量＋距離選 identity、Cr、交配點避開識別位元、goods 菁英、
searcher 直接被較好的投資取代、B_j 用歷史最佳、IBeta(t_b, t_a, e)；原始程式都沒照做，本實作以原始程式為準。

## RL-Ring-SETS 系列（2026-10-13 移入，檔案在 `rl_ring/`）

使用者決定 RL 部分重新設計，整批移出正式程式。**不在 registry、不在測試套件內**；import 路徑仍指向 `Algorithm.se.*`，不能直接執行。

| 檔案 | 原位置 | 內容 |
| --- | --- | --- |
| `RL_Ring_SETS.py`、`RL_Ring_SETSv2.py`、`RL_Ring_SETSpriority.py`、`RL_Ring_SETSpriorityv2.py` | `Algorithm/se/` | 每組 searcher-good 用 D3QN 選區段（v2：v2 特徵＋分級回饋、γ=0；priority 版用 C4） |
| `LinUCB_Ring_SETSv2.py`、`LinUCB_Ring_SETSpriorityv2.py` | `Algorithm/se/` | v2 特徵的 LinUCB 版 |
| `ring_segment_policy.py` | `Algorithm/se/` | **搬移前的完整版本**（PairSegmentMarket、LinUCB、D3QN、v2、`environment_contract`）；正式版只留 PairSegmentMarket＋LinUCB |
| `d3qn.py` | `Algorithm/se/` | NumPy D3QN |
| `train_rl_ring_sets.py` | 專案根目錄 | 訓練腳本 |
| `smoke_ring_segment_policy.py`、`smoke_rl_training_maps.py` | `tests/` | 搬移前完整的測試（正式版 `tests/smoke_ring_segment_policy.py` 只測 LinUCB） |
| `parameter_basis_removed.json` | — | 從 `experiments/parameter_basis.json` 移除的 8 筆說明（含下面兩支） |

**仍留在原處：** `Models/` 的 checkpoint（`rl_ring_sets*`、`rl_ring_setspriority*`、`v2design_*`、`validation_snapshots.pkl.gz`）不動；
`--rl-model` 參數與 `RL_SETS_ALGORITHMS`（現為空集合）保留給下一版 RL。`tmp/` 裡 import `ring_segment_policy` 的 D3QN 部分或 `d3qn` 的舊實驗腳本不能再直接執行。
注意：這些 checkpoint 也因 2026-10-13 SensorEncoding 拿掉 0.5 而不再相容（environment_contract 雜湊）。

**正式程式留下的 LinUCB：** `LinUCB_Ring_SETS`（registry 名 `linucb_ring_se`）、`LinUCB_Ring_SETSpriority`（`linucb_ring_sepri`，C4），good 更新仍是 EE。

**同日移入 `algorithms/se/`、`tests/`：** `CodingSEv2.py`（2021 target 編碼 SE）＋`smoke_codingsev2.py`。`SA_SETS_Target` 同日又依使用者要求搬回正式程式 `Algorithm/se/sets/`。

## PriorityEncoding 其他版本（2026-10-13 移入，檔案在 `priority_encoding/`）

使用者決定 PriorityEncoding 只留 EF（C5：開啟看電量 E/E_max、路由 E/E_max + d/d_max）。這裡是**搬移前的完整檔案**：

| 檔案 | 原位置 | 內容 |
| --- | --- | --- |
| `PriorityEncoding.py` | `State/` | 可切換的 base：開啟 sum／cost（C4）／energy、路由 sum／linear／energy（C4）／far（EF）；level 規則 gain／efficiency／total；單基因版（split_priority=False） |
| `Ring_SETSpriority.py` | `Algorithm/se/` | 定義 `PriorityEncodingC4`、`PriorityEncodingEF` 子類別的版本 |
| `LinUCB_Ring_SETSpriority.py` | `Algorithm/se/` | C4 版 LinUCB（正式版已改 EF） |
| `smoke_priority_decode_kernel.py`、`smoke_ring_setspriority.py` | `tests/` | 含 C4／單基因的舊測試 |

正式版 `State/PriorityEncoding.py` 現在就是 EF（不再有 `PriorityEncodingEF` 類別名），對舊 EF 300 組染色體（電量不均）解碼逐欄位相同。
`_delivery_cost`（C_s，給 C4 的 cost base 與 QUBO 用）移到 `Qubo/sensor_levels.py`。PriorityEncoding 的 0.5 偏移仍保留。
