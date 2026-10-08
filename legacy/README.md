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
