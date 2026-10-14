# TN-GEO：WSN 實驗原型

這是集中式、GEO-inspired 候選解生成器，沿用 `SensorEncoding` 的感測選項／優先權編碼、目前預設解碼器及專案 fitness（**總分越大越好**）。所有新程式均位於本目錄；未註冊到正式演算法清單。不是分散式協定，也不是原論文投資組合實驗的逐項重現。

## 實作內容

- `models.py`：NumPy 實數 MPS Born machine：`p(x)=|ψ(x)|²/Z`，包含雙層張量收縮求正規化、逐變數條件抽樣、完整加权 NLL 解析梯度（含 log Z）、梯度裁切與 Adam。各次收縮縮放避免長鏈下溢，不建立完整機率表。零或接近零振幅／非有限數值明確報錯，不偷偷退回另一個模型。
- `optimizer.py`：`TNGEO(Algorithm)`，回傳共用 `AlgorithmResult`。保留資料庫、最高分 archive、softmax 重抽訓練資料、更新模型、抽樣與評估。softmax 對「分数」取正號，溫度為 archive 分數標準差乘 `temperature`。
- `run.py`：固定電量或完整壽命入口，可選 `mps` / `independent`、`identity` / `random` / `coverage` 排序，支援 discrete / bucketed / exact。
- `test_tn_geo.py`：枚舉機率、抽樣頻率、有限差分梯度、兩顆三態互補關係學習、200 變數數值穩定性與真實 WSN 整合測試。

NumPy 是唯一新增模型所需的數值依賴；其餘沿用專案研究環境。已用 `/home/g114056067/miniconda3/envs/wsnenv/bin/python` 驗證。

## 公平性與限制

初始種群使用既有 GA/SETS 的 CCS 方式（target 逐一寫入 sensor level，保留原覆寫行為），並為初始化配置獨立亂數流；相同 seed/問題/initial-size 下，不同模型及排序有相同初始解。模型種子、探索及排序亂數流分開。

`identity` 是 Problem 目前的 sensor 順序，預設可能已經是 ring 排序，不代表原始檔案順序。`coverage` 以最大感測能力的 target 重疊數，貪婪串接 sensor；每顆 sensor 的 level/priority 相鄰。這是覆蓋排序原型，不代表已經包含路由交互圖。

預設新解來源：70% 模型、10% 完全均勻探索、20% 對 archive 解修改最多兩顆 sensor 的 level/priority。兩個模型都使用同樣探索機制。重複染色體不消耗 FE；批次缺額用均勻探索補足，並計入獨立 fallback 計數。重試有上限，空間耗盡／找不到新解可提早停止；不承諾有限預算遍歷全部狀態。新染色體即使解碼相同仍消耗 FE，另外記錄解碼後唯一配置數。

獨立 categorical 基準以同一批訓練樣本直接做最大似然頻率估計（另加微小數值平滑）；MPS 以有限步梯度近似。因此两者訓練求解程序不同，`train_steps` / learning rate 不影響獨立基準。比較時同時報 FE 和含訓練的牆鐘時間。

此版使用固定 rank 和 Adam，沒有原始 MPS 訓練的自適應 bond dimension／兩站點 SVD sweep。三態教學例子已推廣為每個 sensor 真正合法的選項數。每次 run 都清空資料庫並重新建立模型，避免跨電量狀態沿用舊分數。最佳解依既有 fitness 排序，不另加隱藏可行性排序；可行性用 LifeCheck 另行報告。

## 執行

以下指令可由工作區根目錄執行：

```bash
/home/g114056067/miniconda3/envs/wsnenv/bin/python codes/2027WSN/Algorithm/experiment/tn_geo/run.py --map MAP01 --budget 1000 --model mps --order coverage --output run_mps
/home/g114056067/miniconda3/envs/wsnenv/bin/python codes/2027WSN/Algorithm/experiment/tn_geo/run.py --map MAP01 --budget 1000 --model independent --order coverage --output run_independent
/home/g114056067/miniconda3/envs/wsnenv/bin/python codes/2027WSN/Algorithm/experiment/tn_geo/run.py --map MAP01 --budget 1000 --model mps --order random --output run_random_order
/home/g114056067/miniconda3/envs/wsnenv/bin/python codes/2027WSN/Algorithm/experiment/tn_geo/run.py --mode lifetime --budget 1000 --max-lifetime 2000 --output run_lifetime
/home/g114056067/miniconda3/envs/wsnenv/bin/python codes/2027WSN/Algorithm/experiment/tn_geo/test_tn_geo.py
```

`--budget` 是每次重新最佳化的真實 fitness 呼叫上限；lifetime 的 `total_evaluations` 是整場累計。`--max-lifetime` 與 `--max-segments` 截斷會標示 `truncated=true`。`infeasible_solution` 表示此次搜尋最佳解無法服務，不證明不存在可行解。固定狀態執行不扣能量，也不把其分數當 lifetime。

輸出強制放在本目錄子資料夾，已有相同輸出時報錯，避免覆寫。JSON 保存設定、地圖、每次評估最佳值 history、每批次 trace、FE、可行評估數、染色體／物理解唯一數、重複提案數、各來源 FE、模型訓練與總耗時、最佳編碼及解碼結果。

程式 API：

```python
from Algorithm.experiment.tn_geo.optimizer import TNGEO
result = TNGEO(seed=7, rank=4, order='coverage').run(problem, budget=1000)
state = result.best_state
```

## 已完成驗證

7 個自動測試通過。MAP01 的 MPS 與獨立模型各完成 96 FE 的固定狀態 CLI 驗證；MPS 的 lifetime 入口以 40 FE、2 slots 上限驗證能量推進與截斷。這些是功能測試，不能據此宣稱效能優於 SETS 或獨立模型。小測試結果位於 `smoke_mps`、`smoke_independent`、`smoke_lifetime`。

## 來源

- GEO： https://www.nature.com/articles/s41467-024-46959-5
- 可讀原稿（Figure 1 / Appendix A）： https://arxiv.org/pdf/2101.06250
- MPS Born machine： https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.031012

研究下一步是固定多張地圖、電量狀態及種子，以相同 FE 與相同時間兩種條件比較，而非立即啟動大量超參數搜尋。
