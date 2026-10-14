# MEALPY GA 直接解 WSN

直接呼叫 [MEALPY 3.0.3](https://pypi.org/project/mealpy/3.0.3/) 的 `GA.BaseGA.solve()`；選擇、交配、突變與存活選擇均由套件執行，沒有重寫 GA。WSN 端沿用專案的 `SensorEncoding`、解碼器、fitness 與 lifetime runner。

## 環境與執行

MEALPY 3.0.3 要求 NumPy <=1.26.0，因此使用本資料夾下的獨立 `.venv`，不變更原本 wsnenv。以下在 `codes/2027WSN` 執行，建議 Python 3.12：

```bash
python -m venv Algorithm/experiment/mealpy_ga/.venv
Algorithm/experiment/mealpy_ga/.venv/bin/python -m pip install -r Algorithm/experiment/mealpy_ga/requirements.txt
```

直接跑 GA（預設每張地圖單次最佳化，6 張地圖、seed 7、各 1000 次評估）：

```bash
Algorithm/experiment/mealpy_ga/.venv/bin/python Algorithm/experiment/mealpy_ga/run.py
```

直接跑 lifetime：

```bash
Algorithm/experiment/mealpy_ga/.venv/bin/python Algorithm/experiment/mealpy_ga/run.py \
  --mode lifetime --runs 1 --evaluate 1000
```

也已註冊共用入口，可一起比較專案 GA 與 QEA：

```bash
Algorithm/experiment/mealpy_ga/.venv/bin/python experiment_algorithms.py \
  --algorithm mealpy_ga ga qea --mode lifetime \
  --runs 3 --seed 7 --evaluate 1000 \
  --result-dir Algorithm/experiment/results --experiment-name mealpy_ga_comparison
```

`run.py` 在載入 NumPy 前預設 BLAS/OMP 使用 1 個執行緒，以降低小型矩陣的多執行緒開銷；若你已設定環境變數則保留設定。直接使用共用入口時，可在命令前加 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。

`mealpy_ga` 有額外套件需求，必須明確指定；不加入預設 `--algorithm all`。`run.py` 支援原本共用入口的地圖、fitness、感測模式、移動、繪圖與執行模式參數。`--draw-save true` 可輸出 lifetime 每段能量／路由圖。Windows 請使用 `.venv/Scripts/python.exe`。

## 編碼與參數

每個 sensor 有 `[感測等級, 排序參數]` 兩個整數基因，100 sensors 是 200 維。使用 MEALPY `IntegerVar`，感測域採各節點 `radius_option_counts`，排序域為 0–9。透過 `IntegerVar.decode()` 取得合法整數後交給 `SensorEncoding`，由共享解碼器決定最終感測配置與路由；基因不是節點座標。

使用 `minmax='max'` 最大化專案目標值總和。預設 fitness v2、routing v1、discrete sensing，與共用入口一致。

- 族群 30。
- 交配率 pc=0.95、突變率 pm=0.025。
- Tournament selection（k_way=0.2）、uniform crossover、multipoint flip mutation。
- 參數在 `experiments/presets.py` 的 `mealpy_ga` 項目。這些是初始設定，尚未調參。

MEALPY 3.0.3 每代以配對產生子代，故此 adapter 限制族群為 >=10 的偶數；也確保 20% tournament 有兩個候選父母。

## 預算與 lifetime

`--evaluate` 計算真實 WSN objective 呼叫，包含初始族群。設定已知 scalar objective metadata，避免 MEALPY 額外試算目標維度。預算中途耗盡時，以專用例外停止 `solve()`，回傳所有已評估候選中的最佳 WSN 狀態。因此最後一代可能未完成，MEALPY 自己的 `g_best`／epoch history 不一定更新到末筆；專案回傳的 `best_state`、`best_candidate`、history 與 evaluations 才是完整評估紀錄。沒有替換套件的 GA 更新運算。

每次 lifetime 重新最佳化建立新的 MEALPY model 與族群，從演算法物件的固定 seed 亂數流取得下一個 seed，確保整次實驗可重現。fitness 預算是每次搜尋的預算；總評估次數由共用 runner 另行記錄。

Lifetime 沿用共用 runner：可完整覆蓋、連通且支付能耗才計入一個 slot；同配置服務至不可行後重新搜尋。搜尋找不到可行解時停止，這不代表已證明所有可能配置都不可行。達 `max_rounds`／`lifetime_cap` 為截尾資料。slot 不等同於秒或首個節點死亡時間。

## 檔案與驗證

- `MealpyGA.py`：MEALPY 與 WSN 介面的 adapter。
- `run.py`：直接執行入口，呼叫既有 `experiment_algorithms.py`。
- `requirements.txt`：獨立環境版本。
- `test_mealpy_ga.py`：檢查真實 objective 次數、三種感測模式、最佳解重新解碼、同 seed 重現、迭代上限與 lifetime。

```bash
Algorithm/experiment/mealpy_ga/.venv/bin/python Algorithm/experiment/mealpy_ga/test_mealpy_ga.py
```

`run.py` 的預設輸出位於 `Algorithm/experiment/results/mealpy_ga/`。共用 runner 會建立實驗資料夾、參數報告、各地圖 `summary.csv` 及 lifetime 的 `trace.csv`；若指定名稱已存在會加序號。使用終端顯示的 `result_dir` 找到本次結果。

## 已執行的驗證

在獨立環境中通過 `pip check` 與 `test_mealpy_ga.py`。經 `run.py --mode lifetime --runs 1 --evaluate 1000 --experiment-name lifetime_pilot` 實跑 MAP01–MAP06、seed 7，lifetime 依序為 1036、947、935、968、814、969 slots，平均 944.83。六次均以 `infeasible_solution` 結束，未達上限。每張地圖的 trace 加總與 summary lifetime、真實評估次數一致。這是單 seed 功能驗證，尚未做演算法優劣比較。結果在 `Algorithm/experiment/results/mealpy_ga/lifetime_pilot/`。

## 與專案原本 GA 比較

使用相同環境、地圖、seed、fitness 與 lifetime runner，比較原本 `ga` 與 `mealpy_ga` 的現有設定。原本 GA 初始化 30 次，每代 30 個體再重評當代最佳一次，因此選 **1022 = 30 + 32 × 31** 避免它超出 1000 次預算；MEALPY adapter 則精確停止在 1022 次。這裡等化真實 fitness 呼叫，並非新子代數、總生命週期算力或運算時間。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
Algorithm/experiment/mealpy_ga/.venv/bin/python experiment_algorithms.py \
  --algorithm ga mealpy_ga --mode lifetime --maps maps/maps_100100100.json \
  --runs 3 --seed 7 --evaluate 1022 --fitness-service v2 \
  --sensing-mode discrete --moving false --execution parallel --jobs 4 \
  --result-dir Algorithm/experiment/results/mealpy_ga \
  --experiment-name ga_comparison_1022

Algorithm/experiment/mealpy_ga/.venv/bin/python \
  Algorithm/experiment/mealpy_ga/compare_results.py \
  Algorithm/experiment/results/mealpy_ga/ga_comparison_1022
```

原本 GA 有覆蓋需求初始化、4 人 tournament、區段交配（cu=.8）、以 .1 機率改一個基因，並保留其 decoded snapshot 快取行為；MEALPY 採均勻整數初始化、20% tournament、uniform crossover（pc=.95）與 multipoint flip（pm=.025）。因此比較的是兩個實作及設定的整體效果，不能把差異全歸因於使用套件；兩種突變率的作用層級也不同。此處沒有修正或改造原本 GA。

`compare_results.py` 會核對配對案例、共同設定、逐段真實評估次數與 lifetime 加總，輸出 `comparison.json`、`comparison.md`、`comparison.png`。每張圖上的點是一組相同地圖／seed；對角線上方表示 MEALPY lifetime 較長。平行執行的耗時不是隔離的效能基準。

本次 6 地圖 × seeds 7–9 的結果：原本 GA 平均 lifetime 806.06，MEALPY GA 923.94 slots（平均增加 14.63%）；18 組配對均由 MEALPY 較長。兩者皆以搜尋不可行結束，未碰到上限。整段 lifetime 的平均真實 fitness 呼叫分別為 22370.44 與 27537.22，故不是等總算力比較。這是現有設定下的小型實驗，不是套件或 GA 變體的一般優越性證明。

## 新增 baseline 設定（保留兩個舊版本）

`mealpy_ga_baseline` 使用 n=30、pc=.8、pm=`1/d`（d=2×sensor_count；100 sensors 為 .005）。n、pc 對齊專案 GA，pm 按維度縮放，使每個子代平均嘗試替換 1 個基因，實際改變可能少於一次。均勻交配、20% tournament、multipoint flip 與均勻整數初始化維持 MEALPY 原有機制。

1/d 是可解釋的起始設計，例如 [pymoo Mutation 的預設每變數機率](https://github.com/anyoptimization/pymoo/blob/main/pymoo/core/mutation.py) 也隨維度縮放；此處只借用機率設計，沒有改成 pymoo 的突變算子。這不是 GA 唯一的「標準參數」，也不是已在獨立訓練地圖調好的最佳參數。pm 的 `1/d` 規則會在每次搜尋重新計算，參數報告中保留字串以便追蹤。

重跑三個設定（原本 GA、原 MEALPY、新 baseline）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
Algorithm/experiment/mealpy_ga/.venv/bin/python experiment_algorithms.py \
  --algorithm ga mealpy_ga mealpy_ga_baseline --mode lifetime \
  --maps maps/maps_100100100.json --runs 3 --seed 7 --evaluate 1022 \
  --fitness-service v2 --sensing-mode discrete --moving false \
  --execution parallel --jobs 4 \
  --result-dir Algorithm/experiment/results/mealpy_ga \
  --experiment-name baseline_comparison_1022

Algorithm/experiment/mealpy_ga/.venv/bin/python Algorithm/experiment/mealpy_ga/compare_results.py \
  Algorithm/experiment/results/mealpy_ga/baseline_comparison_1022 \
  --methods ga mealpy_ga mealpy_ga_baseline
```

此回同時改了 pc 與 pm，不能只根據結果判定是哪一個參數造成差異。和專案 GA 也未對齊所有運算子，因此仍是設定整體比較。這些地圖已被用於先前觀察，後續正式驗證需要另留測試資料，避免把重複觀察當成未見資料的泛化證據。

已完成 6 地圖 × 3 seeds 的 54 次 lifetime 執行。平均 lifetime：專案 GA 806.06、原 MEALPY 923.94、新 baseline 805.00 slots。新 baseline 對專案 GA 配對 10 勝 8 敗，平均差 -1.06 slots；對原 MEALPY 配對 1 勝 17 敗。新 baseline 的平均總評估次數 22313.67，專案 GA 22370.44，原 MEALPY 27537.22；各次重新最佳化均為實際 1022 次評估，且 54 次皆以 `infeasible_solution` 停止。原本兩組的 18 筆配對結果與上一輪完全一致。本結果顯示新 baseline 在這批地圖上沒有改善 lifetime；由於同時改動 pc 與 pm，不能判定個別參數的因果作用。

## 可引用的 GA baseline：套件官方預設與公平比較

先前的 `mealpy_ga_baseline`（n=30、pc=.8、pm=1/d）是本專案自行設計的**探索性設定**，沒有一篇論文同時指定這三個值；它在同一批資料上測試過，不可寫成「照論文參數」或未見資料的最終 baseline。

若需要可追溯、無主觀混搭的外部 GA 參考，`mealpy_ga_official` 精確使用 [MEALPY v3.0.3 的 `GA.BaseGA` 原始碼](https://github.com/thieu1995/mealpy/blob/v3.0.3/mealpy/evolutionary_based/GA.py)建構子預設：`pop_size=100`、`pc=.95`、`pm=.025`，並保留 `tournament`（`k_way=.2`）、`uniform` crossover、multipoint `flip` mutation。這是**套件官方預設**，不是聲稱 [MEALPY 2023 套件論文](https://www.sciencedirect.com/science/article/pii/S1383762123000504)提出適用所有 WSN 的 GA 最佳參數。模型仍須使用本專案 `IntegerVar -> SensorEncoding -> WSN fitness` 的目標轉接；評估預算控制會在最後一代中途停止，並保留所有已評估解的最佳結果。

實際比較通常保留有來源的預設或作者實作，讓各方法使用相同地圖、目標函數、seed 與真實 fitness 評估上限，報告配對結果、失敗原因和總成本。若要調參，應事先指定另外的調參資料與相同調參資源，最後才在未見地圖測試；不能看到測試結果後挑一組有利參數當成原始論文設定。[NIST 的 GA 參數研究](https://www.nist.gov/publications/determining-relative-importance-and-best-settings-genetic-algorithm-control-parameters)也指出 GA 控制參數的建議在既有研究中有分歧；[動態 GA benchmark 研究](https://link.springer.com/article/10.1007/s11047-020-09799-y)指出只固定代數會導致不同方法消耗不同評估次數。這是為什麼本文固定真實 fitness 呼叫數。

已跑過的 MAP01–MAP06 及 seeds 7–9 用來檢查程式和探索參數，不能再當作全新的獨立測試集。正式碩論若主張泛化，應新增未參與設定決策的 WSN 地圖／seed；不同演算法可採各自文獻預設，但必須逐項寫明差異和來源。

官方預設比較指令（原本專案 GA 與 `mealpy_ga_official`）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
Algorithm/experiment/mealpy_ga/.venv/bin/python experiment_algorithms.py \
  --algorithm ga mealpy_ga_official --mode lifetime \
  --maps maps/maps_100100100.json --runs 3 --seed 7 --evaluate 1022 \
  --fitness-service v2 --sensing-mode discrete --moving false \
  --execution parallel --jobs 4 \
  --result-dir Algorithm/experiment/results/mealpy_ga \
  --experiment-name official_default_comparison_1022

Algorithm/experiment/mealpy_ga/.venv/bin/python Algorithm/experiment/mealpy_ga/compare_results.py \
  Algorithm/experiment/results/mealpy_ga/official_default_comparison_1022 \
  --methods ga mealpy_ga_official
```

此處 `--evaluate 1022` 是每次重新最佳化的實際 fitness 呼叫上限；100 個體的 MEALPY GA 最後可只評估一部分子代，因而未必完成整代。即使每次上限一致，整段 lifetime 的重新最佳化次數仍可能不同，應另外報告總評估數。除了官方套件預設，也保留原本專案 GA 作為 WSN 領域內的實務對照；兩者的初始化及運算子不同，不應用結果推論單一參數的效果。

官方預設 6 地圖 × 3 seeds 的結果：專案 GA 平均 lifetime 806.06，MEALPY 官方預設 889.72 slots（+10.38%）；配對 18 勝 0 敗。每次搜尋都精確使用 1022 次真實評估，整段 lifetime 的平均總評估數分別是 22370.44 與 27650.78；兩者在此批案例均因搜尋未找到下一個可行解而結束。這些地圖已在前述探索中被看過，本結果是可重現的既有地圖比較，不能當作未見資料的泛化驗證。結果與配對原始值見 `Algorithm/experiment/results/mealpy_ga/official_default_comparison_1022/comparison.json`。
