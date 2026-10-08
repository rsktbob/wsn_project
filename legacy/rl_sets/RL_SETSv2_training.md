# RL_SETSv2 與跨地圖訓練（2026-09-07）

> 2026-09-09 更新：V1、V2 現已共用新的商品更新 reward；下面的初版 reward
> 描述屬於歷史紀錄。公式、短期比較及模型續訓規則請看
> [reward 修正紀錄](RL_SETS_reward_20260909.md)。

## 結論與目前狀態

新增 `Algorithm/se/RL_SETSv2.py`；原本的 `RL_SETS.py`、`d3qn.py` 與 `Models/rl_sets_d3qn.npz` 保留。V2 是待實驗的新方法，不代表已經勝過 SA-SETS。已完成實體地圖生成、跨地圖訓練入口與短流程測試；尚未執行完整 600-episode 訓練，也沒有可交付的正式 V2 訓練模型。

建議採用「預先產生、固定保存的多地圖資料集 + 每輪打亂順序」，而非每 episode 臨時生成不留檔的地圖。這仍然是隨機生成的訓練地圖，但能重現、排除測試資料重複，也方便公平比較不同方法。

## 三份地圖清單

| JSON | 用途 | 實際資料 |
| --- | --- | --- |
| `maps/maps_100100100.json` | 正式六圖實驗 | MAP01–MAP06（已按三重覆蓋規範重生） |
| `maps/maps_100100100_a2.json` | 原始單圖測試 | A2（既有） |
| `maps/maps_100100100_train.json` | 訓練與獨立驗證 | 200 張 RLTRAIN + 40 張 RLTRAINVAL（本次產生） |

第三份 JSON 的 `maps` 只放訓練資料集名稱，`validation_maps` 只放驗證資料集名稱。以下是縮小範例，不是將座標塞進 JSON：

```json
{
  "maps": ["RLTRAIN01", "RLTRAIN02"],
  "validation_maps": ["RLTRAINVAL01"]
}
```

訓練與實驗共用 `experiment_algorithms.load_map_specs`。例如 `RLTRAIN01` 讀取 `data/100100100_RLTRAIN01/sensor.csv`。亦可沿用實驗入口的物件格式指定 `file`、`boundary`、`sensors`、`targets` 等。沒有指定規模時使用原本預設值。各地圖的生成 seed 與座標指紋保存在資料夾內的 `metadata.json`，另附 `sensor.svg` 預覽。

240 張皆為 100×100 區域、100 sensors、100 targets、BS=(3,3)、初始能量 10。訓練地圖 seed=20001–20200，驗證地圖 seed=40001–40040；所有 sensor 座標組互不重複，亦未與 A2 或六張正式實驗地圖重複。2026-09-07 已依每個 target 至少三顆 sensor 候選的規範全部重生；指紋排序座標後計算，不因更名或重新排列 sensor id 而漏掉重複資料。

「隨機地圖」目前指 sensor 座標隨機，targets 仍沿用原始固定網格；不是 targets 與 BS 也隨機。生成器透過 `Problem(FILE=None)` 的原生重抽流程，要求每個 target 在最大感測半徑下至少有三顆 sensor 候選。這只是三重 coverage 檢查，不是 routing 或完整可行排程的證明。因此這批資料針對同規模、同分布的泛化，不宣稱涵蓋不同大小、障礙物或任意 BS 分布。

已執行的生成命令如下；一般情況會拒絕同名資料，只有明確加上 `--overwrite` 才會原位取代生成檔案：

```powershell
python CreateMapData.py --count 200 --seed 20001 --prefix RLTRAIN --split train --validation-count 40 --validation-seed 40001 --manifest maps/maps_100100100_train.json --overwrite
```

產生其他批次時改用新 `--prefix`、`--seed`、`--validation-seed` 與 manifest 路徑；可設定 `--boundary`、`--sensors`、`--targets`。訓練同一模型的這份清單目前要求規模與初始能量一致。

## 舊實驗失敗：證據與假說要分開

以下是既有 `experiment_results` 的五次 A2 壽命結果；不是本次新跑的正式比較。

| 結果資料夾 | 演算法 | 記錄的 fitness / routing / decode | 五次平均 lifetime |
| --- | --- | --- | ---: |
| `20260906_221006_rl_sets` | 舊 RL-SETS | v3 / v1 / 未記錄 | 121.8 |
| `20260906_224743_rl_sets` | 舊 RL-SETS | v5 / v1 / 未記錄 | 336.0 |
| `20260906_232017_sa_sets` | SA-SETS | v5 / v1 / v2 | 942.6 |
| `20260906_235201_sa_sets` | SA-SETS | v5 / v1 / v3 | 932.2 |

可確認的事情：

- 第一組舊 RL 五次最後都 coverage=1，但仍各有 1 顆 disconnected sensor，fitness 約 0.977–0.984；高 fitness 不等於排程可以執行。
- 第二組切到 fitness v5 後仍明顯落後；最後失敗狀態包含 uncovered、disconnected，部分還有 energy failure。`infeasible_solution` 只表示這次搜尋回傳不可執行解，不能直接推論整張網路已物理耗盡、沒有其他可行解。
- 舊 checkpoint 記錄 2,723,760 environment steps、680,877 updates，但沒有訓練地圖與 fitness/routing/decode 的來源資訊。無法證明它與當時每一組實驗使用完全相同環境。舊 RL 結果的 decode 缺漏，因此上述平均差距不能當成版本因素已被控制的因果比較。
- 原本 trainer 的 episode 都建立 A2；換搜尋 seed 只改搜尋路徑，不等於看過新地圖。

根據程式提出的失敗假說（尚未逐項消融證實）：

1. **單圖過擬合與過早結束探索。** 舊 epsilon 每個 action step 乘 0.9995，約 6,000 actions 已到 0.05 下限；一張圖的完整 lifetime 就可能超過這個量。即使只增加地圖數，探索也可能在見到足夠地圖前結束。
2. **操作取代了 SA 的關鍵探索。** 舊 action 2–5 從 `good.copy()` 做局部編輯，並沒有先做原始 SA searcher/good crossover。學到過度偏好某種修補操作時，可能失去 SA 原本的組合探索。
3. **reward 對平台期與微小改善的訊號不理想。** 舊成功率項在完全沒變好時固定給負值；fitness 接近 1 時，小幅但有用的改善可能難與其他訊號區分。新 reward 是有待驗證的設計，不是已找到唯一根因。
4. **FIFO replay 忘記早期地圖。** 20,000 容量的舊 FIFO 在長 episode 後可能主要只剩近期地圖，削弱多地圖訓練效果。

額外小額診斷：舊 frozen checkpoint、A2、seed 7、請求 1,000 evaluations（完整搜尋輪實際 1,040）時，1,024 次投資中 `energy_balanced_replacement` 使用 384 次（37.5%），相對較佳父代的改善只有 5 次（約 1.3%）。它並非完全只選一種操作；這只是操作偏好值得檢查的線索，不能由一次搜尋推論全部失敗原因。

## V2 修改

- 保持 SA-SETS 的分區、選區、編碼、評估預算流程。每個投資輪預留約 25% 候選給原始 SA 操作，其餘交給 RL；不是額外加算一輪 SA，所以不偷偷增加 fitness evaluations。這個配額不保證結果至少等於 SA。
- action 0 是原始 SA crossover；action 1 保留較大幅度交換；action 2–5 改為「先 SA crossover，再 coverage / critical target / energy / routing 修補」。
- reward 綜合相對 good 的平均改善與相對較佳父代的最佳改善；以父代 fitness 分散程度縮放（下限 1e-4）、tanh 限幅。不變的候選 reward=0；強制 SA 配額排除於被選 action 的直接功勞統計之外。
- 觀察值加入 disconnected 與 energy-failed 比例，取代原本最低能量與能量 CV；仍保留平均能量、coverage 與 target 稀缺程度等訊號。
- 改成 reservoir replay，讓早期 transitions 有機會保留到後續地圖。這是對所有 transitions 均勻，不是每張圖等量；壽命較長的圖仍貢獻更多 transitions。
- V2 checkpoint 有獨立 schema。新舊 checkpoint 不可互換；不能直接拿舊 `rl_sets_d3qn.npz` 當 V2。

上述改動只屬於新 V2。短測試確認：V2 關閉保留配額並強制選 action 0 時，相同 seed 的 coding、fitness、evaluation 次數與原始 SA 完全一致。

## 訓練方式與成本

在本專案程式目錄執行：

```powershell
python train_rl_sets.py
```

等同預設 `--algorithm rl_setsv2 --maps maps/maps_100100100_train.json --passes 3 --seed 7`，共 200×3=600 個 episode。**每個 episode 是一張圖的完整 lifetime，不是一次 optimizer call**；每次重新最佳化預設 10,000 evaluations、最多 2,000 segments，因此完整訓練可能很耗時，沒有以短測試推算或承諾完成時間。

- 地圖順序每 pass 打亂，但全部看過才重複。搜尋 seed 依 episode 為 7、8、9……；三個 pass 中同張地圖使用不同搜尋 seed。生成地圖的 seed 是另一件事，不會因搜尋 seed 改變座標。
- 四個平行分區沿用既有從主 seed 決定子亂數串流的方法。可重現不代表四個分區強制共用完全相同亂數序列。
- 模型、optimizer 與 replay 跨地圖保留；每個 episode 重設地圖初始能量與搜尋 RNG。
- epsilon 預設以「走過多少張地圖」衰減，跨兩個完整 pass 從 1 到 0.05；不是前幾千個 action 就歸零探索。可用 `--exploration-passes` 調整。
- `--episodes N`（也接受 `--episode N`）指定總 episode 數；不是每張圖跑 N 次。想每張圖訓練五次可用 `--passes 5`。手動設定 episodes 少於 200 時只會看部分訓練圖，不能稱為完整一輪。
- 預設每 200 episode 與最後一次結束時，於 40 張驗證圖 × seeds 7007–7009 做 frozen lifetime 驗證，並用完全相同 map/seed/budget 跑 SA baseline。第一次驗證 SA 基準會快取，後續只重跑 RL。
- 以獨立驗證圖的平均 lifetime 選最佳 checkpoint，不拿訓練 reward 或最終六張測試圖挑模型；另外記錄對 SA 的平均 lifetime ratio、勝率與零壽命個案。ratio 分母至少 1，SA 為零的個案請另看原始 lifetime，不應解讀為通常意義的倍率。
- `--validation-maps another.json` 可從另一個標準 `maps` 清單載入驗證圖；未指定時讀第三份 JSON 的 `validation_maps`。`--validation-every 0` 可暫停驗證，但不會產生 `.best.npz`，不建議用於正式選模型。

小額流程檢查範例（不是用來判斷演算法優劣，請用獨立輸出路徑）：

```powershell
python train_rl_sets.py --episodes 2 --evaluate 40 --max-rounds 1 --max-lifetime 1 --validation-every 0 --rl-model Models/rl_setsv2_smoke.npz
```

正式輸出：`Models/rl_setsv2_generalized.npz`（最後完成 episode）、`Models/rl_setsv2_generalized.best.npz`（驗證最佳）、`Models/rl_setsv2_generalized.training.json`（完整設定、地圖指紋、每次 map/seed/lifetime/actions、驗證比較）。這些正式檔案要執行完整訓練才會出現；本次測試使用臨時目錄。

已存在的模型預設拒絕覆寫。續跑用 `--resume` 並提供相同設定；已完成 episode 會跳過，可增大 `--episodes`。目前是保留權重與 optimizer 的 warm resume，**不會恢復 replay 或 learner RNG 狀態，不等於不中斷訓練的逐位元重現**。中斷中的 episode 不覆寫最後完成的 checkpoint；驗證中斷可在 resume 時重試。

## 與正式實驗保持一致

`environment_defaults.py` 集中預設：fitness **v5**、routing **v1**、sensor decode **v3**、sensing **discrete**。trainer 使用 experiment 的 `build_problem`、`run_optimizer` 與 `prepare_result_state`。新模型嵌入實際類別與相關來源檔（包含繼承服務）的 SHA-256；載入時若環境不同會拒絕，而非默默沿用模型。來源檔註解變更也會觸發檢查，這是保守保護，不是判斷兩版數學等價的工具。

訓練完成後，先使用驗證最佳模型做正式六圖實驗：

```powershell
python experiment_algorithms.py --algorithm sa_sets rl_setsv2 --rl-model Models/rl_setsv2_generalized.best.npz
```

實驗預設每張圖 seeds 7–11，共六圖×五次。模型在實驗中 frozen、epsilon=0、不更新網路、不累積 replay。舊 RL 與 V2 的 checkpoint 不相容，因此兩個版本請各自執行一次，並分別用 `--rl-model` 指定模型。

## 泛化與下一步判準

200 訓練 + 40 驗證是工程起點，不是論文保證的充分數量。CoinRun 研究顯示 RL 可能嚴重過擬合，應隔離 train/test；Procgen 研究支持程序生成環境的多樣性，但這些論文不是 WSN，不能把其地圖數量直接當成本專案的理論門檻。[CoinRun 原論文](https://proceedings.mlr.press/v97/cobbe19a.html)、[Procgen 原論文](https://proceedings.mlr.press/v119/cobbe20a.html)。

下一階段應看驗證曲線與操作使用量，比較舊方法、V2、SA 的同預算結果，再做保留 SA、reward、replay 等單項消融。正式結論至少搭配多個獨立訓練 seed（例如 7、17、27，分開輸出 checkpoint），不能將同一模型的五個搜尋 seed 當成五次獨立訓練。使用配對 map/seed 的 lifetime 差異與不確定性，避免只報最好一回；少量 run 下 RL 評估容易有不穩定結論。[Agarwal 等人的 RL 評估研究](https://agarwl.github.io/rliable/)。

若驗證佳、六張測試仍差，應擴大獨立測試集、檢查分布與不確定性，不要把測試圖加入訓練後再宣稱原測試上的泛化成功。
