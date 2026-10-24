# E07 使用方法

所有指令都在 `codes/2027WSN` 底下執行。先設兩個變數（每開一個新終端機設一次）：

```bash
cd /home/g114056067/wsn_project/codes/2027WSN
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
```

## 1. 收資料（某個預算第一次用時才需要）

```bash
BUDGET=1000 bash $E/run_collect.sh
```

- 收 RLTRAIN01–200 → `results/data/b1000/{train,val,test}/`（140／30／30 張），已收過的圖會跳過。
- 時間：預算 1000 約 35 分鐘、3000 約 1.5 小時、10000 約 4 小時。
- 目前已收：**b1000**。

## 2. 訓練

```bash
$PY $E/train_sup.py --budget 1000 --maps 30 --label diff_cur --val --epochs 20
```

| 參數 | 意思 | 常用值 |
| --- | --- | --- |
| `--budget` | 用哪個預算的資料 | 1000 |
| `--maps` | 用 train 的前幾張地圖 | 30、60、140 |
| `--label` | 學哪一種標籤 | `diff_cur`（主要）、`diff_prev`、`oracle_choice` |
| `--loss` | `huber`：回歸 4 個標籤值；`rank`：拿掉 4 個全為 0 的組，只學「最好的分數最高」（資料夾加 `_rank`） | `huber`（預設） |
| `--val` | 用 val 30 張決定最好的 epoch（早停） | 建議都加 |
| `--epochs` | 最多看幾遍資料 | 20 |
| `--seed` | 訓練的亂數種子 | 0（預設）、1、2… |

- 模型存到 `results/sup_b{預算}_m{張數}_{標籤}/seed{seed}/best.pt`，例如 `results/sup_b1000_m30_diff_cur/seed0/best.pt`。
- 訓練完會自動在 test 30 張上算「猜得準不準」，寫在同一個資料夾的 `log.jsonl`。
- 時間：30 張約 3 分鐘、60 張約 6 分鐘。
- 兩個可以同時訓練，在指令最後加 `&`，再用 `wait` 等兩個都結束。

## 3. 跑 lifetime

在 test 地圖（RLTRAIN171–200 × seed 7–11，150 場）上跑：

```bash
SPLIT=test BUDGET=1000 TAGS="sup_b1000_m30_diff_cur sup_b1000_m60_diff_cur" bash $E/run_eval_maps.sh
```

- `TAGS`：要跑的模型資料夾名稱（第 2 步的資料夾名），可以一次放好幾個，用空格分開。
- `SPLIT`：`test`（`maps_test.json`）、`val`（`maps_val.json`）、`train`（`maps_train.json`）。
- `BUDGET`：跑 lifetime 時每段的預算（1000、3000、10000）。
- 原版 ring_se 會自動跑一次當對照，跑過就不再重跑。
- 結果：`experiment_results/{SPLIT}711_{模型}_eval{BUDGET}/`。
- 時間：預算 1000 每個模型約 7 分鐘（原版第一次另外約 4 分鐘）。

在 MAP01–06 上跑（和以前的實驗比）：

```bash
E07_CKPT=$E/results/sup_b1000_m30_diff_cur/seed0/best.pt $PY $E/evaluate.py --algorithm e07_net --runs 5 --seed 7 --evaluate 1000 --fitness-service v3 --maps maps/maps_100100100.json --execution parallel --jobs 8 --experiment-name s711_e07_sup_b1000_m30_diff_cur_eval1000
```

## 4. 看結果

```bash
$PY $E/compare.py --split test --budget 1000
```

- 列出這個 split、預算下所有跑過的模型：lifetime、對原版的差距和 p 值、test 上的猜準度、best epoch。
- 只看某幾個：在最後加模型名稱，例如 `$PY $E/compare.py --split test --budget 1000 sup_b1000_m30_diff_cur`。

## 一次做完：訓練 → 跑 test → 看結果

```bash
$PY $E/train_sup.py --budget 1000 --maps 30 --label diff_prev --val --epochs 20 && SPLIT=test BUDGET=1000 TAGS="sup_b1000_m30_diff_prev" bash $E/run_eval_maps.sh && $PY $E/compare.py --split test --budget 1000
```

跑很久的話可以在最前面加 `nohup`、最後加 `> my_run.log 2>&1 &`，讓它在背景跑，關掉終端機也不會停。
