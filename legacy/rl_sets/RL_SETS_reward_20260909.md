# RL-SETS reward 修正與短期比較（2026-09-09）

## 問題與比較範圍

舊 V1 公式在所有投資結果與父代相同時仍回傳 -0.15，因為成功率項是
`0.15 * (2 * success_rate - 1)`。fitness 接近 1 時，微小改善除以約 1.1 後，
很難抵銷此固定項。負 reward 本身不代表 DQN 不能學習，問題是平台期的回饋
幾乎都擠在同一個負值附近。

本次比較固定 sensing=discrete、fitness=v5、routing=v1、sensor encoding=v3，
沒有 moving。n=8、h=4、w=2、mu=0.4；每次重新最佳化都實際評估 10000 次。
診斷與訓練使用 RL-SETS V1 的相同六個動作，僅替換 reward；V2 另外做相容性測試。

所有原始輸出位於 `experiment_results/diagnostics/rl_reward_20260909/`。
比較腳本是 `tests/benchmark_rl_sets_reward.py`。這些小型模型是診斷產物，不是正式訓練完成的模型。

## 固定策略診斷

地圖 RLTRAIN01、02、03、117，各使用 seeds 7、19；epsilon=1，不更新網路。
在同一批投資結果上計算各公式，沒有為替代公式額外做 fitness 評估。
每輪有四個區域決策；前 31 輪約為前 2000 次評估，後 125 輪為剩餘預算。

| 公式 | 後期 reward 平均 | 後期正值比例 | 後期零值比例 | 後期負值比例 |
| --- | ---: | ---: | ---: | ---: |
| legacy：舊公式 | -0.149630 | 0.025% | 0% | 99.975% |
| selected_goods：保留商品改善量，平均與最佳混合 | 0.019713 | 24.525% | 55.300% | 20.175% |
| parent_credit：保留解超越較佳父代才給正分 | 0.005745 | 2.300% | 97.700% | 0% |

selected_goods 在 4992 個決策中，有 1153 次給正分卻沒有保留解超越其較佳父代。
這些分數代表商品確實改善，但可能只是承接較佳 searcher；不能解讀成 1153 次全新突破。
這種更新可能仍有用，因此另做策略訓練與壽命比較，沒有僅依 reward 正負比例選方案。

## 短期訓練

每種公式各訓練 RLTRAIN01（seed 7）與 RLTRAIN02（seed 8），每張圖跑四次
重新最佳化並實際扣能量，共 80000 evaluations、4992 transitions、1185 次網路更新。
兩張圖的 epsilon 分別固定 0.6、0.2；其餘 D3QN 參數沿用預設。
這是兩段受限 lifetime 訓練，沒有涵蓋完整的晚期低電量分布。

凍結策略後，在 RLTRAINVAL01、02 × seeds 7007、7008 測單次求解。

| 公式 | 平均 fitness | 這個解可連續使用的平均時槽 |
| --- | ---: | ---: |
| legacy | 0.99820818 | 71.75 |
| selected_goods | 0.99856894 | 114.75 |
| parent_credit | 0.99851019 | 95.25 |
| selected_mean：只用保留商品的平均改善 | 0.99835889 | 101.25 |

「這個解可連續使用的時槽」不是整個網路 lifetime，也不是重新最佳化後的累積壽命。
`train.json` 的早期三組驗證僅將 `agent.training=False`；包裝演算法的 training 欄位仍顯示 true，
但 epsilon=0、training_steps 一直維持 1185，沒有新增 replay 或更新網路。
腳本現已同步設定包裝旗標；`train_mean.json` 及完整 lifetime 使用凍結策略。

## 模型與訓練紀錄

完整 lifetime 比較使用 RLTRAINVAL01、seed 7007，策略全程凍結，每次重新最佳化
仍為 10000 次評估，未設定 lifetime 上限。結果如下：

| 公式／演算法 | 完整 lifetime | optimizer 呼叫數 | 實際總評估數 |
| --- | ---: | ---: | ---: |
| 舊 reward | 964 | 25 | 250000 |
| selected_goods | 900 | 21 | 210000 |
| selected_mean | 1000 | 25 | 250000 |
| parent_credit（採用） | 1031 | 30 | 300000 |
| SA-SETS | 1046 | 41 | 410000 |

各次求解預算相同，完整 lifetime 的總評估數並不相同，因為解可使用的長度、
重新求解次數及停止時間不同。這不是等總計算量的比較。執行時間也有測試程序重疊，
不使用這次的牆鐘時間判斷哪一種比較快。

採用 parent_credit 的依據是平台期為零、不把直接複製父代當新突破，以及這次
完整 lifetime 對照較有潛力。1031 比 964 多約 7%，但仍少於 SA 的 1046。
只有一張驗證圖與一個訓練 seed，不能視為穩定的壽命提升證據。

## 正式採用的公式

對每個 good j，依市場原本的規則，找出投資 fitness 最大的 searcher i*。

```text
winner[j] = argmax_i investment_fitness[i, j]
source[j] = max(goods_fitness[j], searcher_fitness[winner[j]])
gain[j] = max(0, investment_fitness[winner[j], j] - source[j])
scale = max(1e-5, 父代 fitness 對其中位數絕對偏差的第 75 百分位數)
reward = mean(tanh(gain / scale))
```

改善量不超過 1e-12 時視為零。reward 範圍為 [0,1]，完整停滯是 0，
微小的新改善仍能經過 scale 放大。未成功的探索與商品退化都不給正分，
也不再施加固定負分；`degraded_goods` 另外記錄真正退化的商品數。
因此零 reward 不保證商品沒有退化，而是表示這次動作沒有可歸因的新改善。

例如 good=0.90、searcher=0.95：新解為 0.95 時 reward=0；高於 0.95 才能得到正分。
同時產生很多未被保留的差解，不會稀釋這個成功。此 reward 仍是 fitness 改善的代理訊號，
不是 lifetime 本身，也尚未解決所有長期能量分配的信用分配問題。

正式實作位於 `Algorithm/se/RL_SETS.py` 的 `_calculate_investment_reward`，
版本為 `selected_parent_improvement/1`。回歸測試用隨機輸入比對本次 benchmark
的 parent_credit 公式，確保交付程式與實驗方案相同。

這個設計讓 DQN 學習哪些動作較有機會帶來改善；零分的動作仍可能破壞商品，
所以保留退化計數作為診斷，不以平均 reward 大於零判定演算法成功。

## 訓練輸出與相容性

V1 與 V2 共用 reward 計算。V2 預留 SA 的候選若贏得商品，不把這次成功歸給 RL 選的動作；
使用市場相同的 argmax 與同分順序，避免把未被保留的候選當成實際更新。

訓練輸出新增 `reward_pos`、`reward_zero`、`reward_neg`。它們以「區域動作決策」為分母；
原本 action success_rate 仍以投資候選數為分母，且仍判斷是否超越較佳父代。
兩種統計回答不同問題，不應直接比較百分比。

training JSON 記錄各動作及前 20%／後 80% 評估預算的 reward 計數。
checkpoint 與訓練 protocol 加入 reward_version，舊 reward 的模型可以凍結推論，
但直接接續新 reward 訓練會被拒絕。應指定新的模型檔名，重新訓練。

延續先前的 60 episodes、暫不驗證設定，明確指定 V1：

```powershell
python train_rl_sets.py --algorithm rl_sets --episodes 60 --validation-every 0 --rl-model Models/rl_sets_reward2_60maps.npz
```

每次求解預設為 10000 評估。若使用 V2，將 algorithm 改為 rl_setsv2，並換另一個模型檔名。
不加 --algorithm 時 trainer 原本的預設仍為 rl_setsv2。正式訓練應另外使用獨立地圖驗證。

## 交付檢查

- 平台期、微小改善、複製父代、被淘汰的差解、V2 保留 SA 歸因均通過語意測試。
- 正式公式與 benchmark 的 parent_credit 在 100 組隨機輸入上完全一致。
- 固定動作 0 時，SA 與 RL 的 coding、fitness、評估次數相同。
- V1／V2 的存檔、載入、凍結驗證、續訓、舊 reward 拒絕續訓檢查均通過。
- 正式 trainer 另跑 RLTRAIN117、兩次 10000 評估，共 20000 evaluations、249 次更新。
  每次求解的平均 reward 為 0.015756、0.017361；後 80% 預算的 1000 個區域決策
  有 38 次正 reward、962 次零 reward，沒有固定的 -0.15 懲罰。
  此流程檢查只有兩個 segments，不是完整 lifetime 成績。

## 重現命令

請在 `codes/2023以恩MWSN` 執行，並使用未存在的輸出資料夾：

```powershell
python tests/benchmark_rl_sets_reward.py --phase diagnostic --output experiment_results/diagnostics/rl_reward_repeat
python tests/benchmark_rl_sets_reward.py --phase train --output experiment_results/diagnostics/rl_reward_repeat
python tests/benchmark_rl_sets_reward.py --phase train --variants selected_mean --report-name train_mean --output experiment_results/diagnostics/rl_reward_repeat
python tests/benchmark_rl_sets_reward.py --phase lifetime --variants legacy selected_goods sa_sets --output experiment_results/diagnostics/rl_reward_repeat
python tests/benchmark_rl_sets_reward.py --phase lifetime --variants parent_credit selected_mean --report-name lifetime_alternatives --output experiment_results/diagnostics/rl_reward_repeat
```

這些比較只有一個獨立訓練 seed，不能宣稱統計上勝過 SA-SETS。正式結論需要多個獨立訓練 seed，
以及未參與選 reward 的地圖；不能因為本次某一張驗證圖較好就稱為泛化成功。
