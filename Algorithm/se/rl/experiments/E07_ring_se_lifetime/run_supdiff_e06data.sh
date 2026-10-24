#!/bin/bash
# 資料對照：E07 的簡單網路（SupNet）用 E06 的訓練資料（作弊版推進、預算 1000、E01 快照，409,600 組）
# 回歸 E06 的差異獎勵標籤（Huber）。E06 資料量約 E07 的 1/4 → 20 epoch（總更新步數 ≈ E07 的 5 epoch）。
# 評估：預算 1000，MAP01–06 × seed 7–11，取最大 → experiment_results/s711_e07_supdiff_e06data_eval1000
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
E6=legacy/rl_experiments/E06_ring_se_span_context
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
OUT=$R/supervised_diff_e06data/seed0
$PY $E/train_sup.py --seed 0 --target diff --labels $E6/results/data/labels --out $OUT --epochs 20 \
  > $R/logs/supdiff_e06data_seed0.log 2>&1 || { echo "train failed"; exit 1; }
echo "supdiff_e06data train done $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
E07_CKPT=$OUT/final.pt $PY $E/evaluate.py --algorithm e07_net --runs 5 --seed 7 --evaluate 1000 --fitness-service v3 \
  --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8} --experiment-name s711_e07_supdiff_e06data_eval1000 \
  > $R/logs/eval_supdiff_e06data_1000.log 2>&1
echo "s711_e07_supdiff_e06data_eval1000 exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
$PY $E/summary.py
