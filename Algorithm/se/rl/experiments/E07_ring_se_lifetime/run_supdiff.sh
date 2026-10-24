#!/bin/bash
# 監督式（差異獎勵 Huber 回歸，同 E06 的目標；網路同 SupNet）：train_sup.py --target diff → lifetime 評估
# 預算 1000、10000，MAP01–06 × seed 7–11，取最大。結果：experiment_results/s711_e07_supdiff_eval{B}
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
SEED=${SEED:-0}
$PY $E/train_sup.py --seed $SEED --target diff > $R/logs/supdiff_seed$SEED.log 2>&1 || { echo "train failed"; exit 1; }
echo "supdiff train done $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
export E07_CKPT=$R/supervised_diff/seed$SEED/final.pt
for B in 1000 10000; do
  $PY $E/evaluate.py --algorithm e07_net --runs 5 --seed 7 --evaluate $B --fitness-service v3 \
    --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8} --experiment-name s711_e07_supdiff_eval$B \
    > $R/logs/eval_supdiff_$B.log 2>&1
  echo "s711_e07_supdiff_eval$B exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
  $PY $E/summary.py
done
