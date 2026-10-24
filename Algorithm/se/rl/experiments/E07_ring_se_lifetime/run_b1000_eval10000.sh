#!/bin/bash
# 預算 1000 標籤訓練的差異獎勵模型（results/supervised_diff_b1000/seed0/final.pt，30 張圖）拿去跑預算 10000，
# MAP01–06 × seed 7–11，取最大 → experiment_results/s711_e07_supdiff_b1000_eval10000。等 run_b1000_m60.sh 跑完才開始。
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
until ! pgrep -f "E07_ring_se_lifetime/run_b1000_m60.sh" > /dev/null; do sleep 30; done
E07_CKPT=$R/supervised_diff_b1000/seed0/final.pt $PY $E/evaluate.py --algorithm e07_net --runs 5 --seed 7 --evaluate 10000 \
  --fitness-service v3 --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8} \
  --experiment-name s711_e07_supdiff_b1000_eval10000 > $R/logs/eval_supdiff_b1000_10000.log 2>&1
echo "s711_e07_supdiff_b1000_eval10000 exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
$PY $E/summary.py
