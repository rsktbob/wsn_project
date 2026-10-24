#!/bin/bash
# 對照：E06 監督式（E06 results/supervised/seed0/best.pt，預算 1000 的標籤訓練）在預算 3000、10000 的 lifetime，
# MAP01–06 × seed 7–11，取 Q 最大。等 run_supdiff.sh 跑完才開始（避免搶 CPU）。結果：experiment_results/s711_e06_sup_eval{B}
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
E6=legacy/rl_experiments/E06_ring_se_span_context
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
until ! pgrep -f "E07_ring_se_lifetime/run_supdiff.sh" > /dev/null; do sleep 30; done
export E06_CKPT=$E6/results/supervised/seed0/best.pt
for B in 3000 10000; do
  $PY $E6/evaluate.py --algorithm e06_net --runs 5 --seed 7 --evaluate $B --fitness-service v3 \
    --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8} --experiment-name s711_e06_sup_eval$B \
    > $R/logs/eval_e06sup_$B.log 2>&1
  echo "s711_e06_sup_eval$B exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
  $PY $E/summary.py
done
