#!/bin/bash
# 監督式兩版先評估（不等 RL 訓練完）；run_eval.sh 看到 eval_started_{name} 就等它、不重跑
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E06_ring_se_span_context
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
COMMON="--runs 30 --seed 7 --evaluate 1000 --fitness-service v3 --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8}"
for pair in sup:supervised sup_l0:supervised_l0; do
  name=${pair%%:*}; dir=${pair##*:}
  touch $R/eval_started_$name
  E06_CKPT=$R/$dir/seed0/best.pt $PY $E/evaluate.py --algorithm e06_net $COMMON --experiment-name s736_e06_${name}_eval1000 > $R/logs/eval_$name.log 2>&1
  echo "$name exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
done
