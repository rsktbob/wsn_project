#!/bin/bash
# D3QN（從零、γ=0）：RLTRAIN01–30 完整 lifetime、每段預算 1000（config.json 的 rl）→ results/d3qn_b1000/seed0
# 訓練完評估預算 1000，MAP01–06 × seed 7–11，選 Q 最大 → experiment_results/s711_e07_d3qn_b1000_eval1000
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
SEED=${SEED:-0}
OUT=$R/d3qn_b1000/seed$SEED
$PY $E/train_rl.py --seed $SEED > $R/logs/rl_b1000_seed$SEED.log 2>&1 || { echo "train_rl failed"; exit 1; }
echo "d3qn_b1000 train done $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
E07_CKPT=$OUT/final.pt $PY $E/evaluate.py --algorithm e07_net --runs 5 --seed 7 --evaluate 1000 --fitness-service v3 \
  --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8} --experiment-name s711_e07_d3qn_b1000_eval1000 \
  > $R/logs/eval_d3qn_b1000_1000.log 2>&1
echo "s711_e07_d3qn_b1000_eval1000 exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
$PY $E/summary.py
