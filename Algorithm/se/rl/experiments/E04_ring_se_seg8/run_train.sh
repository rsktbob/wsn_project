#!/bin/bash
# 兩個網路平行訓練（預算 1000）；完成寫 {dueling_concat|attention}/results/TRAIN_DONE_seed{SEED}
cd /home/g114056067/wsn_project/codes/2027WSN
E=Algorithm/se/rl/experiments/E04_ring_se_seg8
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
SEED=${SEED:-0}
for pair in concat:dueling_concat attention:attention; do
  net=${pair%%:*}; dir=${pair##*:}
  mkdir -p $E/$dir/results
  ( $PY $E/train.py --net $net --seed $SEED > $E/$dir/results/train_seed$SEED.log 2>&1; touch $E/$dir/results/TRAIN_DONE_seed$SEED ) &
done
wait
