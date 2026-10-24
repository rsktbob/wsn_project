#!/bin/bash
# seed 0：監督式（2 層、0 層消融）與 RL γ=0 從零同時開始；監督式完成後接 RL γ=0 暖啟動、γ=0.5 暖啟動。
# 完成寫 results/TRAIN_DONE_seed{SEED}
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E06_ring_se_span_context
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
SEED=${SEED:-0}
LOG=$E/results/logs
mkdir -p $LOG
$PY $E/train_sup.py --seed $SEED > $LOG/sup_seed$SEED.log 2>&1 &
SUP=$!
$PY $E/train_sup.py --seed $SEED --sensor-layers 0 > $LOG/sup_l0_seed$SEED.log 2>&1 &
$PY $E/train_rl.py --gamma 0 --seed $SEED > $LOG/rl_g0_scratch_seed$SEED.log 2>&1 &
wait $SUP
INIT=$E/results/supervised/seed$SEED/best.pt
$PY $E/train_rl.py --gamma 0 --seed $SEED --init $INIT > $LOG/rl_g0_warm_seed$SEED.log 2>&1 &
$PY $E/train_rl.py --gamma 0.5 --seed $SEED --init $INIT > $LOG/rl_g05_warm_seed$SEED.log 2>&1 &
wait
touch $E/results/TRAIN_DONE_seed$SEED
