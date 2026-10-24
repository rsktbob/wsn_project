#!/bin/bash
# E04 lifetime：MAP01–06 × seed 7–36（180 場），預算 1000，fitness v3。
# 等兩個網路訓練完（TRAIN_DONE_seed0），依序跑 8 段隨機、瓶頸規則、設計 A、設計 B（best.pt）。
# 結果：experiment_results/s736_e04_{name}_eval1000；進度寫 results/eval_progress.txt
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E04_ring_se_seg8
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
LOG=$E/results
mkdir -p $LOG
COMMON="--runs 30 --seed 7 --evaluate 1000 --fitness-service v3 --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8}"
while [ ! -f $E/dueling_concat/results/TRAIN_DONE_seed0 ] || [ ! -f $E/attention/results/TRAIN_DONE_seed0 ]; do sleep 60; done
run() {  # name algorithm [ckpt]
  E04_CKPT=$3 $PY $E/evaluate.py --algorithm $2 $COMMON --experiment-name s736_e04_$1_eval1000 > $LOG/eval_$1.log 2>&1
  echo "$1 exit=$? $(date '+%m/%d %H:%M')" >> $LOG/eval_progress.txt
}
run random e04_random
run bottleneck e04_bottleneck
run concat e04_concat $E/dueling_concat/results/seed0/best.pt
run attention e04_attention $E/attention/results/seed0/best.pt
touch $LOG/EVAL_DONE
