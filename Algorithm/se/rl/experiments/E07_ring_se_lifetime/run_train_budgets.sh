#!/bin/bash
# 依序訓練 b1000、b3000、b10000（diff_cur、140 張、val 早停、最多 20 epoch）；每個預算等 COLLECT_DONE 才開始。
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
LOG=$E/results/logs
for B in 1000 3000 10000; do
  until [ -f $E/results/data/b$B/COLLECT_DONE ]; do sleep 60; done
  echo "train b$B start $(date '+%m/%d %H:%M')"
  $PY $E/train_sup.py --budget $B --maps 140 --label diff_cur --val --epochs 20 > $LOG/train_sup_b${B}_m140_diff_cur.log 2>&1
  echo "train b$B exit=$? $(date '+%m/%d %H:%M')" | tee -a $E/results/eval_progress.txt
done
