#!/bin/bash
# 用 b3000、b10000 標籤各訓練一個 E12（soft τ=1），各跑 1000／3000／10000 lifetime。
# 先補 E08 的 ②③⑤ 特徵（E08 build_features.py，寫到 E08 results/data/b{B}），再訓練＋評估（run.sh）。
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
E=Algorithm/se/rl/experiments
R=$E/E12_ring_se_relflag/results
for TB in 3000 10000; do
  $PY $E/E08_ring_se_relations/build_features.py --budget $TB --jobs 12 > $R/logs/features_b$TB.log 2>&1
  echo "features b$TB exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
  E12_CHUNK=$([ $TB = 10000 ] && echo 512 || echo 2048) TRAIN_BUDGET=$TB bash $E/E12_ring_se_relflag/run.sh
done
