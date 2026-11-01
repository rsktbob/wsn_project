#!/bin/bash
# E12：訓練 RelFlagNet（E08＋範圍旗標、②③ 併上範圍內）（b1000、plain_delta、soft τ=1、140 張、val 早停），再跑 MAP01–06 × seed 7–11 的 lifetime（1000／3000／10000）。
# TRAIN_BUDGET=3000 bash run.sh → 用 b3000 標籤訓練（先要有 E08 的 b3000 特徵）；EPOCHS=30 bash run.sh → 30 epoch 版（資料夾加 _e30）。
# 結果：experiment_results/s711_e12_relflag_b{TB}_m140_plain_delta_soft1_eval{B}
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E12_ring_se_relflag
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
EP=${EPOCHS:-20}
TB=${TRAIN_BUDGET:-1000}
TAG=relflag_b${TB}_m140_plain_delta_soft1$([ "$EP" != 20 ] && echo _e$EP)
if [ ! -f $R/$TAG/seed0/TRAIN_DONE ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $E/train_relflag.py --budget $TB --maps 140 --label plain_delta --loss soft --tau 1 --val --epochs $EP > $R/logs/train_$TAG.log 2>&1
  echo "train $TAG exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
fi
for B in 1000 3000 10000; do
  NAME=s711_e12_relflag_${TAG#relflag_}_eval$B
  E12_CKPT=$R/$TAG/seed0/best.pt $PY $E/evaluate_relflag.py --algorithm e12_relflag --runs 5 --seed 7 --evaluate $B \
    --fitness-service v3 --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-5} \
    --experiment-name $NAME > $R/logs/eval_${TAG}_$B.log 2>&1
  echo "$NAME exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
done
