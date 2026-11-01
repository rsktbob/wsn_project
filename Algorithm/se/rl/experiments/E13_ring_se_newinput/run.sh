#!/bin/bash
# E13：訓練 Net13（E12 結構、新輸入）（b1000、plain_delta、soft τ=1、140 張、val 早停），再跑 MAP01–06 × seed 7–11 的 lifetime（1000／3000／10000）。
# 先要有 E13 輸入（run_collect.sh）、E07 標籤、E08 ②③⑤（都是 b1000）。
# 結果：experiment_results/s711_e13_b1000_m140_plain_delta_soft1_eval{B}
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E13_ring_se_newinput
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
TAG=e13_b1000_m140_plain_delta_soft1
mkdir -p $R/logs
if [ ! -f $R/$TAG/seed0/TRAIN_DONE ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $E/train13.py --budget 1000 --maps 140 --label plain_delta --loss soft --tau 1 --val --epochs 20 > $R/logs/train_$TAG.log 2>&1
  echo "train $TAG exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
fi
for B in 1000 3000 10000; do
  NAME=s711_${TAG}_eval$B
  E13_CKPT=$R/$TAG/seed0/best.pt $PY $E/evaluate13.py --algorithm e13 --runs 5 --seed 7 --evaluate $B \
    --fitness-service v3 --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-5} \
    --experiment-name $NAME > $R/logs/eval_${TAG}_$B.log 2>&1
  echo "$NAME exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
done
