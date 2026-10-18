#!/bin/bash
# 等標籤收完（COLLECT_DONE）→ 設計 A、B 監督式訓練（平行）→ 各跑 lifetime（MAP01–06 × seed 7–36，預算 1000）
# 結果：experiment_results/s736_e04_sup_{concat,attention}_eval1000；進度寫 results/progress.txt
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E04_ring_se_seg8
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/supervised/results
mkdir -p $R
while [ ! -f $E/data/labels/COLLECT_DONE ]; do sleep 60; done
for net in concat attention; do
  ( $PY $E/supervised/train_sup.py --net $net > $R/train_$net.log 2>&1; echo "train $net exit=$? $(date '+%m/%d %H:%M')" >> $R/progress.txt ) &
done
wait
COMMON="--runs 30 --seed 7 --evaluate 1000 --fitness-service v3 --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8}"
for net in concat attention; do
  E04_CKPT=$R/$net/best.pt $PY $E/evaluate.py --algorithm e04_$net $COMMON \
    --experiment-name s736_e04_sup_${net}_eval1000 > $R/eval_$net.log 2>&1
  echo "eval $net exit=$? $(date '+%m/%d %H:%M')" >> $R/progress.txt
done
touch $R/SUP_DONE
