#!/bin/bash
# 監督式（10000 標籤訓練的 final.pt）在預算 1000、3000 的 lifetime：取最大、抽樣各一次，MAP01–06 × seed 7–11。
# 等預算 10000 的抽樣評估跑完才開始（避免搶 CPU）。結果：experiment_results/s711_e07_sup{,_sample}_eval{B}
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
export E07_CKPT=$R/supervised/seed0/final.pt
until ! pgrep -f "experiment-name s711_e07_sup_sample_eval10000" > /dev/null; do sleep 30; done
for B in 1000 3000; do
  for V in net net_sample; do
    NAME=s711_e07_sup_eval$B
    [ $V = net_sample ] && NAME=s711_e07_sup_sample_eval$B
    $PY $E/evaluate.py --algorithm e07_$V --runs 5 --seed 7 --evaluate $B --fitness-service v3 \
      --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8} --experiment-name $NAME \
      > $R/logs/eval_${V}_$B.log 2>&1
    echo "$NAME exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
  done
done
$PY $E/summary.py
