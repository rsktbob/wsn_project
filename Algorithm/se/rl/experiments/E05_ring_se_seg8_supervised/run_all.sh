#!/bin/bash
# E05：A、B 平行訓練 → 自己推進檢查 → lifetime（A／B × 抽樣／取最大，MAP01–06 × seed 7–36，預算 1000）
# 結果：experiment_results/s736_e05_{net}_{mode}_eval1000；進度寫 results/progress.txt，完成寫 results/ALL_DONE
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E05_ring_se_seg8_supervised
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
mkdir -p $R
for net in concat attention; do
  mkdir -p $E/$net/results
  ( $PY $E/train.py --net $net > $E/$net/results/train_seed0.log 2>&1; echo "train $net exit=$? $(date '+%m/%d %H:%M')" >> $R/progress.txt ) &
done
wait
$PY $E/rollout_check.py > $R/rollout_check.log 2>&1
echo "rollout_check exit=$? $(date '+%m/%d %H:%M')" >> $R/progress.txt
COMMON="--runs 30 --seed 7 --evaluate 1000 --fitness-service v3 --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8}"
for net in attention concat; do
  for mode in sample argmax; do
    E05_CKPT=$E/$net/results/seed0/best.pt $PY $E/evaluate.py --algorithm e05_${net}_${mode} $COMMON \
      --experiment-name s736_e05_${net}_${mode}_eval1000 > $R/eval_${net}_${mode}.log 2>&1
    echo "eval ${net}_${mode} exit=$? $(date '+%m/%d %H:%M')" >> $R/progress.txt
  done
done
touch $R/ALL_DONE
