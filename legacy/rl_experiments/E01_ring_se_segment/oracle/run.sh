#!/bin/sh
# E01 步驟 1：ring_se 作弊版天花板，MAP01–06 × seed 7–11，預算 1000／3000，fitness v3
# 結果：experiment_results/e01_oracle_6maps_eval{B}（experiment_results 本身已是指向 runs/ 的 symlink）
ROOT=/home/g114056067/wsn_project/codes/2027WSN
cd $ROOT || exit 1
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
D=Algorithm/se/rl/experiments/E01_ring_se_segment/oracle
LOG=$ROOT/tmp/e01_oracle
mkdir -p $LOG
for B in 1000 3000; do
  $PY $D/run.py --algorithm ring_se_oracle --runs 5 --seed 7 \
    --evaluate $B --maps maps/maps_100100100.json --execution parallel --jobs 6 \
    --experiment-name e01_oracle_6maps_eval$B > $LOG/eval$B.log 2>&1
  echo "$B exit=$?" >> $LOG/progress.txt
done
echo done > $LOG/DONE
