#!/bin/bash
# lifetime：MAP01–06 × seed 7–36（180 場），預算 1000，fitness v3。等 TRAIN_DONE_seed0 後依序跑各版本 best.pt。
# 結果：experiment_results/s736_e06_{name}_eval1000；進度寫 results/eval_progress.txt
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E06_ring_se_span_context
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
COMMON="--runs 30 --seed 7 --evaluate 1000 --fitness-service v3 --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8}"
while [ ! -f $R/TRAIN_DONE_seed0 ]; do sleep 60; done
run() {  # name ckpt；run_eval_sup.sh 已經開始跑的版本就等它跑完、不重跑
  if [ -f $R/eval_started_$1 ]; then
    until grep -q "^$1 exit" $R/eval_progress.txt 2>/dev/null; do sleep 60; done
    return
  fi
  E06_CKPT=$2 $PY $E/evaluate.py --algorithm e06_net $COMMON --experiment-name s736_e06_$1_eval1000 > $R/logs/eval_$1.log 2>&1
  echo "$1 exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
}
run sup $R/supervised/seed0/best.pt
run g0_warm $R/d3qn/g0_warm_seed0/best.pt
run g0_scratch $R/d3qn/g0_scratch_seed0/best.pt
run g05_warm $R/d3qn/g05_warm_seed0/best.pt
run sup_l0 $R/supervised_l0/seed0/best.pt
touch $R/EVAL_DONE
