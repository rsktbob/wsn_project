#!/bin/bash
# 在驗證圖 RLTRAIN71–85（maps_val.json，15 張）上跑 lifetime，不用 MAP01–06。seed 7–11 → 75 場。
# 用法：BUDGET=1000 TAG=sup_b1000_m30_diff_cur bash run_val_eval.sh（TAG 是 results/ 底下的模型資料夾，用 seed0/best.pt）
# 原版 ring_se 同場跑一次當對照（已跑過就跳過）。結果：experiment_results/val711_{ring_se,TAG}_eval{BUDGET}
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
B=${BUDGET:-1000}; TAG=${TAG:?要給 TAG}
COMMON="--runs 5 --seed 7 --evaluate $B --fitness-service v3 --maps $E/maps_val.json --execution parallel --jobs ${JOBS:-8}"
BASE=val711_ring_se_eval$B
if [ ! -f experiment_results/$BASE/lifetime_tables.md ]; then
  $PY experiment_algorithms.py --algorithm ring_se $COMMON --experiment-name $BASE > $R/logs/eval_$BASE.log 2>&1
  echo "$BASE exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
fi
NAME=val711_${TAG}_eval$B
E07_CKPT=$R/$TAG/seed0/best.pt $PY $E/evaluate.py --algorithm e07_net $COMMON --experiment-name $NAME > $R/logs/eval_$NAME.log 2>&1
echo "$NAME exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
