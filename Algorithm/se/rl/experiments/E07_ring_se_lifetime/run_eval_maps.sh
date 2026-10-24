#!/bin/bash
# 在指定的地圖清單上跑 lifetime（seed 7–11）：SPLIT=test → maps_test.json（RLTRAIN171–200）、val → maps_val.json、train → maps_train.json。
# 用法：SPLIT=test BUDGET=1000 TAGS="sup_b1000_m30_diff_cur sup_b1000_m60_diff_cur" bash run_eval_maps.sh
# TAGS 是 results/ 底下的模型資料夾（用 seed0/best.pt）。原版 ring_se 同場跑一次當對照（已跑過就跳過）。
# 結果：experiment_results/{SPLIT}711_{ring_se 或 TAG}_eval{BUDGET}
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
S=${SPLIT:-test}; B=${BUDGET:-1000}
COMMON="--runs 5 --seed 7 --evaluate $B --fitness-service v3 --maps $E/maps_$S.json --execution parallel --jobs ${JOBS:-8}"
BASE=${S}711_ring_se_eval$B
if [ ! -f experiment_results/$BASE/lifetime_tables.md ]; then
  $PY experiment_algorithms.py --algorithm ring_se $COMMON --experiment-name $BASE > $R/logs/eval_$BASE.log 2>&1
  echo "$BASE exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
fi
for TAG in $TAGS; do
  NAME=${S}711_${TAG}_eval$B
  E07_CKPT=$R/$TAG/seed0/best.pt $PY $E/evaluate.py --algorithm e07_net $COMMON --experiment-name $NAME > $R/logs/eval_$NAME.log 2>&1
  echo "$NAME exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
done
