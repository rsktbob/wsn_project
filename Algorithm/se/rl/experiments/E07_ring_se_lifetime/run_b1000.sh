#!/bin/bash
# 預算 1000 版：設定同 E07（RLTRAIN01–30、完整 lifetime、原版推進、作弊版每 3 段標一段），只把每段預算改成 1000。
# 標籤 → results/data/b1000_m30_train；差異獎勵 Huber 回歸 20 epoch → results/supervised_diff_b1000/seed0；
# 評估預算 1000，MAP01–06 × seed 7–11，取最大 → experiment_results/s711_e07_supdiff_b1000_eval1000
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
LAB=$R/data/b1000_m30_train
OUT=$R/supervised_diff_b1000/seed0
for i in 0 1 2 3 4 5 6 7; do
  $PY $E/collect.py --shard $i --shards 8 --budget 1000 --out $LAB > $R/logs/collect_b1000_$i.log 2>&1 &
done
wait
n=$(ls $LAB/RLTRAIN*.npz 2>/dev/null | grep -vc tmp)
echo "b1000 collect done: $n maps $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
[ "$n" -eq 30 ] || exit 1
$PY $E/train_sup.py --seed 0 --target diff --labels $LAB --out $OUT --epochs 20 > $R/logs/supdiff_b1000_seed0.log 2>&1 || exit 1
echo "supdiff_b1000 train done $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
E07_CKPT=$OUT/final.pt $PY $E/evaluate.py --algorithm e07_net --runs 5 --seed 7 --evaluate 1000 --fitness-service v3 \
  --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8} --experiment-name s711_e07_supdiff_b1000_eval1000 \
  > $R/logs/eval_supdiff_b1000_1000.log 2>&1
echo "s711_e07_supdiff_b1000_eval1000 exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
$PY $E/summary.py
