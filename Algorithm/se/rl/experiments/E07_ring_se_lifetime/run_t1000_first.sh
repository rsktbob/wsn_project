#!/bin/bash
# 3×3 的第一列先做：60 張圖、預算 1000 標籤、驗證早停（驗證圖 RLTRAIN61–66）→ supervised_diff_m60v_t1000/seed0/best.pt，
# 跑預算 1000、3000、10000（MAP01–06 × seed 7–11）→ experiment_results/s711_e07_m60v_t1000_eval{E}。作弊版用現成 s736_oracle_eval{E}。
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
D=$R/data
for i in 0 1 2 3 4 5; do
  $PY $E/collect.py --shard $i --shards 6 --start 60 --maps 66 --budget 1000 --out $D/b1000_val > $R/logs/collect_val_1000_$i.log 2>&1 &
done
wait
n=$(ls $D/b1000_val/RLTRAIN*.npz 2>/dev/null | grep -vc tmp)
echo "val 1000 collect done: $n maps $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
[ "$n" -eq 6 ] || exit 1
OUT=$R/supervised_diff_m60v_t1000/seed0
$PY $E/train_sup.py --seed 0 --target diff --labels $D/b1000_m60_train --val-labels $D/b1000_val --epochs 20 \
  --out $OUT > $R/logs/supdiff_m60v_t1000_seed0.log 2>&1 || exit 1
echo "m60v t1000 train done $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
for EB in 1000 3000 10000; do
  NAME=s711_e07_m60v_t1000_eval$EB
  E07_CKPT=$OUT/best.pt $PY $E/evaluate.py --algorithm e07_net --runs 5 --seed 7 --evaluate $EB --fitness-service v3 \
    --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8} --experiment-name $NAME > $R/logs/eval_m60v_t1000_$EB.log 2>&1
  echo "$NAME exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
  $PY $E/summary.py | tail -9
done
