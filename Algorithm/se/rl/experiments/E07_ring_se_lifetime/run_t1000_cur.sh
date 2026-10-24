#!/bin/bash
# 標籤尺度改用「這一回合」：預算 1000、60 張圖（＋驗證 RLTRAIN61–66）重收（推進與 seed 和 b1000_m60_train 相同，多存 label_cur），
# 差異獎勵回歸用 label_cur、驗證早停 → supervised_diff_m60v_t1000cur/seed0；評估 1000、3000、10000（MAP01–06 × seed 7–11）
# → experiment_results/s711_e07_m60v_t1000cur_eval{E}
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
D=$R/data
for i in 0 1 2 3 4 5 6 7; do
  $PY $E/collect.py --shard $i --shards 8 --maps 60 --budget 1000 --out $D/b1000_m60_train_scalecur > $R/logs/collect_cur_$i.log 2>&1 &
done
wait
for i in 0 1 2 3 4 5; do
  $PY $E/collect.py --shard $i --shards 6 --start 60 --maps 66 --budget 1000 --out $D/b1000_val_scalecur > $R/logs/collect_cur_val_$i.log 2>&1 &
done
wait
n=$(ls $D/b1000_m60_train_scalecur/RLTRAIN*.npz | grep -vc tmp); m=$(ls $D/b1000_val_scalecur/RLTRAIN*.npz | grep -vc tmp)
echo "cur collect done: $n + $m maps $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
[ "$n" -eq 60 ] && [ "$m" -eq 6 ] || exit 1
OUT=$R/supervised_diff_m60v_t1000cur/seed0
$PY $E/train_sup.py --seed 0 --target diff --label-key label_cur --labels $D/b1000_m60_train_scalecur \
  --val-labels $D/b1000_val_scalecur --epochs 20 --out $OUT > $R/logs/supdiff_m60v_t1000cur_seed0.log 2>&1 || exit 1
echo "m60v t1000cur train done $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
for EB in 1000 3000 10000; do
  NAME=s711_e07_m60v_t1000cur_eval$EB
  E07_CKPT=$OUT/best.pt $PY $E/evaluate.py --algorithm e07_net --runs 5 --seed 7 --evaluate $EB --fitness-service v3 \
    --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8} --experiment-name $NAME > $R/logs/eval_m60v_t1000cur_$EB.log 2>&1
  echo "$NAME exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
  $PY $E/summary.py | tail -9
done
