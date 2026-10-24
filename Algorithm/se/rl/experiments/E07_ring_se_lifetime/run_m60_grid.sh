#!/bin/bash
# 3×3 實驗：60 張圖（RLTRAIN01–60）差異獎勵 Huber 回歸，標籤預算 T ∈ {1000, 3000, 10000}，三個模型只差在標籤預算；
# 驗證圖 RLTRAIN61–66（同樣設定、同樣預算各收一份），每個 epoch 看驗證損失，最多 20 epoch、連續 4 個沒進步就停，用 best.pt。
# 每個模型跑預算 E ∈ {1000, 3000, 10000} 的 lifetime，MAP01–06 × seed 7–11 → experiment_results/s711_e07_m60v_t{T}_eval{E}
# 等 run_m60_collect.sh 收完（results/data/M60_COLLECT_DONE）才開始。已經做完的（run_t1000_first.sh 的 t1000）會跳過。
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
D=$R/data
declare -A TRAIN=([1000]=$D/b1000_m60_train [3000]=$D/b3000_m60_train [10000]=$D/b10000_m60_train)
declare -A VAL=([1000]=$D/b1000_val [3000]=$D/b3000_val [10000]=$D/b10000_val)
while [ ! -f $D/M60_COLLECT_DONE ]; do sleep 60; done

for T in 1000 3000 10000; do       # 驗證圖：RLTRAIN61–66，6 個 shard 一次收完
  for i in 0 1 2 3 4 5; do
    $PY $E/collect.py --shard $i --shards 6 --start 60 --maps 66 --budget $T --out ${VAL[$T]} > $R/logs/collect_val_${T}_${i}.log 2>&1 &
  done
  wait
  n=$(ls ${VAL[$T]}/RLTRAIN*.npz 2>/dev/null | grep -vc tmp)
  echo "val $T collect done: $n maps $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
  [ "$n" -eq 6 ] || exit 1
done

for T in 1000 3000 10000; do
  [ -f $R/supervised_diff_m60v_t$T/seed0/TRAIN_DONE ] && continue
  $PY $E/train_sup.py --seed 0 --target diff --labels ${TRAIN[$T]} --val-labels ${VAL[$T]} --epochs 20 \
    --out $R/supervised_diff_m60v_t$T/seed0 > $R/logs/supdiff_m60v_t${T}_seed0.log 2>&1 || exit 1
  echo "m60v t$T train done $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
done

for EB in 1000 3000 10000; do
  for T in 1000 3000 10000; do
    NAME=s711_e07_m60v_t${T}_eval$EB
    [ -f experiment_results/$NAME/lifetime_tables.md ] && continue
    E07_CKPT=$R/supervised_diff_m60v_t$T/seed0/best.pt $PY $E/evaluate.py --algorithm e07_net --runs 5 --seed 7 \
      --evaluate $EB --fitness-service v3 --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8} \
      --experiment-name $NAME > $R/logs/eval_m60v_t${T}_$EB.log 2>&1
    echo "$NAME exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
  done
  $PY $E/summary.py
done
touch $R/M60V_DONE
