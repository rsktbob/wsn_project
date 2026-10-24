#!/bin/bash
# 監督式：訓練（results/data/b{BUDGET}，train 前 MAPS 張、val 早停、test 離線指標）→ lifetime 評估（MAP01–06 × seed 7–11）。
# 用法：BUDGET=1000 MAPS=30 LABEL=diff_cur EVAL="1000" bash run_sup.sh
# 模型：results/sup_b{BUDGET}_m{MAPS}_{LABEL}/seed{SEED}/best.pt；結果：experiment_results/s711_e07_sup_b{BUDGET}_m{MAPS}_{LABEL}_eval{E}
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
B=${BUDGET:-1000}; M=${MAPS:-70}; L=${LABEL:-diff_cur}; SEED=${SEED:-0}; EVALS=${EVAL:-1000}
TAG=sup_b${B}_m${M}_${L}
OUT=$R/$TAG/seed$SEED
mkdir -p $R/logs
$PY $E/train_sup.py --budget $B --maps $M --label $L --val --epochs 20 --seed $SEED > $R/logs/${TAG}_seed$SEED.log 2>&1 \
  || { echo "train failed"; exit 1; }
echo "$TAG seed$SEED train done $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
for EB in $EVALS; do
  NAME=s711_e07_${TAG}_eval$EB
  [ "$SEED" != 0 ] && NAME=${NAME}_seed$SEED
  E07_CKPT=$OUT/best.pt $PY $E/evaluate.py --algorithm e07_net --runs 5 --seed 7 --evaluate $EB --fitness-service v3 \
    --maps maps/maps_100100100.json --execution parallel --jobs ${JOBS:-8} --experiment-name $NAME > $R/logs/eval_${TAG}_$EB.log 2>&1
  echo "$NAME exit=$? $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
done
