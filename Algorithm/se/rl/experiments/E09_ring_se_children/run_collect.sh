#!/bin/bash
# E09 收資料：RLTRAIN01–200 → results/data/b{BUDGET}/{train,val,test}，8 shard 平行；已收的圖會跳過。
# 用法：BUDGET=1000 bash run_collect.sh；完成寫 results/data/b{BUDGET}/COLLECT_DONE
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E09_ring_se_children
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
B=${BUDGET:?要給 BUDGET}
LOG=$E/results/logs
mkdir -p $LOG
for i in 0 1 2 3 4 5 6 7; do
  $PY $E/collect_codes.py --budget $B --shard $i --shards 8 > $LOG/collect_b${B}_$i.log 2>&1 &
done
wait
D=$E/results/data/b$B
n=$(ls $D/train/RLTRAIN*.npz $D/val/RLTRAIN*.npz $D/test/RLTRAIN*.npz 2>/dev/null | grep -vc tmp)
echo "b$B collect done: $n maps $(date '+%m/%d %H:%M')" | tee -a $E/results/eval_progress.txt
[ "$n" -eq 200 ] && touch $D/COLLECT_DONE
