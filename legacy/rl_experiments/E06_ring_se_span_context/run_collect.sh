#!/bin/bash
# 標準化統計（隨機 40 段）＋作弊版標籤 8 shard 平行；完成寫 results/data/COLLECT_DONE
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E06_ring_se_span_context
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
LOG=$E/results/logs
mkdir -p $LOG
$PY $E/collect.py --policy random --norm > $LOG/norm.log 2>&1 &
for i in 0 1 2 3 4 5 6 7; do
  $PY $E/collect.py --policy oracle --shard $i --shards 8 > $LOG/collect$i.log 2>&1 &
done
wait
touch $E/results/data/COLLECT_DONE
