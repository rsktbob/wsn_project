#!/bin/bash
# 8 個 shard 平行收標籤（共 800 個 episode）；完成寫 ../data/labels/COLLECT_DONE
cd /home/g114056067/wsn_project/codes/2027WSN
E=Algorithm/se/rl/experiments/E04_ring_se_seg8
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
mkdir -p $E/data/labels
for i in 0 1 2 3 4 5 6 7; do
  $PY $E/supervised/collect.py --shard $i --shards 8 --episodes 100 > $E/data/labels/collect$i.log 2>&1 &
done
wait
touch $E/data/labels/COLLECT_DONE
