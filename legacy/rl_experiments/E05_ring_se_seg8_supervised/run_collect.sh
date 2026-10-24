#!/bin/bash
# 190 張訓練圖，8 個 shard 平行跑完整 lifetime（8 段隨機推進、作弊版每 3 段標答案、起點依地圖錯開）
# 完成寫 data/labels_lifetime/COLLECT_DONE；中斷後重跑會跳過已完成的圖
cd /home/g114056067/wsn_project/codes/2027WSN
E=Algorithm/se/rl/experiments/E05_ring_se_seg8_supervised
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
mkdir -p $E/data/labels_lifetime
for i in 0 1 2 3 4 5 6 7; do
  $PY $E/collect.py --shard $i --shards 8 > $E/data/labels_lifetime/collect$i.log 2>&1 &
done
wait
touch $E/data/labels_lifetime/COLLECT_DONE
