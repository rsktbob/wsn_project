#!/bin/bash
# 60 張圖的標籤：預算 3000（RLTRAIN01–60 全收）→ results/data/b3000_m60_train；
# 預算 10000（01–30 的設定、seed 和 labels 相同 → 直接複製，只收 31–60）→ results/data/b10000_m60_train。
# 等 run_t1000_first.sh（3×3 第一列）跑完才開始。完成寫 results/data/M60_COLLECT_DONE
cd /home/g114056067/wsn_project/codes/2027WSN || exit 1
E=Algorithm/se/rl/experiments/E07_ring_se_lifetime
PY=/home/g114056067/miniconda3/envs/wsnenv/bin/python
R=$E/results
until ! pgrep -f "E07_ring_se_lifetime/run_t1000_first.sh" > /dev/null; do sleep 30; done
collect() {  # budget out tag
  for i in 0 1 2 3 4 5 6 7; do
    $PY $E/collect.py --shard $i --shards 8 --maps 60 --budget $1 --out $2 > $R/logs/collect_$3_$i.log 2>&1 &
  done
  wait
  n=$(ls $2/RLTRAIN*.npz 2>/dev/null | grep -vc tmp)
  echo "$3 collect done: $n maps $(date '+%m/%d %H:%M')" >> $R/eval_progress.txt
  [ "$n" -eq 60 ]
}
collect 3000 $R/data/b3000_m60_train b3000_m60 || exit 1
mkdir -p $R/data/b10000_m60_train
cp -n $R/data/b10000_m30_train/RLTRAIN*.npz $R/data/b10000_m60_train/
collect 10000 $R/data/b10000_m60_train b10000_m60 || exit 1
touch $R/data/M60_COLLECT_DONE
