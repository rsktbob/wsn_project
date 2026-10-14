"""SensorEncoding2023 對同一組染色體解碼，和學長 CodingState.Decode 逐顆比對。
先在 2023 專案跑 tmp/sa_sets_2023/orig_side.py 產生 npz，再執行：python verify_decode.py orig.npz（RING=0 用距離排序）。"""
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, "/home/g114056067/wsn_project/codes/2027WSN/legacy/sa_sets2023")
sys.path.insert(0, "/home/g114056067/wsn_project/codes/2027WSN")
from SensorEncoding2023 import SensorEncoding2023
from experiments.maps import load_map_specs
from experiments.problem_setup import build_problem

o = np.load(sys.argv[1])
spec = load_map_specs("/home/g114056067/wsn_project/codes/2027WSN/maps/maps_100100100_a2.json")[0]
import os
ring = os.environ.get("RING", "1") == "1"
p = build_problem(SimpleNamespace(_current_map=spec, fitness_service="v2", sensor_ring_id=ring), 7)
print("sensor_ring_id =", ring)
N = p.SENSOR_NUMBER
pos_r = np.asarray(p.sensor, float); pos_o = o["sensor"]
d = np.linalg.norm(pos_r[:, None, :] - pos_o[None, :, :], axis=2)
r2o = d.argmin(axis=1)                       # 你的 sensor r 對應學長的 sensor r2o[r]
assert len(set(r2o)) == N and d.min(axis=1).max() < 1e-9, "座標對不上"
print("座標一一對應：OK；option counts 全為 6：", set(np.asarray(p.radius_option_counts).tolist()))
# 連線容量表、封包量：學長的 P.cap[i, level, j]（j 含 target 與 BS）對你的 link_capacity
dev_o = o["cap"].shape[2]; bs_o = int(o["bsid"])
lc = np.asarray(p.link_capacity, float)
cap_o = o["cap"][r2o][:, :, list(r2o) + [bs_o]]          # 依你的 sensor 順序、只取 sensor + BS
cap_r = lc[:, :, list(range(N)) + [int(p.BSID)]]
print("連線容量表最大差", float(np.nanmax(np.abs(cap_o - cap_r))), "｜封包量相同", np.array_equal(o["pack"][:N][r2o], np.asarray(p.generated_load, float)[:N]))

same_open = same_level = same_hop = 0; cost_err = []; first_bad = None
for t in range(len(o["codes"])):
    oc = o["codes"][t]
    code = np.empty(2 * N, int)
    code[0::2] = oc[0::2][r2o]; code[1::2] = oc[1::2][r2o]
    st = SensorEncoding2023(code).decode(p)
    lv = np.asarray(st.levels)[:N]; hops = np.asarray(st.next_hops)[:N]
    olv = o["sch"][t][r2o]; orou = o["rou"][t][r2o]
    # 下一跳換成學長的編號：BS → 學長的 BSID；sensor → r2o
    mapped = np.array([int(o["bsid"]) if h == p.BSID else (int(r2o[h]) if 0 <= h < N else -1) for h in hops])
    on = lv > 0
    same_open += np.array_equal(on, olv > 0)
    same_level += np.array_equal(lv, olv)
    same_hop += np.array_equal(mapped[on], orou[on])
    radii = np.asarray(p.state_radius(st), float)
    cr = np.asarray(p.calculate_total_cost(st, sensing_radii=radii), float)[:N]
    cost_err.append(np.max(np.abs(cr - o["cost"][t][r2o])))
    if first_bad is None and not (np.array_equal(lv, olv) and np.array_equal(mapped[on], orou[on])):
        first_bad = (t, np.flatnonzero(lv != olv)[:5], np.flatnonzero(on & (mapped != orou))[:5])
n = len(o["codes"])
print(f"開啟的 sensor 完全相同：{same_open}/{n}")
print(f"感測等級完全相同：{same_level}/{n}")
print(f"下一跳完全相同：{same_hop}/{n}")
print(f"每顆 sensor 耗能最大差：中位數 {np.median(cost_err):.2e}、最大 {np.max(cost_err):.2e}")
print("第一個不一致：", first_bad)

# 診斷：不一致的案例中，出問題的 sensor 是否都是「路由基因 = 0（優先值 0、同分）」
tie_only = 0; bad = 0
for t in range(len(o["codes"])):
    oc = o["codes"][t]
    code = np.empty(2 * N, int); code[0::2] = oc[0::2][r2o]; code[1::2] = oc[1::2][r2o]
    st = SensorEncoding2023(code).decode(p)
    lv = np.asarray(st.levels)[:N]; olv = o["sch"][t][r2o]
    hops = np.asarray(st.next_hops)[:N]
    mapped = np.array([int(o["bsid"]) if h == p.BSID else (int(r2o[h]) if 0 <= h < N else -1) for h in hops])
    on = lv > 0
    diff = np.flatnonzero((lv != olv) | (on & (mapped != o["rou"][t][r2o])))
    if len(diff):
        bad += 1
        zero_gene = (code[1::2] % 10 == 0)
        # 第一個出錯的 sensor：依處理順序，看它或它之前是否有同分的 0 優先值 sensor
        tie_only += bool(zero_gene[lv != olv].all()) if (lv != olv).any() else 0
        print(f"案例 {t}：不一致 sensor {diff[:6].tolist()}；其中等級不同的 sensor 路由基因=0：{zero_gene[lv != olv].tolist()}")
print(f"不一致 {bad} 組，其中等級差異全部出在路由基因=0 的 sensor：{tie_only} 組")
