import math
import random
import time

import numpy as np

from Algorithm.misc.BaseGWO import BaseGWO
from Algorithm.Scheduling.SGA import ScheduleState


class CBGWO_State(ScheduleState):  # 自定義的State
    def __init__(self):
        super().__init__()
        self.levels = None
        self.next_hops = None
        self.open = None
        self.code = None
        self.st_sch = None

    def Copy(self):
        n = CBGWO_State()
        if self.levels is not None:
            n.levels = self.levels.copy()
        if self.next_hops is not None:
            n.next_hops = self.next_hops.copy()
        if self.code is not None:
            n.code = self.code.copy()
        if self.open is not None:
            n.open = self.open.copy()
        if self.st_sch is not None:
            n.st_sch = self.st_sch.copy()
        return n


class CBGWO(BaseGWO):
    def __init__(self, P, n=10):
        super().__init__(P, n)
        self.Fname = ["平均剩餘電量/平均花費"]
        self.bet = 10  # 門檻

    def decode_state(self, P, s):
        s.next_hops = np.zeros(P.SENSOR_NUMBER).astype(int) - 1
        s.levels = s.st_sch.copy()
        s.tx_load = np.zeros(P.SENSOR_NUMBER).astype(int)
        for i in range(P.SENSOR_NUMBER):  # 自己的傳送量
            s.tx_load[i] = P.generated_load[i]

        # --------計算灰狼前5%的做cluster head

        code_ch = np.argsort(s.code)  # 排序code
        # ch = code_ch[int(0.95*len(s.code))]#cluster head
        # kid = code_ch[:int(0.95*len(s.code))]#child
        ch = s.open[code_ch[int(0.95 * len(s.code)) :]]  # cluster head
        kid = s.open[code_ch[: int(0.95 * len(s.code))]]  # child
        # print(np.where(s.sch[ch]==0))
        # print(np.where(s.sch[kid]==0))
        dis = P.G.CalDistance(P.sensor[kid], P.sensor[ch])
        fa = np.argmin(dis, 1)  # 和child距離最近的fa

        s.next_hops[kid] = ch[fa]
        s.tx_load[fa] += np.sum(P.generated_load[kid])

        # -----relay配對-----------------
        for i in ch:
            for j in ch:
                if (
                    i != j
                    and s.next_hops[i] == -1
                    and s.next_hops[j] == -1
                ):
                    # print(P.dis[i][j]*1.8169,P.dis[j][P.BSID])
                    v = abs(
                        P.distances[i][j] * 1.8169
                        - P.distances[j][P.BSID]
                    )

                    if v < self.bet:
                        s.next_hops[i] = j
                        s.tx_load[j] += s.tx_load[i]  # i連到 j j需要多負擔i的封包量
                        s.next_hops[j] = P.BSID
            if s.next_hops[i] == -1:
                s.next_hops[i] = P.BSID

        # -------找其他nd 最近的head連到他--------------------

        # print(len(fa),len(kid))
        # print(s.sch[s.open[fa]])

        # print(ch[fa])
        # print(ch)
        # print(s.rou[ch])
        # print('-------')
        if (
            np.sum(
                s.levels[
                    s.next_hops[
                        np.where(
                            (s.next_hops != P.BSID)
                            & (s.next_hops != -1)
                        )[0]
                    ]
                ]
                == 0
            )
            > 0
        ):  # 確認有無開啟但是連到的
            print("----------------------------")
            print(ch)
            print(s.levels[ch])
            print(s.next_hops[ch])
            print(
                np.where(
                    s.levels[
                        s.next_hops[
                            np.where(
                                (s.next_hops != P.BSID)
                                & (s.next_hops != -1)
                            )[0]
                        ]
                    ]
                    == 0
                )
            )
            self.on_iteration_finish(
                problem=P,
                state=s,
                Name="Test",
                time_cost=None,
                stop=False,
                scale=6,
            )
        return s

    def _create_state(self, P, sch_s):  # 自定義
        s = CBGWO_State()
        s.st_sch = sch_s.levels.copy()
        s.levels = sch_s.levels.copy()
        s.next_hops = np.zeros(P.SENSOR_NUMBER).astype(int) - 1
        s.open = np.where(sch_s.levels != 0)[0]
        s.code = np.random.uniform(self.lb, self.ub, len(s.open))
        return s

    def _evaluate_state(self, P, s, DEBUG=False):
        # TD = red/cost
        st = time.time()
        s = self.decode_state(P, s)  # 找出排程與路由結果
        if DEBUG:
            print(time.time() - st)
        cost = P.calculate_total_cost(s)
        # -----------------檢查有無滿足電量約束-----------------
        loss = np.where(P.energy - cost < 0)[0]
        if DEBUG:
            print(time.time() - st)
        if len(loss) > 0:  # 有電量<0
            if DEBUG:
                print(time.time() - st)
                print("---------")
            return [-len(loss)]
        if DEBUG:
            print(time.time() - st)
        # ---------------CBGWO 的fitness --------------------------
        TD = (P.energy[cost != 0]) / cost[cost != 0]  # 開啟中的取平均
        # print(np.sum(TD<1))
        F2 = np.mean(TD) / np.max(TD)  # 越大越好
        # --------------額外增加的fitness----------------------
        # F0,F1 = P.CalFitness(s)
        if DEBUG:
            print(time.time() - st)
        # loss_t = P.CheckFailTargetInSensor(s.sch)
        # F2 = (len(P.target)-len(loss_t))/len(P.target)

        if DEBUG:
            print(time.time() - st)
            print("---")
        return [F2]  # 避免不滿足電量約束
