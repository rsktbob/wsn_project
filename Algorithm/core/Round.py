"""執行 WSN 逐輪模擬、重新最佳化與結果紀錄。"""

import csv
import math
import os
import random
import time
from multiprocessing import Pipe, Process

import numpy as np

from Problem.CodingProblem import CodingProblem as CP
from Problem.services.mobility_service import MobilityService
from Draw import Draw


def run_by_budget(algorithm, problem, budget, state=None):
    """以統一契約執行一次並取出最佳 state。"""
    result = algorithm.run(
        problem,
        budget=budget,
        state=state,
    )
    return result.best_state


def evaluate_state(algorithm, problem, state):
    """直接以 Problem 評估演算法回傳的標準 State。"""
    evaluator = getattr(state, "Evaluate", None)
    if evaluator is not None:
        return evaluator(problem)
    return problem.evaluate_state(state)

# ----------資料集 DS1----------
# S = 100
# B = 100
# T = 25
# F = 10

# #----------DS2--------------
# S = 100
# B = 100
# T = 100
# F = 10

# ----------DS3--------------
# S = 100
# B = 100
# T = 400
# F = 10

# #----------DS4--------------
S = 100
B = 100
T = 400
F = 10

# #----------DS5--------------
# S = 300
# B = 100
# T = 100
# F = 10

# #----------DS8--------------
# S = 300
# B = 200
# T = 100
# F = 10


def RequestHead(P, SAI, RAI):
    """建立每輪實驗報表的欄位名稱。"""
    # print("(11)")
    oname = [
        "CV",
        "適應值總和",
        "平均感測器耗能",
        "感測器總剩餘能量",
        "最大目標剩餘能量目標 ID",
        "最大目標剩餘能量",
        "平均目標剩餘能量",
        "最小目標剩餘能量目標 ID",
        "最小目標剩餘能量",
    ]
    if RAI is None:
        name = ["round"] + SAI.Fname + oname
    else:
        name = ["round"] + SAI.Fname + RAI.Fname + oname

    return name



def RequestAllFitness(P, SAI, RAI, s):
    if RAI is None:
        return evaluate_state(SAI, P, s)
    else:
        return list(evaluate_state(SAI, P, s)) + list(
            evaluate_state(RAI, P, s)
        )



def CreateReport(P, report, SAI, RAI=None):
    # print("(10)")

    name = RequestHead(P, SAI, RAI)


    # print("(12)")

    file = open(report, "w", newline="")
    for i in range(P.SENSOR_NUMBER):
        name.append("Sch" + str(i))
    for i in range(P.SENSOR_NUMBER):
        name.append("Rou" + str(i))
    file = csv.writer(file)
    file.writerow(name)
    return file











def PrintProblemStatus(P, s, cost, i, SAI, RAI=None, test=False, report=None):
    name = RequestHead(P, SAI, RAI)
    value = [i]
    F = RequestAllFitness(P, SAI, RAI, s)

    # print("F===", F)
    for f in F:
        value.append(f)

    target_red = P.calculate_target_remaining_energy(P.energy - cost)
    cv, _, _, _ = Mobile_Threshold(target_red, 1, printout=False)
    value.append(cv)





    value = value + [
        sum(F),
        np.mean(cost),
        np.sum(P.energy - cost),
        np.argmax(target_red),
        np.max(target_red),
        np.mean(target_red),
        np.argmin(target_red),
        np.min(target_red),
    ]

    if test:  ##DEBUG??
        for i in range(len(value)):
            if "能量" in name[i]:
                v = str(value[i]) + "mj"
            else:
                v = value[i]
            print(name[i], ":", v)
        print("本輪排程與路由狀態已完成統計。")

        print("------------------------------")
    if report is not None:
        value = value + s.levels.tolist() + s.next_hops.tolist()
        report.writerow(value)



def normalize(data):
    """將資料正規化至 1 到 2 的範圍。"""
    max_val = max(data)
    min_val = min(data)
    if max_val == min_val:
        return [0 for val in data]
    else:
        return np.array([((val - min_val) / (max_val - min_val)) + 1 for val in data])



def Mobile_Threshold(target_red, w, printout=False):
    """Ian"""

    avg = sum(target_red) / len(target_red)


    diff_sq_sum = sum((num - avg) ** 2 for num in target_red)

    std_dev = math.sqrt(diff_sq_sum / len(target_red))
    cv = std_dev / avg

    m = w * std_dev
    if printout == True:
        print(
            "***變異係數={}, 目標平均剩餘能量={}, 標準差={}, 移動門檻={}***".format(
                cv, avg, std_dev, m
            )
        )
    return cv, avg, std_dev, m







def OnePhase(
    P, AI, iteration=200, ID=0, mw=1
):
    OldMV = True
    mobile_data = []
    max_lifetime = 2000
    life = 0
    mobile_weight = mw

    if not OldMV:
        AI_path = (
            P.FILE
            + AI.name
            + "_"
            + str(P.initial_energy)
            + "mJ"
            + "_"
            + str(ID)
            + "_mw"
            + str(mobile_weight)
            + "/"
        )
        MAI_path = (
            P.MFILE
            + AI.name
            + "_"
            + str(P.initial_energy)
            + "mJ"
            + "_"
            + str(ID)
            + "_mw"
            + str(mobile_weight)
            + "/"
        )
    else:
        AI_path = f"{P.FILE}{AI.name}_{P.initial_energy}mJ_{ID}_OldMV/"
        MAI_path = f"{P.MFILE}{AI.name}_{P.initial_energy}mJ_{ID}_OldMV/"

    if not os.path.isdir(AI_path):
        os.mkdir(AI_path)
        os.mkdir(MAI_path)
    else:
        print("結果資料夾已存在，將沿用現有路徑。")

    report = CreateReport(P, AI_path + "life.csv", AI)


    # print("(13)")


    for i in range(1, max_lifetime):
        s = run_by_budget(AI, P, iteration)

        cost = P.calculate_total_cost(s)



        target_red = P.calculate_target_remaining_energy(P.energy - cost)
        if OldMV == False:
            _, _, _, mt = Mobile_Threshold(target_red, mobile_weight, printout=True)

            complex_target = np.array([(t[0], t[1]) for t in P.target])
            target_base_dis = np.sqrt(
                np.sum((complex_target - np.array([3, 3])) ** 2, axis=1)
            )
            norm_target_base_dis = normalize(target_base_dis.reshape(-1))
            # norm_target_base_dis = norm_target_base_dis.reshape(10,10)    # Reshape
            # print(norm_target_base_dis)

            norm_target_red = norm_target_base_dis * target_red

            if P.LifeCheck(s, cost) == False or np.min(norm_target_red) < mt:

                target_red = P.calculate_target_remaining_energy(
                    P.energy - cost
                )
                fail_target = P.find_uncovered_targets(s.levels)
                min_tid = []
                for j in range(P.TARGET_NUMBER):
                    if norm_target_red[j] < mt:
                        min_tid.append(j)
                if min_tid != []:
                    M = MobilityService(P, s)
                    update_position, updated_power, mobile_data = M.Move(
                        P, s, min_tid, norm_target_red, fail_target
                    )
                    P = CP(
                        B=B,
                        S=S,
                        T=T,
                        F=F,
                        FILE=None,
                        update_position=update_position,
                        updated_power=updated_power,
                        Target_position=P.target,
                    )
                    P.mobile_id = mobile_data.copy()
                    # print(P.J)
                s = run_by_budget(AI, P, iteration)
                cost = P.calculate_total_cost(s)
                target_red = P.calculate_target_remaining_energy(
                    P.energy - cost
                )
                _, _, _, mt = Mobile_Threshold(target_red, mobile_weight, printout=True)
                print("重新最佳化後適應值：", evaluate_state(AI, P, s))
                if P.LifeCheck(s, cost, True) == False:
                    # update_position, updated_power = M.Move(P, s, min_tid, target_red)
                    print("第", i, "輪失敗")
                    Draw.ShowImage(
                        P,
                        Name="Test",
                        s=s,
                        time_cost=None,
                        stop=True,
                        scale=7,
                        save=MAI_path + "round" + str(i) + "(fail).png",
                    )
                    break
        else:
            if P.LifeCheck(s, cost) == False or np.min(target_red) < 10:

                target_red = P.calculate_target_remaining_energy(
                    P.energy - cost
                )
                fail_target = P.find_uncovered_targets(s.levels)
                min_tid = []
                for j in range(P.TARGET_NUMBER):
                    if target_red[j] < 10:
                        min_tid.append(j)
                if min_tid != []:
                    M = MobilityService(P, s)
                    update_position, updated_power, mobile_data = M.Move(
                        P, s, min_tid, target_red, fail_target
                    )
                    P = CP(
                        B=B,
                        S=S,
                        T=T,
                        F=F,
                        FILE=None,
                        update_position=update_position,
                        updated_power=updated_power,
                        Target_position=P.target,
                    )
                    P.mobile_id = mobile_data.copy()
                    # print(P.J)
                s = run_by_budget(AI, P, iteration)
                cost = P.calculate_total_cost(s)
                target_red = P.calculate_target_remaining_energy(
                    P.energy - cost
                )
                print("重新最佳化後適應值：", evaluate_state(AI, P, s))
                if P.LifeCheck(s, cost, True) == False:
                    # update_position, updated_power = M.Move(P, s, min_tid, target_red)
                    print("第", i, "輪失敗")
                    Draw.ShowImage(
                        P,
                        Name="Test",
                        s=s,
                        time_cost=None,
                        stop=True,
                        scale=7,
                        save=MAI_path + "round" + str(i) + "(fail).png",
                    )
                    break


        Draw.ShowImage(
            P,
            Name="Test",
            s=s,
            time_cost=None,
            stop=False,
            scale=7,
            save=MAI_path + "round" + str(i) + ".png",
        )
        T_sqrt = int(math.sqrt(T))
        if OldMV == False:
            Z = norm_target_base_dis.reshape(T_sqrt, T_sqrt) * target_red.reshape(
                T_sqrt, T_sqrt
            )
        else:
            Z = target_red.reshape(T_sqrt, T_sqrt)

        # plt.imshow(Z, cmap="hot")
        # plt.colorbar()
        # plt.savefig(MAI_path + "HOT_round" + str(i) + ".png")
        # plt.clf()
        # ##---------------------------
        cost = P.calculate_total_cost(s)
        PrintProblemStatus(P, s, cost, life, AI, report=None, test=True)


        for j in range(0, 5000):
            if P.LifeCheck(s, cost) == False:
                # print(np.where(P.J<0))
                P.prepare_coding_cache(cost * j)
                break
            PrintProblemStatus(P, s, cost, life, AI, report=report, test=False)
            P.energy = P.calculate_remaining_energy(cost)
            life += 1
        print("第", life, "輪成功")
        # PrintProblemStatus(P,s,cost,life,report,AI)







def SAC_Phase(
    P, AI, RLAI, iteration=200, ID=0
):
    # print("(9)")
    mobile_data = []
    max_lifetime = 2000
    life = 0

    AI_path = (
        P.FILE
        + RLAI.name
        + "_"
        + AI.name
        + "_"
        + str(P.initial_energy)
        + "mJ"
        + "_"
        + str(ID)
        + "/"
    )
    MAI_path = (
        P.MFILE
        + RLAI.name
        + "_"
        + AI.name
        + "_"
        + str(P.initial_energy)
        + "mJ"
        + "_"
        + str(ID)
        + "/"
    )
    if not os.path.isdir(AI_path):
        os.mkdir(AI_path)
        os.mkdir(MAI_path)
    else:
        print("結果資料夾已存在，將沿用現有路徑。")

    report = CreateReport(P, AI_path + "life.csv", AI)


    # print("(13)")





    #     target_red = P.CalTargetRed(P.J-cost)

    #         max_lifetime = 1

    for i in range(1, max_lifetime):
        # else:
        # s =
        RLAI.run_by_evaluation_limit(
            P,
            1,
            iteration,
            creat_new_model=False,
        )
        break
        # M = MobilityService(P, s)
        # update_position, updated_power = M.Move(P, s, min_tid, target_red)
        # sdsds()

        cost = P.calculate_total_cost(s)

        target_red = P.calculate_target_remaining_energy(P.energy - cost)
        if P.LifeCheck(s, cost) == False or np.min(target_red) < 10:

            target_red = P.calculate_target_remaining_energy(
                P.energy - cost
            )
            fail_target = P.find_uncovered_targets(s.levels)
            min_tid = []
            for j in range(P.TARGET_NUMBER):
                if target_red[j] < 10:
                    min_tid.append(j)
            if min_tid != []:
                M = MobilityService(P, s)
                update_position, updated_power, mobile_data = M.Move(
                    P, s, min_tid, target_red, fail_target
                )
                P = CP(
                    B=B,
                    S=S,
                    T=T,
                    F=F,
                    FILE=None,
                    update_position=update_position,
                    updated_power=updated_power,
                    Target_position=P.target,
                )
                P.mobile_id = mobile_data.copy()
                # print(P.J)
            s = run_by_budget(AI, P, iteration)
            cost = P.calculate_total_cost(s)
            print("重新最佳化後適應值：", evaluate_state(AI, P, s))
            if P.LifeCheck(s, cost, True) == False:
                # update_position, updated_power = M.Move(P, s, min_tid, target_red)
                print("第", i, "輪失敗")
                Draw.ShowImage(
                    P,
                    Name="Test",
                    s=s,
                    time_cost=None,
                    stop=True,
                    scale=7,
                    save=MAI_path + "round" + str(i) + "(fail).png",
                )
                break


        Draw.ShowImage(
            P,
            Name="Test",
            s=s,
            time_cost=None,
            stop=False,
            scale=7,
            save=MAI_path + "round" + str(i) + ".png",
        )
        cost = P.calculate_total_cost(s)
        PrintProblemStatus(P, s, cost, life, AI, report=None, test=True)


        for j in range(0, 5000):
            if P.LifeCheck(s, cost) == False:
                # print(np.where(P.J<0))
                P.prepare_coding_cache(cost * j)
                break
            PrintProblemStatus(P, s, cost, life, AI, report=report, test=False)
            P.energy = P.calculate_remaining_energy(cost)
            life += 1
        print("第", life, "輪成功")
        # PrintProblemStatus(P,s,cost,life,report,AI)
