import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Algorithm.core.Round import OnePhase
from Draw import Draw
from Problem.CodingProblem import CodingProblem as CP
from State.CodingState import CodingState

F = 10
Draw.SHOW = 3
Draw.SCALE = 8
# ------------------DS1-------------------------
B = 100
S = 100
T = 25
F = 10
FILE = "DS1_0924_新測試資料"
# #------------------DS2-------------------------
# B = 100
# S = 100
# T = 100
# F = 10
# FILE = 'DS2'
# #------------------DS3-------------------------
# B = 100
# S = 100
# T = 400
# F = 10
# FILE = 'DS3'
# #------------------DS4-------------------------
# B = 100
# S = 200
# T = 100
# F = 10
# FILE = 'DS4'
# #------------------DS5-------------------------
# B = 100
# S = 300
# T = 100
# F = 10
# FILE = 'DS5'
# #------------------DS6-------------------------
# B = 100
# S = 100
# T = 100
# F = 10
# FILE = 'DS6'
# #------------------DS7-------------------------
# B = 100
# S = 100
# T = 100
# F = 10
# FILE = 'DS7'
# #------------------DS8-------------------------
# B = 200
# S = 300
# T = 100
# F = 10
# FILE = 'DS8'

# input 參數
test = str(B) + str(S) + str(T) + "_" + str(FILE)  #'10010025_DS1'
data1_name = "CodingSEv2_8420.4_" + str(F) + "mJ"
j = 0
data_name = "data/" + test + "/" + data1_name + "_" + str(j) + "/life.csv"
d0_5 = []


def draw(P, s, cost, data, i):
    Draw.ShowImage(P, Name="Test", s=s, time_cost=cost, stop=True, scale=8)
    if i == len(data) - 1:
        Draw.ShowImage(P, Name="Test", s=s, time_cost=None, stop=True, scale=8)

    # print(i,d0_5[i])


def run():
    ##讀取某筆life資料
    try:
        data = pd.read_csv(data_name, encoding="ansi")
    except:
        data = pd.read_csv(data_name, encoding="utf-8")
    P = CP(B=B, S=S, T=T, F=F, FILE=FILE)
    data = data.to_numpy()
    ##-----模擬重生life資料--------
    for i in range(len(data)):
        # print(data[i],len(data[i]))
        # -------計算狀態----------
        s = CodingState(P)
        s.levels = data[i][12 : 12 + S].copy().astype("int")
        s.next_hops = data[i][12 + S : 12 + S * 2].copy().astype("int")
        cost = P.calculate_total_cost(s)
        # -----畫出來-------
        draw(P, s, cost, data, i)
        P.energy = P.calculate_remaining_energy(cost)
        d0_5.append(sum(P.energy < 0.5))
    return P.energy


if __name__ == "__main__":
    run()
