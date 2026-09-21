import numpy as np
import math
import pandas as pd
'''
此文件用以定義產生資料、公式定義的一些單位參數

'''
#-------------公式用參數-------------
#---------  每個 Sensor 和 Target 距離的計算 --------------
mj = 0.001*1000
uj = 0.000001*1000
nj = 0.000000001*1000
pj = 0.000000000001*1000

#----用以產生位置---------
class Cal():
	#-------------- 一、問題的產生，產生位置---------------------------
	#-------- 1-1 Sensor 位置隨機決定-----------------
	def CreateTestSensor(G,self,DIFF=3):#尚未改動-> 改成sensor彼此距離大於diff以上，
		#--------- 1-1-1 產生最初的位置---------
		sensor = np.random.randint(DIFF,self.BOUNDARY-DIFF,size=(self.SENSOR_NUMBER,2))#隨機給予sensor位置
		sensor,repeat_count= np.unique(sensor,axis=0,return_counts=True)#計算重複的位置(P)數量
		#------- 1-1-2 去除重複直到個數滿足 SENSOR_NUMBER----------
		while len(sensor)<self.SENSOR_NUMBER:
			temp = np.random.randint(DIFF,self.BOUNDARY-DIFF,size=(self.SENSOR_NUMBER-len(sensor),2))#隨機給予sensor位置
			sensor = np.concatenate((sensor,temp))#做合併
			sensor,repeat_count= np.unique(sensor,axis=0,return_counts=True)#計算重複的位置(P)數量
		return sensor
	#-------- 1-2 Target Sample 位置固定決定-------------------
	def CreateTestTargetSample(G,self):
		
		#------1-2-1 找出固定座標差SP-----------------
		#線段數量 = 取根號後(代表一列的節點數)+1
		segment_number = math.ceil(math.sqrt(self.TARGET_NUMBER))+1
		#每一列的節點數
		point_number = int(segment_number-1)
		#兩節點之間的空間
		segment_space = np.repeat(self.BOUNDARY/segment_number,point_number)
		#紀錄一行線段中所有節點的位置(SP)
		segment_position = np.cumsum(segment_space)
		
		#--------- 1-2-2 根據SP去去產生target_sample-----------------
		target_sample = []
		for p1 in segment_position:
			for p2 in segment_position:
				target_sample.append([p1,p2])
				if len(target_sample)==self.TARGET_NUMBER:
					break
			if len(target_sample)==self.TARGET_NUMBER:
				break		
		return np.array(target_sample)
	#--------容錯-----------
	def CheckTargetInKSensor(G,self,s):
		detect_count = np.zeros(self.TARGET_NUMBER).astype(int)#紀錄 target有幾個sensor連接
		#------------connect------------
		#connect[:,SENSOR_ID,s] 因為SENSOR_ID和s 數量相同使得會是一個二維陣列 shape(t_num,s_num)
		tid,sid = np.where(self.coverage_table[:,self.SENSOR_ID,s]==1)
		t,repeat_count= np.unique(tid,return_counts=True)#找出重複的tid 表示有幾個sensor連接
		detect_count[t] = repeat_count
		return detect_count
	def CheckFailTargetInKSensor(G,self,s,k=3):
		target_count = G.CheckTargetInKSensor(self,s)
		fail_target = np.where(target_count<k)[0]
		return fail_target
	#CalDistance ：計算兩點群之間的歐式距離-------
	#Input:
	# 	 T : 點群一(預設帶入所有目標點座標)
	#	 S : 點群二(預設帶入所有感測器座標)
	#Output
	#    dis : 二維陣列 長度 [len(T),len(S)]紀錄兩點群每個點彼此之間的歐式距離
	def CalDistance(G,T,S):
		complex_target = np.array([complex(t[0], t[1]) for t in T])#做成複數
		complex_sensor =np.array([complex(s[0], s[1]) for s in S])#做成複數
		v,h = np.meshgrid(complex_sensor,complex_target)
		dis = abs(v-h)
		return dis
