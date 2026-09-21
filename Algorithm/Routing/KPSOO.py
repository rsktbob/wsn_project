import math

import numpy as np

from Algorithm.misc.BasePSO import BasePSO


class KPSOO(BasePSO):#只做Rou
	def __init__(self,P,n=50,w=0.7968,end_w = 0.4,vmax=0.5,vmin=-0.5,fmax=1.0,fmin=0):
		super().__init__(n,w,end_w,vmax,vmin,fmax,fmin)
		self.Fname=['最小剩餘電量/花費)']
		self.hop=None ##紀錄每個sensor可行的候選轉送點

	def _create_state(self,P,sch_s): ##自定義 要有s
		#-------這部分應該寫在init 但是因為不知道sch_s只能寫這，理論上也只會在run的開始執行--------
		self.hop = []
		dsch = np.concatenate((sch_s.levels,[1]))
		for i in range(P.SENSOR_NUMBER):##計算每個點的候選轉送點
			hop_open_id = np.where((dsch[P.next_hop[i]]!=0)&(P.distances[i][P.next_hop[i]]<=600))[0]#找有開的 且比BS近的 (<=80->限制距離減少候選點加速收旂)
			self.hop.append(P.next_hop[i][hop_open_id])
		#-----------------------------------
		s = sch_s.Copy()
		s.code = np.random.uniform(self.fmin,self.fmax,P.SENSOR_NUMBER)
		#print(s.code)
		return s
	def decode_particle(self,DP,s):#
		s.next_hops = np.full(DP.SENSOR_NUMBER,-1)
		for i in range(DP.SENSOR_NUMBER):
			
				if s.levels[i]==0:##沒開不用連
					s.next_hops[i]=-1
				else:	
					try:
					#print(s.code[i],len(hop_open))
						choose_id = math.ceil(s.code[i]*len(self.hop[i]))-1##選擇目標的rou點
						s.next_hops[i] = self.hop[i][choose_id]
					except:
						print(i,s.code[i],len(self.hop[i]))
						s.next_hops[i] = DP.BSID#多餘
			

		#path = DP.FindForward(s)#找到 path
		#s.use = DP.Findtb(path,s)#找到use
		return s
	def _evaluate_state(self,P,s):
		#cap = np.full(P.SENSOR_NUMBER,0)
		#for i in s.rou:
		#	if i !=P.BSID:
		#		cap[i]+=1
		cost = P.calculate_routing_cost(s)
	
		#print(cost)
		F2 = np.min(P.energy[s.levels>0]/cost[s.levels>0])/np.max(P.energy[s.levels>0]/cost[s.levels>0])
		#F0,F1 = P.CalFitness(s)

		return [F2]
