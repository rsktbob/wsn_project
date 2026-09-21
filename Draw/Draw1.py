import cv2
import numpy as np
import math
import random
#------------- 三、顯示-----------------------
SHOW = 0
SCALE=3
def DrawSensor(problem,s,scale=7):
	if s is not None:
		sch_s = s.levels
		rou_s = s.next_hops
	else:
		sch_s = None
		rou_s = None
	img = np.zeros((problem.BOUNDARY*scale,problem.BOUNDARY*scale,3),np.uint8)
	img.fill(0)
	if sch_s is None:
		sch_s = np.zeros(problem.SENSOR_NUMBER,np.int)

	fail_target = problem.find_uncovered_targets(sch_s)
	#print(fail_target)
	
	#-------------------- sensor 畫出來------------------------
	for sid in range(len(problem.sensor)):
		p = problem.sensor[sid]
		#debug
		#if sid==14:
		#   cv2.circle(img,(int(p[0]*scale),int(p[1]*scale)),3, (255, 244, 0), 1)
		if sch_s[sid]!=0:
			cv2.circle(img,(int(p[0]*scale),int(p[1]*scale)),int(problem.sensing_radius(sid, sch_s[sid])*scale), (0,255, 0), 1)
			cv2.circle(img,(int(p[0]*scale),int(p[1]*scale)),scale//3+1, (255, 244, 0), 1)
			cv2.putText(img,str(sid), (int(p[0]*scale),int(p[1]*scale)), cv2.FONT_HERSHEY_DUPLEX,0.6, (255, 244, 255), 1, cv2.LINE_AA)
		else:
			cv2.circle(img,(int(p[0]*scale),int(p[1]*scale)),scale//3+1, (39,127, 255), 1)
			cv2.putText(img,str(sid), (int(p[0]*scale),int(p[1]*scale)), cv2.FONT_HERSHEY_DUPLEX,0.6, (39,127, 255), 1, cv2.LINE_AA)
	#-----------------target sample 畫出來---------------------
	for tid in range(len(problem.target)):
		t = problem.target[tid]
		if tid in fail_target:
			cv2.circle(img,(int(t[0]*scale),int(t[1]*scale)), 2, (0, 0, 255), -1)
		else:
			cv2.circle(img,(int(t[0]*scale),int(t[1]*scale)), 2, (0, 255, 255), -1)
	#--------------BS Draw--------------
	cv2.circle(img,(int(problem.BS[0]*scale),int(problem.BS[1]*scale)), 2, (255, 0,120), -1)
	#-----------------畫出 Routing-------------------------------
	if rou_s is not None:
		S = problem.device
		#-----找到loss_device---------
		loss_device = problem.find_disconnected(s)
		
		#--------------------------
		for sid in range(len(rou_s)):
			st = sid
			ed = rou_s[sid]
			#if ed!=problem.SENSOR_NUMBER:#----------Sensor---------
				#cv2.line(img,(int(S[st,0]*scale),int(S[st,1]*scale)),(int(problem.BS[0]*scale),int(problem.BS[1]*scale)),(225, 150,0),1)
			#---------------當 PRE 非 -1時 表示在樹內顯示邊--------------------
			#elif ed>=0:#------BS----------
			if ed>=0:
				if st not in loss_device:
					cv2.arrowedLine(img,(int(S[st,0]*scale),int(S[st,1]*scale)),(int(S[ed,0]*scale),int(S[ed,1]*scale)),(random.randint(150,150), random.randint(150,150),0),tipLength=3/problem.distances[st][ed])
				else:
					cv2.arrowedLine(img,(int(S[st,0]*scale),int(S[st,1]*scale)),(int(S[ed,0]*scale),int(S[ed,1]*scale)),(203, 192,255),tipLength=3/problem.distances[st][ed])
	return img

def DrawPower(problem,time_cost=None,scale=1):
	if time_cost is None:
		time_cost=np.zeros(problem.SENSOR_NUMBER)
	colNumber = len(problem.sensor)//60+1
	col = 200#每個col的空間
	rowNumber = 60
	row = 15#每個row的空間
	img = np.zeros((row*rowNumber*scale+10,col*colNumber*scale,3),np.uint8)

	for sid in range(problem.SENSOR_NUMBER):
		i = sid%rowNumber
		j = sid//rowNumber
		stY = (13+row*i)*scale # 13 是離天花板 13 
		stX = int(10+col*j)*scale # 10 離左邊 10 
		
		#輸出 節點 : sid 剩餘能量 : J 座標
		cv2.putText(img,'N'+str(sid)+': '+str(np.round(problem.energy[sid],3)), (stX,stY), cv2.FONT_HERSHEY_DUPLEX,0.35, (0, 255, 255), 1, cv2.LINE_AA)
		
		stX = (90+j*col)*scale#col
		stY = (4+row*i)*scale #空隙 + 每個分配的空間row
		mid1X = stX+int(max(problem.energy[sid],0)/(problem.initial_energy/100))*scale
		mid2X = mid1X+int(max(time_cost[sid],0)/(problem.initial_energy/100))*scale
		edY = (row*(1+i))*scale#結束的空間座標 = i+1
		edX = (90+j*col+100)*scale # 離左邊90 + 100個能量(100%的位置)
		
		cv2.rectangle(img,(stX,stY),(mid1X,edY), (0, 255, 0), -1)
		cv2.rectangle(img,(mid1X,stY),(mid2X,edY), (0, 100, 100), -1)
		cv2.rectangle(img,(mid2X,stY),(edX,edY), (0, 0, 100), -1)
	return img
	
def ShowImage(problem,Name="",s=None,time_cost=None,stop=False,scale=7,save=None,SOLID=False):
	scale = SCALE
	img = DrawSensor(problem,s,scale)
	img2 = DrawPower(problem,time_cost)
	
	if save is not None:#存檔
		print(save)
		r = np.random.randint(0,10,1)[0]
		if r > 3 or SOLID:##節省硬碟空間
			cv2.imwrite(save,img)
	else:
		if int(SHOW/2)==1:
			cv2.imshow(Name+'_cost:',img2)
		if SHOW%2==1:
			cv2.imshow(Name+'_map',img)
		if stop ==True:
			cv2.waitKey(0)
		else:
			if stop!=False:
				cv2.waitKey(stop)
			else:
				cv2.waitKey(10)
	#cv2.destroyAllWindows()
	
#------------------顯示每個sensor的候選路徑----------------
def DrawCandidate(problem,device_conn,sch_s=None,scale=10):
	img = np.zeros((problem.BOUNDARY*scale,problem.BOUNDARY*scale,3),np.uint8)
	img.fill(0)
	if sch_s is None:
		sch_s = np.zeros(problem.SENSOR_NUMBER,np.int)
		
	for sid in range(len(problem.sensor)):
		p = problem.sensor[sid]
		#debug
		#if sid==14:
		#   cv2.circle(img,(int(p[0]*scale),int(p[1]*scale)),3, (255, 244, 0), 1)
		if sch_s[sid]!=0:
			cv2.circle(img,(int(p[0]*scale),int(p[1]*scale)),int(problem.sensing_radius(sid, sch_s[sid])*scale), (0,255, 0), 1)
			cv2.circle(img,(int(p[0]*scale),int(p[1]*scale)),scale//3+1, (255, 244, 0), 1)
			cv2.putText(img,str(sid), (int(p[0]*scale),int(p[1]*scale)), cv2.FONT_HERSHEY_DUPLEX,0.6, (255, 244, 255), 1, cv2.LINE_AA)
		else:
			cv2.circle(img,(int(p[0]*scale),int(p[1]*scale)),scale//3+1, (39,127, 255), 1)
			cv2.putText(img,str(sid), (int(p[0]*scale),int(p[1]*scale)), cv2.FONT_HERSHEY_DUPLEX,0.6, (39,127, 255), 1, cv2.LINE_AA)
			
	S = problem.device
	print(device_conn.shape[1])
	for sid in range(device_conn.shape[1]):
		st = device_conn[0,sid]
		ed = device_conn[1,sid]
		if ed>=0 and st>=0:
			cv2.arrowedLine(img,(int(S[st,0]*scale),int(S[st,1]*scale)),(int(S[ed,0]*scale),int(S[ed,1]*scale)),(225, 150,0),1)
	cv2.imshow('A',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
