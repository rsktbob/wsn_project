import cv2
import numpy as np
import math
import random
#------------- 三、顯示-----------------------
SHOW = 0
SCALE = 8

def DrawSensor(problem,s,scale=8):
	B=problem.BOUNDARY
	if s is not None:
		sch_s = s.levels
		rou_s = s.next_hops
	else:
		sch_s = None
		rou_s = None
	# Reserve a fixed right-hand legend panel.  The old width expression was
	# only large enough for SCALE=8; after live display used scale=5, the
	# legend text could be clipped at the right edge.
	legend_width = 330
	img = np.zeros(((B+10)*scale, B*scale + legend_width, 4), np.uint8)
	img.fill(255)
	#-------------------邊線--------------------------
	cv2.line(img, (0, 0), (B*scale, 0), (0, 0, 0), 1)  #上橫
	cv2.line(img, (0, B*scale), (B*scale, B*scale), (0, 0, 0), 1)  #下橫
	cv2.line(img, (0, 0), (0, B*scale), (0, 0, 0), 1)  #左
	cv2.line(img, (B*scale, 0), (B*scale, B*scale), (0, 0, 0), 1)  #右
	
	#-------------------座標--------------------------
	no = 0
	for x in range(10+1):
		axis_pos = int(B * scale * no / 20)
		axis_value = int(B * no / 20)
		cv2.line(img, (axis_pos, B*scale), (axis_pos, B*scale+10), (0, 0, 0), 1)  #x軸
		cv2.line(img, (B*scale, axis_pos), (B*scale+7, axis_pos), (0, 0, 0), 1)   #y軸
		if no != 0:
			cv2.putText(img, str(axis_value), (axis_pos-11, B*scale+32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)  #x軸
			cv2.putText(img, str(axis_value), (B*scale+15, axis_pos+8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)   #y軸
		else:
			cv2.putText(img, str(0), (100*no+1, B*scale+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)  #x軸
			cv2.putText(img, str(0), (B*scale+20, 100*no+18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA) #y軸
		no+=2

	#-------------------圖例-BS--------------------------
	cv2.putText(img, "Base Station", (B*scale+110, B*scale-150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
	cv2.circle(img,(B*scale+90,B*scale-160),5, (30, 144, 255), -1)  #開啟範圍
	cv2.circle(img,(B*scale+90,B*scale-160),8, (30, 144, 255), 1)  #開啟範圍
	cv2.circle(img,(B*scale+90,B*scale-160),12, (30, 144, 255), 1)  #開啟範圍
	cv2.circle(img,(B*scale+90,B*scale-160),14, (30, 144, 255), 1)  #開啟範圍
	
	cv2.line(img, (B*scale+90,B*scale-155), (B*scale+85,B*scale-139), (0, 0, 0), 2)  #x軸
	cv2.line(img, (B*scale+90,B*scale-155), (B*scale+95,B*scale-139), (0, 0, 0), 2)  #x軸
				

	# #畫星星
	# phi = 4*np.pi/5
	# rotations = [[[np.cos(i*phi), -np.sin(i*phi)], [i*np.sin(phi), np.cos(i*phi)]] for i in range(1, 5)]
	# pentagram = np.array([[[[0, -1]] + [np.dot(m, (0, -1)) for m in rotations]]], dtype=np.float)
	# #位置
	# pentagram = np.round(pentagram*10 + np.array([B*scale+90, B*scale-124])).astype(np.int) 
	# #5個點連線
	# cv2.polylines(img, pentagram, True, (0, 0, 255), 1)
### 畫三角形
# 	cv2.putText(img, "Close Sensor", (B*scale+110, B*scale-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
# 	#畫填滿的三角形
# 	point = (B*scale+90, B*scale-65)
# 	a = pow(3,1/2)/2*scale
# 	triangle_cnt2 = np.array([(point[0]+a,point[1]+a), (point[0]-a,point[1]+a), (point[0],point[1]-a)]).astype(int)
# 	cv2.drawContours(img, [triangle_cnt2], 0, (205, 205, 201), 1)
	#-------------------圖例-Open Sensor--------------------------
	cv2.putText(img, "Open Sensor", (B*scale+110, B*scale-120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
	#畫填滿的SENSOR菱形
	point = (B*scale+90, B*scale-125)
	a = pow(4,1/2)/2*scale
	triangle_cnt1 = np.array([(point[0],point[1]-a), (point[0]-a,point[1]-a/2), (point[0],point[1]+a), (point[0]+a,point[1]-a/2)]).astype(int)
	cv2.drawContours(img, [triangle_cnt1], 0, (34, 139, 34), -1)

	#-------------------圖例-Close Sensor--------------------------
	cv2.putText(img, "Close Sensor", (B*scale+110, B*scale-90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
	#畫空心的SENSOR菱形
	point = (B*scale+90, B*scale-95)
	a = pow(4,1/2)/2*scale
	triangle_cnt2 = np.array([(point[0],point[1]-a), (point[0]-a,point[1]-a/2), (point[0],point[1]+a), (point[0]+a,point[1]-a/2)]).astype(int)
	cv2.drawContours(img, [triangle_cnt2], 0, (0, 0, 0), 1)
	#-------------------圖例-mobile Sensor--------------------------
	cv2.putText(img, "Mobile Sensor", (B*scale+110, B*scale-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
	#畫空心的三角形
	point = (B*scale+90, B*scale-65)
	a = pow(3,1/2)/2*scale
	triangle_cnt2 = np.array([(point[0]+a,point[1]+a), (point[0]-a,point[1]+a), (point[0],point[1]-a)]).astype(int)
	cv2.drawContours(img, [triangle_cnt2], 0, (255, 156, 56), 2)
	#-------------------圖例-Covered target--------------------------
	cv2.putText(img, "Covered Target", (B*scale+110, B*scale-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
	cv2.circle(img,(B*scale+90, B*scale-35), 5, (0, 0, 0), -1)
	#-------------------圖例-Lose target--------------------------
	cv2.putText(img, "Lose Target", (B*scale+110, B*scale), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
	cv2.circle(img,(B*scale+90, B*scale-5), 5, (255, 0, 0), -1)
	
	if sch_s is None:
		sch_s = np.zeros(problem.SENSOR_NUMBER,np.int)

	fail_target = problem.find_uncovered_targets(sch_s)
	#print(fail_target)
	
	#-------------------- sensor 畫出來------------------------
	zeros1 = np.zeros(((B+10)*scale, B*scale + legend_width, 4), np.uint8)
	zeros1.fill(255)
	s_pow = pow(4,1/2)/2
	for sid in range(len(problem.sensor)):
		p = problem.sensor[sid]
		if sch_s[sid]!=0:
			radius = problem.sensing_radius(sid, sch_s[sid])
			cv2.circle(zeros1,(int(p[0]*scale),int(p[1]*scale)),int(radius*scale), (175, 238, 238), -1)  #開啟範圍
			cv2.circle(img,(int(p[0]*scale),int(p[1]*scale)),int(radius*scale), (0, 204, 204), 1)  #開啟範圍
			#畫填滿的SENSOR菱形
			triangle_cnt = np.array([(int((p[0])*scale),int((p[1]-s_pow)*scale)), (int((p[0]-s_pow)*scale),int((p[1]-s_pow/2)*scale)), (int((p[0])*scale),int((p[1]+s_pow)*scale)), (int((p[0]+s_pow)*scale),int((p[1]-s_pow/2)*scale))])
			cv2.drawContours(img, [triangle_cnt], 0, (34, 139, 34), -1)
			# cv2.putText(img,str(sid), (int(p[0]*scale),int(p[1]*scale)), cv2.FONT_HERSHEY_DUPLEX,0.6, (255, 244, 255), 1, cv2.LINE_AA)
		else:
			#畫未填滿的SENSOR菱形
			triangle_cnt = np.array([(int((p[0])*scale),int((p[1]-s_pow)*scale)), (int((p[0]-s_pow)*scale),int((p[1]-s_pow/2)*scale)), (int((p[0])*scale),int((p[1]+s_pow)*scale)), (int((p[0]+s_pow)*scale),int((p[1]-s_pow/2)*scale))])
			cv2.drawContours(img, [triangle_cnt], 0, (0, 0, 0), 1)

	# cv2.imshow("img", img)
	# cv2.imshow("zeros1", zeros1)
	img = cv2.addWeighted(img, 0.9, zeros1, 0.1 ,0)
	# cv2.imshow("result", img)
	# cv2.waitKey()
	# cv2.destroyAllWindows()
	
	#-----------------target sample 畫出來---------------------
	for tid in range(len(problem.target)):
		t = problem.target[tid]
		if tid in fail_target:
			cv2.circle(img,(int(t[0]*scale),int(t[1]*scale)), 5, (255, 0, 0), -1)
		else:
			cv2.circle(img,(int(t[0]*scale),int(t[1]*scale)), 5, (0, 0, 0), -1)
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
					cv2.arrowedLine(img,(int(S[st,0]*scale),int(S[st,1]*scale)),(int(S[ed,0]*scale),int(S[ed,1]*scale)),(153, 153, 204),tipLength=3/problem.distances[st][ed])
				else:
					cv2.arrowedLine(img,(int(S[st,0]*scale),int(S[st,1]*scale)),(int(S[ed,0]*scale),int(S[ed,1]*scale)),(255, 0, 0),tipLength=3/problem.distances[st][ed])
	#--------------BS Draw--------------
	cv2.circle(img,(problem.BS[0]*scale-10, problem.BS[0]*scale-10),5, (30, 144, 255), -1)  #開啟範圍
	cv2.circle(img,(problem.BS[0]*scale-10, problem.BS[0]*scale-10),8, (30, 144, 255), 1)  #開啟範圍
	cv2.circle(img,(problem.BS[0]*scale-10, problem.BS[0]*scale-10),12, (30, 144, 255), 1)  #開啟範圍
	cv2.circle(img,(problem.BS[0]*scale-10, problem.BS[0]*scale-10),14, (30, 144, 255), 1)  #開啟範圍
	
	cv2.line(img, (problem.BS[0]*scale-10,problem.BS[0]*scale-5), (problem.BS[0]*scale-15,problem.BS[0]*scale+11), (0, 0, 0), 2)  #x軸
	cv2.line(img, (problem.BS[0]*scale-10,problem.BS[0]*scale-5), (problem.BS[0]*scale-5,problem.BS[0]*scale+11), (0, 0, 0), 2)  #x軸
				
	#--------------mobile sensor--------------
	if problem.mobile_id != []:
		for sid1 in range(len(problem.mobile_id)):
			triangle_cnt = np.array( [(int((problem.mobile_id[sid1][1]+pow(3,1/2)/2)*scale),int((problem.mobile_id[sid1][2]+pow(3,1/2)/2)*scale)), (int((problem.mobile_id[sid1][1]-pow(3,1/2)/2)*scale),int((problem.mobile_id[sid1][2]+pow(3,1/2)/2)*scale)), (int((problem.mobile_id[sid1][1])*scale),int((problem.mobile_id[sid1][2]-pow(3,1/2)/2)*scale))])
			cv2.drawContours(img, [triangle_cnt], 0, (255, 145, 36), 1)
			cv2.arrowedLine(img,(int(problem.mobile_id[sid1][1]*scale),int(problem.mobile_id[sid1][2]*scale)),(int(problem.mobile_id[sid1][3]*scale),int(problem.mobile_id[sid1][4]*scale)),(255, 145, 36),thickness=2 ,tipLength=0.08)
		# print("problem.mobile_id==========", problem.mobile_id)

	# problem.sensor 已在 Problem.sort_nodes_by_bs_distance() 依 ring
	# 順序重排，因此 sid 就是 chromosome 中的 ring-order ID。
	# 最後才畫標籤，避免 sensing range、target 或 routing 線蓋住編號。
	font_scale = max(0.35, min(0.55, scale / 16.0))
	for sid, p in enumerate(problem.sensor):
		label = str(sid)
		label_pos = (int(p[0] * scale) + 4, int(p[1] * scale) - 4)
		cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_DUPLEX,
			font_scale, (255, 255, 255), 3, cv2.LINE_AA)
		cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_DUPLEX,
			font_scale, (0, 0, 0), 1, cv2.LINE_AA)

	#cv2.circle(img,(int(problem.BS[0]*scale),int(problem.BS[1]*scale)), 2, (255, 0,120), -1)
	#畫星星
	# phi = 4*np.pi/5
	# rotations = [[[np.cos(i*phi), -np.sin(i*phi)], [i*np.sin(phi), np.cos(i*phi)]] for i in range(1, 5)]
	# pentagram = np.array([[[[0, -1]] + [np.dot(m, (0, -1)) for m in rotations]]], dtype=np.float)
	# #位置
	# pentagram = np.round(pentagram*10 + np.array([(int(problem.BS[0]*scale),int(problem.BS[1]*scale))])).astype(np.int)
	 
	# # 将5个顶点作为多边形顶点连线，得到五角星
	# cv2.polylines(img, pentagram, True, (0, 0, 255), 1)
	return img


def DrawPower(problem,time_cost=None,scale=8):
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
		
		cv2.rectangle(img,(stX,stY),(mid1X,edY), (205, 201, 165), -1)
		cv2.rectangle(img,(mid1X,stY),(mid2X,edY), (255, 250, 205), -1)
		cv2.rectangle(img,(mid2X,stY),(edX,edY), (0, 0, 100), -1)
	return img
	
def ShowImage(problem,Name="",s=None,time_cost=None,stop=False,scale=8,save=None,SOLID=False):
	# Keep the map readable without creating an enormous per-sensor power
	# panel.  The legacy code used SCALE for both images, which makes the
	# power panel 60 rows * 15 pixels * 8 for a 100-sensor map.
	scale = max(1, int(scale))
	# if problem.mobile_id != []:
	# 	print("problem.mobile_id==========", problem.mobile_id)
	# # 	fdgdfgd()
	img = DrawSensor(problem,s,scale)
	power_scale = scale if save is not None else 1
	img2 = DrawPower(problem,time_cost,power_scale)
	
	if save is not None:#存檔
		print(save)
		# r = np.random.randint(0,10,1)[0]
		# if r > 3 or SOLID:##節省硬碟空間
		# if SOLID:##節省硬碟空間
		image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		cv2.imwrite(save,image_rgb)
	else:
		if int(SHOW/2)==1:
			# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
			cv2.imshow(Name+'_cost:',img2)
		if SHOW%2==1:
			# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
def DrawCandidate(problem,device_conn,sch_s=None,scale=8):
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
