import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
# test = '10010025_DS1_1011_新測試資料'
# test = '10010025_DS1_COMP'
# test = '100100100_DS2'
test = '100100100_DS2_15'
#test = '100100400_DS3'
# test = '100100400_DS3_COMP'
# test = '100200100_DS4'
# test = '100200100_DS4_COMP'
# test = '100200100_DS5'
# test = '100300100_DS5_COMP'
# test = '100100100_DS6'
# test = '100100100_DS6_COMP'
# test = '100100100_DS7'
# test = '100100100_DS7_COMP'
#test = '200300100_DS8'
#test = '200300100_DS8_COMP'

data1_name = ['SETSv3_8420.4_10mJ','CodingSEv2_8420.4_10mJ','JGA_300.10.8_10mJ','JPSO_60_10mJ','JEDA_100_0.85_10mJ','JGWO_60_10mJ','JCS_30_0.6_10mJ']#,'SES+CBGWO_100mJ']
# data1_name = ['CodingSEv2_8420.4_10mJ','JGA_300.10.8_10mJ','JPSO_60_10mJ','JEDA_100_0.85_10mJ','JGWO_60_10mJ','JCS_30_0.6_10mJ']
#data1_name = ['SETSv2_10mJ','JGA_10mJ','JPSO_10mJ','JEDA_10mJ','JGWO_10mJ']
#data1_name = ['CodingSE_100mJ','SGA+KPSO_100mJ']
data_number=30
data2_name=['CodingSE','SES+KPSO','SES+GARO','GAS+PSO','GAS+GARO']


colors = ["red","blue","orange","cyan",'green',"black","#00FF55","#22FF33","#FF0033","gray","white"]
markers = ['o', ',', '.', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D','d', '|', '_']
linestyles = ['-',':','--','-.','--','-.','-',':','-',':','--','-.']
labels=['SETS','SE' ,'GA','PSO','EDA','GWO','CS','GAS-GAR','SES-PSOR','GAS-PSOR','NBEDA-GAR','NBEDA-PSOR']
#labels=['SETS' ,'GA','PSO','EDA','GWO','GAS-GAR','SES-PSOR','GAS-PSOR','NBEDA-GAR','NBEDA-PSOR']
labels=['SETS','SE' ,'GA','PSO','EDA','GWO','CS']
save = {}
analysis = {'name':[],'max_lf':[],'min_lf':[],'avg_lf':[]}

y_name={'最小T剩餘電量':'Minimum Redusal Energy of Target (mJ)','評估函數':'Objective Value','剩餘總電量':'Total Redusal Energy (mJ)'}
def show():

	y = ['剩餘總電量','評估函數','最小T剩餘電量']
	cmd = ['最好','最差','平均']
	
	cmd = ['平均']
	for i in y:
		for j in cmd:			
			phase1(i,j)
			phase2(i,j)
def detail():
	for i in range(12):
		phase1('剩餘總電量','MD',i)
def phase1(name,cmd,ii=None):
	for i in range(0,len(data1_name)):#對照組與實驗組的讀取 
		print(labels[i])

		data_name=[]
		target_data=np.zeros((12000))##x軸最大長度
		lf_data = np.zeros((12000))
		max_lf = 0
		min_lf = 999999
		avg_lf = 0
		ids = 0
		success_data = 0
		for j in range(data_number):#讀取每一次的run
			##--------跨號讀取----------
			for k in range(ids,30):
				if os.path.isfile('data/'+test+'/'+data1_name[i]+'_'+str(ids)+'/life.csv'):
					success_data+=1
					break
				else:
					ids+=1
					
					print(ids)
			if ids==30:
				break
			#---------------------------
			#print(j)
			#print(ids)
			data_name.append('data/'+test+'/'+data1_name[i]+'_'+str(ids)+'/life.csv')
			
			try:
				
				data=pd.read_csv(data_name[j],encoding='utf-8')
			except:
				data=pd.read_csv(data_name[j],encoding='ansi')
				print('ooo')
			if cmd == '平均':
				#print(data,np.array(data.loc[:,name]))
				target_data[0:len(data)]=target_data[0:len(data)] + np.array(data.loc[:,name])
			#target_data[0:len(data)]+=np.array(data.loc[:,'評估函數'])#這一次的run把它加入進去到]
			#target_data[0:len(data)]+=np.array(data.loc[:,'最小T剩餘電量'])
			#print(np.array(data.loc[:,'最小T剩餘電量']))
			
			if max_lf<len(data):
				max_lf = max(max_lf,len(data))
				if cmd =='最好':
					target_data[0:len(data)]=np.array(data.loc[:,name])
			if min_lf>len(data):
				min_lf = min(min_lf,len(data))
				print("最差：：：：", j," ", min_lf)
				if cmd=='最差':
					target_data=np.zeros((12000))
					target_data[0:len(data)]=np.array(data.loc[:,name])
			if cmd =='MD' and ii==j:
				target_data=np.zeros((12000))
				target_data[0:len(data)]=np.array(data.loc[:,name])
			
			avg_lf+=len(data)##計算平均壽命
			# print(ids,j,len(data))
			ids+=1
			lf_data[0:len(data)]+=1##計算有到達這個round的實驗次數
		print("avg_lf", avg_lf, "   success_data", success_data)
		# try:
		#     avg_lf=avg_lf/success_data
		# except ZeroDivisionError:
		#     avg_lf = 0

		if success_data == 0:
		    avg_lf = 0
		else:
		    avg_lf = avg_lf / success_data
		print('life:',max_lf,min_lf,avg_lf,success_data)
		analysis['name'].append(labels[i])
		analysis['max_lf'].append(max_lf)
		analysis['min_lf'].append(min_lf)
		analysis['avg_lf'].append(avg_lf)
		avg_lf = int(avg_lf)
		#target_data[:max_lf]/=lf_data[:max_lf]
		if cmd=='平均' and name=='最小T剩餘電量':
			target_data/=success_data
		elif   cmd=='平均':
			target_data[:int(avg_lf)]/=lf_data[:avg_lf]
			target_data[avg_lf:]=0
			#for k in range(max_lf):
			#    if k> 2 and target_data[k-1]< target_data[k]:
			#        target_data[k]=max(2*target_data[k-1]-target_data[k-2],0.001)
		
		x = np.arange(12000)[target_data>0]
		y = target_data[target_data>0]
		save[labels[i]]=target_data
		plt.plot(x[x%1==0],y[x%1==0],label=labels[i],linestyle=linestyles[i],color=colors[i],marker=markers[i],markevery=200)

	td = pd.DataFrame.from_dict(save)
	

	plt.legend(loc=0, numpoints=1)
	plt.grid(linestyle="--")  # 设置背景网格线为虚线
	ax = plt.gca()
	ax.spines['top'].set_visible(False)  # 去掉上边框
	ax.spines['right'].set_visible(False)  # 去掉右边框
	if not os.path.isdir('result/'+test+'/'):
		os.makedirs('result/'+test+'/')
	td.to_csv('result/'+test+'/'+test+name+cmd+'.csv',index=False)
	#plt.savefig('total.png')
	if cmd=='MD':
		plt.savefig('result/'+test+'/MD/'+test+name+cmd+'_'+str(ii)+'.png')
	#plt.show()
	plt.cla()
	anyd = pd.DataFrame.from_dict(analysis)
	anyd.to_csv('result/'+test+'/'+test+'_analysis.csv')##
def phase2(name,cmd):
	dt = pd.read_csv('result/'+test+'/'+test+name+cmd+'.csv')
	data = {}
	for i in dt:
		if i=='Proposed' or i=='JGA' or i=='JPSO' or True:
			data[i]=dt[i] 
		else:
			listd =[dt[i][0]]
			
			for j in range(1,len(dt[i])):
				if dt[i][j]==0:#壽命到了
					break
				else:
					value = listd[len(listd)-1]
					sub = (dt[i][j-1]-dt[i][j])*0.5
					listd.append(value-sub)
					#listd.append(value-sub)
					#if j%2==0:
					value = listd[len(listd)-1]
					sub = (dt[i][j-1]-dt[i][j])*0.
					listd.append(value-sub)
			data[i] = np.zeros(12000)
			data[i][0:len(listd)] = np.array(listd)
		print(len(data[i]))
			
	td = pd.DataFrame.from_dict(data)
	i = 0
	for t in td:
		print(t)
		#plt.plot(td[t])
		x = np.arange(12000)[td[t]>0]
		y = td[t][td[t]>0]
		plt.plot(x[x%1==0],y[x%1==0],label=labels[i],linestyle=linestyles[i],color=colors[i],marker=markers[i],markevery=100)
		i+=1
	plt.legend(loc=0, numpoints=1)
	plt.grid(linestyle="--")  # 设置背景网格线为虚线
	ax = plt.gca()
	ax.spines['top'].set_visible(False)  # 去掉上边框
	ax.spines['right'].set_visible(False)  # 去掉右边框
	ax.set(xlabel='Round (time)',ylabel=y_name[name])

	plt.savefig('result/'+test+'/'+test+name+cmd+'_changed.png',dpi=300)
	plt.show()
	plt.cla()
	lft = []
	for t in td:
		#plt.plot(td[t])
		x = np.arange(12000)[td[t]>0]
		lft.append(len(x))
		#plt.plot(x[x%1==0],y[x%1==0],label=labels[i],linestyle=linestyles[i],color=colors[i],marker=markers[i],markevery=200)
		#i+=1
	xx = range(0,len(lft))
	print(lft)
	barlist = plt.bar(xx,lft,align='center',width=0.5)
	color_data = ['#1FB445','#5DCD7A','#178B35','#16D748','#1FB445',"#12B445","#12B445"]
	for i in range(len(lft)):
		barlist[i].set_color(color_data[i])
	for a,b in zip(xx,lft):
		plt.text(a, b, b, ha='center', va='bottom', fontsize=14, color='red')

	plt.xticks(xx, labels)
	#plt.show()

	#plt.grid(linestyle="--")  # 设置背景网格线为虚线
	ax = plt.gca()
	ax.spines['top'].set_visible(False)  # 去掉上边框
	ax.spines['right'].set_visible(False)  # 去掉右边框
	ax.set(ylabel='Round (time)')
	#

	plt.savefig('result/'+test+'/'+test+'_changed_A.png',dpi=300)
	plt.cla()
	#plt.show()
def main(data,n=30,it=1):
	global test,data_number
	test = data
	#data_number = n
	show()
if __name__ == '__main__':
	if len(sys.argv)>=2:
		main(sys.argv[1])#,int(sys.argv[2]),int(sys.argv[2]))
	else:
		# main('10010025_DS1_1011_新測試資料',8,1)
		# main('10010025_DS1_COMP',8,1)
		# main('100100100_DS2',8,1)
		main('100100100_DS2_15',8,1)
		#main('100100400_DS3',8,1)
		# main('100100400_DS3_COMP',8,1)
		# main('100200100_DS4',8,1)
		# main('100200100_DS4_COMP',8,1)
		# main('100300100_DS5',8,1)
		# main('100300100_DS5_COMP',8,1)
		# main('100100100_DS6',8,1)
		# main('100100100_DS6_COMP',8,1)
		# main('100100100_DS7',8,1)
		# main('100100100_DS7_COMP',8,1)
		#main('200300100_DS8',8,1)
		#main('200300100_DS8_COMP',8,1)
