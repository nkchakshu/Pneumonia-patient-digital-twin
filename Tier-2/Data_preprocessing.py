import numpy as np
import os

#This file is for preprocessing of data --> downloaded waveform data using Read_database.py



# Import data


def get_pat_array(pat_id,file_num):
	
	
		file_qual_flag=0
		s=open('check_'+pat_id+'_'+str(file_num)+'.csv','r')

		g=s.readline().split(',')
		eer=["'Elapsed time'","'HR'","'PULSE'","'RESP'","'SpO2'"] 

		hji=np.zeros(11)
		we=0
		for ss in eer:
			if ss in g:
				hji[we]=g.index(ss)
				we+=1
						
			elif ss=='HR' or ss=='PULSE':
				we+=1
				continue
			else:
				file_qual_flag=1
				break	
			
		for ty in g:
			if 'Sys' in ty:
				if 'ART' in ty:
					hji[5]=g.index(ty)
				if 'ABP' in ty:
					hji[5]=g.index(ty)
				if 'NBP' in ty:
					hji[6]=g.index(ty)

			if 'Dias' in ty:
				if 'ART' in ty:
					hji[7]=g.index(ty)
				if 'ABP' in ty:
					hji[7]=g.index(ty)
				if 'NBP' in ty:
					hji[8]=g.index(ty)
				
			if 'Mean' in ty:
				if 'ART' in ty:
					hji[9]=g.index(ty)
				if 'ABP' in ty:
					hji[9]=g.index(ty)
				if 'NBP' in ty:
					hji[10]=g.index(ty)
		
		if file_qual_flag==1:
			return None,False

		hr=np.zeros([100000,8])
		line_count=0
		s.readline()
		for i in range(0,100000):

			line1=s.readline()
			if line1=='':
				break
			line_count+=1
			line=line1.split(',')
			line[-1]=line[-1].strip()
			for ssd in line:
				if ssd=='-':
					ids=line.index(ssd)
					line[ids]=0.0
		#elapsed time:
			hr[i,0]=float(line[0])
			
		#heart rate
			if hji[1]!=0 and hji[2]!=0:
				hr[i,1]=float(line[int(hji[1])])
				hr[i,2]=float(line[int(hji[2])])
			
			#if either HR or PULSE is missing
			elif hji[1]!=0 and hji[2]==0:
				hr[i,1]=float(line[int(hji[1])])
				hr[i,2]=hr[i,1]	

			elif hji[1]==0 and hji[2]!=0:
				hr[i,2]=float(line[int(hji[1])])
				hr[i,1]=hr[i,2]
			
			else:
				print('bad file--no hr or pulse')
				file_qual=1
				break

		
		#Respiration
			hr[i,3]=float(line[int(hji[3])])
			hr[i,4]=float(line[int(hji[4])])

		#Blood pressure
			if (hji[5]!=0 and hji[6]!=0):
				if(hji[5]!=0 and hji[7]!=0 and hji[9]!=0):
					hr[i,5]=float(line[int(hji[5])])
					hr[i,6]=float(line[int(hji[7])])
					hr[i,7]=float(line[int(hji[9])])
			elif (hji[5]==0 and hji[6]!=0):
				if(hji[6]!=0 and hji[8]!=0 and hji[10]!=0):
					hr[i,5]=float(line[int(hji[6])])
					hr[i,6]=float(line[int(hji[8])])
					hr[i,7]=float(line[int(hji[10])])
			elif (hji[5]!=0 and hji[6]==0):
				if(hji[5]!=0 and hji[7]!=0 and hji[9]!=0):
					hr[i,5]=float(line[int(hji[5])])
					hr[i,6]=float(line[int(hji[7])])
					hr[i,7]=float(line[int(hji[9])])

		
		output_array=np.zeros([line_count,8])
		output_array=hr[0:line_count,:]
		output_array,iis=data_treat(output_array,8)
		if iis:
			return None,False
		return output_array,True



#Data preprocessing

def data_treat(inp,no_cols):
	lenn=len(inp[:,0])
	
	if lenn<5:
		return inp,True

	for kk in range(0,no_cols):
		if inp[0,kk]==0.0: #condition before first positive value appears for each parameter
			for g3 in range(1,lenn):
				if inp[g3,kk]!=0.0:
					inp[0,kk]=inp[g3,kk]
					break
		for g2 in range(1,lenn):
			if inp[g2,kk]==0.0:
				inp[g2,kk]=inp[g2-1,kk]

	for kk in range(0,no_cols):
		if inp[int(lenn/2),kk]==0.0:
			print('Wrong -------------')
			return inp,True
			
	return inp,False
		
			

#People with pneumonia

file_pn=open('Use_actual_diagnosis_pn.csv','r')
file_pn.readline()

y=np.zeros(8998)
pn_check=np.zeros(8998)


for jk in range(0,8998):
	ff=file_pn.readline()
	uu=ff.split(',')
	#print(jk)
	y[jk]=int(uu[1])
	pn_check[jk]=int(uu[2])

file_pn.close()



#check if records are available:

qq=open('index_for_use_1.txt','r')
file_checky= ["" for x in range(2274)]
file_num=np.zeros(2274)

qqpos=open('hamid1.txt','r')
hamdid=np.zeros(5691)

for jl in range(0,5691):
	gs=qqpos.readline().split(',')
	hamdid[jl]=gs[0]

for fd in range(0,2274):
	asd=qq.readline().split(',')
	
	file_checky[fd]=asd[0]
	asd[1]=asd[1].strip()
	file_num[fd]=int(asd[1])


#Load subject ids who needed both vent and had pneumonia

file_ventipn=open('Use_people_both_vent_and_pn.csv','r')  #consists of patients who required both ventilator and had pneumonia(extracted from PROCEDURES_ICD.csv)
 
file_ventipn.readline()

venti_check=np.zeros(6483) #list of subjects

for ds in range(0,6483):
	fgg=file_ventipn.readline().split(',')
	venti_check[ds]=int(fgg[2])


g_file=open('ADMISSIONS_redone.csv','r')
g_file.readline()

death_icu=np.zeros([58976,2])
for j in range(0,58976):
	q=g_file.readline().split(',')
	death_icu[j,0]=q[2]
	q[-1]=q[-1].strip()
	death_icu[j,1]=int(q[4])

#for age and gender
file_o_lab=open('ADMISSIONS.csv','r')
file_o_lab.readline()

sub_id=np.zeros(58976)
age1=np.zeros(58976)

for fd in range(0,58976):
	asd=file_o_lab.readline().split(',')
	
	sub_id[fd]=int(asd[1])
	
	aww=asd[3].split('-')

	age1[fd]=int(aww[0])


print(age1)

file_o_pat=open('PATIENTS.csv','r')
file_o_pat.readline()


sub_id2=np.zeros(46520)
deaths=np.zeros(46520)
gender=np.zeros(46520)
age=np.zeros(46520)

for fd in range(0,46520):
	asd=file_o_pat.readline().split(',')
	
	xdc=int(asd[1])
	#print(asd[2])
	if asd[2]=='"F"':
		gender[fd]=-1
	if asd[2]=='"M"':
		gender[fd]=1
	aww=asd[3].split('-')
	res=np.where(sub_id==xdc)
	iid=res[0][0]
	age[fd]=(int(age1[iid])-int(aww[0]))
	asd[-1]=asd[-1].strip()
	deaths[fd]=int(asd[-1])
	sub_id2[fd]=xdc
	

input_wave=[]
input_label=[]
death_label=[]
z_count=0
z_countx=0
agex=[]
genderx=[]
pat_id=[]

for ool in range(0,2274):#2274
	os.chdir('Database_waveform')
	h=file_checky[ool]	
	fi=int(h[2:])
	
	
	
	dee=file_checky.index(h)
	aaa=int(file_num[dee])
	for ff in range(0,aaa):
		
		
		if hamdid[z_count] in pn_check:
			aw,ind=get_pat_array(h,ff)
			
			if ind:
				input_wave.append(aw)
				np.savetxt('data_act/'+str(z_countx)+'.csv', input_wave[z_countx], delimiter=",")
				ssa=np.where(sub_id2==fi)
				pat_id.append(hamdid[z_count])
				idxs=np.where(death_icu[:,0]==hamdid[z_count])
				death_label.append(int(death_icu[int(idxs[0][0]),1]))
				
				agex.append(age[ssa])
				genderx.append(gender[ssa])
				if hamdid[z_count] in venti_check:
					input_label.append(1)
				else:
					input_label.append(0)
				z_count+=1
				z_countx+=1
			else:
				z_count+=1
				continue
		else:
			z_count+=1
			continue

np.savetxt('data_act/label_venti.csv', input_label, delimiter=",")
np.savetxt('data_act/label_age.csv', agex, delimiter=",")
np.savetxt('data_act/label_gender.csv', genderx, delimiter=",")
np.savetxt('data_act/label_death.csv', death_label, delimiter=",")
np.savetxt('data_act/label_pat.csv', pat_id, delimiter=",")

	






