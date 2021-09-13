import os
import os.path
import numpy as np



file1=open('Use_actual_diagnosis_pn.csv','r')    #diagnosis_pneumonia.csv has only pneumonia patients-filtered from DIAGNOSIS_ICD.csv  
file1.readline()


usable_dirnames=[]
usable_dirpaths=[]

for u in range(0,8998):			#reading from mimiciii matched subset waveforms for pneumonia patients
	a1=file1.readline().split(',')
	a2=format(int(a1[1]),'05d')
	

	if ('p0'+a2) in usable_dirnames:
		continue
	else:
		usable_dirnames.append('p0'+a2)	
		usable_dirpaths.append('/mimic3wdb/matched/'+('p0'+a2[0])+'/'+('p0'+a2)+'/')
			 

file1.close()

data_path='ADMISSIONS_redone.csv'		# contains only ROW_ID,SUBJECT_ID,HADM_ID,ADMITTIME,HOSPITAL_EXPIRE_FLAG columns from ADMISSIONS.csv
datax = np.genfromtxt(data_path, delimiter=',',names=True,dtype=[int,int,int,'|S10',int])
data = np.loadtxt(data_path, delimiter=',',usecols=(0,1,2),skiprows=1)

print(data)
file_side=open('hamid1.txt','w')

check_actual_number=0
kl=np.zeros(8998)
for w in range(0,len(usable_dirnames)):
	try:
		os.chdir(usable_dirpaths[w])
	except:
		continue
	check_actual_number+=1
	prefixed = [filename for filename in os.listdir('.') if filename.startswith(usable_dirnames[w])]
	ww=0
	places=np.where(data[:,1]==float(usable_dirnames[w][2:]))
	for filenamex in np.sort([f for f in prefixed if f.endswith("n.hea")]):

		d_po=filenamex.split('-')
		d_po1=(d_po[1]+'-'+d_po[2]+'-'+d_po[3])
		kl=0
		print(len(places[0]))
		for o in range(0,len(places[0])):	
			idx=int(places[0][o])
			if (datax[idx][3].decode("utf-8"))==d_po1:
				print(d_po1+',',usable_dirnames[w]+','+str(idx))
				kl=int(data[idx,2])
				break
		filez=open(filenamex,'r')
		filez.readline()
		d=filez.readline().split(' ')
		s=d[0].split('.')
		e=s[0]	
		filez.close()
		os.system('rdsamp -r mimic3wdb/'+e[0:2]+'/'+e[:-1]+'/'+e+' -c -p -v > /Database_waveform/check_'+usable_dirnames[w]+'_'+str(ww)+'.csv')
		file_side.write(str(kl)+','+'\n')
		ww+=1
	
	os.chdir('/')

print(check_actual_number)

























