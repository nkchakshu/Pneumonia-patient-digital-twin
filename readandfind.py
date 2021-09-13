import numpy as np

#Uses data from eICU Collaborative Research Database v2.0
# Objective: FInd out all entries from 'apacheApsVar.csv'
# Pre-requisites: patients_pneumonia.csv  --- manually filtered data from 'patients.csv' for pneumonia patients.


file1=open('apacheApsVar.csv','r')
k=file1.readline()

file2=open('patients_pneumonia.csv','r')
file2.readline()


y=np.zeros([5789])

for jk in range(0,5789):
	ff=file2.readline()
	uu=ff.split(',')
	y[jk]=int(uu[0])

file2.close()

out=open('Use_apache_var_new_pn.csv','w')	# Entries of apacheApsVar matched with patients_pneumonia
out.write(k)


for u in range(0,171177):	
	tr=file1.readline()
	pp=tr.split(',')
	if int(pp[1]) in y:
		out.write(tr)

file1.close()
out.close()
