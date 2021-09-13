
import numpy as np
from scipy import interp
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import regularizers
from numpy import std,mean
import matplotlib.pyplot as plt
from keras.regularizers import l1, l2, l1_l2

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold



# This code is to predict death based on simple inputs (such breathing rate, heart rate, age etc with ventilator also as input) using an MLP. The idea is to predict death by running the model twice once with ventilator input on and again with ventilator off to see the probability of death. This helps in estimating the severity.



def standardize(x):
	sd=std(x)
	mea=mean(x)
	return mea,sd,(x-mea)/(2*sd)




fo_data=np.loadtxt('new_potassium.csv',delimiter=',') 

past_data=np.loadtxt('new_pastHistory.csv',delimiter=',')

apache=np.loadtxt('apachePatientResult.csv',delimiter=',',usecols=(0,1,7,8),skiprows=1)
print(apache)


jk=open('Use_apache_var_new_pn.csv','r')
						#Since the data is in two files, two files have to be read to create one input array.
ao=open('patients_pneumonia.csv','r')


inp_fields1=np.zeros([5789,2])	#for ao

inp_fields2=np.zeros([5789,20])	#for jk

death1=np.zeros(5789)



fio2=np.zeros(5789)


jk.readline()

ao.readline()



i=0
for l in range(0,5789):
	jj=jk.readline().split(',')
	ff=ao.readline().split(',')


	if int(jj[1])==int(ff[0]) and (float(jj[20])!=-1 and float(jj[21])!=-1): #(float(jj[2])!=1 or float(jj[3])!=1)


		pat_id=jj[1]

	
		if ff[2]=='Male':
			inp_fields1[i,0]=1		#Gender
		elif ff[2]=='Female':
			inp_fields1[i,0]=-1
		

		inp_fields1[i,1]=float(ff[3]) #AGE

		###############################
		if int(ff[15]) ==0:

			death1[i]=1 #Our required death output. 'Death' has been converted as 1 and 'Alive' as 0 in patients_pneumonia.csv
		else:
			death1[i]=0
		################################

			

		ih=np.where(fo_data[:,0]==float(pat_id))
		iq=np.where(past_data[:,0]==float(pat_id))
		


		if float(jj[2])==0:
			inp_fields2[i,0]=-1 #Intubation status   #Set intubation status to 0 when using for Tier 1.
		elif float(jj[2])==1:
			inp_fields2[i,0]=1

		if float(jj[3])==0:
			inp_fields2[i,1]=-1 #Ventilator status 
		elif float(jj[3])==1:
			inp_fields2[i,1]=1

		if float(jj[4])==0:
			inp_fields2[i,2]=-1 #Dialysis status 
		elif float(jj[4])==1:
			inp_fields2[i,2]=1		

		## Glasgow Coma scale

		inp_fields2[i,3]=float(jj[5]) #Eye opening
		inp_fields2[i,4]=float(jj[6]) #Motor response
		inp_fields2[i,5]=float(jj[7]) #Verbal response

		inp_fields2[i,6]=float(jj[10]) #White blood cell count 
	
		inp_fields2[i,7]=float(jj[11]) #Body Temperature in C

		inp_fields2[i,8]=float(jj[12]) #Respiratory rate (breaths per minute)
		
		

		inp_fields2[i,9]=float(jj[14]) #Heart rate

		inp_fields2[i,10]=float(jj[15]) #Mean BP
		
		inp_fields2[i,11]=float(jj[23]) #Glucose levels #check for redundancy

		inp_fields2[i,12]=float(jj[20]) #Partial pressure of oxygen(PaO2)
 
		inp_fields2[i,13]=float(jj[21]) #Partial pressure of carbon dioxide(PaCO2) levels
		
		if float(jj[13])!=-1.0:
			inp_fields2[i,14]=float(jj[13]) #Sodium levels
		else:
			inp_fields2[i,14]=130.0

		inp_fields2[i,15]=float(jj[24])		#bilirubin
		inp_fields2[i,16]=float(jj[22])	#Blood Urea Nitrogen(BUN) levels 

		try: 		
			
			index_past=iq[0][0]
			
			inp_fields2[i,18]=float(past_data[index_past,1])		#chronic disease history
		except:
			
			inp_fields2[i,18]=0.0

		try:
			index_pot=ih[0][0]
			inp_fields2[i,17]=float(fo_data[index_pot,1]) #potassium levels

		except:
			inp_fields2[i,17]=4.5
		
		inp_fields2[i,19]=float(jj[25]) #fiO2


		i+=1

	
jk.close()
ao.close()

meano=np.zeros(22)
stdo=np.zeros(22)



for ii in range(0,2):
	meano[ii],stdo[ii],inp_fields1[:,ii]=standardize(inp_fields1[:,ii])

for ii in range(0,20):
	meano[ii+2],stdo[ii+2],inp_fields2[:,ii]=standardize(inp_fields2[:,ii])



inp_fields=np.zeros([i,22])

inp_fields[:,0:2]=inp_fields1[0:i,:]

inp_fields[:,2:22]=inp_fields2[0:i,:]

death=np.zeros(i)

death=death1[0:i]


model = Sequential()
model.add(Dense(48, input_dim=22, activation='relu'))	
model.add(Dropout(0.5))		
model.add(Dense(16, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


cv = StratifiedKFold(n_splits=10)


X=inp_fields
y=death



fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X, y)):


    model = Sequential()
    model.add(Dense(42, input_dim=22, activation='relu', activity_regularizer=l1(0.0003)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    history=model.fit(X[train], y[train], epochs=100, batch_size=150, verbose=2)







model.save('Tier1_prob_death.h5')

############## AUROC ################

pred = model.predict(inp_fields)


fpr, tpr, _ = roc_curve(death, pred)
roc_auc = auc(fpr, tpr)


lw = 2
plt.plot(fpr, tpr, color='darkorange',linewidth=2, label='Model (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.001])
plt.ylim([0.0, 1.001])

plt.xlabel('False Positive Rate',fontsize=16)
plt.ylabel('True Positive Rate',fontsize=16)
plt.title('RNN model for probability of death',fontsize=16)
plt.rcParams.update({'font.size': 16})
plt.legend(loc="lower right")
plt.show()








