from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from numpy import std,mean
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.layers import Merge
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from keras.regularizers import l1, l2, l1_l2

def standardize(x):
	sd=std(x)
	mea=mean(x)

	return sd,mea,(x-mea)/(2*sd)


##################################### Load Data ##############################################

data_path='bloodvals_24.csv'
data = np.genfromtxt(data_path, delimiter=',',missing_values=0.0,filling_values=0.0,usecols=(1,5,6,7,8,9,10,11,12,13,14,16,17,19,20,21,23,24,25,27,28,31,0))


K.tensorflow_backend._get_available_gpus()


input_wave=[]



file1=open('data_act/label_age.csv','r')
file2=open('data_act/label_gender.csv','r')
file3=open('data_act/label_venti.csv','r')
file4=open('data_act/label_death.csv','r')

fileid=open('data_act/label_pat.csv','r')


input_seq=np.zeros([493,100000,8])
input_dis=np.zeros([493,21])
output_val=np.zeros([493,1])
pat1=[]

for i in range(0,493):
	ds=open('data_act/'+str(i)+'.csv','r')


	d1=file1.readline()
	d2=file2.readline()
	d3=file3.readline()
	d4=file4.readline()
	id_real=fileid.readline()

	count=0
	while True:	
		qq1=ds.readline()
		if qq1=='' and count<100000:
			for gi in range(count,100000):
				input_seq[i,count,:]=0.0
			break
		if qq1=='' and count==100000:
			break
		qq=qq1.split(',')
		qq[-1]=qq[-1].strip()

		for t in range(0,8):

			input_seq[i,count,t]=float(qq[t])

		count+=1

	id_real=id_real.strip()
	idx=np.where(data[:,0]==float(id_real))

	if len(idx)>1:
	
		for b in range(1,20):
			ki=data[int(idx[0][0]):int(idx[0][-1]),b]
			input_dis[i,b+1]=np.amin(ki[np.nonzero(ki)])
		pat1.append(data[idx[0][0],21])	
	else:
		for b in range(1,20):
			input_dis[i,b+1]=data[idx[0][0],b]
		pat1.append(data[idx[0][0],21])	

	d1=d1.strip()
	d2=d2.strip()
	input_dis[i,0]=float(d1)	
	input_dis[i,1]=float(d2)
	
	output_val[i,0]=int(d4[0])

	ds.close()

for k in range(0,21):
	_,_,input_dis[:,k]=standardize(input_dis[:,k])

in_wave=np.zeros([493,1000,8])
d_count=0
for jj in range(0,100000):
	if jj%100==0:
		in_wave[:,d_count,:]=input_seq[:,int(jj/100),:]
		d_count+=1




#######################Training#####################################################

cv = StratifiedKFold(n_splits=10)



X=in_wave						#No data split has been given. Split the data as required. In the paper a split of 80% Training and 20% Testing has been used.
y=output_val




for i, (train, test) in enumerate(cv.split(X, y,input_dis)):


	###################RNN Model############################################
	
    left_branch = Sequential()
    left_branch.add(
        keras.layers.CuDNNLSTM(64, input_shape=(None, 8), recurrent_regularizer=l1(0.02), return_sequences=True,
                               activity_regularizer=l1_l2(0.02),
                               kernel_initializer=keras.initializers.glorot_normal(30),
                               recurrent_initializer=keras.initializers.glorot_normal(30)))
    left_branch.add(keras.layers.Dropout(0.3))
    left_branch.add(
        keras.layers.CuDNNLSTM(32, return_sequences=True, recurrent_regularizer=l1(0.02), activity_regularizer=l1(0.02),
                               recurrent_initializer=keras.initializers.glorot_normal(30),
                               kernel_initializer=keras.initializers.glorot_normal(30)))
    left_branch.add(keras.layers.Dropout(0.3))
    left_branch.add(
        keras.layers.CuDNNLSTM(32, return_sequences=False, recurrent_regularizer=l1(0.02),
                               activity_regularizer=l1(0.02),
                               kernel_initializer=keras.initializers.glorot_normal(30),
                               recurrent_initializer=keras.initializers.glorot_normal(30)))
    left_branch.add(keras.layers.Dense(64, activation='relu', kernel_initializer=keras.initializers.glorot_normal(30)))
    left_branch.add(keras.layers.Dropout(0.3))
    left_branch.add(keras.layers.Dense(64, activation='relu', kernel_initializer=keras.initializers.glorot_normal(30)))
    left_branch.add(keras.layers.Dropout(0.3))


    right_branch = Sequential()
    right_branch.add(
        Dense(128, input_dim=21, activation='relu', kernel_initializer=keras.initializers.glorot_normal(30),
              activity_regularizer=l1(0.01)))
    right_branch.add(keras.layers.Dropout(0.3))
    right_branch.add(Dense(128, activation='relu', kernel_initializer=keras.initializers.glorot_normal(30)))
    right_branch.add(keras.layers.Dropout(0.3))
    right_branch.add(Dense(64, activation='relu', kernel_initializer=keras.initializers.glorot_normal(30)))
    right_branch.add(keras.layers.Dropout(0.3))
    right_branch.add(Dense(64, activation='relu', kernel_initializer=keras.initializers.glorot_normal(30)))
    right_branch.add(keras.layers.Dropout(0.3))

    merged = Merge([left_branch, right_branch], mode='concat')

    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(128, activation='relu'))
    final_model.add(keras.layers.Dropout(0.3))
    final_model.add(Dense(64, activation='relu', kernel_initializer=keras.initializers.glorot_normal(30)))
    final_model.add(keras.layers.Dropout(0.3))
    final_model.add(Dense(32, activation='relu', kernel_initializer=keras.initializers.glorot_normal(30)))
    final_model.add(keras.layers.Dropout(0.3))
    final_model.add(Dense(32, activation='relu', kernel_initializer=keras.initializers.glorot_normal(30)))
    final_model.add(keras.layers.Dropout(0.3))
    final_model.add(Dense(32, activation='relu', kernel_initializer=keras.initializers.glorot_normal(30)))
    final_model.add(Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.glorot_normal(30)))

    final_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    final_model.summary()

    print(train)
    history=final_model.fit([X[train],input_dis[train]], y[train],validation_data=([X[test],input_dis[test]], y[test]), epochs=100, batch_size=10, verbose=2)




final_model.save('Tier2_prob_death_wo_mechvent.h5')


################################ AUROC Plot ###################################

pred = final_model.predict([in_wave, input_dis])

print(metrics.roc_auc_score(output_val, pred))

fpr, tpr, _ = roc_curve(output_val, pred)
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






