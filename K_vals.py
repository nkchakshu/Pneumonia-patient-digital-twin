hm=open('new_lab.csv','r')# new_lab.csv are the potassium records from lab.csv
lo=open('new_potassium.csv','w')
hj=0
val=0
count=0
for j in range(0,1493259):
	k=hm.readline().split(',')
	if k=='':
		break
	
	hl=int(k[1])
	print(count)
	if hl==hj:
		try:
			if val>float(k[5]):

				
					val=float(k[5])

		except:


			count+=1
			continue		
	else:
		lo.write(str(hj)+','+str(val)+'\n')
		hj=hl
		try:
			val=float(k[5])
		except:
			val=4.5
	count+=1

hm.close()
lo.close()
