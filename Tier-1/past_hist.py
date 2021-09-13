
#Loads history of chronic disease in patients.

jk=open('pastHistory.csv','r')
jk.readline()

ll=open('new_pastHistory.csv','w')
for k in range(0,1149180):
	p=jk.readline()
	lq=p.split(',')

	if ('lymphoma' in p) or ('leukemia' in p) or ('myeloma' in p):
		ll.write(lq[1]+','+str(10)+'\n') 
	if ('Metastases' in p):
		
		ll.write(lq[1]+','+str(9)+'\n') 
	if ('AIDS' in p):
		print('hi')
		ll.write(lq[1]+','+str(17)+'\n')

ll.close() 
		
