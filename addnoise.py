import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
import math
import csv
from random import random
from tqdm import tqdm


reader = csv.reader(open("electrode__clean_data/try5/F7.csv", "r"), delimiter=",")
x = list(reader)
F7 = np.array(x).astype("float")


reader = csv.reader(open("electrode__clean_data/try5/AF3.csv", "r"), delimiter=",")
x = list(reader)
AF3 = np.array(x).astype("float")

reader = csv.reader(open("electrode__clean_data/try5/F3.csv", "r"), delimiter=",")
x = list(reader)
F3 = np.array(x).astype("float")

reader = csv.reader(open("electrode__clean_data/try5/FC5.csv", "r"), delimiter=",")
x = list(reader)
FC5 = np.array(x).astype("float")

reader = csv.reader(open("electrode__clean_data/try5/T7.csv", "r"), delimiter=",")
x = list(reader)
T7 = np.array(x).astype("float")

reader = csv.reader(open("electrode__clean_data/try5/P7.csv", "r"), delimiter=",")
x = list(reader)
P7 = np.array(x).astype("float")

reader = csv.reader(open("electrode__clean_data/try5/O1.csv", "r"), delimiter=",")
x = list(reader)
O1 = np.array(x).astype("float")


reader = csv.reader(open("electrode__clean_data/try5/O2.csv", "r"), delimiter=",")
x = list(reader)
O2 = np.array(x).astype("float")

reader = csv.reader(open("electrode__clean_data/try5/P8.csv", "r"), delimiter=",")
x = list(reader)
P8 = np.array(x).astype("float")

reader = csv.reader(open("electrode__clean_data/try5/T8.csv", "r"), delimiter=",")
x = list(reader)
T8 = np.array(x).astype("float")

reader = csv.reader(open("electrode__clean_data/try5/FC6.csv", "r"), delimiter=",")
x = list(reader)
FC6 = np.array(x).astype("float")

reader = csv.reader(open("electrode__clean_data/try5/F4.csv", "r"), delimiter=",")
x = list(reader)
F4 = np.array(x).astype("float")

reader = csv.reader(open("electrode__clean_data/try5/F8.csv", "r"), delimiter=",")
x = list(reader)
F8 = np.array(x).astype("float")

reader = csv.reader(open("electrode__clean_data/try5/AF4.csv", "r"), delimiter=",")
x = list(reader)
AF4 = np.array(x).astype("float")

#------------------------Defining Gaussian Noise------------------------------------------------------------------------------------

'''
reader = csv.reader(open("electrode_trainingdata/trainingdatasai.csv", "r"), delimiter=",")
x = list(reader)
CleanArray= np.array(x).astype("float")
'''

'''

def noise(sigma):
	x = np.linspace(0,1,128)	
	noise=(10*stats.norm.pdf(x, 0.2, sigma))
	return noise
#print(noise.shape)
'''
'''

def add_noise(arr,p,m):   
## arr is the array of 128 samples, p is the random index between (40,80) , m is the random spike of amplitude between (50,80)
	arr=arr.astype(np.float32)
	#ax1=plt.subplot(211)
	#ax1.set_ylim([-50,50])
	#plt.plot(arr)
	arr[p]+=m
	for c in range(0,p/2):
		m=m/2
		arr[p+p/2]+=m
		p=p/2
		arr[p]+=m
	#ax2=plt.subplot(212)
	#ax2.set_ylim([-50,50])
	#plt.plot(arr)
	#plt.show()
	#plt.close()
	return arr
'''


def addnoise_2(arr,p):
	arrnoise=np.zeros(128)
	x = np.linspace(0,1,128)
	noise = 100*stats.norm.pdf(x,p,0.3)
	for i in range(0,128):
		arrnoise[i]=arr[i]+noise[i]
	return(arrnoise)
	
		
F7noise=np.zeros((len(F7),128))


#sigma = np.rand()%0.5
#mean = np.rand()%1.5
	
'''
plt.plot(x,noise)
plt.draw()
plt.pause(1)
plt.close()
'''


noises="gaussian"
#print(len(F7))

for idx in range(0,len(F7)):
	#print(idx)
	p=random()
	#m=random.randint(50,80)
	#print(idx)
	
	F7noise[idx]=addnoise_2(F7[idx],p)
	#F7noise[idx] = [x+p for x in F7noise[idx]]
	print("noise added to F7 in {} row ".format(idx+1))


print("\n{} noise addition completed to F7".format(noises))
	
#print(F7noise.shape)
for i in range(0,10):
	ax1=plt.subplot(2,1,1)
	plt.subplots_adjust(hspace=0.6)	
	plt.plot(F7[i])
	ax1.set_title('Clean data')
	ax1.set_ylim([-280,280])

	ax2=plt.subplot(2,1,2)
	ax2.set_title('Noisy data')
	plt.plot(F7noise[i])
	ax2.set_ylim([-280,280])
	plt.show()


#----------------- Adding Noise to all 14 electrodes----------------------------------------------------------------------------------

'''
for idx in tqdm(range(len(F7))):
	mu =random.random()
	variance=random.randint(1,10)
	sigma = math.sqrt(variance)
  
	if noise_ctf7<(len(F7)):
		noise_ctf7+=1
		F7[idx]=add_noise(F7[idx],mu,sigma)
		F7[idx] = [x+p for x in F7[idx]]
		print("noise added to F7 in {} row ".format(idx))

    
	else:
		print("\n{} noise addition completed to F7".format(noises))


  if noise_ctaf3<(len(AF3)/2):
    noise_ctaf3+=1
    AF3[idx]=add_noise(AF3[idx],mn,sd,noise_type=noises)
    AF3[idx] = [x+p for x in AF3[idx]]
    
  else:
    print("\n{} noise addition completed to AF3".format(noises))

  if noise_ctaf4<(len(AF4)/2):
    noise_ctaf4+=1
    AF4[idx]=add_noise(AF4[idx],mn,sd,noise_type=noises)
    AF4[idx] = [x+p for x in AF4[idx]]
    
  else:
    print("\n{} noise addition completed to AF4".format(noises))

  if noise_ctf3<(len(F3)/2):
    noise_ctf3+=1
    F3[idx]=add_noise(F3[idx],mn,sd,noise_type=noises)
    F3[idx] = [x+p for x in F3[idx]]
  else:
    print("\n{} noise addition completed to F3".format(noises))

  if noise_ctfc5<(len(FC5)/2):
    noise_ctfc5+=1
    FC5[idx]=add_noise(FC5[idx],mn,sd,noise_type=noises)
    FC5[idx] = [x+p for x in FC5[idx]]
    
  else:
    print("\n{} noise addition completed to FC5".format(noises))

  if noise_ctt7<(len(T7)/2):
    noise_ctt7+=1
    T7[idx]=add_noise(T7[idx],mn,sd,noise_type=noises)
    T7[idx] = [x+p for x in T7[idx]]
    
  else:
    print("\n{} noise addition completed to T7".format(noises))

  if noise_ctp7<(len(P7)/2):
    noise_ctp7+=1
    P7[idx]=add_noise(P7[idx],mn,sd,noise_type=noises)
    P7[idx] = [x+p for x in P7[idx]]
    
  else:
    print("\n{} noise addition completed to P7".format(noises))

  if noise_ctt8<(len(T8)/2):
    noise_ctt8+=1
    T8[idx]=add_noise(T8[idx],mn,sd,noise_type=noises)
    T8[idx] = [x+p for x in T8[idx]]
    
  else:
    print("\n{} noise addition completed to T8".format(noises))

  if noise_ctp8<(len(P8)/2):
    noise_ctp8+=1
    P8[idx]=add_noise(P8[idx],mn,sd,noise_type=noises)
    P8[idx] = [x+p for x in P8[idx]]
    
  else:
    print("\n{} noise addition completed to P8".format(noises))

  if noise_cto1<(len(O1)/2):
    noise_cto1+=1
    O1[idx]=add_noise(O1[idx],mn,sd,noise_type=noises)
    O1[idx] = [x+p for x in O1[idx]]
    
  else:
    print("\n{} noise addition completed to O1".format(noises))

  if noise_cto2<(len(O2)/2):
    noise_cto2+=1
    O2[idx]=add_noise(O2[idx],mn,sd,noise_type=noises)
    O2[idx] = [x+p for x in O2[idx]]
    
  else:
    print("\n{} noise addition completed to O2".format(noises))

  if noise_ctfc6<(len(FC6)/2):
    noise_ctfc6+=1
    FC6[idx]=add_noise(FC6[idx],mn,sd,noise_type=noises)
    FC6[idx] = [x+p for x in FC6[idx]]
    
  else:
    print("\n{} noise addition completed to FC6".format(noises))

  if noise_ctf4<(len(F4)/2):
    noise_ctf4+=1
    F4[idx]=add_noise(F4[idx],mn,sd,noise_type=noises)
    F4[idx] = [x+p for x in F4[idx]]
   
  else:
    print("\n{} noise addition completed to F4".format(noises))

  if noise_ctf8<(len(F8)/2):
    noise_ctf8+=1
    F8[idx]=add_noise(F8[idx],mn,sd,noise_type=noises)
    F8[idx]=[x+p for x in F8[idx]]
    
  else:
    print("\n{} noise addition completed to F8".format(noises))
 
'''
#-----------------------------printing 14 electrodes to see noise-------------------------------
'''
for i in range(0,10):
	plt.title('Plotting Row number : '+str(i))
	plt.figure(figsize=(20,12))
	plt.plot(F7[i])
	plt.ylim(-280,280)

	plt.subplot(7,2,2)
	plt.plot(AF3[i])
	plt.ylim(-280,280)

	plt.subplot(7,2,3)
	plt.plot(F3[i])
	plt.ylim(-280,280)

	plt.subplot(7,2,4)
	plt.plot(FC6[i])
	plt.ylim(-280,280)

	plt.subplot(7,2,5)
	plt.plot(FC5[i])
	plt.ylim(-280,280)

	plt.subplot(7,2,6)
	plt.plot(T7[i])
	plt.ylim(-280,280)

	plt.subplot(7,2,7)
	plt.plot(P7[i])
	plt.ylim(-280,280)

	plt.subplot(7,2,8)
	plt.plot(O1[i])
	plt.ylim(-280,280)

	plt.subplot(7,2,9)
	plt.plot(O2[i])
	plt.ylim(-280,280)

	plt.subplot(7,2,10)
	plt.plot(T8[i])
	plt.ylim(-280,280)

	plt.subplot(7,2,11)
	plt.plot(P8[i])
	plt.ylim(-280,280)

	plt.subplot(7,2,12)
	plt.plot(F4[i])
	plt.ylim(-280,280)

	plt.subplot(7,2,13)
	plt.plot(F8[i])
	plt.ylim(-280,280)

	plt.subplot(7,2,14)
	plt.plot(AF4[i])
	plt.ylim(-280,280)

	plt.draw()
	plt.pause(1)
	plt.close()


'''
'''
a,b=(AF3.shape)
print(AF3.shape)
newarr=[[]]
#print(a)
for i in range(0,a):
	newarr=F7
	#print(newarr.shape)	
	newarr=np.hstack((newarr,AF3))
	newarr=np.hstack((newarr,AF4))
	newarr=np.hstack((newarr,F3))
	newarr=np.hstack((newarr,F4))
	newarr=np.hstack((newarr,FC5))
	newarr=np.hstack((newarr,FC6))
	newarr=np.hstack((newarr,O1))
	newarr=np.hstack((newarr,O2))
	newarr=np.hstack((newarr,P7))
	newarr=np.hstack((newarr,T7))
	newarr=np.hstack((newarr,P8))
	newarr=np.hstack((newarr,T8))
	newarr=np.hstack((newarr,F8))	
		
	

with open("electrode_trainingdata/trainingdatatestnoise.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(newarr)

print(newarr.shape)
'''
