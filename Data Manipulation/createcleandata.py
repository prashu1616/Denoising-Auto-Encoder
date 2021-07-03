#Creating Dataset for Training Denoising Auto-Encoder
#Data taken on 19/09/2019,Patient: Prashanth Seshadri 
#Author: Prashanth Seshadri

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
import pandas as pd
import keyboard
import csv
#import cv2
#import os

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
# Obtaining Data from all Electrodes from csv File

#file_to_read = input("Enter the File to be read : ")

data1 =pd.read_csv('anunew_2019.01.04_10.49.28.csv')
#counter=data1.COUNTER

AF3=data1.AF3
F7=data1.F7
F3=data1.F3
FC5=data1.FC5
T7=data1.T7
P7=data1.P7
O1=data1.O1
O2=data1.O2
P8=data1.P8
T8=data1.T8
FC6=data1.FC6
F4=data1.F4
F8=data1.F8
AF4=data1.AF4

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------

#print(counter)
time = np.linspace(0,len(AF3)/128,len(AF3))
'''
print(time)
plt.plot(time,F7)
plt.show()
'''

length =len(F7)
#plt.figure(figsize=(20,12))
window=1

fc=40
fc1=1
fs=128.0
w = fc/(fs / 2) # Normalize the frequency
w1 = fc1/(fs / 2)

#print(w)
#print(w1)




# Filtering all EEG electodes


b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, F7)
b, a = signal.butter(5, w1, 'high')
F7 = signal.filtfilt(b, a, output)

b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, AF3)
b, a = signal.butter(5, w1, 'high')
AF3 = signal.filtfilt(b, a, output)


b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, F3)
b, a = signal.butter(5, w1, 'high')
F3 = signal.filtfilt(b, a, output)


b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, FC5)
b, a = signal.butter(5, w1, 'high')
FC5 = signal.filtfilt(b, a, output)

b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, T7)
b, a = signal.butter(5, w1, 'high')
T7 = signal.filtfilt(b, a, output)

b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, P7)
b, a = signal.butter(5, w1, 'high')
P7 = signal.filtfilt(b, a, output)

b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, O1)
b, a = signal.butter(5, w1, 'high')
O1 = signal.filtfilt(b, a, output)

b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, O2)
b, a = signal.butter(5, w1, 'high')
O2 = signal.filtfilt(b, a, output)

b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, T8)
b, a = signal.butter(5, w1, 'high')
T8 = signal.filtfilt(b, a, output)

b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, P8)
b, a = signal.butter(5, w1, 'high')
P8 = signal.filtfilt(b, a, output)

b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, FC6)
b, a = signal.butter(5, w1, 'high')
FC6 = signal.filtfilt(b, a, output)

b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, F4)
b, a = signal.butter(5, w1, 'high')
F4 = signal.filtfilt(b, a, output)

b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, F8)
b, a = signal.butter(5, w1, 'high')
F8 = signal.filtfilt(b, a, output)

b, a = signal.butter(5, w, 'low')
output = signal.filtfilt(b, a, AF4)
b, a = signal.butter(5, w1, 'high')
AF4 = signal.filtfilt(b, a, output)



# Using F7 as the main electrode for Selecting clean data

j=0
key=0    #to determine the segment to add...


#plt.plot()


# Slicing all signals into windows of our choosing

for i in range(0,int(length-(window*128)),window):
	plt.close()	
	key=0#to determine the segment to add...
	s=128*i
	e=128*(i+window)
	n=e-s
	F71=F7[s:e]
	AF31=AF3[s:e]
	F31=F3[s:e]
	FC51=FC5[s:e]
	AF41=AF4[s:e]
	T71=T7[s:e]
	P71=P7[s:e]
	O11=O1[s:e]
	O21=O2[s:e]
	P81=P8[s:e]
	T81=T8[s:e]
	FC61=FC6[s:e]
	F41=F4[s:e]
	F81=F8[s:e]
	#print(F71)
	
	plt.plot(F71)
	plt.ylim(-280,280)
	plt.draw()
	plt.pause(0.3)
	
	print(i)

	print('\n Enter "m" for clean data and "n" for noisy data') 
 
	while True:
		if keyboard.read_key()=='m':
			#print("pressed m")
			key=1
			break
		elif keyboard.read_key()=='n':
			#print("pressed n")
			key=0
			break
		elif keyboard.read_key()=='q':
			key=2
			break
#	key = int(input('Enter 1 for clean data and 2 for noisy data : '))
	if key == 1:
		if j==0:
			F7new = F71			
			AF3new= AF31
			F3new=F31
			FC5new=FC51
			AF4new=AF41
			T7new=T71
			P7new=P71
			O1new=O11
			O2new=O21
			P8new=P81
			T8new=T81
			FC6new=FC61		
			F4new=F41
			F8new=F81
			#print('once')
			j+=1
		else:
			if len(F71)==window*128:

				F7new=np.vstack((F7new,F71))
				AF3new=np.vstack((AF3new,AF31))
				F3new=np.vstack((F3new,F31))
				FC5new=np.vstack((FC5new,FC51))
				T7new=np.vstack((T7new,T71))
				P7new=np.vstack((P7new,P71))
				O1new=np.vstack((O1new,O11))
				O2new=np.vstack((O2new,O21))
				P8new=np.vstack((P8new,P81))
				T8new=np.vstack((T8new,T81))
				FC6new=np.vstack((FC6new,FC61))
				F4new=np.vstack((F4new,F41))
				F8new=np.vstack((F8new,F81))
				AF4new=np.vstack((AF4new,AF41))
			#print('added')
			j+=1
	elif key==2:
		break



print('Loop over \n')
#print(F7new.shape)
print(j)


'''
for i in range(0,j):
	plt.plot(F7new[i])
	plt.show()
	plt.pause(1)


plt.close()
'''


#trainfile= open("trainingdata.csv","w+")
#traindatfile.close()


for i in range(0,j):
	for k in range(0,127):
		#traindatfile=open("trainingdata.csv","a")
		trainfile.write('%s' %F7new[i][k])
		trainfile.write(',')
	trainfile.write('\n')

		
trainfile.close()
'''

'''
with open("electrode__clean_data/try5/F7.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(F7new)


with open("electrode__clean_data/try5/AF3.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(AF3new)


with open("electrode__clean_data/try5/F3.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(F3new)


with open("electrode__clean_data/try5/FC5.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(FC5new)

with open("electrode__clean_data/try5/T7.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(T7new)


with open("electrode__clean_data/try5/P7.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(P7new)


with open("electrode__clean_data/try5/O1.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(O1new)


with open("electrode__clean_data/try5/O2.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(O2new)


with open("electrode__clean_data/try5/T8.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(T8new)


with open("electrode__clean_data/try5/P8.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(P8new)


with open("electrode__clean_data/try5/FC6.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(FC6new)


with open("electrode__clean_data/try5/F4.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(F4new)


with open("electrode__clean_data/try5/F8.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(F8new)	


with open("electrode__clean_data/try5/AF4.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(AF4new)




