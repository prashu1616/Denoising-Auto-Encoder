import csv
import numpy
import matplotlib.pyplot as plt
import numpy as np

#plt.figure(figsize=(20,12))

#datafile = open('electrode__clean_data/F7.csv', 'r')
'''
reader = csv.reader(open("EEGTestData/allEEGF.csv", "r"), delimiter=",")
x = list(reader)
ALLEEG = numpy.array(x).astype("float")

'''
reader = csv.reader(open("EEGTestData/F7.csv", "r"), delimiter=",")
x = list(reader)
F7 = numpy.array(x).astype("float")


reader = csv.reader(open("EEGTestData/AF3.csv", "r"), delimiter=",")
x = list(reader)
AF3 = numpy.array(x).astype("float")

reader = csv.reader(open("EEGTestData/F3.csv", "r"), delimiter=",")
x = list(reader)
F3 = numpy.array(x).astype("float")

reader = csv.reader(open("EEGTestData/FC5.csv", "r"), delimiter=",")
x = list(reader)
FC5 = numpy.array(x).astype("float")

reader = csv.reader(open("EEGTestData/T7.csv", "r"), delimiter=",")
x = list(reader)
T7 = numpy.array(x).astype("float")

reader = csv.reader(open("EEGTestData/P7.csv", "r"), delimiter=",")
x = list(reader)
P7 = numpy.array(x).astype("float")

reader = csv.reader(open("EEGTestData/O1.csv", "r"), delimiter=",")
x = list(reader)
O1 = numpy.array(x).astype("float")


reader = csv.reader(open("EEGTestData/O2.csv", "r"), delimiter=",")
x = list(reader)
O2 = numpy.array(x).astype("float")

reader = csv.reader(open("EEGTestData/P8.csv", "r"), delimiter=",")
x = list(reader)
P8 = numpy.array(x).astype("float")

reader = csv.reader(open("EEGTestData/T8.csv", "r"), delimiter=",")
x = list(reader)
T8 = numpy.array(x).astype("float")

reader = csv.reader(open("EEGTestData/FC6.csv", "r"), delimiter=",")
x = list(reader)
FC6 = numpy.array(x).astype("float")

reader = csv.reader(open("EEGTestData/F4.csv", "r"), delimiter=",")
x = list(reader)
F4 = numpy.array(x).astype("float")

reader = csv.reader(open("EEGTestData/F8.csv", "r"), delimiter=",")
x = list(reader)
F8 = numpy.array(x).astype("float")

reader = csv.reader(open("EEGTestData/AF4.csv", "r"), delimiter=",")
x = list(reader)
AF4 = numpy.array(x).astype("float")

newarr=[[]]

a,b=(AF3.shape)

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
		
	

with open("Testdata.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(newarr)

print(newarr.shape)

'''

data=[]
count=0
for row in datareader:
	data.append(row)
	count+=1
#data.astype(float)

'''
'''
for i in range(0,AF3.shape[0]):
	plt.title('Plotting Row number : ',i)
	plt.figure(figsize=(20,12))
	plt.subplot(7,2,1)
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
#print(count)
'''	
