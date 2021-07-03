import csv
import numpy
import matplotlib.pyplot as plt
import numpy as np

reader = csv.reader(open("electrode_trainingdata/trainingdata[2]clean.csv", "r"), delimiter=",")
x = list(reader)
CleanArray2= np.array(x).astype("float")

reader = csv.reader(open("electrode_trainingdata/trainingdata[2]noise.csv", "r"), delimiter=",")
x = list(reader)
NoiseArray2= np.array(x).astype("float")

reader = csv.reader(open("electrode_trainingdata/trainingdata[3]clean.csv", "r"), delimiter=",")
x = list(reader)
CleanArray3= np.array(x).astype("float")

reader = csv.reader(open("electrode_trainingdata/trainingdata[3]noise.csv", "r"), delimiter=",")
x = list(reader)
NoiseArray3= np.array(x).astype("float")

reader = csv.reader(open("electrode_trainingdata/trainingdata[4]clean.csv", "r"), delimiter=",")
x = list(reader)
CleanArray4= np.array(x).astype("float")

reader = csv.reader(open("electrode_trainingdata/trainingdata[4]noise.csv", "r"), delimiter=",")
x = list(reader)
NoiseArray4= np.array(x).astype("float")

reader = csv.reader(open("electrode_trainingdata/trainingdata[5]clean.csv", "r"), delimiter=",")
x = list(reader)
CleanArray5= np.array(x).astype("float")

reader = csv.reader(open("electrode_trainingdata/trainingdata[5]noise.csv", "r"), delimiter=",")
x = list(reader)
NoiseArray5= np.array(x).astype("float")

reader = csv.reader(open("electrode_trainingdata/trainingdata[6]clean.csv", "r"), delimiter=",")
x = list(reader)
CleanArray6= np.array(x).astype("float")

reader = csv.reader(open("electrode_trainingdata/trainingdata[6]noise.csv", "r"), delimiter=",")
x = list(reader)
NoiseArray6= np.array(x).astype("float")

reader = csv.reader(open("electrode_trainingdata/trainingdata[7]clean.csv", "r"), delimiter=",")
x = list(reader)
CleanArray7= np.array(x).astype("float")

reader = csv.reader(open("electrode_trainingdata/trainingdata[7]noise.csv", "r"), delimiter=",")
x = list(reader)
NoiseArray7= np.array(x).astype("float")

reader = csv.reader(open("electrode_trainingdata/trainingdata[8]clean.csv", "r"), delimiter=",")
x = list(reader)
CleanArray8= np.array(x).astype("float")

reader = csv.reader(open("electrode_trainingdata/trainingdata[8]noise.csv", "r"), delimiter=",")
x = list(reader)
NoiseArray8= np.array(x).astype("float")

print(CleanArray2.shape)
print(CleanArray3.shape)
print(CleanArray4.shape)
print(CleanArray5.shape)
print(CleanArray6.shape)
print(CleanArray7.shape)
print(CleanArray8.shape)

print(NoiseArray2.shape)
print(NoiseArray3.shape)
print(NoiseArray4.shape)
print(NoiseArray5.shape)
print(NoiseArray6.shape)
print(NoiseArray7.shape)
print(NoiseArray8.shape)

with open("electrode_trainingdata/trainingdatacombclean.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(CleanArray2)
    csvWriter.writerows(CleanArray3)
    csvWriter.writerows(CleanArray4)
    csvWriter.writerows(CleanArray5)
    csvWriter.writerows(CleanArray6)
    csvWriter.writerows(CleanArray7)
    csvWriter.writerows(CleanArray8)


with open("electrode_trainingdata/trainingdatacombnoise.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(NoiseArray2)
    csvWriter.writerows(NoiseArray3)
    csvWriter.writerows(NoiseArray4)
    csvWriter.writerows(NoiseArray5)
    csvWriter.writerows(NoiseArray6)
    csvWriter.writerows(NoiseArray7)
    csvWriter.writerows(NoiseArray8)


reader = csv.reader(open("electrode_trainingdata/trainingdatacombclean.csv", "r"), delimiter=",")
x = list(reader)
combCleanArray= np.array(x).astype("float")

reader = csv.reader(open("electrode_trainingdata/trainingdatacombnoise.csv", "r"), delimiter=",")
x = list(reader)
combNoiseArray= np.array(x).astype("float")


print(combCleanArray.shape)
print(combNoiseArray.shape)

'''

'''
#combCleanArray=[[]]
#combNoiseArray=[[]]

'''
CleanArray.extend(comb1CleanArray)
combCleanArray=CleanArray

NoiseArray.extend(comb1NoiseArray)
combNoiseArray=NoiseArray


#combCleanArray=CleanArray
#combNoiseArray=NoiseArray
if(len(comb1CleanArray[0])==1792):
	combCleanArray=np.vstack((combCleanArray,comb1CleanArray))
	combNoiseArray=np.vstack((combNoiseArray,comb1NoiseArray))


with open("electrode_trainingdata/trainingdatacomb2clean.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(combCleanArray)

with open("electrode_trainingdata/trainingdatacomb2noise.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(combNoiseArray)

		


print(combCleanArray.shape)
print(combNoiseArray.shape)
'''
