import numpy as np
#from scipy import signal
#from keras.datasets import mnist
import matplotlib.pyplot as plt
#from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import csv
import random
from tqdm import tqdm


#train_loader= DataLoader(train_dataset, batch_size=128,shuffle=True,num_signals=1,pin_memory=True)



#-------------------------- Preparing the Dataset for Passing lists into the array ---------------------------------------------------

class noisedDataset(Dataset):
  
  def __init__(self,datasetnoised,datasetclean,transform):
    self.noise=datasetnoised
    self.clean=datasetclean
    self.transform=transform
  
  def __len__(self):
    return len(self.noise)

  def __getitem__(self,idx):
    xNoise=self.noise[idx]
    xClean=self.clean[idx]
    
    if self.transform != None:
      xNoise=self.transform(xNoise)
      xClean=self.transform(xClean)
      
    
    return (xNoise,xClean)

# generate noisy array from noise.csv
# generate clean array from clean.csv

reader = csv.reader(open("electrode_trainingdata/trainingdatacombclean.csv", "r"), delimiter=",")
x = list(reader)
CleanArray= np.array(x).astype('float')

reader = csv.reader(open("electrode_trainingdata/trainingdatacombnoise.csv", "r"), delimiter=",")
x = list(reader)
NoisyArray= np.array(x).astype('float')

reader = csv.reader(open("Testdata.csv", "r"), delimiter=",")
x = list(reader)
NoisyTest= np.array(x).astype('float')

print('segment 1 passed')
#CleanArray =pd.read_csv('electrode_trainingdata/trainingdatacombclean.csv')
#NoisyArray =pd.read_csv('electrode_trainingdata/trainingdatacombnoise.csv')

#print(CleanArray.shape)
#print(NoisyArray.shape)

#tsfms=transforms.Compose([transforms.ToTensor()])
tsfms=None
trainset=noisedDataset(NoisyArray,CleanArray,None)
testset=noisedDataset(NoisyTest,CleanArray,None)

trainloader=DataLoader(trainset,batch_size=16,num_workers=8,shuffle=True)
testloader=DataLoader(testset,batch_size=16,num_workers=8,shuffle=True)


#---------------------------------------------------------------------------------------------------------------------------------
'''

def add_noise(arr,noise_type="gaussian"):
  
  row,col=798,1792
  arr=arr.astype(np.float32)
  
  if noise_type=="gaussian":
    mean=0
    var=10
    sigma=var**.5
    noise=np.random.normal(-10,1,arr.shape)
    noise=noise.reshape(col)
    arr=arr+noise
    return arr




noises="gaussian"
noise_ct=0
traindata=np.zeros((798,1792))



for idx in tqdm(range(len(CleanArray))):
   
  if noise_ct<(len(CleanArray)/2):
    noise_ct+=1
    traindata[idx]=add_noise(CleanArray[idx],noise_type=noises)
    
  else:
    print("\n{} noise addition completed to lists".format(noises))
    noise_ct=0


NoisyArray=traindata


for i in range(0,10):
	plt.ylim(-280,280)
	plt.plot(NoisyArray[i])
	plt.show()
	plt.close()

#print(NoisyArray.shape)
p=0
plarr=0
for i in range(0,1):
	q=max(NoisyArray[i])
	print(q)
	for j in range(0,13):
		t=0
		s=128*j
		e=128*(j+1)
		p=random.randint(128*j,128*(j+1))
		NoisyArray[i,p] = 50+q
		plarr=NoisyArray[i,128*j:i,128*(j+1)]
		plarr=NoisyArray[i]
		print(plarr.shape)
		plarr1=plarr[s:e]
		print(plarr1.shape)
		plt.plot(plarr1)
		plt.ylim(-280,280)	
		plt.show()
		plt.pause(0.3)
		plt.close()
			
#print(NoisyArray.shape)

'''












# ------------ Auto-encoder model ----------------------------------------------------------------------------------


class denoising_model(nn.Module):
  def __init__(self):
    super(denoising_model,self).__init__()
    self.encoder=nn.Sequential(
		  nn.Linear(1792,896),
                  nn.ReLU(True),
		  nn.Linear(896,672),
                  nn.ReLU(True),                  
		  nn.Linear(672,512),
                  nn.ReLU(True),
                  nn.Linear(512,256),
                  nn.ReLU(True),
		  )
    
    self.decoder=nn.Sequential(
		  nn.Linear(256,512),
                  nn.ReLU(True),
                  nn.Linear(512,672),
                  nn.ReLU(True),
                  nn.Linear(672,896),
                  nn.ReLU(True),
                  nn.Linear(896,1792),
                  nn.Hardtanh(min_val=-100,max_val=100),
                  )
    
 
  def forward(self,x):
    x=self.encoder(x)
    x=self.decoder(x)
    
    return x


#------------------------Training the Auto-encoder -----------------------------------------------------------------------------


if torch.cuda.is_available()==True:
  device="cuda:0"
else:
  device ="cpu"


print(device)

epochs=int(input('Enter the Epoch Number: '))
l=len(trainloader)
losslist=list()
epochloss=0
running_loss=0

model=denoising_model()
model.load_state_dict(torch.load('sai/sai.model'))
model.train()

criterion=nn.L1Loss()
#optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.8,weight_decay=1e-5)
optimizer=optim.SGD(model.parameters(),lr=0.001,weight_decay=1e-5)

'''
model=denoising_model()
model=torch.load('sai/sai.model.pth')
'''


#def RMSELoss(yhat,y):
    #return torch.sqrt(torch.mean((yhat-y)**2))



'''
#criterion = RMSELoss
#model=denoising_model().to(device)
'''
'''

for epoch in range(epochs):
  
  print("Entering Epoch: ",epoch)
  for noisy,clean in tqdm((trainloader)):
    
    noisy=noisy.view(noisy.size(0),-1).type(torch.FloatTensor)
    clean=clean.view(clean.size(0),-1).type(torch.FloatTensor)
    noisy,clean=noisy.to(device),clean.to(device)
    #-----------------Forward Pass----------------------
    output=model(noisy)
    loss=criterion(output,clean)
    #-----------------Backward Pass---------------------
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    running_loss+=loss.item()
    epochloss+=loss.item()
 #-----------------Log-------------------------------
  #running_loss=running_loss**(0.5)
  print(output)
  print(noisy)
  print('\n running loss is {}'.format(running_loss/l)) 	
  losslist.append(running_loss/l)
  running_loss=0
  print("======> epoch: {}/{},lossitem : {} ".format(epoch,epochs,loss.item()))
  


torch.save(model.state_dict(),'sai/sai.model')

#torch.save(model,'sai/sai.model')



plt.plot(range(len(losslist)),losslist)
plt.xlabel('no of epochs')
plt.ylabel(' Loss Value')
plt.show()
plt.close()
'''
print(" Training Done\n ")

print(" now testing phase\n ")

#a,b=NoisyTest.shape
#model.test()
#f,axes= plt.subplots(6,2,figsize=(20,20))
#axes[0,1].set_title("Dirty Image")
#axes[0,1].set_title("Cleaned Image")
#print(testloader.shape)
#print(" segment 2 passed" )


test_imgs=np.random.randint(0,3280,size=6)
output=[]
for idx in range((6)):
	print(idx)
        dirty=testset[test_imgs[idx]][0]
        #clean=testset[test_imgs[idx]][1]
	#dirty=dirty.view(dirty.size(0),-1).type(torch.FloatTensor)
	dirty=np.array(dirty).astype("float")
	dirty=torch.Tensor(dirty)	
	dirty=dirty.to(device)
	output=np.array(output).astype("float")
	output=model(dirty)
	
	
	
	#dirtyarr=noisy.numpy()
	#denoisedarr=output.numpy()
	output=output.view(1792)
	#output=output.permute(1,2,0).squeeze(2)
	output=output.detach().cpu().numpy()

	print(output)
	
	dirty=dirty.view(1792)
	#dirty=dirty.permute(1,2,0).squeeze(2)
	dirty=dirty.detach().cpu().numpy()
	plt.subplot(211)
	plt.ylim(-200,200)
	plt.plot(dirty)
	plt.title('dirty')
	
	plt.subplot(212)
	plt.plot(output)
	plt.ylim(-100,100)
	plt.title('clean')
	plt.subplots_adjust(hspace=0.6)
	plt.show()

  #clean=clean.permute(1,2,0).squeeze(2)
  #clean=clean.detach().cpu().numpy()
	#axes[idx,0].imshow(dirty)
	#axes[idx,1].imshow(output)



