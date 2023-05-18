
import numpy as np
from sklearn.neural_network import MLPClassifier
import random
import torch
from scipy.special import erf
import matplotlib.pyplot as plt
from math import*
from scipy.integrate import quad as itg
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

"""# Preprocessing MNIST"""

def preprocess_data(dataset):
    '''
    Taking the pyTorch MNIST dataset, flattening the 28x28 images in a 784 dimensional
    vector, centering the data and rescaling. See Appendix D for more details.
    '''
    n, d_x, d_y = dataset.data.shape
    d = d_x * d_y
    X = torch.clone(dataset.data).view(n,-1).float()
    y = torch.clone(dataset.targets).view(n,).float()
    data, labels = [], []
    # Extract digits and create labels. We only keep 1s and 7s for simplicity
    for k,label in enumerate(y):
        if label in [1]:
            data.append(X[k].numpy())
            labels.append(1)
        elif label in [7]:
            data.append(X[k].numpy())
            labels.append(-1)
            
    data = np.array(data)
    data -= data.mean(axis=0)
    data /= 400
    return np.array(data), np.array(labels)

# Load MNIST 
mnist_trainset= datasets.MNIST(root='data', train=True, 
                               download=True, transform=None)
mnist_testset= datasets.MNIST(root='data', train=False, 
                               download=True, transform=None)
# Pre-process
X_train, y_train = preprocess_data(mnist_trainset)

X_plus = X_train[y_train == 1]       #1s
X_minus = X_train[y_train == -1]     #7s
X_plus=X_plus[:X_minus.shape[0]]     #We balance the clusters for simplicity, by keeping as many 1s as 7s

X_balanced=np.vstack((X_plus,X_minus))



X_test, y_test = preprocess_data(mnist_testset)
X_test_tot=np.vstack((X_test[y_test==1][:1028],X_test[y_test==-1]))  #We do the same for the test set.

mean_plus = np.mean(X_plus, axis=0)   #cluster mean of the 1s
mean_minus = np.mean(X_minus, axis=0) #cluster mean of the 7s
center=.5*mean_plus+.5*mean_minus

X_plus-=np.tile(center,(X_plus.shape[0],1))
X_minus-=np.tile(center,(X_minus.shape[0],1))

X_test_tot-=np.tile(center,(X_test_tot.shape[0],1))
X_balanced-=np.tile(center,(X_balanced.shape[0],1))

"""# Training the autoencoder"""

d=784

class generate_data(Dataset):
  def __init__(self,n,μ,sigma_e=.5):
    #sampling 
    self.n=n
    self.idx=random.sample(range(X_balanced.shape[0]),self.n)
    
    X=torch.from_numpy(X_balanced[self.idx])
    Y=X*np.sqrt(1-sigma_e**2)+torch.randn(n,X.shape[1])*sigma_e
    
    self.X,self.Y=X.to(device),Y.to(device)
    self.μ=μ

    self.sigma_e=sigma_e
    self.samples=n
    

  def __getitem__(self,idx):
    return self.X[idx].to(device),self.Y[idx].to(device)

  def __len__(self):
    return self.samples

class AE_tied(torch.nn.Module):
    def __init__(self, d):
        super(AE_tied, self).__init__()
        
        self.b=torch.nn.Parameter(torch.Tensor([1]))
        self.w=torch.nn.Parameter(torch.randn(d))

    def forward(self, x):
        identity=x
        h=torch.tanh(x@self.w/np.sqrt(d))
        yhat = h.reshape(x.shape[0],1)@self.w.reshape(1,d)/np.sqrt(d)
        yhat+=self.b*identity
        return yhat

class rescale(torch.nn.Module):
    def __init__(self, d):
        super(rescale, self).__init__()    
        self.b=torch.nn.Parameter(torch.Tensor([1]))

    def forward(self, x):
        yhat=self.b*x
        return yhat

def quadloss(ypred, y):
    
    return torch.sum((ypred-y)**2)/2

def train(train_loader, X_test,y_test,μ,σ_e,tol=1e-5,verbose=False,tied=False):  
    if not tied:
      ae=AE(d).to(device)
    else:
      ae=AE_tied(d).to(device)
    optimizer = torch.optim.Adam([{'params': [ae.w],"weight_decay":1e-1},{'params': [ae.b],"weight_decay":0.}],lr=.05)
   
    gen_Loss_list=[]
    q1_=[]
    norm=[]
    skip=[]

    for t in range(2000):
        if t%500==0: print(t)
        for x,y in train_loader:
          
          y_pred = ae(y)

          loss = quadloss(y_pred,x)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        if t%100==0 and t>1400 : 
          
            gen_Loss_list.append(float(torch.mean((X_test-ae(y_test))**2))) 
            
            w1=ae.w.cpu()

            if torch.norm(w1)<1e-12:
              q1=0
            else:
              q1=np.abs(float(w1@μ/torch.norm(w1)))
              
            q1_.append(q1)
            norm.append(float(torch.norm(w1))**2/d)
            skip.append(float(ae.b))
            noise=(x-y).cpu()

    return np.mean(gen_Loss_list[-5:]),np.mean(q1_[-5:]),np.mean(norm[-5:]),np.mean(skip[-5:])

def train_scalar(train_loader, X_test,y_test,μ,tol=1e-5,verbose=False):  
    b=rescale(d).to(device)
    

    optimizer = torch.optim.Adam(b.parameters(), lr=1e-2,weight_decay=.001)
    
    gen_Loss_list=[]
    

    for t in range(2000):
        if t%500==0: print(t)
        for x,y in train_loader:
          
          y_pred = b(y)

          loss = quadloss(y_pred,x)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        if t%100==0: 
          
            gen_Loss_list.append(float(torch.mean((X_test-b(y_test))**2))) 
    
    return np.mean(gen_Loss_list[-2:])

def get_errors(sigmas,N_iter=5, tied=False):
    global X_test
    Eg=None
    
    first=True
    for sig in sigmas:
        
        eg=[]      #test MSE for the full DAE
        b=[]       # test MSE of the rescaling component 
        Q1=[]      #cosine similarity
        skip=[]    #skip connection strength
        norm=[]    #weights squared norm
      

        σ_e=sig    #noise level

        alpha=1    #sample complexity
        n=int(np.round(alpha*d))
        
        for j in range(N_iter):
            μ=torch.from_numpy((mean_plus-mean_minus)/2)

            print("############################################",j)
            id_test=random.sample(range(2056),1000)
            X_test=torch.from_numpy(X_test_tot[id_test]).to(device)
            y_test=X_test*np.sqrt(1-σ_e**2)+σ_e*torch.randn(1000,X_test.shape[1]).to(device)
            X_train=generate_data(n,μ,sigma_e=σ_e)
            train_loader=DataLoader(X_train,batch_size=int(n))
            
            e,q1,nrm,sk=train(train_loader,X_test,y_test,μ,σ_e,tied=tied)
            e_b=train_scalar(train_loader,X_test,y_test,μ)

            eg.append(e)
            b.append(e_b)
            skip.append(sk)
            norm.append(nrm)
            Q1.append(q1)
            
        eg=np.array(eg)
        Q1=np.array(Q1)
        b=np.array(b)
        skip=np.array(skip)
        norm=np.array(norm)
        
        if first:
            first=False
            Eg=eg
            Ov1=Q1
            B=b
            Norm=norm
            Skip=skip
        else:
            Eg=np.vstack((Eg,eg))
            B=np.vstack((B,b))
            Ov1=np.vstack((Ov1,Q1))
            Norm=np.vstack((Norm,norm))
            Skip=np.vstack((Skip,skip))
            
        print("sigma {}  eg{} ".format(sig,Eg[-1]))
    return Eg, B,  Ov1, Norm, Skip

sigmas=np.linspace(0.05,.95,10)

Eg, B,Q1,Norm, Skip=get_errors(sigmas,tied=True)