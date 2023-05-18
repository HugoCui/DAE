
import numpy as np
from sklearn.neural_network import MLPClassifier
import random
import torch
from scipy.special import erf
import matplotlib.pyplot as plt
from math import*
from scipy.integrate import quad as itg
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

d=500


μ=torch.ones(d)/np.sqrt(d)              #centroid


# As an example, we take the cluster variances to be independetly drwan from the Wishart-Laguerre ensemble
seed1=np.random.RandomState(4)
seed2=np.random.RandomState(456)

F1=seed1.randn(d,700)/np.sqrt(700)
F2=seed2.randn(d,600)/np.sqrt(600)

Σp=F1@F1.T*.1
Σm=F2@F2.T*.1


#Diagonalizing
Dp,Up=np.linalg.eigh(Σp)
Dp=np.maximum(Dp,Dp*0)
Sp=torch.from_numpy((Up@np.diag(Dp**.5)@Up.T)).to(torch.float)#squareroot

Dm,Um=np.linalg.eigh(Σm)
Dm=np.maximum(Dm,Dm*0)
Sm=torch.from_numpy(Um@np.diag(Dm**.5)@Um.T).to(torch.float)

def get_x(n,μ): #Generate the synthetic clean data
    s=np.sign(np.random.randn())
    if s>0:
      S=Sp
    else:
      S=Sm
    x=μ*s+S@torch.randn(d)
    for i in range(n-1):
        s=np.sign(np.random.randn())
        if s>0:
          S=Sp
        else:
          S=Sm
        x=torch.vstack((x,μ*s+S@torch.randn(d)))
    return x


def get_y(n,μ,σ_e):#Generate the synthetic noisy data
    x=get_x(n,μ)
    y=x*np.sqrt(1-σ_e**2)+torch.randn(n,d)*σ_e
    return x.to(device),y.to(device)

class generate_data(Dataset): # Dataset class implementing the Gaussian mixture 
  def __init__(self,n,μ,sigma=.5,sigma_e=.5):
    self.X,self.Y=get_y(n,μ,sigma_e) 
    self.μ=μ
    self.sigma=sigma
    self.sigma_e=sigma_e
    self.samples=n
    

  def __getitem__(self,idx):
    return self.X[idx].to(device),self.Y[idx].to(device)

  def __len__(self):
    return self.samples

def PCA(y,k=1): #PCA simulations; k is the number of principal components kept.
    n,d=y.shape
    Σ=y.T@y/n
    e,v=torch.linalg.eigh(Σ)
    pca=(v[:,-k:]).T
    return pca[-1]

class AE_tied(torch.nn.Module): #DAE
    def __init__(self, d):
        super(AE_tied, self).__init__()
        
        self.b=torch.nn.Parameter(torch.Tensor([1])) #skip connection
        self.w=torch.nn.Parameter(torch.randn(d)) #weights. Here we take p=1

    def forward(self, x):
        identity=x
        h=torch.tanh(x@self.w/np.sqrt(d))
        yhat = h.reshape(x.shape[0],1)@self.w.reshape(1,d)/np.sqrt(d)
        yhat+=self.b*identity
        return yhat

class rescale(torch.nn.Module): #trainable rescaling
    def __init__(self, d):
        super(rescale, self).__init__()    
        self.b=torch.nn.Parameter(torch.Tensor([1]))

    def forward(self, x):
        yhat=self.b*x
        return yhat

def quadloss(ypred, y):  #loss function
    
    return torch.sum((ypred-y)**2)/2

def train(train_loader, X_test,y_test,μ,σ_e,tol=1e-5,verbose=False,tied=False):  
    
    ae=AE_tied(d).to(device)
    
    optimizer = torch.optim.Adam([{'params': [ae.w],"weight_decay":0.1},{'params': [ae.b],"weight_decay":0.}],lr=5e-2)
   
    gen_Loss_list=[]  #test MSEs
    Loss_list=[]      #training MSEs
    q_=[]
    norm=[]
    skip=[]

    for t in range(2000):
  
        for x,y in train_loader:
          
          y_pred = ae(y)
          loss = quadloss(y_pred,x)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        if t%100==0 and t>1400 : 
            Loss_list.append(loss.item())
            gen_Loss_list.append(float(torch.mean((X_test-ae(y_test))**2))) 
            w=ae.w.cpu()
              
            
            if torch.norm(w)<1e-12 :
              q=0
            else:
              q=np.abs(float(w@μ/torch.norm(w)))
              
            
            q_.append(q)
            norm.append(float(torch.norm(w))**2/d)
            skip.append(float(ae.b))

    return np.mean(gen_Loss_list[-5:]),np.mean(q_[-5:]),np.mean(norm[-5:]),np.mean(skip[-5:]),np.mean(Loss_list[-5:])

def train_scalar(train_loader, X_test,y_test,μ,tol=1e-5,verbose=False):  
    b=rescale(d).to(device)
    optimizer = torch.optim.Adam(b.parameters(), lr=1e-2,weight_decay=.001)
    
    gen_Loss_list=[]
    Loss_list=[]
    

    for t in range(2000):
        
        for x,y in train_loader:
          
          y_pred = b(y)

          loss = quadloss(y_pred,x)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        if t%100==0: 
            Loss_list.append(loss.item())
            gen_Loss_list.append(float(torch.mean((X_test-b(y_test))**2))) 
   
    return np.mean(gen_Loss_list[-2:]),np.mean(Loss_list[-2:])

def get_errors(sigmas,N_iter=1, tied=False):
    Eg=None
    
    first=True
    for sig in sigmas:
        print("sigma={} ############################################".format(sig))
        
        eg=[]      #test MSE of the full DAE
        b=[]       #test MSE of the rescaling component
        pca=[]     #PCA MSE
        Q=[]       #cosine similarity
        Qpca=[]    #PCA cosine similarity
        skip=[]    #skip connection strength
        norm=[]    #weights norm
        et=[]      #full DAE train MSE
        etb=[]     #rescaling train MSE

        σ_e=sig    #noise level sqr(Delta)
 
        alpha=1    #sample complexity
        n=int(np.round(alpha*d))    #number of samples
    
        for j in range(N_iter):
            
            X_test,y_test=get_y(1000,μ,σ_e)
            X_train=generate_data(n,μ,sigma_e=σ_e)
            train_loader=DataLoader(X_train,batch_size=int(n))
            
            e,q,nrm,sk,t=train(train_loader,X_test,y_test,μ,σ_e,tied=tied)
            e_b,ebt=train_scalar(train_loader,X_test,y_test,μ)

            eg.append(e)
            et.append(t)
            b.append(e_b)
            etb.append(ebt)
            skip.append(sk)
            norm.append(nrm)

            eigv=PCA(X_train.X,1)
            projected=((y_test@eigv).reshape(1000,1))@(eigv.reshape(1,d))
            qpca=np.abs(float(eigv@μ.to(device)))           
            pca.append(float(torch.mean((projected-X_test)**2)))

            Q.append(q)
            Qpca.append(qpca)

            
        eg=np.array(eg)
        et=np.array(et)
        etb=np.array(etb)
        pca=np.array(pca)
        Q=np.array(Q)
        Qpca=np.array(Qpca)
        b=np.array(b)
        skip=np.array(skip)
        norm=np.array(norm)
        
        if first:
            first=False
            Eg=eg
            Et=et
            Pca=pca
            Ov=Q
            B=b
            Etb=etb
            Ovpca=Qpca
            Norm=norm
            Skip=skip
        else:
            Eg=np.vstack((Eg,eg))
            Et=np.vstack((Et,et))
            Etb=np.vstack((Etb,etb))
            B=np.vstack((B,b))
            Pca=np.vstack((Pca,pca))
            Ov=np.vstack((Ov,Q))
            Ovpca=np.vstack((Ovpca,Qpca))
            Norm=np.vstack((Norm,norm))
            Skip=np.vstack((Skip,skip))
            
    return Eg, B,Pca, Ov, Ovpca, Norm, Skip, Et,Etb

sigmas=np.linspace(0.01,.95,10)

Eg, B,Pca, Q,Qpca,Norm, Skip, Et,Etb=get_errors(sigmas,tied=True)

