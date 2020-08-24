#!/usr/bin/env python
# coding: utf-8

# In[53]:


import sys
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy import fftpack as fft

import pickle

import warnings


# In[ ]:


warnings.simplefilter(action='ignore',category=FutureWarning)


# In[54]:


class Patient():
    def __init__(self,cgm):
        self.cgm = cgm
        
    def preprocess(self):
        
        #drop last column as it has many missing values for all patients
        self.cgm=self.cgm.iloc[:,:30]
        
        #reset the indices
        self.cgm.reset_index(inplace=True,drop=True)
        
        #interpolate the remaining missing values
        self.cgm.interpolate(method='polynomial',order=3,inplace=True)
        self.cgm.bfill(inplace=True)
        self.cgm.ffill(inplace=True)
        self.cgm=self.cgm.astype('float64')
        
    def fft(self,):
        ndarr = fft.rfft(self.cgm, n=5, axis=1)
        df= pd.DataFrame(data=ndarr)
        df.columns=['fft'+str(i) for i in range(1,df.shape[1]+1)]
        return df
        
    def rolling_mean(self,win,olap):
        df=self.cgm.rolling(window=win,axis=1).apply(np.mean).dropna(axis=1).iloc[:,::olap]
        df.columns=['rm'+str(i) for i in range(1,df.shape[1]+1)]
        return df
    
    def kurtosis(self,win,olap):
        df=self.cgm.rolling(window=win,axis=1).apply(kurtosis).dropna(axis=1).iloc[:,::olap]
        df.columns=['kt'+str(i) for i in range(1,df.shape[1]+1) ]
        return df

    def stdev(self,win,olap):
        df=self.cgm.rolling(window=win,axis=1).apply(np.std).dropna(axis=1).iloc[:,::olap]
        df.columns=['st'+str(i) for i in range(1,df.shape[1]+1)]
        return df
    
    def featureMatrix(self):
        self.preprocess()
        df=pd.concat([self.fft(),self.rolling_mean(10,5),self.stdev(10,5),self.kurtosis(10,5)],axis=1)
        return df


# In[55]:


args=sys.argv


# In[56]:


obj=Patient(pd.read_csv(args[1]))


# In[57]:


obj=obj.featureMatrix()


# In[58]:


file=open('pca.pkl','rb')
p=pickle.load(file)
file.close()


# In[59]:


obj=pd.DataFrame(p.transform(obj))


# In[60]:


file=open('model.pkl','rb')
model=pickle.load(file)
file.close()


# In[61]:


predictions=pd.DataFrame(model.predict(obj))


# In[62]:


predictions.to_csv('output.csv',index=False,header=False)

