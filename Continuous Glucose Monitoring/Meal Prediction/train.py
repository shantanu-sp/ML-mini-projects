#!/usr/bin/env python
# coding: utf-8

# In[41]:


import sys
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy import fftpack as fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

import pickle
import warnings


# In[ ]:


warnings.simplefilter(action='ignore',category=FutureWarning)


# In[42]:


class Patient():
    def __init__(self,cgm):
        self.cgm = cgm
        
    def preprocess(self):
        #drop rows with 30% of values missing
        self.cgm=self.cgm.loc[self.cgm.isnull().mean(axis=1)<0.3,:]
        
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


# In[ ]:


args=sys.argv


# In[ ]:


p1m=Patient(pd.read_csv(args[1]))
p1nm=Patient(pd.read_csv(args[2]))

p2m=Patient(pd.read_csv(args[3]))
p2nm=Patient(pd.read_csv(args[4]))

p3m=Patient(pd.read_csv(args[5]))
p3nm=Patient(pd.read_csv(args[6]))

p4m=Patient(pd.read_csv(args[7]))
p4nm=Patient(pd.read_csv(args[8]))

p5m=Patient(pd.read_csv(args[9]))
p5nm=Patient(pd.read_csv(args[10]))


# In[44]:


p1m=p1m.featureMatrix()
p1nm=p1nm.featureMatrix()

p2m=p2m.featureMatrix()
p2nm=p2nm.featureMatrix()

p3m=p3m.featureMatrix()
p3nm=p3nm.featureMatrix()

p4m=p4m.featureMatrix()
p4nm=p4nm.featureMatrix()

p5m=p5m.featureMatrix()
p5nm=p5nm.featureMatrix()


# In[45]:


alldata=p1m.append([p1nm,p2m,p2nm,p3m,p3nm,p4m,p4nm,p5m,p5nm])
mdata=p1m.append([p2m,p3m,p4m,p5m])
nmdata=p1nm.append([p2nm,p3nm,p4nm,p5nm])


# In[46]:


stdscaler = StandardScaler()
mat = stdscaler.fit_transform(alldata)
p = PCA(n_components=5)
p.fit(mat)

filename = open('pca.pkl','wb')
pickle.dump(p,filename)
filename.close()

mdata=pd.DataFrame(p.transform(mdata))
nmdata=pd.DataFrame(p.transform(nmdata))

mdata['label'] = 1
nmdata['label'] = 0

alldata=mdata.append(nmdata)


# In[47]:


data=alldata.iloc[:,:5]
labels=alldata.iloc[:,5]


# In[48]:


model=MLPClassifier(hidden_layer_sizes=(100,60),learning_rate='adaptive',random_state=7)
results=[]
kfold=StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
cv_results=cross_val_score(model,data,labels,cv=kfold,scoring='f1')
results.append(cv_results)
print("f1 score for cross validation: ",cv_results.mean())


# In[49]:


model.fit(data,labels)


# In[50]:


filename=open("model.pkl", 'wb')
pickle.dump(model,filename)
filename.close()


# In[52]:


print(filename)

