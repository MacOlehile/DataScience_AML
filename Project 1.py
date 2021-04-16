#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[51]:


n = 100
X = np.random.normal(30,15,n)
Y = 1+3*X 
eps = np.random.normal(0,10,n)
Y_hat = Y + eps
X_new = np.random.uniform(20,5,n)

class LinearRegression:
    
    def __init__(self,X,Y_hat):
        self.X = X
        self.Y_hat = Y_hat
    
    def X_Design(self):
        return np.c_[np.ones(len(self.X)),self.X]
    
    def fit(self):    
        Z = self.X_Design()
        return np.linalg.inv(Z.T.dot(Z)).dot(Z.T).dot(Y_hat)
    
    def mle(self):
        Z = self.X_Design()
        U = self.fit()
        sig = np.transpose(Y-Z.dot(U)).dot(Y-Z.dot(U))
        sigmasqr =  sig*(1/n)
        return np.random.normal(Z.dot(U),sigmasqr)
    
    def predictOLS(self,X_new):
        L = self.fit()
        M = self.X_Design()
        sig = np.transpose(Y-M.dot(L)).dot(Y-M.dot(L))
        sigmasqr =  sig*(1/n)
        predictionsOLS = L[0] + L[1]*X_new + eps
        D = np.c_[np.ones(len(X_new)),X_new]
        predictionMLE = np.random.normal(D.dot(L),sigmasqr)
        return predictionsOLS,predictionMLE
    
    def predictMLE(self,X_new):
        L = self.fit()
        M = self.X_Design()
        sig = np.transpose(Y-M.dot(L)).dot(Y-M.dot(L))
        sigmasqr =  sig*(1/n)
        D = np.c_[np.ones(len(X_new)),X_new]
        predictionMLE = np.random.normal(D.dot(L),sigmasqr)
        return predictionMLE
    
    def visualOLS(self):
        s = self.fit()
        q = print(plt.plot(self.X,Y_hat,'.'),plt.plot(self.X,Y))
        return q
    
    def visualMLE(self):
        t = self.mle()
        r = print(plt.plot(self.X,t,'.'),plt.plot(self.X,Y),plt.plot(self.X,Y - np.std(Y), c = 'y'),
                 plt.plot(self.X,Y + np.std(Y), c = 'y'))
        return r




model = LinearRegression(X,Y_hat)

choice = input("OLS OR MLE METHOD?:")

if choice == 'MLE':
    MLEmodel = model.mle()
    prediction_ = model.predictMLE(X_new)
    visual = model.visualMLE()
    print(visual)
    print(prediction_)
elif choice == 'OLS':
    OLSmodel = model.fit()
    prediction = model.predictOLS(X_new)
    visuals = model.visualOLS()
    print('parameters are:',OLSmodel[0],OLSmodel[1])
    print(visuals)
    print(prediction)
     
    













# In[ ]:




