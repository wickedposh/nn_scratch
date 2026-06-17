#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:47:51 2019

@author: apple
"""

import numpy as np
import pandas as pd
    

###Build 3 hidden layer MLP manually with Backpropagation 
##(GD and mini-batch(when self.num_bn_layers=1 it becomes SGD) with momemtum)
def sigmoid(x):
    return 1/(1+np.exp(-x))
class MLP(): 
    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum, num_bn_layers, threshold):
        self.input_size = input_size
        self.output_size = output_size
        self.hiddens = hiddens
        self.activations = activations
        self.weight_init_fn=weight_init_fn
        self.bias_init_fn=bias_init_fn
        if self.weight_init_fn == None:
            self.W0=self.random_normal_weight_init(self.input_size,self.hiddens[0])
            self.W1=self.random_normal_weight_init(self.hiddens[0],self.hiddens[1])
            self.W2=self.random_normal_weight_init(self.hiddens[1],self.hiddens[2])
            self.W3=self.random_normal_weight_init(self.hiddens[2],self.output_size)
        if self.bias_init_fn==None:
            self.bias0=self.zeros_bias_init(self.hiddens[0])
            self.bias1=self.zeros_bias_init(self.hiddens[1])
            self.bias2=self.zeros_bias_init(self.hiddens[2])
            self.bias3=self.zeros_bias_init(self.output_size)
        self.criterion=criterion
        self.lr=lr
        self.momentum=momentum
        self.num_bn_layers=num_bn_layers
        self.threshold=threshold


    def forward(self, training):
        x=training[:,:-1]
        y=training[:,-1].reshape(-1,1)
        activation=self.activations
        self.z1=np.dot(x, self.W0)+self.bias0
        self.h1=activation(self.z1)
        self.z2=np.dot(self.h1,self.W1)+self.bias1
        self.h2=activation(self.z2)
        self.z3=np.dot(self.h2,self.W2)+self.bias2
        self.h3=activation(self.z3)
        self.z4=np.dot(self.h3, self.W3)+self.bias3
        pred=activation(self.z4)
        Err=sum((y-pred)**2/2)
        return pred, Err
    
    def backward(self, training):
        x=training[:,:-1]
        y=training[:,-1].reshape(-1,1)
        pred, Err = self.forward(training)
        dif = pred-y
        jaz4 = np.zeros((self.output_size, self.output_size))
        dely = dif

        delz4 = dely*pred*(1-pred)
        delh3 = delz4@self.W3.T
        delw3 = self.h3.T @ delz4
        delb3 = delz4.sum(axis=0)
        delz3=delh3*self.h3*(1-self.h3)

        delh2 = delz3@self.W2.T
        delw2 = self.h2.T@delz3
        delb2 = delz3.sum(axis=0)
        

        delz2 = delh2*self.h2*(1-self.h2)
        delh1 = delz2@self.W1.T
        delw1 = self.h1.T@delz2
        delb1 = delz2.sum(axis=0)

        delz1 = delh1*self.h1*(1-self.h1)
        delw0 = x.T@delz1
        delb0 = delz1.sum(axis=0)
        return delw0, delw1, delw2, delw3, delb0, delb1, delb2, delb3     

            
                    
    def train(self, training):
        Err=100
        ##mini-batch training
        if self.num_bn_layers!=None:
            pred, Err = self.forward(training)
            while Err>self.threshold:
                x=pd.dataframe(training)
                x[pred]=pred
                x=np.random.shuffle(x)
                pred=x[:,-1]
                x.drop(columns=['pred'])
                y=x[:,-1]
                a=x.columns.values
                x.drop(columns=[a[-1]])
                delw3=np.zeros((self.hiddens[2], self.output_size))
                delw2=np.zeros((self.hiddens[1],self.hiddens[2]))
                delw1=np.zeros((self.hiddens[0], self.hiddens[1]))
                delw0=np.zeros((self.input_size,self.hiddens[0]))
                
                delw0p=np.zeros((self.input_size, self.hiddens[0]))
                delw1p=np.zeros((self.hiddens[0], self.hiddens[1]))
                delw2p=np.zeros((self.hiddens[1], self.hiddens[2]))
                delw3p=np.zeros((self.hiddens[2], self.output_size))
                delb0p=np.zeros((self.hiddens[0]))
                delb1p=np.zeros((self.hiddens[1]))
                delb2p=np.zeros((self.hiddens[2]))
                delb3p=np.zeros((self.output_size))                
                     
                for l in range(int(x.shape[0]/self.num_bn_layers)):
                    dif = -y[l*self.num_bn_layers,(l-1)*self.num_bn_layers-1]+pred[l*self.num_bn_layers,(l-1)*self.num_bn_layers-1]
                    dely = dif
                    jaz4 = np.zeros((self.output_size, self.output_size))
                    for i in range(len(dely)):
                        for j in range(len(dely)):
                            if i==j:
                                jaz4[i,j]+=pred[i]*(1-pred[i])
                            else:
                                jaz4[i,j]+=pred[i]*pred[j]
                    delz4 = np.dot(dely, jaz4)
                    delh3 = np.dot(self.W3, delz4)
                    delw3 = np.dot(self.h3, delz4)
                    delb3 = delz4
    
                    jaz3 = np.zeros((self.hiddens[2], self.hiddens[2]))
                    for i in range(len(self.hiddens[2])):
                        for j in range(len(self.hiddens[2])):
                            if i==j:
                                jaz3[i,j]+=self.h3[i]*(1-self.h3[i])
                            else:
                                jaz3[i,j]+= -self.h3[i]*self.h3[j]
                    delz3 = np.dot(delh3, jaz3)
                    delh2 = np.dot(self.W2, delz3)
                    delw2 = np.dot(self.h2, delz3)
                    delb2 = delz3
        
                    jaz2 = np.zeros((self.hiddens[1], self.hiddens[1]))
                    for i in range(len(self.hiddens[1])):
                        for j in range(len(self.hiddens[1])):
                            if i==j:
                                jaz2[i,j]+=self.h2[i]*(1-self.h2[i])
                            else:
                                jaz2[i,j]+= -self.h2[i]*self.h2[j]
                    delz2 = np.dot(delh2, jaz2)
                    delh1 = np.dot(self.W1, delz2)
                    delw1 = np.dot(self.h1, delz2)
                    delb1 = delz2   
        
                    jaz1 = np.zeros((self.hiddens[0], self.hiddens[0]))
                    for i in range(len(self.hiddens[0])):
                        for j in range(len(self.hiddens[0])):
                            if i==j:
                                jaz1[i,j]+=self.h1[i]*(1-self.h1[i])
                            else:
                                jaz1[i,j]+= -self.h1[i]*self.h1[j]
                    delz1 = np.dot(delh1, jaz1)
                    delw0 = np.dot(x, delz1)
                    delb0 = delz1


                    self.W0+= -self.lr*delw0 + delw0p*self.momentum
                    self.W1+=self.momentum*delw1p-self.lr*delw1
                    self.W2+=self.momentum*delw2p-self.lr*delw2
                    self.W3+=self.momentum*delw3p-self.lr*delw3
                    self.b0+=self.momentum*delb0p-self.lr*delb0
                    self.b1+=self.momentum*delb1p-self.lr*delb1
                    self.b2+=self.momentum*delb2p-self.lr*delb2
                    self.b3+=self.momentum*delb3p-self.lr*delb3                
                    pred, Err = self.forward(training)
                    
                    delw0p, delw1p, delw2p, delw3p, delb0p, delb1p, delb2p, delb3p = delw0, delw1, delw2, delw3, delb0, delb1, delb2, delb3
                     


        if self.momentum == 0:
            i=0
            while np.sum(Err)>self.threshold and i<=10000:
            
                delw3=np.zeros((self.hiddens[2], self.output_size))
                delw2=np.zeros((self.hiddens[1],self.hiddens[2]))
                delw1=np.zeros((self.hiddens[0], self.hiddens[1]))
                delw0=np.zeros((self.input_size,self.hiddens[0]))
                delw0, delw1, delw2, delw3, delb0, delb1, delb2, delb3 = self.backward(training)
                self.W0-=self.lr*delw0
                self.W1-=self.lr*delw1
                self.W2-=self.lr*delw2
                self.W3-=self.lr*delw3
                self.bias0-=self.lr*delb0
                self.bias1-=self.lr*delb1
                self.bias2-=self.lr*delb2
                self.bias3-=self.lr*delb3
                pred, Err = self.forward(training)
                i+=1
                if i % 100 == 0:
                    print(f"step {i}, loss {np.sum(Err):.4f}")
        else:
            delw0p=np.zeros((self.input_size, self.hiddens[0]))
            delw1p=np.zeros((self.hiddens[0], self.hiddens[1]))
            delw2p=np.zeros((self.hiddens[1], self.hiddens[2]))
            delw3p=np.zeros((self.hiddens[2], self.output_size))
            delb0p=np.zeros((self.hiddens[0]))
            delb1p=np.zeros((self.hiddens[1]))
            delb2p=np.zeros((self.hiddens[2]))
            delb3p=np.zeros((self.output_size))
            
            while np.sum(Err)>self.threshold and i<10000 :
            
                delw3=np.zeros((self.hiddens[2], self.output_size))
                delw2=np.zeros((self.hiddens[1],self.hiddens[2]))
                delw1=np.zeros((self.hiddens[0], self.hiddens[1]))
                delw0=np.zeros((self.input_size,self.hiddens[0]))
                delw0, delw1, delw2, delw3, delb0, delb1, delb2, delb3 = self.backward(training)
                                         

                self.W0+= -self.lr*delw0 + delw0p*self.momentum
                self.W1+=self.momentum*delw1p-self.lr*delw1
                self.W2+=self.momentum*delw2p-self.lr*delw2
                self.W3+=self.momentum*delw3p-self.lr*delw3
                self.bias0+=self.momentum*delb0p-self.lr*delb0
                self.bias1+=self.momentum*delb1p-self.lr*delb1
                self.bias2+=self.momentum*delb2p-self.lr*delb2
                self.bias3+=self.momentum*delb3p-self.lr*delb3
                
                delw0p, delw1p, delw2p, delw3p, delb0p, delb1p, delb2p, delb3p = delw0, delw1, delw2, delw3, delb0, delb1, delb2, delb3
                pred, Err = self.forward(training)
                i+=1
                if i % 100 == 0:
                    print(f"step {i}, loss {np.sum(Err):.4f}")
   
            
    def random_normal_weight_init(self,a,b):
        d=np.random.normal(0,1,(a,b))
        return d
    def zeros_bias_init(self,a):
        b=np.zeros(a)
        return b    
            
data= np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])


m=MLP(input_size=2, output_size=1, hiddens=[4,4,4], activations=sigmoid,weight_init_fn=None,
             bias_init_fn=None,  lr=0.01, momentum=0,criterion=None,threshold=0.01,num_bn_layers=None)

m.train(data)

