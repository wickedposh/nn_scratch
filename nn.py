import numpy as np
import pandas as pd

##activation function
def sigmoid(x):return 1/(1+np.exp(-x))
def relu(x):return np.maximum(0,x)
def tanh(x):return np.tanh(x)
def dsigmoid(a): return a * (1 - a)
def dtanh(a):    return 1 - a ** 2
def drelu(a):    return (a > 0).astype(float)
acts = {"sigmoid": (sigmoid, dsigmoid),"tanh": (tanh, dtanh),"relu": (relu, drelu),}


def normalize(x):
    return x/np.sum(x)

class NeuralNetwork:
    def __init__(self,input_size,hidden_size,output_size,activation,lr,momentum,threshold):
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.activation,self.actd=acts[activation]
        self.lr=lr
        self.momentum=momentum
        self.threshold=threshold
        self.W=[]
        self.b=[]
        size=[input_size]+hidden_size+[output_size]
        for i in range(len(size)-1):
            self.W.append(np.random.rand(size[i],size[i+1]))
            self.b.append(np.random.rand(size[i+1]))

    def forward(self, training,label):
        x=training.reshape(1,-1)
        y=label.reshape(1,-1)
        activation=self.activation
        self.z=[]
        self.act=[x]
        a=x
        for i in range(len(self.W)):
            zz=a@self.W[i]+self.b[i]
            self.z.append(zz)
            a=activation(zz)
            self.act.append(a)
        err=np.sum((y-a)**2)/2

        return a,err
    def backward(self, training,label):
        dW,db=[None]*len(self.W),[None]*len(self.W)
        x=training
        y=label.reshape(-1,1)
        pred, Err = self.forward(training)
        delta=(pred-y)*self.actd(self.act[-1])

        for i in reversed(range(len(self.W))):
            dW[i] = self.act[i].T @ delta
            db[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self.actd(self.act[i])
        return dW,db
    def train(self,training,label):
        x=training
        y=label.reshape(-1,1)
        vW = [np.zeros_like(w) for w in self.W]
        vb = [np.zeros_like(b) for b in self.b]
        i = 0
        pred, err = self.forward(training)
        while np.sum(err) > self.threshold and i < 10000:
            dW, db = self.backward(training)
            for k in range(len(self.W)):
                vW[k] = self.momentum * vW[k] - self.lr * dW[k]   # momentum update
                vb[k] = self.momentum * vb[k] - self.lr * db[k]
                self.W[k] += vW[k]
                self.b[k] += vb[k]
            pred, err = self.forward(training)
            i += 1
            if i % 100 == 0:
                print(f"step {i}, loss {np.sum(err):.4f}")

data = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
net = NeuralNetwork(input_size=2, hidden_size=[4,4,4], output_size=1,
                      activation="sigmoid", lr=0.5, momentum=0.9, threshold=0.01)



