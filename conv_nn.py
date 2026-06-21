from torchvision import datasets, transforms
import numpy as np
from nn import *


train = datasets.MNIST(root='./data', train=True, download=True,
                         transform=transforms.ToTensor())
test  = datasets.MNIST(root='./data', train=False, download=True,
                         transform=transforms.ToTensor())

X_t=train.data.numpy()
y_t=train.targets.numpy()
X_t=X_t.astype(np.float32)
X_small,y_small=X_t[:500],y_t[:500]
acts = {"sigmoid": (sigmoid, dsigmoid),"tanh": (tanh, dtanh),"relu": (relu, drelu),}

def relu(x):    return np.maximum(0,x)
def drelu(a):    return (a > 0).astype(float)
def normalize(x):   return x/np.sum(x)
def padding_(image,padding):
    return np.pad(image,pad_width=((0,0),(padding,padding),(padding,padding)),mode='constant',constant_values=0)
def conv2d(image,kernel,padding,stride):
    if padding>0:
        image=padding_(image,padding)
    out_,in_,K,_=kernel.shape
    H,W=image.shape[1],image.shape[2]
    out=np.zeros((out_,(H-K)//stride+1,(W-K)//stride+1))
    for i in range(out_):
        ox=0
        for x in range(0,H-K+1,stride):
            oy=0
            for y in range(0,W-K+1,stride):
                patch=image[:,x:x+K,y:y+K]
                out[i,ox,oy]=np.sum(patch*kernel[i])
                oy+=1
            ox+=1
    return out

def pool2d(image,kernel,stride,act):
    H,W=image.shape[1],image.shape[2]
    K=kernel.shape[0]
    a,b,c=(H-kernel.shape[0])//stride+1,(W-kernel.shape[1])//stride+1,image.shape[0]
    out=np.zeros((c,a,b))
    for i in range(c):
        ox=0
        for x in range(0,a,stride):
            oy=0
            for y in range(0,b,stride):
                patch = image[i, x:x + K, y:y + K]
                if act=="L2":
                    out[i, ox, oy] = np.linalg.norm(patch)
                elif act=="mean":
                    out[i, ox, oy] = np.mean(patch)
                else:
                    out[i, ox, oy] = np.max(patch)
                oy+=1
            ox+=1
    return out

class CNN:
    def __init__(self,input_size, output_size,stride,activate,lr,momentum,threshold,padding,K,pooling=False):
        self.input_size = input_size
        self.output_size = output_size
        self.stride = stride
        self.threshold = threshold
        self.padding = padding
        self.K = K
        self.W=np.random.randn(output_size,input_size,K,K)*0.1
        self.b=np.random.randn(output_size)
        self.pooling=pooling
        self.momentum=momentum
        self.lr=lr
        self.classifier=None
        self.activation,self.actd=acts[activate]
        self.conv_size=None

    def conv_forward(self,image,y):
        image=conv2d(image,self.W,self.padding,self.stride)
        y=np.eye(10)[y]
        if self.pooling!=False:
            image=pool2d(image,self.W.shape[0],self.stride,self.pooling)
        self.conv_size=image.shape
        image=image.flatten()
        image=self.activation(image)
        if self.classifier==None:
            self.classifier=NeuralNetwork(input_size=len(image),hidden_size=[4,4,2],output_size=10,lr=0.01,momentum=0.9,activation="sigmoid",threshold=0.01)

        logits,err=self.classifier.forward(image,y)
        return err,logits

    def backward(self,training,y):
        err,pred=self.conv_forward(training,y)
        dW,db=[None]*len(self.classifier.W),[None]*len(self.classifier.W)
        delta=(pred-y)*self.classifier.actd(self.classifier.act[-1])
        for i in reversed(range(len(self.classifier.W))):
            dW[i] = self.classifier.act[i].T @ delta
            db[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.classifier.W[i].T) * self.classifier.actd(self.classifier.act[i])
        delta_input = delta @ self.classifier.W[0].T  # propagate through first layer to input
        d0=delta_input.reshape(self.conv_size)
        oo,ii,K,_=self.W.shape
        dF_c=np.zeros_like(self.W)
        for j in range(oo):
            for i in range(ii):
                for k in range(K):
                    for l in range(K):
                        dF_c[j, i, k, l] = np.sum(d0[j] * training[i, k:k + d0.shape[1], l:l + d0.shape[2]])
        db_c=d0.sum(axis=(1,2))
        return dF_c,db_c,dW,db
    def train(self,training,y):
        vF = np.zeros_like(self.W)
        vb_c = np.zeros_like(self.b)
        i = 0
        err, pred = self.conv_forward(training,y)
        while np.sum(err) > self.threshold and i < 10000:
            dF_c,db_c,dW,db = self.backward(training,y)
            vb_c=self.momentum*vb_c-self.lr * db_c
            vF = self.momentum*vF-self.lr * dF_c
            self.b += vb_c
            self.W += vF
            for k in range(len(self.classifier.W)):
                self.classifier.W[k] -= self.lr*dW[k]
                self.classifier.b[k] -= self.lr*db[k]
            err, pred = self.conv_forward(training,y)
            i += 1
            if i % 100 == 0:
                print(f"step {i}, loss {np.sum(err):.4f}")



img = X_small[0].reshape(1, 28, 28)
cnn = CNN(input_size=1, output_size=4, stride=1, activate="sigmoid",
          lr=0.01, momentum=0.9, threshold=0.01, padding=0, K=3)
cnn.train(img, y_small[0])







