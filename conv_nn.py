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
    def __init__(self,input_size, output_size,stride,threshold,padding,K):
        self.input_size = input_size
        self.output_size = output_size
        self.stride = stride
        self.threshold = threshold
        self.padding = padding
        self.K = K
        self.W=np.random.randn(output_size,input_size,K,K)*0.1
        self.b=np.random.randn(output_size)








