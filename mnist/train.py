from micrograd.nn import MLP, SGD
from micrograd.useful import *
from micrograd.engine import Value
import sys
import random
sys.setrecursionlimit(10000)

from mnist import MNIST
mndata = MNIST("data")
mndata.gz = True
images, labels = mndata.load_training()
temp = list(zip(images,labels))
random.shuffle(temp)
images, labels = zip(*temp)
images, labels = list(images), list(labels)

nn = MLP(784,[16,16,10])
optim = SGD(nn, 0.1)
batch_size = 128

#nn.loadParams("weights/weights_1664544949.74629.txt")

def labelVec(l:int):
    return [1 if x == l else 0 for x in range(10)] #0 except 1 at index of label

i = 1
loss = Value(0)
for image,label in zip(images,labels):
    out = nn(image)
    error = sub(out, labelVec(label))
    loss += dot(error, error)/len(error) #square error and get mean
    if i%batch_size== 0:
        nn.zero_grad()
        loss.backward()
        optim.stepGrads()
        print(f"---loss: {loss.data/batch_size} ---")
        loss = Value(0)
        nn.saveParams("weights")
        #if i%1000 == 0:
        #    optim.lr *= 0.1
    i+=1
