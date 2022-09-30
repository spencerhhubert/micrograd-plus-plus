from micrograd.nn import MLP, SGD
from micrograd.useful import *
from micrograd.engine import Value
import sys
sys.setrecursionlimit(10000)

from mnist import MNIST
mndata = MNIST("data")
mndata.gz = True
images, labels = mndata.load_training()

nn = MLP(784,[128,10])
optim = SGD(nn, 0.01)
batch_size = 100

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
        nn.saveParams("weights.txt")
        #if i%100 == 0:
        #    optim.lr += 0.01
    i+=1
