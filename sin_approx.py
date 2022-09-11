from micrograd.nn import MLP,SGD
from micrograd.engine import Value
from micrograd.useful import randomList,sub,dot
from math import sin
import random
import sys

sys.setrecursionlimit(10000)

def smallSin(x):
    return Value(sin(x)/2.5)

inputs = randomList(100)
targets = list(map(smallSin, inputs))
inputs = list(map(lambda x : [Value(x)], inputs))
nn = MLP(1,[32,128,32,1])
optim = SGD(nn,1)
iterations = 100

for i in range(iterations):
    outs = list(map(nn, inputs))
    #print(outs)
    #print(targets)
    diff = sub(outs,targets)
    loss = dot(diff,diff)/len(outs)
    loss.backward()
    optim.lr = 1.0 - 0.9*i/100
    optim.lr = 0.0001
    print(f"loss: {loss.data}")
    print(f"learning rate: {optim.lr}")
    optim.stepGrads()
    if i != (iterations-1):
        nn.zero_grad()

print(f"Test with value of 2.2, model: {nn([2.2]).data}, actual: {smallSin(2.2).data}")
