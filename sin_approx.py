from micrograd.nn import MLP,SGD
from micrograd.engine import Value
from micrograd.useful import randomList,sub,dot
from math import sin
import random
import sys

def smallSin(x):
    return Value(sin(x)/2.5)

nn = MLP(1,[32,128,32,1])
optim = SGD(nn,1)
optim.lr = 0.001
iterations = 1000
nn.loadParams("approx_sin_weights.txt")

for i in range(iterations):
    inputs = randomList(50,-1,9)
    targets = list(map(smallSin, inputs))
    inputs = list(map(lambda x : [Value(x)], inputs))
    outs = list(map(nn, inputs))
    #print(outs)
    #print(targets)
    diff = sub(outs,targets)
    loss = dot(diff,diff)/len(outs)
    loss.backward()
    print(f"loss: {loss.data}")
    print(f"learning rate: {optim.lr}")
    optim.stepGrads()
    if i % 100 == 0:
        optim.lr = optim.lr - 0.1
        optim.lr = 0.001
    if i != (iterations-1):
        nn.zero_grad()

nn.saveParams("approx_sin_weights.txt")

print(f"Test with value of 2.2, model: {nn([2.2]).data}, actual: {smallSin(2.2).data}")
