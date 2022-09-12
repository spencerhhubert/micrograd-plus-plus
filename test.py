from micrograd.nn import MLP,SGD
from micrograd.engine import Value
from micrograd.useful import randomList,sub,dot
from math import sin
import random
import sys

nn = MLP(1,[32,128,32,1])
optim = SGD(nn,1)
optim.lr = 0.001
nn.loadParams("approx_sin_weights.txt")
print(nn([3.25]))
