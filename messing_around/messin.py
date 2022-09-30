from micrograd.nn import MLP,SGD
from micrograd.engine import Value
from micrograd.useful import randomList,sub,dot
from math import sin
import random
import sys

nn = MLP(1,[3,4,3,1])
print(nn.parameters())
path = "weights.txt"
nn.saveParams(path)
mm = MLP(1,[3,4,3,1])
print(mm.parameters())
mm.loadParams(path)
print(mm.parameters())
optim = SGD(nn,1)

exit()
