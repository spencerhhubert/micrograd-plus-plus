from micrograd.nn import MLP, SGD
from micrograd.useful import *
from micrograd.engine import Value

import sys
sys.setrecursionlimit(10000)

import random

from mnist import MNIST

mndata = MNIST("data")
mndata.gz = True

data = mndata.load_testing()

def getRandomData():
    val = int(random.random()*len(data[0]))
    image, label = data[0][val], data[1][val]
    return (image, label)

nn = MLP(784,[32,16,10])
nn.loadParams("weights/weights_1664544822.615671.txt")


def labelVec(l:int):
    return [1 if x == l else 0 for x in range(10)] #0 except 1 at index of label

def getPrediction(outs:list):
    values = list(map(lambda x : x.data, outs))
    high = values[0]
    high_idx = 0
    for val,i in zip(values,range(len(values)-1)):
        if val > high:
            high = val
            high_idx = i
    return range(len(outs))[high_idx]

def testAccuracy():
    how_many_right = 0
    tries = 100
    for i in range(tries):
        image, label = getRandomData()
        prediction = getPrediction(nn(image))
        if prediction == label:
            how_many_right += 1
        print(mndata.display(image))
        print(f"actual: {label}")
        print(f"prediction: {prediction}")
        print("---")
    return how_many_right / tries

image, label = getRandomData()
out = nn(image)
print(testAccuracy())
