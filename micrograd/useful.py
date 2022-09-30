import random
import os
import micrograd.nn as nn
import micrograd.engine as engine

def randomList(l,lb,ub):
    out = []
    for i in range(l):
        out.append(random.uniform(lb,ub))
    return out

def sub(A:list,B:list):
    out = []
    for a,b in zip(A,B):
        out.append(a-b)
    return out

def dot(A:list,B:list):
    out = []
    for a,b in zip(A,B):
        out.append(a*b)
    return sum(out)

def mean(A:list):
    return sum1(A)/len(A)

def sum1(A:list):
    out = 0
    for x in A:
        out += x
    return x


