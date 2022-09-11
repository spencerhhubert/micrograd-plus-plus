import random

def randomList(l):
    out = []
    for i in range(l):
        out.append(random.uniform(-1,1))
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
    return sum(A)/len(A)
