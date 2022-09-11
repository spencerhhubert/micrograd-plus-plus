import random
from micrograd.engine import Value
import os
import datetime

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        self.nin = nin
        self.nouts = nouts
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def saveParams(self,path:str):
        if not os.path.isfile(path):
            time = datetime.datetime.now().timestamp()
            os.path.join(path,f"weights_{time}")
        with open(path,'w') as file:
            out = f"architecture: {[self.nin]+self.nouts}\n"
            for l in self.layers:
                for n in l.neurons:
                    for w in n.w:
                        out += f"{w.data}\n"
                    out += f"{n.b.data}\n"
            file.write(out)

    def loadParams(self,path:str):
        with open(path,'r') as file:
            arch = file.readline()
            for l in self.layers:
                for n in l.neurons:
                    for w in n.w:
                        w.data = float(file.readline())
                    n.b.data = float(file.readline())

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

#optimizers
class SGD():
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def stepGrads(self):
        for param in self.model.parameters():
            param.data -= self.lr * param.grad


