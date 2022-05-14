'''
# Reference: http://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html
import torch as torch
import torch.optim as optim

import functions.basic_nn as basic

# Custom Packages
import sys
sys.path.append('./utils')
import param

optimizer_SGD = optim.SGD(basic.parameters(), lr=param.learning_rate, momentum=param.momentum)

# Custom Packages
import sys
sys.path.append('../utils')
sys.path.append('./src/functions')

import basic_nn as basic
#import param as param


class SGD(optim.Optimizer):
    def __init__(self):
        super(SGD, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.old = None
        if self.old is None:
            self.old = [torch.zeros_like(p) for p in model.parameters()]
            
    def step(self):
        with torch.no_grad():
            update = lambda old, grad: self.momentum * old + self.learning_rate * grad
            self.old = [update(old, params.grad) for old, params in zip(self.old, self.model.parameters())]
            for parameters, old in zip(self.model.parameters(), self.old):
                parameters -= old   
    
    


#def SGD(): optimizer = torch.optim.SGD(params=basic.parameters(), lr=param.learning_rate, momentum=param.momentum)
'''