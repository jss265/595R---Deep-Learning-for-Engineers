import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


class MLP(nn.Module):

    def __init__(self, hlayers, width):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(1, width))
        layers.append(nn.Tanh())

        for _ in range(hlayers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, t):
        return self.network(t)
    
def residual(model, t, params): # t is a vector of collocation points size: (ncol, 1)  !!NOTE!! Torch is flipped size expected 
    # unpack parameters
    m, mu, k = params

    # evaluate model
    y = model(t)

    # compute derivatives
    go = torch.ones_like(y)
    dydt = torch.autograd.grad(y, t, grad_outputs=go, create_graph=True)[0] # back propogation of y with respect to t
    d2ydt2 = torch.autograd.grad(dydt, t, grad_outputs=go, create_graph=True)[0] # still need create_graph for backprop

    # compute residual
    return m*d2ydt2 + mu*dydt + k*y

def boundary(model, tbc): # tbc size nbc x 1
    ybc = model(tbc)

    dydt = torch.autograd.grad(ybc, tbc, grad_outputs=torch.ones_like(ybc), create_graph=True)[0]
    
    # Compare against:
    # y(tbc) = 1
    # dydt(tbc) = 0 
    return ybc - 1, dydt
    
def datapoints():
    tdata = torch.zeros(1, 1) # size (1, 1) one initial condition at time=zero

    ncol = 50 # free to choose
    tcol = torch.linspace(0, 1, ncol).reshape(ncol, 1) # arbitrarily decided to use space evenly with linspace. Could chose points anywhere
    ...

def train(tbc, tcoll, params, model, lossfn, optimizer):

    model.train()
    optimizer.zero_grad()

    bc1, bc2 = boundary(model, tbc)
    lossbc1 = torch.mean(bc1**2)
    lossbc2 = torch.mean(bc2**2)

    rcol = residual(model, tcoll, params)
    loss_col = torch.mean(rcol**2)

    lambda1, lambda2 = 1e-1, 1e-4

    ...
    # print(loss)

if __name__ == '__name__':
    m = 1; mu = 4; k = 400
    params = (m, mu, k)
    
    hlayers = 4; width = 32
    model = MLP(hlayers, width)