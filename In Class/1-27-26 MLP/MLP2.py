import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) # add parent folder to Python path so sibling modules can be imported
from root import HelpersJSS as jss

# ----------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np

df = pd.read_csv('HW 1 - MLP/auto+mpg/auto-mpg.data', sep=r'\s+', header=None, quotechar='"') # sep in place of delim_whitespace=True
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
assert len(column_names) == df.shape[1], f"Expected {df.shape[1]} column names, but got {len(column_names)}"
df.columns = column_names

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
    
def physics(model, t, params): # t is a vector of collocation points size: (ncol, 1)  !!NOTE!! Torch is flipped size expected 
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
    tbc = torch.zeros(1, 1, requires_grad=True) # size (1, 1) one initial condition at time=zero

    ncol = 50 # free to choose
    tcol = torch.linspace(0, 1, ncol, requires_grad=True).reshape(ncol, 1) # arbitrarily decided to use space evenly with linspace. Could chose points anywhere
    
    return tbc, tcol

def train(tbc, tcol, params, model, optimizer): # normally pass in the lossfn but here we will write it ourselves

    # set model to training mode
    model.train()

    #zero out gradients
    optimizer.zero_grad()

    # compute boundary condition loss
    bc1, bc2 = boundary(model, tbc)
    lossbc1 = torch.mean(bc1**2)
    lossbc2 = torch.mean(bc2**2)

    # compute physics loss
    rcol = physics(model, tcol, params)
    losscol = torch.mean(rcol**2)

    # total loss
    lambda1 = 1e-1; lambda2 = 1e-4
    loss = lossbc1 + lambda1*lossbc2 + lambda2*losscol

    # backpropogate
    loss.backward()

    # update parameters
    optimizer.step()

    # return training loss
    return loss.item()

def exact_solution(t, params): # useful to plot against model. Usually we won't have this.
    m, mu, k = params

    delta = mu / (2*m)
    omega0 = np.sqrt(k/m)

    omega = np.sqrt(omega0**2 - delta**2)
    phi = np.arctan(-delta/omega)
    A = 1/(2*np.cos(phi))
    y = np.exp(-delta*t)*2*A*np.cos(phi + omega*t)
    return y
    

# if __name__ == '__name__': # blocked because parent folder added

# setup pararmeters, network, optimizer
m = 1; mu = 4; k = 400
params = (m, mu, k)

# setup data points, netwokr, optimizer
hlayers = 4; width = 32
model = MLP(hlayers, width)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# generate data/evaluation points
tbc, tcol = datapoints()

epochs = 4_000

train_losses = np.zeros(epochs)

for t in range(epochs):
    train_losses[t] = train(tbc, tcol, params, model, optimizer)

    if t % 100 == 0:
        print(train_losses[t])

plt.figure()
plt.plot(range(epochs), train_losses)
plt.xlabel('Epochs')
# plt.yscale('log')
plt.ylabel('Training Loss')

t_test = torch.linspace(0, 1, 200).reshape(-1, 1)
y_hat = model(t_test).detach()
y_exact = exact_solution(t_test, params)

plt.figure()
plt.plot(t_test, y_hat, '--')
plt.plot(t_test, y_exact)
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend(['PINN Prediction', 'Exact Solution'])

plt.show()