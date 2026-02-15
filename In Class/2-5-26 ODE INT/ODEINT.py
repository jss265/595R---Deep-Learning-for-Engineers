import torch
from torch import nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import os


class NeuralODE(nn.Module):

    def __init__(self, nstates, hlayers, width):
        super(NeuralODE, self).__init__()

        layers = []
        layers.append(nn.Linear(nstates, width))
        layers.append(nn.SiLU())

        for _ in range(hlayers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(width, nstates))

        self.network = nn.Sequential(*layers)

    def odefunc(self, t, y):  # output dy/dt
        return self.network(y)

    def forward(self, y0, tsteps):
        return odeint(self.odefunc, y0, tsteps)  # TODO assert size

def train(y_train, t_train, optimizer, lossfn):
    model.train()
    optimizer.zero_grad()
    yhat = model(y0[0, :], t_train)
    loss = lossfn(yhat, y_train)
    loss.backward()
    optimizer.step()

    return loss.item()

        
if __name__ == '__main__':

    data_path = os.path.join(os.path.dirname(__file__), 'odedata.txt')
    data = np.loadtxt(data_path)
    t_train = torch.tensor(data[:, 0], dtype=torch.float64)  # nt
    y_train = torch.tensor(data[:, 1:], dtype=torch.float64)  # nt x 2
    # print(y_train.shape)

    nstates = 2
    hlayers = 3
    width = 24
    model = NeuralODE(nstates, hlayers, width).double()

    y0 = y_train[0, :]
    with torch.no_grad():
        yhat = model(y0, t_train)
    

    plt.figure()
    plt.plot(t_train, yhat[:, 0], 'r--')
    plt.plot(t_train, yhat[:, 1], 'b--')
    plt.plot(t_train, y_train[:, 0])
    plt.plot(t_train, y_train[:, 1])
    plt.xlabel('time')
    plt.ylabel('population')
    plt.show()