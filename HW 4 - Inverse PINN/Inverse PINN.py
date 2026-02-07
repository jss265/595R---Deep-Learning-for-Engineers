import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add parent folder to Python path so sibling modules can be imported
from root import HelpersJSS as jss
# ----------------------------------------
import torch
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt


class Inverse_PINN(nn.Module):

    def __init__(self, nin, nout, hlayers, width):
        super(Inverse_PINN, self).__init__()

        layers = []
        layers.append(nn.Linear(nin, width))
        layers.append(nn.Tanh())

        for _ in range(hlayers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, nout))

        self.network = nn.Sequential(*layers)

    def forward(self, *args):
        inputs = torch.cat(args, dim=1)
        return self.network(inputs)

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(__file__), 'burgers.txt')
    data = np.loadtxt(data_path)
    x_obs = torch.tensor(data[:, 0], dtype=torch.float32).view(-1, 1)
    t_obs = torch.tensor(data[:, 1], dtype=torch.float32).view(-1, 1)
    u_obs = torch.tensor(data[:, 2], dtype=torch.float32).view(-1, 1)
    # define a neural network to train
    model = Inverse_PINN(2, 1, 3, 32)  # (x, t) -> u

    # define training points over the entire domain, for the physics loss
    x = torch.linspace(0, 1, 30)  # shape (30,)
    t = torch.linspace(-1, 1, 30)  # shape (30,)
    x, t = torch.meshgrid(x, t, indexing='ij')  # shape 30x30 each
    x_train = x.reshape(-1, 1).requires_grad_(True)  # flatten for model input
    t_train = t.reshape(-1, 1).requires_grad_(True)  # flatten for model input

    # treat lambda1 and lambda2 as learnable parameters
    lambda1 = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
    lambda2 = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
    lambdas = []

    # add lambdas to the optimiser
    optimiser = torch.optim.Adam(list(model.parameters())+[lambda1, lambda2], lr=1e-3)

    # train
    for i in range (15001):
        optimiser.zero_grad()

        # compute each term of the PINN loss function above
        # using the following hyperparameters:
        weight1 = 1

        # compute physics loss
        u = model(x_train, t_train)

        u_x = torch.autograd.grad(u, x_train, torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t_train, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_train, torch.ones_like(u), create_graph=True)[0]

        residual = u_t + lambda1*u*u_x + lambda2*u_xx
        loss_phys = torch.mean(residual**2)

        # compute data loss from observed data
        u_pred = model(x_obs, t_obs)

        loss_obs = torch.mean((u_pred - u_obs)**2)

        # backpropogate joint loss, take optimiser step
        loss = loss_phys + weight1*loss_obs
        loss.backward()
        optimiser.step()

        # record lambda values
        lambdas.append([lambda1.item(), lambda2.item()])

        # plot the result as training progresses
        if i % 5000 == 0:
            u = model(x_train, t_train).detach()
            u = u.reshape(30, 30)  # reshape back to 2D grid for plotting
            fig = plt.figure()
            plt.pcolormesh(
                x.numpy(),
                t.numpy(),
                u.numpy(),
                shading='auto')
            plt.colorbar(label='u(t,x)')
            plt.xlabel('t')
            plt.ylabel('x')
            plt.title(f'Inverse PINN Solution Attempt {i}')
            plt.tight_layout()
            jss.savePicInSequence(fig, 'HW 4 - Inverse PINN/figs')
            plt.close(fig)

fig = plt.figure()
plt.title(r'$\lambda_1$ and $\lambda_2$')
plt.plot(lambdas[0], label=r'$\lambda_1$', color='b')
plt.plot(lambdas[1], label=r'$\lambda_2$', color='r')
plt.hlines(1, 0, len(lambdas[0]), colors='b', linestyles='--', label=r'True $\lambda_1$ value')  # these are the true values defined in the homework
plt.hlines(-0.01/np.pi, 0, len(lambdas[1]), colors='r', linestyles='--', label=r'True $\lambda_2$ value')  # these are the true values defined in the homework
jss.savePicInSequence(fig, 'HW 4 - Inverse PINN/figs')

print(f"Final lambda1: {lambdas[-1][0]:.4f} (true: 1.0000)")
print(f"Final lambda2: {lambdas[-1][1]:.6f} (true: {-0.01/np.pi:.6f})")
