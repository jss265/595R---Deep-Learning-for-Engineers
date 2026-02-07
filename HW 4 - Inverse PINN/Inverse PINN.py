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
    noAttempts = 1
    for _ in range(noAttempts):
        data_path = os.path.join(os.path.dirname(__file__), 'burgers.txt')
        data = np.loadtxt(data_path)
        x_obs = torch.tensor(data[:, 0], dtype=torch.float32).view(-1, 1)
        t_obs = torch.tensor(data[:, 1], dtype=torch.float32).view(-1, 1)
        u_obs = torch.tensor(data[:, 2], dtype=torch.float32).view(-1, 1)
        # define a neural network to train
        model = Inverse_PINN(2, 1, 3, 32)  # (x, t) -> u

        # define training points over the entire domain, for the physics loss
        x = torch.linspace(-1, 1, 20)  # shape (20,)
        t = torch.linspace(0, 1, 20)  # shape (20,)
        x, t = torch.meshgrid(x, t, indexing='ij')  # shape 30x30 each
        x_train = x.reshape(-1, 1).requires_grad_(True)  # flatten for model input
        t_train = t.reshape(-1, 1).requires_grad_(True)  # flatten for model input

        # treat lambda1 and lambda2 as learnable parameters
        lambda1 = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        lambda2 = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        lambdas = []

        # add lambdas to the optimiser
        optimiser = torch.optim.Adam(list(model.parameters())+[lambda1, lambda2], lr=1e-3)
        losses_obs = []
        losses_phys = []
        losses = []

        # train
        epochs = 30001
        for i in range (epochs):
            optimiser.zero_grad()

            # compute each term of the PINN loss function above
            # using the following hyperparameters:
            weight1 = 1e-1

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

            # record lambda values and losses
            lambdas.append([lambda1.item(), lambda2.item()])
            losses_phys.append(loss_phys)
            losses_obs.append(loss_obs)
            losses.append(loss)

            # plot the result as training progresses
            if i % 5000 == 0:
                # Create finer grid for smooth plotting (independent of training grid)
                x_fine = torch.linspace(-1, 1, 100)
                t_fine = torch.linspace(0, 1, 100)
                x_fine, t_fine = torch.meshgrid(x_fine, t_fine, indexing='ij')
                
                u = model(x_fine.reshape(-1, 1), t_fine.reshape(-1, 1)).detach()
                u = u.reshape(100, 100)  # reshape back to 2D grid for plotting
                fig = plt.figure()
                plt.pcolormesh(
                    t_fine.numpy(),
                    x_fine.numpy(),
                    u.numpy(),
                    shading='auto',
                    cmap='rainbow')
                plt.colorbar(label='u(t,x)')
                plt.xlabel('t')
                plt.ylabel('x')
                plt.title(f'Inverse PINN Solution Attempt {i}')
                plt.tight_layout()
                jss.savePicInSequence(fig, 'HW 4 - Inverse PINN/figs')
                plt.close(fig)

                print(f'\n-----Attempt{i}-----')
                print(f'Loss: {loss}')
                print(f'    Physics  Loss: {loss_phys:.6f}')
                print(f'    Observed Loss: {loss_obs:.6f}')

        lambda1_vals = [l[0] for l in lambdas]
        lambda2_vals = [l[1] for l in lambdas]
        iterations = range(len(lambdas))

        fig = plt.figure()
        plt.title(r'$\lambda_1$ and $\lambda_2$')
        plt.plot(iterations, lambda1_vals, label=r'$\lambda_1$', color='b')
        plt.plot(iterations, np.abs(lambda2_vals), label=r'$\lambda_2$ (absolut value)', color='r')
        plt.hlines(1, 0, len(lambdas), colors='b', linestyles='--', label=r'True $\lambda_1$ value')
        plt.hlines(np.abs(-0.01/np.pi), 0, len(lambdas), colors='r', linestyles='--', label=r'True $\lambda_2$ value (absolut value)')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.yscale('log')
        plt.legend()
        jss.savePicInSequence(fig, 'HW 4 - Inverse PINN/figs')

        print(f"\nFinal lambda1: {lambdas[-1][0]:.4f} (true: 1.0000)")
        print(f"Final lambda2: {lambdas[-1][1]:.6f} (true: {-0.01/np.pi:.6f})")

        fig = plt.figure()
        plt.title('Losses v Epochs')
        plt.plot(iterations, [l.detach().numpy() for l in losses], label='Total Weighted Loss')
        plt.plot(iterations, [l.detach().numpy() for l in losses_phys], label='Physical Losses')
        plt.plot(iterations, [l.detach().numpy() for l in losses_obs], label='Observation Losses')
        plt.legend()
        plt.yscale('log')
        jss.savePicInSequence(fig, 'HW 4 - Inverse PINN/figs')