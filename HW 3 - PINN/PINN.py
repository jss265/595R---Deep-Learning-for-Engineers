import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add parent folder to Python path so sibling modules can be imported
from root import HelpersJSS as jss

# ----------------------------------------

import torch
from torch import nn
import numpy as np
from scipy.stats import qmc  # for Latin Hypercube sampling col_p or random sampling
import matplotlib.pyplot as plt


# NETWORK --------------------------------

class PINN(nn.Module):

    def __init__(self, hlayers, width):
        super(PINN, self).__init__()

        layers = []
        layers.append(nn.Linear(2, width))
        layers.append(nn.Tanh())

        for _ in range(hlayers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, tx: torch.Tensor):
        assert isinstance(tx, torch.Tensor), f"tx must be a torch.Tensor, got {type(tx)}"
        assert tx.ndim == 2, f'tx should have 2 dimensions, got {tx.ndim}'
        assert tx.shape[1] == 2, f'tx should be of shape [batch_size, 2], columns = [t, x], got size {tx.shape}'
        return self.network(tx)
    

# IC/BC/COLLOCATION POINTS ---------------

def collocation_points(N_f, sampler):
    # collocation points
    sample_f = sampler.random(N_f)  # shape [N_f, 2] in range [0, 1]
    t_f = sample_f[:, 0:1]  # t in range [0, 1]
    x_f = sample_f[:, 1:2] * 2 - 1  # x in range [-1, 1]

    # convert to tensor (remember requires grad, so we can backpropogate)
    t_f = torch.tensor(t_f, dtype=torch.float32, requires_grad=True)
    x_f = torch.tensor(x_f, dtype=torch.float32, requires_grad=True)

    # concatenate tx
    tx_f = torch.cat([t_f, x_f], dim=1)

    return tx_f

def boundary_condition_point(N_bc, sampler):
    # boundary condition (x=-1 and x=1, t in [0, 1])
    sample_bc = sampler.random(N_bc)  # shape [N_bc, 2] in range [0, 1]
    t_bc = sample_bc[:, 0:1]  # t in [0, 1]
    x_bc = np.random.choice([-1, 1], size=t_bc.shape)  # x = 1 or -1
    u_bc = np.zeros_like(t_bc)  # zeros like of size t_bc (known value)

    # convert to tensor
    t_bc = torch.tensor(t_bc, dtype=torch.float32)
    x_bc = torch.tensor(x_bc, dtype=torch.float32)
    u_bc = torch.tensor(u_bc, dtype=torch.float32)

    # concatenate tx_bc
    tx_bc = torch.cat([t_bc, x_bc], dim=1)

    return tx_bc, u_bc

def initial_condition_points(N_ic, sampler):
    # initial condition (t=0, x in [-1, 1])
    sample_ic = sampler.random(N_ic)  # shape [N_ic, 2] in range [0, 1]
    t_ic = np.zeros((N_ic, 1))  # zeros of size N_ic
    x_ic = sample_ic[:, 0:1] * 2 - 1  # x in [-1, 1]
    u_ic = -np.sin(np.pi * x_ic)  # Known values at points above

    # convert to tensor
    t_ic = torch.tensor(t_ic, dtype=torch.float32)
    x_ic = torch.tensor(x_ic, dtype=torch.float32)
    u_ic = torch.tensor(u_ic, dtype=torch.float32)

    # concatenate tx_ic
    tx_ic = torch.cat([t_ic, x_ic], dim=1)

    return tx_ic, u_ic


# LOSS -----------------------------------

def physics_loss(model, tx_f):
    u_prediction = model(tx_f)  # solution of the PDE
    grads = torch.autograd.grad(u_prediction, tx_f, grad_outputs=torch.ones_like(u_prediction), create_graph=True)[0]
    u_t = grads[:, 0:1]  # du/dt
    u_x = grads[:, 1:2]  # du/dx

    # only take the d2u/dx2 part (other part is d2u/dxdt)
    u_xx = torch.autograd.grad(u_x, tx_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]  # d2u/dxdx

    # PDE residual
    f = u_t + u_prediction*u_x - (0.01/np.pi)*u_xx
    loss_physics = torch.mean(f**2)
    
    return loss_physics

def boundary_condition_loss(model, tx_bc, u_bc):
    u_prediction = model(tx_bc)
    loss_bc = torch.mean((u_prediction - u_bc)**2)  # MSE
    return loss_bc

def inicial_condition_loss(model, tx_ic, u_ic):
    u_prediction = model(tx_ic)
    loss_ic = torch.mean((u_prediction - u_ic)**2)  # MSE
    return loss_ic

def loss(model, tx_f, tx_bc, tx_ic, u_bc, u_ic, lambda1, lambda2, lambda3) -> torch.Tensor:
    loss_physics = physics_loss(model, tx_f)
    loss_bc = boundary_condition_loss(model, tx_bc, u_bc)
    loss_ic = inicial_condition_loss(model, tx_ic, u_ic)
    
    loss = lambda1*loss_physics + lambda2*loss_bc + lambda3*loss_ic
    return loss


# RUN ------------------------------------

if __name__ == '__main__':


    #  HYPER PARAMETERS ------------------
    training_attempts = 5
    
    N_f = 10_000  # num collocation points
    N_u = 100  # num training points

    N_ic = N_u // 2  # num training points for initial condition
    N_bc = N_u - N_ic  # num training points for boundary condition

    lambda1 = 1
    lambda2 = 1
    lambda3 = 1

    epochs = 5000

    hlayers = 3
    width = 20


    #  IMPLEMENTATION --------------------

    #  run training attempts and save plots
    for attempt in range(training_attempts):
        sampler = qmc.LatinHypercube(d=2)  # 2D: t and x

        # select training IC/BC, and collocation points
        tx_f = collocation_points(N_f, sampler)  # [N_f, 2]
        tx_bc, u_bc = boundary_condition_point(N_bc, sampler)  # [N_bc, 2], [N_bc, 1]
        tx_ic, u_ic = initial_condition_points(N_ic, sampler)  # [N_ic, 2], [N_ic, 1]

        # instatiate the model, optimizer, and losses
        model = PINN(hlayers, width)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []

        # training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            L = loss(model, tx_f, tx_bc, tx_ic, u_bc, u_ic, lambda1, lambda2, lambda3)
            losses.append(L.item())
            L.backward()
            optimizer.step()
            

            if (epoch + 1) % 200 == 0 or epoch == 0:
                lp = physics_loss(model, tx_f).item()
                with torch.no_grad(): # save computation expense
                    lbc = boundary_condition_loss(model, tx_bc, u_bc).item()
                    lic = inicial_condition_loss(model, tx_ic, u_ic).item()
                print(f'Attempt {attempt+1}, Epoch {epoch + 1}, Loss: {L.item():.6f}')
                print(f'    L_ph={lp:.6f}')
                print(f'    L_bc={lbc:.6f}')
                print(f'    L_ic={lic:.6f}\n\n')

        # PLOT PINN PREDICTION RESULTS ----

        # create evaluation grid
        t = np.linspace(0, 1, 100)
        x = np.linspace(-1, 1, 100)
        T, X = np.meshgrid(t, x)

        tx_eval = np.hstack([
            T.reshape(-1, 1),
            X.reshape(-1, 1)
        ])

        tx_eval = torch.tensor(tx_eval, dtype=torch.float32)

        # predict
        with torch.no_grad():
            u_pred = model(tx_eval).numpy()

        U = u_pred.reshape(T.shape)

        # plot PDE
        fig_PDE = plt.figure(figsize=(7,5))
        plt.pcolormesh(T, X, U, shading='auto')
        plt.colorbar(label='u(t,x)')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title(f'PINN Solution Attempt {attempt+1}')
        plt.tight_layout()
        jss.savePicInSequence(fig_PDE, 'HW 3 - PINN/try 1')
        plt.close(fig_PDE)

        # Slice at t = 0.75
        t_slice = 0.75
        x_line = np.linspace(-1, 1, 200)
        tx_line = torch.tensor(np.hstack([t_slice*np.ones((200,1)), x_line.reshape(-1,1)]), dtype=torch.float32)
        with torch.no_grad():
            u_line = model(tx_line).numpy()
        fig_line, ax_line = plt.subplots()
        ax_line.plot(x_line, u_line)
        ax_line.set_xlabel('x')
        ax_line.set_ylabel('u(t=0.75, x)')
        ax_line.set_title(f'PINN Slice at t=0.75, Attempt {attempt+1}')
        jss.savePicInSequence(fig_line, 'HW 3 - PINN/try 1')
        plt.close(fig_line)

        # Training loss vs epochs
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(range(1, epochs+1), losses)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'Training Loss, Attempt {attempt+1}')
        jss.savePicInSequence(fig_loss, 'HW 3 - PINN/try 1')
        plt.close(fig_loss)
