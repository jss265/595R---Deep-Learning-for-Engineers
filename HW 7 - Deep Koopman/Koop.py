import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# ------- Hyper Parameters -------
batch_size = 256
autoencoder_layers = []  # list encoder layers only, latent dimension is last element of list
k_init_std = 0.01
activation_function = nn.ELU
loss_function = nn.MSELoss

# ------- Data Prep -------
ntraj = 2148  # number of trajectories
nt = 50  # number of time steps
ny = 7  # number of states

tvec = np.linspace(0, 350, nt)
Y = np.loadtxt('HW 7 - Deep Koopman/kdata.txt').reshape(ntraj, nt, ny)
Ytrain = Y[:2048, :, :]  # 2048 training trajectories
Ytest = Y[2048:, :, :]  # 100 testing trajectoreis
print(f'Loaded data:\n    Ytrain size: {Ytrain.shape}\n    Ytest shape {Ytest.shape}')
print('    Shape: trajectories (trial runs), time steps, states')

Ytrain_torch = torch.tensor(Ytrain, dtype=torch.float32)
Ytest_torch = torch.tensor(Ytest, dtype=torch.float32)

# Creating Vectorized pairs (y_k, y_k+1)
X = Ytrain_torch[:, :-1, :].reshape(-1, ny)
X_next = Ytrain_torch[:, 1:, :].reshape(-1, ny)
print(f'Created Tensors size:\n    X: {X.shape}\n    X_next: {X_next.shape}')

dataset = TensorDataset(X, X_next)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------- Net Creations -------
class Encoder(nn.Module):
        def __init__(self, layer_list, activation):
            super().__init__()
            
            layers = []
            for i in range(len(layer_list) - 1):
                layers.append(nn.Linear(layer_list[i], layer_list[i+1]))
                if i < len(layer_list) - 2:
                    layers.append(activation())
            self.encoder = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, layer_list, activation):
        super().__init__()

        layers = []
        for i in range(len(layer_list) - 1, 0, -1):
            layers.append(nn.Linear(layer_list[i], layer_list[i-1]))
            if i > 1:
                layers.append(activation())
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class KoopmanModel(nn.Module):
    def __init__(self, layer_list, activation, k_init_std):
        super().__init__()
        
        self.encoder = Encoder(layer_list, activation)
        self.decoder = Decoder(layer_list, activation)

        # KOOPMAN OPERATOR
        latent_dim = layer_list[-1]
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.normal_(self.K.weight, mean=0.0, std=k_init_std)

    def forward(self, x):
        z = self.encoder(x)
        z_next = self.K(z)
        x_next_pred = self.decoder(z_next)
        return x_next_pred

# ------- Loss Functions -------
    # MSELoss(x_recon, x)
    # loss_lin_dyn = mse(K(z_k), z_k_next)
    # loss_pred = mse(x_next_pred, x_next)
# ------- Trainings -------
def train(model, loader, optimizer, loss_fn, train_AE_only=False):
    model.train()
    optimizer.zero_grad()

    if train_AE_only:
        ...
    else:
        ...
    
    loss.backward()
    optimizer.step()

    return loss.item()
# ------- Main -------


if __name__ == '__main__':
    # for i in range(7):
    #     traj = Y[i]  # first trajectory

    #     plt.figure()
    #     for i in range(3):  # first three states
    #         plt.plot(tvec, traj[:, i], linestyle='--', label=f'State {i+1}')
        
    #     plt.xlabel('Time')
    #     plt.ylabel('State Value')
    #     plt.legend()
        plt.show()