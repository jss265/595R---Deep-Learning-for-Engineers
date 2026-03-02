import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# ------- Hyper Parameters -------
batch_size = 256
autoencoder_layers = []  # list encoder layers only, latent dimension is last element of list
k_init_std = 0.01
objective_alphas = [1.0, 1.0, 1.0]  # a1 reconstruction, a2 prediction, a3 linear dynamics
decay_alpha = 1e-4
activation_function = nn.ELU
loss_function = nn.MSELoss()

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
    
    def autoencode(self, x):
        z = self.encoder(x)
        x_decoded = self.decoder(z)
        return x_decoded

# ------- Trainings -------
def train(
        model: KoopmanModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        alphas: list,
        train_AE_only: bool=False
        ):

    model.train()
    total_loss = 0.0
    a1, a2, a3 = alphas

    for (traj_batch,) in loader:  # shape: batch_size, nt, ny
        optimizer.zero_grad()

        # Reconstruction Loss
        # x_k ≈ x_k (encoded/decoded only)
        recon = loss_fn(model.autoencode(traj_batch), traj_batch)  # reconstruction loss

        if train_AE_only:
            loss = recon

        else:
            # Prediction Loss
            # x_k ≈ xhat_k
            x_k = traj_batch[:, :-1, :]
            x_k_next = traj_batch[:, 1:, :]

            x_k_next_pred = model(x_k)
            pred = loss_fn(x_k_next_pred, x_k_next)
            
            # Linear Dynamics Loss (comparison inside latent space)
            # z_k+m ≈ K^m z_k
            z = model.encoder(traj_batch)

            z_k = z[:, :-1, :]
            z_k_next = z[:, 1:, :]

            z_k_next_pred = model.K(z_k)
            lin_dyn = loss_fn(z_k_next_pred, z_k_next)
            
            # Combined Loss
            loss = a1*recon + a2*pred + a3*lin_dyn
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss
    
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