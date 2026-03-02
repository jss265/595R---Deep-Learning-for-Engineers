import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) # add parent folder to Python path so sibling modules can be imported
from root import HelpersJSS as jss

# ----------------------------------------
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


# ------- Hyper Parameters -------
HP = {
    'fig_path': r'HW 7 - Deep Koopman\Figs',
    'activation_function': nn.ELU,
    'loss_function': nn.MSELoss(),
    'optimizer_class': torch.optim.AdamW,
    'train_batch_size': 256,
    'test_batch_size': 100,  # recommended full length of test trajectories
    'autoencoder_layers': [7, 17, 22, 27],  # list encoder layers only, latent dimension is last element of list
    'k_init_std': 0.01,
    'objective_alphas': [0.5, 0.75, 1.75],  # a1 reconstruction, a2 prediction, a3 linear dynamics
    'decay_alpha_ae': 1e-4,
    'lr_ae': 1e-3,
    'epochs_ae': 50,  # learn latent representation
    'lr_full': 1e-4, # usually smaller for full model learning
    'epochs_full': 200,
    'decay_alpha_full': 0,
    'rollout_steps': 49,  # how many steps to roll out from z_0 (max = nt-1)
}

# ------- Data Prep -------
ntraj = 2148  # number of trajectories
nt = 50  # number of time steps
ny = 7  # number of states
assert HP['autoencoder_layers'][0] == ny, 'Failed: autoencoder_layers[0] == ny'
assert HP['rollout_steps'] == nt - 1, "Failed: HP['rollout_steps'] == nt - 1"

tvec = np.linspace(0, 350, nt)
Y = np.loadtxt('HW 7 - Deep Koopman/kdata.txt').reshape(ntraj, nt, ny)
Ytrain = Y[:2048, :, :]  # 2048 training trajectories
Ytest = Y[2048:, :, :]  # 100 testing trajectoreis
print(f'Loaded data:\n    Ytrain size: {Ytrain.shape}\n    Ytest shape {Ytest.shape}')
print('    Shape: trajectories (trial runs), time steps, states')

Ytrain_torch = torch.tensor(Ytrain, dtype=torch.float32)
Ytest_torch = torch.tensor(Ytest, dtype=torch.float32)

train_dataset = TensorDataset(Ytrain_torch)
train_loader = DataLoader(train_dataset, batch_size=HP['train_batch_size'], shuffle=True)
test_dataset = TensorDataset(Ytest_torch)
test_loader = DataLoader(test_dataset, batch_size=HP['test_batch_size'], shuffle=False)

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
        rollout_steps: int = 1,
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
            # Rollout from initial latent state z_0
            # z_t = K^t * z_0  for t = 0 ... rollout_steps
            z0 = model.encoder(traj_batch[:, 0, :])  # (B, latent)
            z_roll = [z0]
            for _ in range(rollout_steps):
                z_roll.append(model.K(z_roll[-1]))
            z_roll = torch.stack(z_roll, dim=1)  # (B, rollout_steps+1, latent)

            # Prediction Loss: Decoder(K^t z_0) vs x_t
            x_roll = model.decoder(z_roll)  # (B, rollout_steps+1, ny)
            pred = loss_fn(x_roll, traj_batch[:, :rollout_steps+1, :])

            # Linear Dynamics Loss: K^t z_0 vs Encoder(x_t)
            z_true = model.encoder(traj_batch[:, :rollout_steps+1, :])  # (B, rollout_steps+1, latent)
            lin_dyn = loss_fn(z_roll, z_true)

            # Combined Loss
            loss = a1*recon + a2*pred + a3*lin_dyn
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(
        model: KoopmanModel,
        loader: DataLoader,
        loss_fn: nn.Module,
        alphas: list,
        rollout_steps: int = 1,
        train_AE_only: bool=False):

    model.eval()
    total_loss = 0.0
    a1, a2, a3 = alphas

    with torch.no_grad():
        for (traj_batch,) in loader:  # shape: batch_size, nt, ny

            # Reconstruction Loss
            # x_k ≈ x_k (encoded/decoded only)
            recon = loss_fn(model.autoencode(traj_batch), traj_batch)  # reconstruction loss

            if train_AE_only:
                loss = recon

            else:
                # Rollout from initial latent state z_0
                # z_t = K^t * z_0  for t = 0 ... rollout_steps
                z0 = model.encoder(traj_batch[:, 0, :])  # (B, latent)
                z_roll = [z0]
                for _ in range(rollout_steps):
                    z_roll.append(model.K(z_roll[-1]))
                z_roll = torch.stack(z_roll, dim=1)  # (B, rollout_steps+1, latent)

                # Prediction Loss: Decoder(K^t z_0) vs x_t
                x_roll = model.decoder(z_roll)  # (B, rollout_steps+1, ny)
                pred = loss_fn(x_roll, traj_batch[:, :rollout_steps+1, :])

                # Linear Dynamics Loss: K^t z_0 vs Encoder(x_t)
                z_true = model.encoder(traj_batch[:, :rollout_steps+1, :])  # (B, rollout_steps+1, latent)
                lin_dyn = loss_fn(z_roll, z_true)

                # Combined Loss
                loss = a1*recon + a2*pred + a3*lin_dyn
                
            total_loss += loss.item()

    return total_loss / len(loader)

# ------- Main -------


if __name__ == '__main__':
    
    # instantiate model
    model = KoopmanModel(HP['autoencoder_layers'], HP['activation_function'], HP['k_init_std'])

    # learn latent representation
    for param in model.K.parameters():
        param.requires_grad = False
    
    ae_optimizer = HP['optimizer_class'](
        list(model.encoder.parameters()) + 
        list(model.decoder.parameters()),
        lr=HP['lr_ae'],
        weight_decay=HP['decay_alpha_ae']
    )

    train_losses = []
    test_losses = []
    for i in range(HP['epochs_ae']):
        train_loss = train(
            model,
            train_loader,
            ae_optimizer,
            HP['loss_function'],
            HP['objective_alphas'],
            train_AE_only=True
            )
        train_losses.append(train_loss)
    
        test_loss = evaluate(
            model,
            test_loader,
            HP['loss_function'],
            HP['objective_alphas'],
            train_AE_only=True
        )
        test_losses.append(test_loss)

        print(f'Epoch {i+1}:\n    Train Loss: {train_loss:.6f}\n    Test Loss: {test_loss:.6f}')
    
    fig = plt.figure()
    plt.plot(range(HP['epochs_ae']), train_losses, label='Train Loss')
    plt.plot(range(HP['epochs_ae']), test_losses, label='Test Loss')
    plt.title('Learning Latent Representation with Reconstruction Loss')
    plt.ylabel('MSE Error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.yscale('log')
    jss.text_box_to_fig(fig, HP)
    jss.savePicInSequence(fig, HP['fig_path'])

    # learn Koopman operator and full model
    for param in model.K.parameters():
        param.requires_grad = True

    full_optimizer = HP['optimizer_class'](
        model.parameters(),
        lr=HP['lr_full'],
        weight_decay=HP['decay_alpha_full']
    )
    
    train_losses = []
    test_losses = []
    for i in range(HP['epochs_full']):
        train_loss = train(
            model,
            train_loader,
            full_optimizer,
            HP['loss_function'],
            HP['objective_alphas'],
            rollout_steps=HP['rollout_steps'],
            train_AE_only=False
            )
        train_losses.append(train_loss)
    
        test_loss = evaluate(
            model,
            test_loader,
            HP['loss_function'],
            HP['objective_alphas'],
            rollout_steps=HP['rollout_steps'],
            train_AE_only=False
        )
        test_losses.append(test_loss)

        print(f'Epoch {i+1}:\n    Train Loss: {train_loss:.6f}\n    Test Loss: {test_loss:.6f}')
    
    fig = plt.figure()
    plt.plot(range(HP['epochs_full']), train_losses, label='Train Loss')
    plt.plot(range(HP['epochs_full']), test_losses, label='Test Loss')
    plt.title('Learning Full Koopman Model with Objective')
    plt.ylabel('MSE Error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.yscale('log')
    jss.text_box_to_fig(fig, HP)
    jss.savePicInSequence(fig, HP['fig_path'])

    # Plot First trajectory against prediction
    model.eval()

    with torch.no_grad():
        # First test trajectory
        x_true = Ytest_torch[0:1]   # shape: (1, nt, ny)

        # Get predictions recursively (open-loop rollout)
        x_pred = torch.zeros_like(x_true)
        z = model.encoder(x_true[:, 0:1, :])  # initial state only

        z_roll = z
        for t in range(nt):
            x_pred[:, t, :] = model.decoder(z_roll)
            z_roll = model.K(z_roll)

    # Convert to numpy
    x_true = x_true.squeeze(0).cpu().numpy()
    x_pred = x_pred.squeeze(0).cpu().numpy()

    # Plot first 3 states
    fig = plt.figure()

    for i in range(3):
        plt.plot(tvec, x_true[:, i], '--', label=f'True State {i+1}')
        plt.plot(tvec, x_pred[:, i], '-', label=f'Pred State {i+1}')

    plt.xlabel('Time')
    plt.ylabel('State Value')
    plt.title('Test Trajectory: True vs Predicted')
    plt.legend()
    jss.text_box_to_fig(fig, HP)
    jss.savePicInSequence(fig, HP['fig_path'])
    plt.show()