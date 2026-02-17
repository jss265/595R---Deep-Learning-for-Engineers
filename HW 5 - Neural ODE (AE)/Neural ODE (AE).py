'''
HW 5 - Neural ODE (AE)

This code uses an autoencorder to compress monthly weather samples into a latent space to be
solved by an ODE Solver and the decoded into usable predicion data.

It is designed to be managed through the Hyper Parameters and Settings section.
'''

import torch
from torch import nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------- Hyper Parameters and Settings -------
csv_filename = r'HW 5 - Neural ODE (AE)\Normalized Data.csv'

train_size = 20
ae_sizes = [4, 8, 4, 2]  # (first) num features, AE layers, (last) latent dim
latent_sizes = [2, 16, 2]  # (first) latent dim, hidden layers, (last) latent dim 
activation = nn.ReLU
lr = 0.01
stage_size = 4
max_epochs = 100
tol = 1e-4
patience = 10

# ------- Data Prep -------

# combine data
files = [
    r'HW 5 - Neural ODE (AE)\Daily Climate Delhi 2013-2017\DailyDelhiClimateTest.csv',
    r'HW 5 - Neural ODE (AE)\Daily Climate Delhi 2013-2017\DailyDelhiClimateTrain.csv',
    ]
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

columns = [
    'date',
    'meantemp',
    'humidity',
    'wind_speed',
    'meanpressure'
    ]

# average data over one month
df['date'] = pd.to_datetime(df['date'].str.strip())
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
monthly = df.groupby(['year', 'month'], as_index=False).mean(numeric_only=True)
monthly['index'] = (monthly['year'] - 2013)*12 + monthly['month'] - 1

# normalize data
for col in monthly.columns:
    if col not in ['year', 'month', 'index']:
        monthly[col] = (monthly[col] - monthly[col].mean()) / (monthly[col].std())
    else:
        monthly[col] = monthly[col]
monthly.to_csv(csv_filename, index=False)
print(f'Saved {csv_filename}')

# split into test/train, incrementally
train = monthly[monthly['index'] < train_size]
test = monthly[monthly['index'] >= train_size]

# ------- Build Neural Network -------
class Encoder(nn.Module):
    def __init__(self, ae_sizes, activation):
        super(Encoder, self).__init__()

        layers = []
        for i in range(len(ae_sizes) - 1):
            layers.append(nn.Linear(ae_sizes[i], ae_sizes[i+1]))
            if i < len(ae_sizes) - 2:
                layers.append(activation())
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)
    
class NeuralODE(nn.Module):
    def __init__(self, latent_sizes, activation):
        super(NeuralODE, self).__init__()

        layers = []
        for i in range(len(latent_sizes) - 1):
            layers.append(nn.Linear(latent_sizes[i], latent_sizes[i+1]))
            if i < len(latent_sizes) - 2:
                layers.append(activation())
        self.network = nn.Sequential(*layers)

    def solveODE(self, t, y):  # output dy/dt
        return self.network(y)

    def forward(self, y0, tsteps):
        return odeint(self.solveODE, y0, tsteps)  # should we assert size?
    
class Decoder(nn.Module):
    def __init__(self, ae_sizes, activation):
        super(Decoder, self).__init__()

        layers = []
        for i in range(len(ae_sizes) - 1, 0, -1):
            layers.append(nn.Linear(ae_sizes[i], ae_sizes[i-1]))
            if i > 1:
                layers.append(activation())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)
    
class NeuralODEAutoencoder(nn.Module):
    def __init__(self, ae_sizes, latent_sizes, activation):
        super(NeuralODEAutoencoder, self).__init__()

        self.encoder = Encoder(ae_sizes, activation)
        self.ode = NeuralODE(latent_sizes, activation)
        self.decoder = Decoder(ae_sizes, activation)

    def forward(self, x, tsteps):
        z0 = self.encoder(x)
        zt = self.ode(z0, tsteps)
        xhat = self.decoder(zt)
        return xhat

# ------- Prepare for Training -------
features = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
X_train = torch.tensor(train[features].values, dtype=torch.float32)
X_test = torch.tensor(test[features].values, dtype=torch.float32)

model = NeuralODEAutoencoder(ae_sizes, latent_sizes, activation)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# ------- Train -------
current_end = stage_size
while current_end <= X_train.shape[0]:
    x_stage = X_train[:current_end]
    tsteps = torch.arange(current_end, dtype=torch.float32)
    prev_loss = float('inf')
    wait = 0
    losses = []

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        xhat = model(x_stage, tsteps)
        xhat_diag = xhat[torch.arange(len(x_stage)),
                         torch.arange(len(x_stage))]
        loss = loss_fn(xhat_diag, x_stage)
        loss.backward()
        optimizer.step()

        if prev_loss - loss.item() < tol:
            wait += 1
            if wait >= patience:
                print(f'Converged at epoch {epoch}')
                break
        else:
            wait = 0
        
        prev_loss = loss.item()
        losses.append(prev_loss)

    current_end = min(current_end + stage_size, X_train.shape[0])

    with torch.no_grad():
        xhat_stage = model(x_stage, tsteps).detach().numpy()
        x_stage_np = x_stage.detach().numpy()

    fig, axs = plt.subplots(5, 1, figsize=(8, 12), sharex=False)
    names = ['meantemp', 'humidity', 'wind_speed', 'meanpressure', 'Loss']

    for i in range(4):
        axs[i].plot(x_stage_np[:, i], label='Actual')
        xhat_diag = xhat_stage[np.arange(len(x_stage_np)), 
                               np.arange(len(x_stage_np))]
        axs[i].plot(xhat_diag[:, i], '--', label='Pred')
        axs[i].set_ylabel(names[i])
        axs[i].legend()

    axs[4].plot(range(len(losses)), losses)
    axs[4].set_ylabel('Loss')
    axs[4].set_yscale('log')
    axs[4].set_xlabel('Epoch')

    axs[-2].set_xlabel('Time (months)')  # Since axs[3] is the last feature
    plt.suptitle(f'Stage {current_end}')
    plt.tight_layout()
    plt.show()


