import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

class VanillaNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()

        size1 = 5
        size2 = 5
        size3 = 5
        
        # two hidden layers
        self.network = nn.Sequential(
            nn.Linear(7, size1),
            nn.ReLU(),
            nn.Linear(size1, size2),
            nn.ReLU(),
            nn.Linear(size2, size3),
            nn.ReLU(),
            nn.Linear(size3, 1),
        )

    def forward(self, x):
        return self.network(x)

def train(dataloader, model, loss_fn, optimizer):

    model.train()

    num_batches = len(dataloader)
    train_loss = 0

    for X, y in dataloader:

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        train_loss += loss.item() # .item() returns the number 
        
    train_loss /= num_batches
    print(f"Train loss (MSE): {train_loss:>8f} \n")

    return train_loss

def test(dataloader, model, loss_fn, loss_fn_MAE, u_target, sigma_target):

    model.eval()

    num_batches = len(dataloader)
    test_loss = 0
    test_loss_MAE = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item() # .item() returns the number
            test_loss_MAE += loss_fn_MAE(pred, y).item() 

    test_loss /= num_batches # average test loss
    test_loss_MAE /= num_batches

    test_loss_MAE *= sigma_target

    print(f"Test loss (MSE): {test_loss:>8f} \n")
    print(f"Test loss (MAE and unnormalized): {test_loss_MAE:>8f} \n")

    return test_loss, test_loss_MAE

def loaddata(batch_size):

    data = np.genfromtxt(
        'HW 1 - MLP/auto+mpg/auto-mpg.data',
        usecols=range(8),     # ignore car name
        missing_values='?',   # marks missing values
        filling_values=np.nan # replaces ? with nan
        )
    data = data[~np.isnan(data).any(axis=1)] # removes rows with nan

    # normalization: x_hat = (x - u) / sigma 
    # and get u_target and sigma_target
    for i in range(data.shape[1]):
        u = data[:, i].mean().item()
        sigma = data[:, i].std().item()
        data[:, i] -= u
        data[:, i] /= sigma
        if i == 0:
            u_target = u
            sigma_target = sigma
    
    features = torch.tensor(data[:, 1:], dtype=torch.float32)
    targets = torch.tensor(data[:, :1].reshape(-1, 1), dtype=torch.float32)

    dataset = TensorDataset(features, targets)

    train_ds, test_ds = random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_ds, batch_size=batch_size)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size)

    return train_dataloader, test_dataloader, u_target, sigma_target

if __name__ == '__main__':
    for i in range(5):
        model = VanillaNetwork()
        loss_fn = nn.MSELoss() # objective
        loss_fn_MAE = nn.L1Loss() # MAE loss
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch_size = 64
        
        train_dl, test_dl, u_target, sigma_target = loaddata(batch_size)

        epochs = 1000
        train_losses = []
        test_losses = []
        test_losses_MAE = []
        
        for e in range(epochs):
            print(f"Epoch {e+1}\n-------------------------------")
            train_loss = train(train_dl, model, loss_fn, optimizer)
            test_loss, test_loss_MAE = test(test_dl, model, loss_fn, loss_fn_MAE, u_target, sigma_target)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_losses_MAE.append(test_loss_MAE)

        # create MSE error figure
        plt.figure()
        plt.title('Objective')
        plt.xlabel('Epoch')
        plt.ylabel('Normalized Mean Squared Error')
        plt.plot(range(epochs), train_losses)
        plt.plot(range(epochs), test_losses)
        plt.yscale('log')
        plt.legend(['train', 'test'])

        # save pic
        import os 
        folder_path = 'HW 1 - MLP/Figures'
        os.makedirs(folder_path, exist_ok=True)
        existing_files = os.listdir(folder_path)
        if not existing_files:
            next_num = 1
        else:
            numbers = [int(file_name[:-4]) for file_name in existing_files]
            next_num = np.max(numbers) + 1

        plt.savefig(f'{folder_path}/{next_num}', dpi=300, bbox_inches="tight")
        print(f'Saved figure as {next_num}.png')

        # create MAE error figure
        plt.figure()
        plt.title('Test MAE Error Tracking')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error (MPG)')
        plt.plot(range(epochs), test_losses_MAE)
        plt.yscale('linear')
        plt.text(
            0.95, 0.95,  # position: 95% across x, 95% up y
            f'MAE Error after {epochs} epochs:\n {test_losses_MAE[-1]:.2f} MPG',
            horizontalalignment='right',
            verticalalignment='top',
            transform=plt.gca().transAxes,  # coordinates relative to the axes
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
        )

        # save pic
        import os 
        folder_path = 'HW 1 - MLP/Figures'
        os.makedirs(folder_path, exist_ok=True)
        existing_files = os.listdir(folder_path)
        if not existing_files:
            next_num = 1
        else:
            numbers = [int(file_name[:-4]) for file_name in existing_files]
            next_num = np.max(numbers) + 1

        plt.savefig(f'{folder_path}/{next_num}', dpi=300, bbox_inches="tight")
        print(f'Saved figure as {next_num}.png')

    # plt.show()
