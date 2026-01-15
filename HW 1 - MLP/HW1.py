import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

class VanillaNetwork(nn.Module): 
    
    def __init__(self):
        super().__init__()

        size1 = 20
        size2 = 20
        size3 = 20
        
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
    print(f"Train loss: {train_loss:>8f} \n")

    return train_loss

def test(dataloader, model, loss_fn):

    model.eval()

    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item() # .item() returns the number 

    test_loss /= num_batches # average test loss

    print(f"Test loss: {test_loss:>8f} \n")

    return test_loss

def loaddata(batch_size):

    data = np.genfromtxt(
        'HW 1 - MLP/auto+mpg/auto-mpg.data',
        usecols=range(8),     # ignore car name
        missing_values='?',   # marks missing values
        filling_values=np.nan # replaces ? with nan
        )
    data = data[~np.isnan(data).any(axis=1)] # removes rows with nan

    features = torch.tensor(data[:, 1:], dtype=torch.float32)
    targets = torch.tensor(data[:, :1].reshape(-1, 1), dtype=torch.float32)

    # normalization: x_hat = (x - u) / sigma
    for i in range(features.shape[1]):
        u = features[:, i].mean().item()
        sigma = features[:, i].std().item()
        features[:, i] -= u
        features[:, i] /= sigma

    for i in range(targets.shape[1]):
        u = targets[:, i].mean().item()
        sigma = targets[:, i].std().item()
        targets[:, i] -= u
        targets[:, i] /= sigma
    # -------------

    dataset = TensorDataset(features, targets)

    train_ds, test_ds = random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_ds, batch_size=batch_size)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size)

    return train_dataloader, test_dataloader

if __name__ == '__main__':

    
    model = VanillaNetwork()
    loss_fn = nn.MSELoss() # objective
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 64
    
    train_dl, test_dl = loaddata(batch_size)

    epochs = 1000
    train_losses = []
    test_losses = []
    
    for e in range(epochs):
        print(f"Epoch {e+1}\n-------------------------------")
        train_loss = train(train_dl, model, loss_fn, optimizer)
        test_loss = test(test_dl, model, loss_fn)
        train_losses.append(train_loss)
        test_losses.append(test_loss)


    plt.figure()
    plt.plot(range(epochs), train_losses)
    plt.plot(range(epochs), test_losses)
    plt.legend(['train', 'test'])

    # save pic
    import os

    existing_files = os.listdir('screenshots')
    if not existing_files:
        next_num = 1
    else:
        numbers = [int(file_name[:-4]) for file_name in existing_files]
        next_num = np.max(numbers) + 1
    plt.savefig(f'screenshots/{next_num}', dpi=300, bbox_inches="tight")
    print(f'Saved figure as {next_num}.png')
    plt.show()
