import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def data_setup(batch_size=128):

    # ------------ import data ------------
    # 1000 trajectory sets with 100 time steps split 75/25 for training/testing
    data = np.load("HW 9 - GNN/spring_data.npz")
    # X_train (75000, 4, 5)  data points, 4 particles, 5 states: x, y, Vx, Vy, m (positions, velocities, mass)
    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    # y_train (75000, 4, 2)  data points, 4 particles, 2 states: ax, ay (accelerations)
    y_train = torch.tensor(data['y_train'], dtype=torch.float32)
    # X_test (25000, 4, 5)  data points, 4 particles, 5 states
    X_test = torch.tensor(data['X_test'], dtype=torch.float32)
    # y_test (25000, 4, 5)  data points, 4 particles, 5 states
    y_test = torch.tensor(data['y_test'], dtype=torch.float32)
    # 100 time steps (not really needed)
    times = torch.tensor(data['times'], dtype=torch.float32)

    # ------- Save a few trajectories for plotting -------
    # the data points are currently ordered in time (for each separate trajectory)
    # so I'm going to save one set before shuffling the data.
    # this will make it easier to check how well I'm predicting the trajectories

    nt = len(times)
    train_traj = X_train[:nt, :, :]
    test_traj = X_test[:nt, :, :]

    # # You can comment this out, just showing you how do this
    # # for when you'll want to compare later.
    # plt.figure()
    # for j in range(4):  # plot all 4 particles
    #     plt.plot(train_traj[:, j, 0], train_traj[:, j, 1])
    # plt.xlabel('x position')
    # plt.ylabel('y position')

    # # plotting one set of testing trajectories
    # plt.figure()
    # for j in range(4):
    #     plt.plot(test_traj[:, j, 0], test_traj[:, j, 1])
    # plt.xlabel('x position')
    # plt.ylabel('y position')

    # plt.show()

    # ------ edge index ------
    # this just defines how the nodes (particles) are connected
    # in this case each of the 4 particles interacts with every other particle
    edge_index = torch.tensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
       [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]
    ], dtype=torch.long)

    # -------- Further data prep ----------
    # - espeically while developing (and maybe the whole time) you will want to extract
    #   just a subset of data points
    # - when you put the data into a DataLoader, you'll want to shuffle the data
    #   so that you'll be pulling from different trajectories
    # - note that Data and DataLoader in torch_geometric are a bit different
    train_data_list = []
    for i in range(X_train.shape[0]):
        data_point = Data(
            x=X_train[i],          # (4, 5) features: nodes, (x, y, Vx, Vy, m)
            y=y_train[i],          # (4, 2) targets (Ax, Ay)
            edge_index=edge_index  # (2 nodes, num_edges) NOTE fully connected means 1s with 0s on diagonal or num_nodes**2 - num_nodes
        )
        train_data_list.append(data_point)  # len: X_train.shape[0], type torch_geometric.data.Data

    test_data_list = []
    for i in range(X_test.shape[0]):
        data_point = Data(
            x=X_test[i],
            y=y_test[i],
            edge_index=edge_index
        )
        test_data_list.append(data_point)

    train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_traj, test_traj, edge_index



if __name__ == "__main__":
    data_setup()
