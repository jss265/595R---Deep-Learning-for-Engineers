import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) # add parent folder to Python path so sibling modules can be imported
from root import HelpersJSS as jss

# ----------------------------------------
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
import matplotlib.pyplot as plt
import hw9setup as hw9

HP = {
    'middle layers': [32, 32],
    'activation': nn.ReLU,
    'optim': torch.optim.Adam,
    'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'sch_mode': 'min',
    'sch_factor': 0.5,
    'sch_patience': 3,
    'sch_min_lr':1e-6,
    'epochs': 150,
    'lr': 1e-2,
    'batch_size': 128,
    'loss_fn': nn.L1Loss
}

class GNN(MessagePassing):
    def __init__(self, middle_layers, activation, aggr='add'):  # Davis suggest mean, and I don't know why... lol
        super().__init__(aggr=aggr)

        num_features = 5  # x, y, Vx, Vy, m
        num_targets = 2   # Ax, Ay NOTE remember the point (target) of this GNN is to predict acceleration given input state

        message_in = 2 * num_features  # because message takes in x_i and x_j OR node and it's neighbor constituting an edge
        message_out = 2  # Fx, Fy NOTE the "message" is a sum of forces acting on the particle (node) via all other particles (edges)
        message_layers = [message_in, *middle_layers, message_out]  # encourage depth here for richness NOTE this is where the value of DL is

        update_in = message_out + num_features  # because update takes node state as well as input forces to calculate accelerations
        update_out = num_targets
        update_layers = [update_in, *middle_layers, update_out]  # encourage depth here for richness NOTE this is where the value of DL is

        # Message NN | Sum of the Forces
        layers = []
        for i in range(len(message_layers) - 1):
            layers.append(nn.Linear(message_layers[i], message_layers[i+1]))
            if i < len(message_layers) - 2:
                layers.append(activation())
        self.sum_forces = nn.Sequential(*layers)
        # print(self.sum_forces)

        # Update NN | Current Accelerations
        layers = []
        for i in range(len(update_layers) - 1):
            layers.append(nn.Linear(update_layers[i], update_layers[i+1]))
            if i < len(update_layers) - 2:
                layers.append(activation())
        self.cur_accelerations = nn.Sequential(*layers)
        # print(self.cur_accelerations)

    def forward(self, x, edge_index):
        # pytorch_geometric.nn.MessagePassing.propagate
        # NOTE ^^ takes care of messaging (edge traversal & batching), aggragation, and updating
        # NOTE edge_index is used to parse through x to create x_i and x_j relationships
        return self.propagate(edge_index, x=x)  # num nodes, num targets | (4, 2)
    
    def message(self, x_i, x_j):
        xcat = torch.cat([x_i, x_j], dim=1)  # size: [x,y,Vx,Vy,m]node + [x,y,Vx,Vy,m]neighbor BY num nodes
        return self.sum_forces(xcat)
    
    def update(self, aggr_out, x):
        xcat = torch.cat([aggr_out, x], dim=1)  # size: [Fx, fy]message + [x,y,Vx,Vy,m]node BY num nodes
        return self.cur_accelerations(xcat)

def y_prime(t, y, model, edge_index, constant_terms):
    # y is the state vector [x,y,Vx,Vy] that changes with time
    # solve_ivp always passes in y as a numpy array
    y = torch.tensor(y, dtype=torch.float32)

    # constant_terms are terms that do not change with time and don't need to be calculated with solve_ivp
    # restructure y from flattened array to (4, 4) | 4 nodes BY x,y,Vx,Vy
    y = y.view(4, 4)
    x = torch.cat([y, constant_terms], dim=1)  # (4, 5) | 4 nodes BY x,y,Vx,Vy + m (constant_terms)

    # calculate accelerations
    with torch.no_grad():
        accel = model(x, edge_index)  # 4, 2 | nodes BY Ax, Ay

    # return y_prime -aka- dydt
    dydt = torch.zeros_like(y)
    dydt[:, 0:2] = y[:, 2:4]
    dydt[:, 2:4] = accel
    dydt = dydt.view(-1).numpy()  # flatten to [Vx1, Vy1, Ax1, Ay1, Vx2, ...]
    return dydt

# Training time boyo

if __name__ == "__main__":
    # define model, optimizer, scheduler, and loss function
    model = GNN(HP['middle layers'], HP['activation'])
    optimizer = HP['optim'](model.parameters(), lr=HP['lr'])
    sch_kargs = {
        'mode': HP['sch_mode'],
        'factor': HP['sch_factor'],
        'patience': HP['sch_patience'],
        'min_lr': HP['sch_min_lr']
    }
    scheduler = HP['scheduler'](optimizer, **sch_kargs)
    loss_fn = HP['loss_fn']()

    # import data
    train_loader, test_loader, train_traj, test_traj, edge_index = hw9.data_setup(HP['batch_size'])

    train_losses = []
    test_losses = []

    for epoch in range(HP['epochs']):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            pred = model(batch.x, batch.edge_index)
            loss = loss_fn(pred, batch.y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        scheduler.step(epoch_loss)

        model.eval()
        total_test_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                pred = model(batch.x, batch.edge_index)
                loss = loss_fn(pred, batch.y)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Test Loss = {avg_test_loss:.6f}, Step Size = {scheduler.get_last_lr()[0]}")

    t_span = (0, 5)
    t_eval = np.linspace(0, 5, 100)

    init_state = test_traj[0, :, :4].reshape(-1).numpy()    # first timestep: (1, 4, 4) -> (16,)
    mass = test_traj[0, :, 4:5].clone().detach()

    sol = solve_ivp(y_prime, t_span, init_state, t_eval=t_eval, args=(model, edge_index, mass))

    # NOTE: solve_ivp outputs a (16, 100) vector here, sol.y.
    # ChatGPT helped me ficure out the .y.T.reshape part - explanation needed
    pred_traj = sol.y.T.reshape(-1, 4, 4)  # (time, particles, state)

    fig = plt.figure()

    colors = ["red", "orange", "green", "blue"]

    for j in range(4):
        # true
        plt.plot(test_traj[:, j, 0], test_traj[:, j, 1], color = colors[j], label=f"True {j}")

        # predicted
        plt.plot(pred_traj[:, j, 0], pred_traj[:, j, 1], '--', color = colors[j], label=f"Pred {j}")

    plt.legend()
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("True vs Predicted Trajectories")
    jss.text_box_to_fig(fig, HP)
    jss.savePicInSequence(fig, 'HW 9 - GNN/figs')

    fig = plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel(f"Loss ({HP['loss_fn'].__name__})")
    plt.title("Training vs Testing Loss")
    plt.legend()
    jss.text_box_to_fig(fig, HP)
    jss.savePicInSequence(fig, 'HW 9 - GNN/figs')
    try:
        import winsound as ws
        ws.Beep(1000, 1000)  # this is to make a noise to get my attention. won't work on MAC
    except ImportError:
        print('\a')  # sometimes makes a bell noise in some terminals
        print('done')