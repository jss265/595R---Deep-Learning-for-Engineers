import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

'''
start with an (overly) simple case.  1 trajectory.
given a stream of 20 time steps, predict the next 1 position (20→1 mapping)
just a starting point to understand how to set up the data (introduce windowing)
and model set (LSTM, GRU)
'''

def generate_damped_oscillator(m=1.0, c=0.2, k=1.0, dt=0.1, T=40.0):
    t = np.arange(0, T, dt)
    n = len(t)
    x = np.zeros(n)
    v = np.zeros(n)
    x[0] = 1.0  # Initial displacement
    v[0] = 0.0  # Initial velocity

    for i in range(1, n):
        a = (-c * v[i-1] - k * x[i-1]) / m
        v[i] = v[i-1] + a * dt
        x[i] = x[i-1] + v[i] * dt

    return t, x


def dataprep(SEQ_LEN):
    # Prepare dataset
    t, x = generate_damped_oscillator()

    # this is just one single trajectory, so we need to create multiple samples by sliding a window across it
    inputs  = torch.tensor(np.array([x[i:i+SEQ_LEN] for i in range(len(x) - SEQ_LEN)]), dtype=torch.float32)
    targets = torch.tensor(np.array([x[i+SEQ_LEN]   for i in range(len(x) - SEQ_LEN)]), dtype=torch.float32)
    # print(inputs.shape)  # 380 x 20 (380 samples, each with a sequence of 20 time steps)
    # print(targets.shape)  # 380 (corresponds to the next step after each sequence)

    X = inputs.unsqueeze(-1)  # Add input feature dimension  380 x 20 x 1.  samples x time steps x features
    y = targets.unsqueeze(-1)  # Add output feature dimension. 380 x 1.  samples x features

    N = len(X)
    perm = torch.randperm(N)
    split = int(0.8 * N)   # 80/20

    train_idx = perm[:split]
    test_idx = perm[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, y_train, X_test, y_test, t, x


class RNNPredictor(nn.Module):

    def __init__(self):

        super().__init__()
        # 1 feature input.  # hidden size is a hyper parameter.  # chose one layer.
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
        self.rnn = nn.GRU(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # Defaults to zeros if (h_0, c_0) is not provided
        # print(rnn_out.shape)  # batch x seq x hidden_size
        output = self.fc(rnn_out[:, -1, :])  # Predict next step using last RNN output
        return output



def train(X_train, y_train, model, lossfn, optimizer):

    optimizer.zero_grad()
    predictions = model(X_train)

    loss = lossfn(predictions, y_train)
    loss.backward()
    optimizer.step()

    return loss.item()

def test(X_test, y_test, model, lossfn):

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        loss = lossfn(predictions, y_test)

    return loss.item()


if __name__ == "__main__":
    SEQ_LEN = 20
    X_train, y_train, X_test, y_test, t_traj, x_traj = dataprep(SEQ_LEN)

    model = RNNPredictor()
    lossfn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 200
    for epoch in range(epochs):
        trainloss = train(X_train, y_train, model, lossfn, optimizer)
        testloss = test(X_test, y_test, model, lossfn)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {trainloss}, Test Loss: {testloss}")

    # Evaluate the model
    model.eval()
    predictions = []
    input_seq = torch.tensor(x_traj[:SEQ_LEN], dtype=torch.float32)  # grab starting sequence from the true trajectory
    input_seq = input_seq.unsqueeze(0)  # Add batch dimension
    input_seq = input_seq.unsqueeze(-1)  # Add input feature dimension

    for i in range(len(x_traj) - SEQ_LEN):
        with torch.no_grad():
            pred = model(input_seq).item()  # predict next point
        predictions.append(pred)
        # adjust input sequence to be the last SEQ_LEN points, from which we'll predict another point.
        input_seq[0, :, 0] = torch.cat((input_seq[0, 1:, 0], torch.tensor([pred], dtype=torch.float32)))


    # Plot results
    plt.plot(t_traj, x_traj, "-o", label="True Solution")
    plt.plot(t_traj[SEQ_LEN:], predictions, label="GRU Prediction", linestyle='dashed')
    plt.xlabel("Time")
    plt.ylabel("Displacement")
    plt.legend()
    plt.show()
