import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

'''
modify.  many trajectories.
given a stream of 20 time steps, predict the next 80 position (20→80 mapping)
this is a seq2seq problem.  encoder reads 20 steps, produces context vector (h, c)
decoder conditioned on (h, c) generates 80 steps one at a time.
'''

torch.manual_seed(42)
np.random.seed(42)

N_TRAJ  = 1000   # oscillators with different (ω_n, ζ)
T_TOTAL = 100    # time steps per trajectory
T_IN    = 10     # steps the encoder sees  ("initial condition")
T_OUT   = T_TOTAL - T_IN   # steps the decoder must predict
BATCH = 64

def damped_oscillator():
    omega_n = np.random.uniform(0.5, 3.0)        # natural frequency  [rad/s]
    zeta    = np.random.uniform(0.05, 0.4)        # damping ratio (underdamped)
    omega_d = omega_n * np.sqrt(1 - zeta**2)      # damped natural frequency
    t       = np.linspace(0, 15, T_TOTAL)
    x       = np.exp(-zeta * omega_n * t) * np.sin(omega_d * t)
    return x.astype(np.float32)

def dataprep():
    trajs = np.array([damped_oscillator() for _ in range(N_TRAJ)])

    X = torch.tensor(trajs[:, :T_IN ]).unsqueeze(-1)   # (N, 10,  1)
    Y = torch.tensor(trajs[:, T_IN: ]).unsqueeze(-1)   # (N, 90,  1)

    split = int(0.8 * N_TRAJ)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # print(X_train.shape)  # (800, 10, 1)  batch, time steps, features
    # print(Y_train.shape)  # (800, 90, 1)

    return X_train, Y_train, X_test, Y_test


class RNNPredictor(nn.Module):

    def __init__(self):

        super().__init__()
        # 1 feature input.  # hidden size is a hyper parameter.  # chose one layer.
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
        self.encoder = nn.GRU(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        self.decoder = nn.GRU(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x, y=None, teacher_forcing=False):

        _, h = self.encoder(x)  # pass in 10 time steps
        # h is the same as the last time step of the supressed output (if one layer),
        # it's the "context vector" that the decoder will use to generate the next 80 steps.
        # print(h.shape)  # (num_layers, batch, hidden_size)  (1, 800, 64)

        decoder_input = x[:, -1:, :]  # start the decoder with the last input (batch, 1, features)

        nt = T_OUT  # number of time steps to predict (90)
        nbatch = x.shape[0]
        nout = self.fc.out_features
        outputs = torch.empty(nbatch, nt, nout)  # preallocate output tensor

        for t in range(nt):
            out, h = self.decoder(decoder_input, h)
            pred = self.fc(out)  # pred.shape: (batch, 1, 1) nbatch x time steps x features
            outputs[:, t:t+1, :] = pred  # store prediction

            if teacher_forcing and y is not None:
                decoder_input = y[:, t:t+1, :]  # use true next step as input
            else:
                decoder_input = pred  # feed prediction back in

        return outputs  # (batch, nt, features)


def train(X_train, Y_train, model, lossfn, optimizer):

    total_loss = 0.0

    idx = torch.randperm(len(X_train))
    for i in range(0, len(X_train), BATCH):
        xb   = X_train[idx[i:i+BATCH]]
        yb   = Y_train[idx[i:i+BATCH]]

        optimizer.zero_grad()
        predictions = model(xb, yb, teacher_forcing=True)
        loss = lossfn(predictions, yb)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / (len(X_train) / BATCH)  # return average loss per batch

def test(X_test, Y_test, model, lossfn):

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        loss = lossfn(predictions, Y_test)

    return loss.item()


if __name__ == "__main__":

    X_train, Y_train, X_test, Y_test = dataprep()

    model = RNNPredictor()
    model(X_train, Y_train)

    model = RNNPredictor()
    lossfn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)

    epochs = 100
    trainlosses = []
    testlosses = []
    for epoch in range(epochs):
        trainloss = train(X_train, Y_train, model, lossfn, optimizer)
        testloss = test(X_test, Y_test, model, lossfn)
        trainlosses.append(trainloss)
        testlosses.append(testloss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {trainloss}, Test Loss: {testloss}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_test)



    plt.figure()
    plt.semilogy(range(epochs), trainlosses, label="Train Loss")
    plt.semilogy(range(epochs), testlosses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()


    t_all = np.linspace(0, 15, T_TOTAL)
    t_in  = t_all[:T_IN]
    t_out = t_all[T_IN:]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, ax in enumerate(axes):
        ax.plot(t_in,  X_test[i, :, 0].numpy(), color="gray",      lw=2, label="Observed (input)")
        ax.plot(t_out, Y_test[i, :, 0].numpy(), color="steelblue", lw=2, label="True trajectory")
        ax.plot(t_out, Y_pred[i, :, 0].numpy(), color="tomato",    lw=2, linestyle="--", label="GRU prediction")
        ax.axvline(t_all[T_IN], color="black", linestyle=":", linewidth=1)
        ax.set_title(f"Test oscillator #{i+1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("x(t)")
        if i == 0:
            ax.legend(fontsize=8)
    plt.show()