import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        output = self.fc(rnn_out[:, -1, :])  # Predict next step using last RNN output
        return output