import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence

class RNNPredictor(nn.Module):
    def __init__(self, input_size=10, hidden_size=32, num_classes=4, num_layers=1, batch_first=True):
        super().__init__()
        # input_size=10 because your IMU data has 10 columns (q0-g3, ax-az, gx-gz)
        # num_classes=4 because you are classifying 0, 1, 2, or 3 reps
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # 1. pack_padded_sequence needs lengths to be on the CPU
        lengths = lengths.cpu()
        
        # 2. Pack the padded tensor so the GRU ignores the zeros
        # enforce_sorted=False is needed because we didn't sort the batch by length
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # 3. Pass the packed sequence through the GRU
        _, h_n = self.rnn(packed_x)
        
        # 4. h_n shape is (num_layers, batch_size, hidden_size)
        # We grab the final hidden state of the very top layer
        last_hidden = h_n[-1]
        
        # 5. Pass it through the linear classification layer
        output = self.fc(last_hidden) # these are the raw logits for each class (not probabilities because we will use CrossEntropyLoss which applies softmax internally)
        
        return output
        