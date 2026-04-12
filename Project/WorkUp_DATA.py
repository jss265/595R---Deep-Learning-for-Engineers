import torch
import pandas as pd
import glob
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class IMURepDataset(Dataset):
    def __init__(self, data_dir, valid_subjects, valid_arms, valid_reps):
        """
        Args:
            data_dir (str): Path to the folder containing the CSVs.
            valid_subjects (list of str): Acceptable subjects.
            valid_arms (list of str): Acceptable arms.
            valid_reps (list of str): Acceptable reps.
        """
        file_paths = glob.glob(os.path.join(data_dir, "*.csv"))
        
        # Lists to hold the data in RAM
        self.sequences = []
        self.labels = []
        
        # Parse all files ONCE during initialization
        for file_path in file_paths:
            # 1. Parse Metadata
            meta_df = pd.read_csv(file_path, nrows=5, header=None, index_col=0)
            
            # Filter by acceptable subject, arm, and reps
            subject = str(meta_df.loc['subject', 1]).strip()
            arm = str(meta_df.loc['arm', 1]).strip()
            reps = str(meta_df.loc['reps', 1]).strip()
            
            if subject not in valid_subjects or arm not in valid_arms or reps not in valid_reps:
                continue

            label_val = int(meta_df.loc['label', 1])
            self.labels.append(torch.tensor(label_val, dtype=torch.long))
            
            # 2. Parse Actual Data
            data_df = pd.read_csv(file_path, skiprows=6)
            features = data_df[['q0', 'q1', 'q2', 'q3', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
            
            self.sequences.append(torch.tensor(features, dtype=torch.float32))

        # --- GLOBAL Z-SCORE NORMALIZATION ---
        # 1. Stack all sequences together into one giant tensor along the time dimension
        all_data = torch.cat(self.sequences, dim=0)

        # 2. Calculate the global mean and standard deviation for each of the 10 features
        # Keep dim=0 because we are flattening time and files, keeping only the 10 features
        self.mean = all_data.mean(dim=0)
        self.std = all_data.std(dim=0)

        # Prevent division by zero if a feature is completely constant
        self.std[self.std == 0] = 1.0

        # 3. Apply the global mean and std to standardize every individual sequence in the dataset
        for i in range(len(self.sequences)):
            self.sequences[i] = (self.sequences[i] - self.mean) / self.std
        input('pause')


    def __len__(self):
        # The length of the dataset is the number of samples (files) we have
        return len(self.sequences)

    def __getitem__(self, idx):
        # Returns the pre-parsed sequence and label for the given index when instance[] is called
        # Now this is lighting fast! Just returning from RAM.
        return self.sequences[idx], self.labels[idx]

def pad_collate(batch):
    """
    Custom collate function to handle variable-length IMU sequences.
    To be passed to the DataLoader: `DataLoader(..., collate_fn=pad_collate)`
    """
    # 'batch' is a list of tuples from __getitem__: [(seq1, label1), (seq2, label2), ...]
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    labels = torch.stack(labels) # Convert list of labels to a tensor
    
    # Record the original lengths of each sequence before padding
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    
    # Pad the sequences. batch_first=True -> (batch_size, max_len, features)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    
    return sequences_padded, labels, lengths
