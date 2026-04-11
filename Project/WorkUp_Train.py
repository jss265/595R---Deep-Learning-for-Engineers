import torch
import matplotlib.pyplot as plt
import WorkUp_DATA as data
import WorkUp_GRU as model

if __name__ == "__main__":
    data = data.IMURepDataset("data/train")
    print(f"Number of samples: {len(data)}")