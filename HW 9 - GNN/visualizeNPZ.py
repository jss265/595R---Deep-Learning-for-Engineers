import numpy as np

data = np.load("HW 9 - GNN/spring_data.npz")

# See what's inside
print(data.files)

# Access each array
for key in data.files:
    print(f"{key}:")
    print(data[key])