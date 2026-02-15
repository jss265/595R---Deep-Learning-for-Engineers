import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

files = [
    r'HW 5 - Neural ODE (AE)\Daily Climate Delhi 2013-2017\DailyDelhiClimateTest.csv',
    r'HW 5 - Neural ODE (AE)\Daily Climate Delhi 2013-2017\DailyDelhiClimateTrain.csv'
    ]
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# collumns:
#     date
#     meantemp
#     humidity
#     wind_speed
#     meanpressure

# print(dfs[0].shape)
# print(dfs[1].shape)
# print(df.shape)

