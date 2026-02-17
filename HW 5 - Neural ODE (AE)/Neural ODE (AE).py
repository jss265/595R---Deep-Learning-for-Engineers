'''
HW 5 - Neural ODE (AE).Neural ODE (AE)

This code uses an autoencorder to compress monthly weather samples into a latent space to be
solved by an ODE Solver and the decoded into usable predicion data.

It is designed to be managed through the Hyper Parameters and Settings section.
'''

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------- Hyper Parameters and Settings -------
csv_filename = r'HW 5 - Neural ODE (AE)\Normalized Data.csv'

# ------- Data Prep -------

files = [
    r'HW 5 - Neural ODE (AE)\Daily Climate Delhi 2013-2017\DailyDelhiClimateTest.csv',
    r'HW 5 - Neural ODE (AE)\Daily Climate Delhi 2013-2017\DailyDelhiClimateTrain.csv'
    ]
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

columns = [
    'date',
    'meantemp',
    'humidity',
    'wind_speed',
    'meanpressure'
    ]

df_norm = pd.DataFrame()
for col in df.columns:
    if col != 'date':
        df_norm[col] = (df[col] - df[col].mean()) / (df[col].std())
    else:
        df_norm[col] = df[col]
df_norm.to_csv(csv_filename, index=False)
