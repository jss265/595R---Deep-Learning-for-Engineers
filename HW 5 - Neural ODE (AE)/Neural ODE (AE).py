'''
HW 5 - Neural ODE (AE)

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

train_size = 20
stage_size = 4
activation = nn.ReLU

# ------- Data Prep -------

# combine data
files = [
    r'HW 5 - Neural ODE (AE)\Daily Climate Delhi 2013-2017\DailyDelhiClimateTest.csv',
    r'HW 5 - Neural ODE (AE)\Daily Climate Delhi 2013-2017\DailyDelhiClimateTrain.csv',
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

# average data over one month
df['date'] = pd.to_datetime(df['date'].str.strip())
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
monthly = df.groupby(['year', 'month'], as_index=False).mean(numeric_only=True)
monthly['index'] = (monthly['year'] - 2013)*12 + monthly['month'] - 1

# normalize data
for col in monthly.columns:
    if col not in ['year', 'month', 'index']:
        monthly[col] = (monthly[col] - monthly[col].mean()) / (monthly[col].std())
    else:
        monthly[col] = monthly[col]
monthly.to_csv(csv_filename, index=False)
print(f'Saved {csv_filename}')

# split into test/train, incrementally
train = monthly[monthly['index'] < train_size]
test = monthly[monthly['index'] >= train_size]

# ------- Build Neural Network -------
class Autoencoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Autoencoder, self).__init__()