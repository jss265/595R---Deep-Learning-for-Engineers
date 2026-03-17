import numpy as np
from pysr import *  # import this (juliapkg) before torch

# EXAMPLE FROM pysr WEBSITE
# X = 2 * np.random.randn(100, 5)
# y = 2 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 2
# model = PySRRegressor(binary_operators=["+", "-", "*", "/"])
# model.fit(X, y)
# print(model)

data = np.loadtxt('subset.cvs', delimiter=',')
y = data[:,0]  # outputs
X = data[:,1:]  # inputs

model = PySRRegressor(
    binary_operators=['+', '-', '*', '/', '^'],  # ^ is the julia operator for the python ** NOTE you may want to constrain operators like '^' for example next line
    unary_operators=['exp', 'log'],
    constraints={'^': (-1, 1)}  # -1: base can have any complexity, 1: exponent can only have a number
)

model.fit(X, y)

print(model)

# this is level 3 (Symbolic Regression), we will need to go to level 4 because regression doesn't work well will many variables. 
# level 4 uses NNs so we can increase dimensionality
