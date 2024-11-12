import numpy as np
import pandas as pd

# PART A

# set n
n = 500

# generate x1 and x2 values
x1 = np.random.uniform(0, 1, n)
x2 = np.random.uniform(0, 1, n)

# split into classes
classes = (x1**2 > x2**2).astype(int)

# make dataset into pandas df
dataset = pd.DataFrame({'x1': x1, 'x2': x2, 'class': classes})