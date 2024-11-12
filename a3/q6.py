import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# PART B

plt.figure(figsize=(8, 6))
plt.scatter(x1[classes == 0], x2[classes == 0], color='blue', label='Class 0', alpha=0.6)
plt.scatter(x1[classes == 1], x2[classes == 1], color='red', label='Class 1', alpha=0.6)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Data Points')
plt.legend()
plt.grid(True)
plt.show()