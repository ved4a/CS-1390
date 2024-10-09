import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# for reading the csv
import openpyxl

# Load dataset
file_path = 'perceptron_assignment.csv'
df = pd.read_csv(file_path)

# Store columns
X_train = df[['x', 'y']].values
y_train = df['result'].values

# Visualization
# plot Class 0
plt.plot (
    X_train[y_train == 0, 0],
    X_train[y_train == 0, 1],
    marker = ".",
    color = "b",
    markersize = 5,
    linestyle = "",
    label = "Class 0",
)

# plot Class 1
plt.plot (
    X_train[y_train == 1, 0],
    X_train[y_train == 1, 1],
    marker = ".",
    color = "m",
    markersize = 5,
    linestyle = "",
    label = "Class 1",
)

# make visualization pretty
plt.legend(loc = 2)

plt.xlim([-0.2,1.2])
plt.ylim([-0.2,1.2])

plt.xlabel("Feature $x$", fontsize = 12)
plt.ylabel("Feature $y$", fontsize = 12)

plt.grid()
plt.show()