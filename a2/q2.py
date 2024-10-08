from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
  
# Fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# Data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets

# Examine structure of data
# check dimensions
print(X.shape)
print(y.shape)

# view first 5 rows
print(X.head())
print(y.head())

# column names + data types
print(X.columns)
print(X.dtypes)

# statistical summary
print(X.describe)

# Check if there any missing values, and how many
missing_X = X.isnull().sum()
missing_Y = y.isnull().sum()

print("Missing values in features:\n", missing_X) # none
print("\nMissing values in targets:\n", missing_Y) # none

# Implementing feature scaling using min-max scaling
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)

# Split data into training and test sets
train, test = train_test_split(iris, test_size=0.3)
train = train.reset_index()
test = test.reset_index()

# Implement a softmax regression classifier from scratch
# write function to calculate φ:
def phi(i, theta, x):
    theta_matrix = np.matrix(theta[i]) # create matrix θ
    x_matrix = np.matrix(x) # create matrix x
    numerator = math.exp(np.dot(theta_matrix.T, x_matrix))
    denominator = 0
    for j in range (0, k):
        theta_j_matrix = np.matrix(theta[j])
        denominator += math.exp(np.dot(theta_j_matrix.T, x_matrix))
    phi_i = numerator / denominator
    return phi_i

# create indicator function:
def indicator(a, b):
    if (a == b):
        return 1
    else:
        return 0
    