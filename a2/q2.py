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
print(X.describe())

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

# derivative fun used to update θj:
def derivative(j, theta):
    sum = np.array([0 for i in range(0, n + 1)]) # n is no of features
    for i in range(0, m):
        p = indicator(y[i], j) - phi(j, theta, x.loc[i])
        sum += (x.loc[i] *p)
    gradient = -sum / m # m is no of training examples
    return gradient

# implement actual gradient descent
def gradient_descent(theta, alpha = 0.1, iters = 500):
    for j in range(0, k):
        for iter in range(iters):
            theta[j] = theta[j] - alpha * derivative(j, theta)
    print('Running iterations')
    return theta

# create hypothesis function
def h_theta(x):
    x_matrix = np.matrix(X)
    h_matrix = np.empty((k, 1)) # empty matrix h to store probabilities
    denominator = 0
    for j in range(0, k):
        denominator += math.exp(np.dot(theta_hat[j].T, x_matrix))
    for i in range(0, k):
        h_matrix[i] = math.exp(np.dot(theta_hat[i].T, x_matrix))
    h_matrix = h_matrix/denominator
    return h_matrix

# define features and training examples
n = X.shape[1]
m = X.shape[0]

# define k & map diff species to numbers (0,1,2)
k = len(y.unique())
y = y.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
y.value_counts()

# add column of 1s for intercept term
X[5] = np.ones(X.shape[0])
X.shape

# initialize θ as empty matrix
theta = np.empty((k, n + 1))

# apply gradient descent function
theta_hat = gradient_descent(theta)