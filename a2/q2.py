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

# Split data into training and test sets
# specify the random state as answer to the ultimate question of life, the universe, and everything
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Implementing feature scaling using min-max scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implement a softmax regression classifier from scratch
# write function to calculate φ:
def phi(i, theta, x, k):
    numerator = np.exp(np.dot(theta[i].T, x))
    denominator = np.sum([np.exp(np.dot(theta[j].T, x)) for j in range(k)]) # sum over all classes
    return numerator / denominator

# create indicator function:
def indicator(a, b):
    return 1 if a == b else 0

# derivative fun used to update θj:
def derivative(j, theta, X, y, m, k):
    gradient_sum = np.zeros(theta.shape[1]) # initialize to 0

    for i in range(m):
        xi = X[i]
        yi = y[i]

        error = indicator(yi, j) - phi(j, theta, xi, k)

        gradient_sum += xi * error

    return -gradient_sum / m # avg over m training examples

# implement actual gradient descent
def gradient_descent(theta, X, y, alpha = 0.1, iters = 500):
    m = X.shape[0] # no of examples
    k = theta.shape[0] # no of classes

    for iteration in range(iters):
        for j in range(k): # update for each class
            gradient = derivative(j, theta, X, y, m, k) # compute grad for class j
            theta[j] -= alpha * gradient

        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Theta Updated.")

    return theta

# create hypothesis function
def h_theta(x, theta):
    k = theta.shape[0] # no of classes
    h_matrix = np.zeros(k) # empty matrix h to store probabilities

    denominator = np.sum([np.exp(np.dot(theta[j].T, x)) for j in range(k)])

    for i in range(k):
        h_matrix[i] = np.exp(np.dot(theta[i].T, x)) / denominator
    
    return h_matrix

# define features and training examples
n = X_train_scaled.shape[1]
m = X_train_scaled.shape[0]
k = len(np.unique(y_train))

# map diff species to numbers (0,1,2)
y = y.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# add column of 1s for intercept term for both training and test sets
X_train_scaled = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
X_test_scaled = np.c_[np.ones(X_test_scaled.shape[0]), X_test_scaled]

# initialize θ as empty matrix
theta = np.empty((k, n + 1))

# apply gradient descent function
theta_hat = gradient_descent(theta, alpha=0.1, iters=500)