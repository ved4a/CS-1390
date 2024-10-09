from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
# fetch dataset 
real_estate_valuation = fetch_ucirepo(id=477) 
  
# data (as pandas dataframes) 
X_all = real_estate_valuation.data.features

X = X_all[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]
y = real_estate_valuation.data.targets

# split into 3 training / test sets
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=0.5, random_state=42)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X, y, test_size=0.9, random_state=42)

# reset indices
y_train_1 = y_train_1.reset_index(drop=True)
y_train_2 = y_train_2.reset_index(drop=True)
y_train_3 = y_train_3.reset_index(drop=True)

y_test_1 = y_test_1.reset_index(drop=True)
y_test_2 = y_test_2.reset_index(drop=True)
y_test_3 = y_test_3.reset_index(drop=True)

# Univariate Feature Selection
features = ['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
training_errors = {} # dictionary

def calculate_mse(y_true, y_predicted):
    n = len(y_true)
    mse = np.sum((y_true - y_predicted) ** 2) / n
    return mse

def simple_linear_regression(X, y):
    n = len(y)

    X_mean = np.mean(X)
    y_mean = np.mean(y)

    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)

    theta1 = numerator / denominator
    theta0 = y_mean - theta1 * X_mean

    return theta0, theta1

for feature in features:
    X_feature = X_train_1[feature].values
    theta0, theta1 = simple_linear_regression(X_feature, y_train_1.values)  # Use .values to get NumPy array
    
    y_predicted_train = theta0 + theta1 * X_feature
    
    mse_train = calculate_mse(y_train_1.values, y_predicted_train)
    training_errors[feature] = mse_train

# print training error for each lin reg w respective feature
for feature, error in training_errors.items():
    print(f"Training error (MSE) for {feature}: {error:.4f}")

# identify the feature with the lowest training error
best_feature = min(training_errors, key=training_errors.get)
best_error = training_errors[best_feature]
print(f"The feature with lowest MSE is '{best_feature}' with an MSE of {best_error:.4f}.")