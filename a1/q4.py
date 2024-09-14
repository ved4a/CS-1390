import pandas as pd
import numpy as np

# For reading excel files
import openpyxl

# Load dataset into pandas dataframe
file_path = 'a1/real_estate_data.xlsx'
df = pd.read_excel(file_path)

# Extract relevant columns
features = df[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']]
target = df['Y house price of unit area']

# Create matrix X
X = np.c_[np.ones(features.shape[0]), features] # np.c concatanes the columns
y = target.values

# Training data
X_train = X[:325]
y_train = y[:325]

# Test data
X_test = X[325:]
y_test = y[325:]

# Evaluate (estimated) θ = (X^T * X)^(-1) * X^T * y
X_transpose = X_train.T
theta = np.linalg.inv(X_transpose @ X_train) @ X_transpose @ y_train

# Print estimated θ
print('Estimated coefficient values:', theta)

# Print estimated y (testing the model)
y_test_predicted = X_test @ theta
print('Predicted prices:', y_test_predicted)

# Calculate MSE for test set
mse_test = np.mean((y_test_predicted - y_test) ** 2)
print('Mean Squared Error for Test Set:', mse_test)

# Calculate MSE for training set
y_train_predicted = X_train @ theta
mse_train = np.mean((y_train_predicted - y_train) ** 2)
print('Mean Squared Error for Training Set:', mse_train)
