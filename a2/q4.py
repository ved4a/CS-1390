import pandas as pd
import numpy as np

# for reading the csv
import openpyxl

# Load dataset
file_path = 'perceptron_assignment.csv'
df = pd.read_csv(file_path)

# Store columns
X_train = df[['x', 'y']].values
y_train = df['result'].values

# Compute training errors
def get_training_errors(X, y, thetas):
    errors = []
    for theta in thetas:
        predictions = (X @ theta >= 0).astype(int)
        error = np.mean(predictions != y)
        errors.append(error)
    return errors

# Thetas
thetas = [
    np.array([0, -1]),
    np.array([0.65, -0.22]),
    np.array([0.9, -1]),
    np.array([0.7, -0.5]),
    np.array([0.5, -0.4]),
]

# Calculate training errors
errors = get_training_errors(X_train, y_train, thetas)

# Print errors
for i, error in enumerate(errors, start=1):
    print(f"Training error: {error:.4f}")

n = 500
delta = 0.01
epsilon_hat = 0.0380

# calculate error margin
epsilon = np.sqrt(-np.log(delta/2) / (2 * n))

# calculate bounds
lambda_1 = epsilon_hat - epsilon
lambda_2 = epsilon_hat + epsilon

# Print the results
print(f"Training Error: {epsilon_hat:.4f}")
print(f"Lambda 1: {lambda_1:.4f}")
print(f"Lambda 2: {lambda_2:.4f}")