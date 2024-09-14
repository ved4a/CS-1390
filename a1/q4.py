import pandas as pd
import numpy as np

# for reading excel files
import openpyxl

# load dataset into pandas dataframe
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