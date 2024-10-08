from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np
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