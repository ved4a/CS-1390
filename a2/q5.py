from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
  
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