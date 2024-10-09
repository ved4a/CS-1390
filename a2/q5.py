from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
import matplotlib.pyplot as plt

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
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    n = len(y_true)
    mse = np.sum((y_true - y_predicted) ** 2) / n
    return mse

for feature in features:
    X_feature = X_train_1[[feature]]
    model = LinearRegression()
    model.fit(X_feature, y_train_1)
    
    y_train_predict = model.predict(X_feature)
    mse_train = calculate_mse(y_train_1, y_train_predict)

    training_errors[feature] = mse_train

# print training error for each linear regression with the respective feature
for feature, error in training_errors.items():
    print(f"Training error for {feature}: {error:.2f}") # 2 d.p.

# Identify the feature with the lowest training error
best_feature = min(training_errors, key=training_errors.get)
best_error = training_errors[best_feature]
print(f"The feature with the lowest MSE is '{best_feature}' with an MSE of {best_error:.2f}.") # 2 d.p.

# Incremental Feature Addition

remaining_features = [f for f in features if f != best_feature]  # everything except X3
selected_features = [best_feature]  # start w X3
incremental_training_errors = {} # dictionary

for _ in range(len(remaining_features)):
    errors_with_new_feature = {}  # dictionary
    
    for feature in remaining_features:
        X_current = X_train_1[selected_features + [feature]]
        
        model = LinearRegression()
        model.fit(X_current, y_train_1)
        
        y_train_predict = model.predict(X_current)
        mse_train = calculate_mse(y_train_1, y_train_predict)
        
        errors_with_new_feature[feature] = mse_train
    
    best_new_feature = min(errors_with_new_feature, key=errors_with_new_feature.get)
    best_new_error = errors_with_new_feature[best_new_feature]
    
    selected_features.append(best_new_feature)
    
    remaining_features.remove(best_new_feature)
    
    incremental_training_errors[', '.join(selected_features)] = best_new_error

    print(f"Added '{best_new_feature}' to the model. New training error: {best_new_error:.2f}")

print("\nIncremental Feature Addition Results:")
for feature_set, error in incremental_training_errors.items():
    print(f"Features: {feature_set} | Training Error: {error:.2f}")

# Model Evaluation
train_test_splits = [(X_train_1, X_test_1, y_train_1, y_test_1),
                     (X_train_2, X_test_2, y_train_2, y_test_2),
                     (X_train_3, X_test_3, y_train_3, y_test_3)]

training_errors_per_split = []
testing_errors_per_split = []

for split_idx, (X_train, X_test, y_train, y_test) in enumerate(train_test_splits):
    print(f"\nEvaluating Train-Test Split {split_idx + 1}:")
    
    remaining_features = [f for f in features if f != best_feature]
    selected_features = [best_feature]
    
    training_errors = []
    testing_errors = []
    
    for step in range(len(remaining_features) + 1):
        X_current_train = X_train[selected_features]
        X_current_test = X_test[selected_features]
        
        model = LinearRegression()
        model.fit(X_current_train, y_train)
        
        # Training error
        y_train_predict = model.predict(X_current_train)
        mse_train = calculate_mse(y_train, y_train_predict)
        training_errors.append(mse_train)
        
        # Testing error
        y_test_predict = model.predict(X_current_test)
        mse_test = calculate_mse(y_test, y_test_predict)
        testing_errors.append(mse_test)
        
        print(f"Step {step + 1}: Features = {selected_features}")
        print(f"  Training Error (MSE): {mse_train:.4f}")
        print(f"  Testing Error (MSE): {mse_test:.4f}\n")
        
        if remaining_features:
            errors_with_new_feature = {}
            
            for feature in remaining_features:
                X_current = X_train[selected_features + [feature]]
                
                model = LinearRegression()
                model.fit(X_current, y_train)
                
                y_train_predict = model.predict(X_current)
                mse_train = calculate_mse(y_train, y_train_predict)
                
                errors_with_new_feature[feature] = mse_train
            
            best_new_feature = min(errors_with_new_feature, key=errors_with_new_feature.get)
            selected_features.append(best_new_feature)
            remaining_features.remove(best_new_feature)
    
    training_errors_per_split.append(training_errors)
    testing_errors_per_split.append(testing_errors)