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