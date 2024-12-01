import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Loading
file_path = "country data/Country-data.csv"
data = pd.read_csv(file_path)

# Standardization
features = ['child mort', 'health', 'income', 'inflation', 'life expec', 'total fer', 'gdpp']
data.columns = data.columns.str.strip()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features])

# Prinicpal Component Analysis
cov_matrix = np.cov(scaled_data.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

num_components = np.argmax(cumulative_variance_ratio >= 0.8) + 1

print(f"Eigenvalues: {eigenvalues}")
print(f"Explained Variance Ratios: {explained_variance_ratio}")
print(f"Minimum number of components to explain 80% variance: {num_components}")