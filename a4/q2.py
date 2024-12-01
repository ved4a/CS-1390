import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Loading
file_path = "country data/Country-data.csv"
data = pd.read_csv(file_path)

# Standardization
features = ['child_mort', 'health', 'income', 'inflation', 'life_expec', 'total_fer', 'gdpp']
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

# 2D Analysis and Visualization
pca_2d = PCA(n_components=2)
data_2d = pca_2d.fit_transform(scaled_data)

kmeans_2d = KMeans(n_clusters=4, random_state=42)
labels_2d = kmeans_2d.fit_predict(data_2d)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], hue=labels_2d, palette='viridis', s=100)
plt.title("2D PCA - Clustered Data", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.legend(title="Cluster", fontsize=10)
plt.grid()
plt.show()