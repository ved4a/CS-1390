import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Data Preparation
file_path = "country data/Country-data.csv"
data = pd.read_csv(file_path)

print("Data Shape: ", data.shape)
print("First 5 Rows: \n", data.head())
print("Dataset Description: \n", data.describe())

num_features = data.select_dtypes(include=['float64', 'int64']).columns
data[num_features].hist(figsize=(15, 10), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Histograms of Features", fontsize=16)
plt.show()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[num_features])
scaled_df = pd.DataFrame(scaled_data, columns=num_features)
print("First 5 Rows of Scaled Data: \n", scaled_df.head())

# K-Means Implementation
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

def k_means(data, k=4, max_iters=100, epsilon=1e-6):
    # Randomly initialization
    np.random.seed(42)
    initial_indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[initial_indices]
    
    for iteration in range(max_iters):
        clusters = {}
        for i in range(k):
            clusters[i] = []
        
        for idx, point in enumerate(data):
            distances = euclidean_distance(centroids, point)
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(idx)
        
        new_centroids = []
        for i in range(k):
            if clusters[i]:
                cluster_points = data[clusters[i]]
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(data[np.random.choice(data.shape[0])])
        
        new_centroids = np.array(new_centroids)
        
        if np.all(np.abs(new_centroids - centroids) < epsilon):
            print(f"Converged in {iteration + 1} iterations.")
            break
        
        centroids = new_centroids
    
    cluster_labels = np.zeros(data.shape[0])
    for cluster_idx, indices in clusters.items():
        for idx in indices:
            cluster_labels[idx] = cluster_idx

    return centroids, cluster_labels


def k_means_multiple_initializations(data, k=4, initializations=5):
    best_centroids = None
    best_labels = None
    # Inertia is the sum of squared distances to centroids:
    # inertia = ΣΣ||xi - μj||^2, where first Σ is from j=1 to k, and second Σ is i ∈ cluster j
    best_inertia = float('inf') 

    for i in range(initializations):
        print(f"Initialization {i + 1}:")
        centroids, labels = k_means(data, k)
        
        # Compute inertia
        inertia = sum(
            np.sum((data[np.where(labels == j)] - centroids[j])**2)
            for j in range(k)
        )
        print(f"Inertia: {inertia}")
        
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels

    return best_centroids, best_labels

final_centroids, final_labels = k_means_multiple_initializations(scaled_data)
print("Final Centroids:")
print(final_centroids)
print("Final Labels:")
print(final_labels)