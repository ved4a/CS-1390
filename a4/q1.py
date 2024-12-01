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