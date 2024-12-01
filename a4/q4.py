import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
import keras as keras
from keras import layers, models, regularizers


# Data Preprocessing
file_path = "path/to/AirfoilSelfNoise.csv"  # Replace with the correct path
data = pd.read_csv(file_path, header=None)

# Assign column names
data.columns = [
    "Frequency (Hz)", "Angle of Attack (degrees)",
    "Chord Length (m)", "Free-Stream Velocity (m/s)",
    "Suction Side Displacement Thickness (m)",
    "Scaled Sound Pressure Level (SSPL)"
]

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
