import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
import keras as keras
from keras import layers, models, regularizers


# Data Preprocessing
file_path = "airfoil self noise/AirfoilSelfNoise.csv"
data = pd.read_csv(file_path)

# Assign column names
# data.columns = [
#     "Frequency (Hz)", "Angle of Attack (degrees)",
#     "Chord Length (m)", "Free-Stream Velocity (m/s)",
#     "Suction Side Displacement Thickness (m)",
#     "Scaled Sound Pressure Level (SSPL)"
# ]

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Neural Network Architecture
model = models.Sequential([
    layers.Dense(300, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001), input_shape=(X.shape[1],)),
    layers.Dense(300, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

# Model Training
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Model Evaluation
test_mse = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {test_mse[1]}")

y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual SSPL')
plt.ylabel('Predicted SSPL')
plt.title('Predicted vs Actual SSPL')
plt.show()