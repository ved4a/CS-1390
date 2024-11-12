import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# PART A

# set n
n = 500

# generate x1 and x2 values
x1 = np.random.uniform(0, 1, n)
x2 = np.random.uniform(0, 1, n)

# split into classes
classes = (x1**2 > x2**2).astype(int)

# make dataset into pandas df
dataset = pd.DataFrame({'x1': x1, 'x2': x2, 'class': classes})

# PART B

plt.figure(figsize=(8, 6))
plt.scatter(x1[classes == 0], x2[classes == 0], color='blue', label='Class 0', alpha=0.6)
plt.scatter(x1[classes == 1], x2[classes == 1], color='red', label='Class 1', alpha=0.6)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Data Points')
plt.legend()
plt.grid(True)
plt.show()

# PART C

# combine into 1 dataset to run regression
X = np.column_stack((x1, x2))
y = classes

model = LogisticRegression()
model.fit(X, y)

# PART D

# predict class labels
predicted_classes = model.predict(X)

# plot predicted labels
plt.figure(figsize=(8, 6))
plt.scatter(x1[predicted_classes == 0], x2[predicted_classes == 0], color='blue', label='Class 0', alpha=0.6)
plt.scatter(x1[predicted_classes == 1], x2[predicted_classes == 1], color='red', label='Class 1', alpha=0.6)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Predicted Class Labels')
plt.legend()
plt.grid(True)
plt.show()

# PART E

# create new features
x1_squared = x1 ** 2
x2_squared = x2 ** 2
x1_x2 = x1 * x2

# wanna avoid log(0)
log_x1 = np.log(x1 + 1e-10)
log_x2 = np.log(x2 + 1e-10)

# make 1 big dataset
X_extended = np.column_stack((x1, x2, x1_squared, x2_squared, x1_x2, log_x1, log_x2))

# fit logistic regression
model_extended = LogisticRegression()
model_extended.fit(X_extended, y)

# PART F

# predict class labels
predicted_classes_extended = model_extended.predict(X_extended)

# plot
plt.figure(figsize=(8, 6))
plt.scatter(x1[predicted_classes_extended == 0], x2[predicted_classes_extended == 0], color='blue', label='Predicted Class 0', alpha=0.6)
plt.scatter(x1[predicted_classes_extended == 1], x2[predicted_classes_extended == 1], color='red', label='Predicted Class 1', alpha=0.6)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Predicted Class Labels (Part E)')
plt.legend()
plt.grid(True)
plt.show()

# PART G

# fit linear SVM model
svc_linear = SVC(kernel='linear')
svc_linear.fit(X, y)

# predict class labels
predicted_classes_svc = svc_linear.predict(X)

# plot
plt.figure(figsize=(8, 6))
plt.scatter(x1[predicted_classes_svc == 0], x2[predicted_classes_svc == 0], color='blue', label='Predicted Class 0', alpha=0.6)
plt.scatter(x1[predicted_classes_svc == 1], x2[predicted_classes_svc == 1], color='red', label='Predicted Class 1', alpha=0.6)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Predicted Class Labels (Linear SVM)')
plt.legend()
plt.grid(True)
plt.show()