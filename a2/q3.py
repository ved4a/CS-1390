import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# for reading the csv
import openpyxl

# Load dataset
file_path = 'perceptron_assignment.csv'
df = pd.read_csv(file_path)

# Store columns
X_train = df[['x', 'y']].values
y_train = df['result'].values

# Visualization
# plot Class 0
plt.plot (
    X_train[y_train == 0, 0],
    X_train[y_train == 0, 1],
    marker = ".",
    color = "b",
    markersize = 5,
    linestyle = "",
    label = "Class 0",
)

# plot Class 1
plt.plot (
    X_train[y_train == 1, 0],
    X_train[y_train == 1, 1],
    marker = ".",
    color = "m",
    markersize = 5,
    linestyle = "",
    label = "Class 1",
)

# make visualization pretty
plt.legend(loc = 2)

plt.xlim([-0.2,1.2])
plt.ylim([-0.2,1.2])

plt.xlabel("Feature $x$", fontsize = 12)
plt.ylabel("Feature $y$", fontsize = 12)

plt.grid()
plt.show()

# Implement perceptron
class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = np.array([1.0, 1.0])
        self.bias = 0.0
        self.learning_rate = 0.5
    
    def forward(self, x):
        weighted_sum = np.dot(x, self.weights) + self.bias
        return 1 if weighted_sum > 0 else 0
    
    def update(self, x, true_y):
        prediction = self.forward(x)
        error = true_y - prediction

        self.bias += self.learning_rate * error
        self.weights += self.learning_rate * error * x
        
        return error

def train(model, all_x, all_y):
    iteration = 0
    while True:
        iteration += 1
        error_count = 0
        for x, y in zip(all_x, all_y):
            error = model.update(x, y)
            error_count += abs(error)
        print(f"Iteration {iteration} errors {error_count}")

        # stop if no errors
        if error_count == 0:
            break
    return iteration

perceptron = Perceptron(num_features=2)
num_iterations = train(perceptron, X_train, y_train)

# print results
print(f"Number of iterations until convergence: {num_iterations}")
print(f"Theta: {perceptron.weights}, Bias: {perceptron.bias}")