import numpy as np
import struct
from pathlib import Path
from sklearn.model_selection import train_test_split

import numpy as np

# Data Loading and Preprocessing
def read_idx_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, f"Invalid magic number: {magic}"
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return images

def read_idx_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        assert magic == 2049, f"Invalid magic number: {magic}"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

train_images_path = Path("mnist/train-images.idx3-ubyte")
train_labels_path = Path("mnist/train-labels.idx1-ubyte")
test_images_path = Path("mnist/t10k-images.idx3-ubyte")
test_labels_path = Path("mnist/t10k-labels.idx1-ubyte")

train_images = read_idx_images(train_images_path)
train_labels = read_idx_labels(train_labels_path)
test_images = read_idx_images(test_images_path)
test_labels = read_idx_labels(test_labels_path)

binary_train_filter = np.isin(train_labels, [0, 1])
binary_test_filter = np.isin(test_labels, [0, 1])

binary_train_images = train_images[binary_train_filter]
binary_train_labels = train_labels[binary_train_filter]

binary_test_images = test_images[binary_test_filter]
binary_test_labels = test_labels[binary_test_filter]

binary_train_labels = (binary_train_labels == 1).astype(int)
binary_test_labels = (binary_test_labels == 1).astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    binary_train_images, binary_train_labels, test_size=0.2, random_state=42
)

X_train = X_train / 255.0
X_val = X_val / 255.0
binary_test_images = binary_test_images / 255.0

# Neural Network Implementation
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, reg_lambda=0.4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.reg_lambda = reg_lambda

        limit_hidden = np.sqrt(6 / (input_size + hidden_size))
        limit_output = np.sqrt(6 / (hidden_size + output_size))
        self.W1 = np.random.uniform(-limit_hidden, limit_hidden, (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.uniform(-limit_output, limit_output, (hidden_size, output_size))
        self.b2 = np.zeros(output_size)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)  # Activation function
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def compute_loss(self, y, y_hat):
        N = y.shape[0]
        log_probs = -np.log(y_hat[np.arange(N), y])
        data_loss = np.sum(log_probs) / N
        reg_loss = (self.reg_lambda / 2) * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return data_loss + reg_loss

    def backprop(self, X, y, y_hat):
        N = y.shape[0]

        delta2 = y_hat
        delta2[np.arange(N), y] -= 1
        delta2 /= N

        self.dW2 = np.dot(self.a1.T, delta2) + self.reg_lambda * self.W2
        self.db2 = np.sum(delta2, axis=0)

        delta1 = np.dot(delta2, self.W2.T) * (1 - self.a1**2)
        self.dW1 = np.dot(X.T, delta1) + self.reg_lambda * self.W1
        self.db1 = np.sum(delta1, axis=0)

    def update_weights(self, learning_rate):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

# Training and Optimization
def train_nn(nn, X_train, y_train, X_val, y_val, learning_rate, max_iters=500):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience, wait = 10, 0

    for i in range(max_iters):
        y_train_hat = nn.forward(X_train)
        train_loss = nn.compute_loss(y_train, y_train_hat)
        nn.backprop(X_train, y_train, y_train_hat)

        nn.update_weights(learning_rate)

        y_val_hat = nn.forward(X_val)
        val_loss = nn.compute_loss(y_val, y_val_hat)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at iteration {i}")
                break

        if i % 10 == 0:
            print(f"Iteration {i}: Training Loss = {train_loss}, Validation Loss = {val_loss}")

    return train_losses, val_losses

