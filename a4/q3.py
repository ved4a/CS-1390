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
