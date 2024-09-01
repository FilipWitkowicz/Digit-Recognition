import os

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# Consts
EPOCHS = 5
BATCH_SIZE = 32
FILE_PATH = "models/Digit_recognition_model.keras"


# Function to display an image and its label
def display_digit(image, label):
    plt.imshow(image, cmap="gray")
    plt.title(f"Label: {label}")
    plt.show()


# Function to normalize the data
def data_normalize(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x - x_min) / (x_max - x_min)
    return x


# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to [0, 1] range
data_normalize(x_train)
data_normalize(x_test)

# Display the first image from the training set
display_digit(x_train[0], y_train[0])

# Build a neural network model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28)),  # Input layer
        tf.keras.layers.Flatten(),  # Flatten Input layer to 1 dimension
        tf.keras.layers.Dense(128, activation="relu"),  # 1st hidden layer
        tf.keras.layers.Dense(64, activation="relu"),  # 2nd hidden layer
        tf.keras.layers.Dense(10, activation="softmax"),  # Output layer
    ]
)

# Compile the model
model.compile(
    optimizer="adam",  # Adam optimizer
    loss="sparse_categorical_crossentropy",  # Loss function for multi-class classification
    metrics=["accuracy"],  # Track accuracy metric
)

# Train the model
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# Save and load the model
tf.keras.models.save_model(model, FILE_PATH)
loaded_model = tf.keras.models.load_model(FILE_PATH)

# Check the loaded model
test_loss, test_acc = loaded_model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy for loaded_model: {test_acc:.4f}")

# Summary of the model
loaded_model.summary()
